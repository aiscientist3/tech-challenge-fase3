# Tech Challenge – Fase 3

Predição e inteligência analítica para alfabetização no Brasil (IAST).

Modelo supervisionado para prever se um aluno será considerado **alfabetizado** (`alfabetizado` = 0/1), a partir da camada Gold da Fase 2.

## Contexto do problema

A alfabetização infantil é um indicador-chave do desenvolvimento educacional. Gestores públicos precisam antecipar risco, identificar municípios vulneráveis e entender quais fatores territoriais e socioeconômicos mais se associam ao resultado. A unidade de análise é o **aluno**; quase todas as features preditivas, porém, são contexto do **município** (IVS, PIB per capita, metas, histórico).

## Objetivo analítico

Classificar `alfabetizado` com uma pipeline Scikit-learn reproduzível, evitar data leakage (em especial a memorização municipal) e produzir ranking de risco utilizável em política pública — não apenas uma métrica alta.

## Base utilizada

Camada Gold `s3://{DATALAKE_BUCKET}/gold/br_inep_alfabetizacao/`:

| Tabela | Papel |
|--------|--------|
| `alunos_features` | fato de modelagem (target + contexto já joinado) |
| `contexto_territorio` | apoio EDA |
| `indicador_crianca_alfabetizada_municipio` | metas, gaps, risco municipal |
| `indicador_crianca_alfabetizada_uf` | visão estadual |

Partição de modelagem: **`ano=2024`** (lags `lag1_*` preenchidos; metas calculadas sobre a linha de 2023, portanto informação prévia). Amostra aleatória de 300 mil linhas, seed 42, cache em `data/processed/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Preencha credenciais AWS no `.env`.

## EDA

```bash
python scripts/run_eda.py
jupyter notebook notebooks/01_eda.ipynb
```

Saídas: `reports/eda.md`, `images/eda_*.png`. A amostragem sorteia row groups (não lê o início do parquet, que é ordenado por município).

## Etapas de modelagem

1. Imputação numérica (mediana + indicador de ausência) e categórica (`DESCONHECIDO`).
2. Transformação por família de modelo: scaling + one-hot + target encoding (logística/KNN); ordinal + frequency encoding (árvores).
3. Data leakage: drop de IDs/metadados; split agrupado por `id_municipio`; pré-processamento só no treino (dentro do `Pipeline`); Target Encoding cross-fitted; coerência 2023→2024; holdout único.
4. `Pipeline(prep, clf)` — o mesmo objeto valida e seria o artefato de produção (`models/*.joblib`).
5. `RandomizedSearchCV` + `StratifiedGroupKFold` (scoring `roc_auc`).
6. Seed única, versões em `requirements.txt`, amostra cacheada.

```bash
python scripts/run_modeling.py
jupyter notebook notebooks/02_modelagem.ipynb
```

## Escolha do algoritmo

Problema de **classificação**. Candidatos: Random Forest e Regressão logística. Comparativos: Dummy, árvore e KNN.

**Escolhido: regressão logística** (`C=0,01`, `class_weight=balanced`). Empate técnico com a Random Forest na CV agrupada (AUC 0,667 vs 0,668, diferença < 0,5 pp); a logística é mais simples e entrega odds ratio. Holdout agrupado: ROC AUC **0,684**, PR AUC **0,717**, F1 **0,656**.

## Métricas de avaliação

ROC AUC, PR AUC, F1, precisão, recall, balanced accuracy, Brier, matriz de confusão, curvas ROC/PR, calibração, curva de aprendizado. Comparação split aleatório vs agrupado: 4.854 municípios apareceriam nos dois lados num split por aluno; o relatório usa só o split agrupado (overlap 0).

Números e tabelas: [`reports/modelagem.md`](reports/modelagem.md), [`reports/model_metrics.json`](reports/model_metrics.json).

## Interpretação e insights

Permutation importance: `nivel_alfabetizacao` (municipal), `rede`, metas e `lag1_*` dominam; IVS/PIB quase não adicionam AUC depois do histórico. SHAP da floresta em `images/model_shap_summary.png`.

No holdout, **690 de 1.099** municípios têm taxa prevista abaixo da meta 2024; maior risco médio na região Norte (`reports/risco_municipal.csv`, `reports/risco_regional.csv`).

O teto de performance é o **risco municipal**: não há atributos individuais do aluno. Correlação máxima com o target ~0,31; AUC ~0,68 no holdout agrupado.

## Limitações

- Gold sem features no grain do aluno (além de rede/série; série é constante = 2).
- `lag1_proporcao_aluno_nivel_*` 100% nulo.
- Amostra de 300k, não os 2,1M da partição.
- KNN em no máximo 50k linhas.

## Aplicação prática para políticas públicas

Priorizar municípios com alta probabilidade de não alfabetização e gap negativo em relação à meta; comparar regiões; usar o modelo como **triagem**, não como substituto da avaliação in loco.

## Evoluções futuras

Enriquecer com Censo Escolar no grain da escola/aluno; usar a partição completa; calibrar por UF; monitoramento temporal 2025+.

## Estrutura

```
data/  notebooks/  src/{preprocessing,modeling,evaluation,visualization}
reports/  images/  models/  scripts/  requirements.txt
```
