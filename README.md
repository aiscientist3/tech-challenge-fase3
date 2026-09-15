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
| `alunos_features` | fato de modelagem (`GOLD_TABLE` no `.env`); metadados de pipeline são descartados no `build_model_frame` |
| `alunos_analytic` | mesma granularidade, já sem colunas de pipeline |
| `contexto_territorio` | apoio EDA |
| `indicador_crianca_alfabetizada_municipio` | metas, gaps, risco municipal |
| `indicador_crianca_alfabetizada_uf` | visão estadual |

Partição de modelagem: **`ano=2024`** (lags `lag1_*` preenchidos; metas calculadas sobre a linha de 2023, portanto informação prévia). Amostra aleatória de 300 mil linhas, seed 42, cache em `data/processed/`. Das 79 colunas da Gold sobram **18 features** no X após descartar colunas 100% nulas, constantes, redundantes e de leakage.

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

1. Imputação numérica (mediana) e categórica (`DESCONHECIDO`).
2. Transformação única: `StandardScaler` nas numéricas + One-Hot nas categóricas (`rede`, região, UF).
3. Data leakage: drop de IDs/metadados; exclusão de `nivel_alfabetizacao` (agregado municipal contemporâneo); split agrupado por `id_municipio`; pré-processamento só no treino (dentro do `Pipeline`); coerência 2023→2024; holdout único.
4. `Pipeline(prep, clf)` — o mesmo objeto valida e seria o artefato de produção (`models/*.joblib`).
5. `RandomizedSearchCV` + `StratifiedGroupKFold` (scoring `roc_auc`).
6. Seed única, versões em `requirements.txt`, amostra cacheada.

```bash
python scripts/run_modeling.py
jupyter notebook notebooks/02_modelagem.ipynb
```

## Escolha do algoritmo

Problema de **classificação**. Candidatos: Random Forest e Regressão logística. Comparativos: Dummy, árvore e KNN.

Comparação no holdout agrupado (18 features, 4.392 municípios em treino e 1.099 em teste, sem overlap):

| Modelo | ROC AUC | PR AUC | F1 | Recall | Precisão | Brier | CV AUC |
|--------|---------|--------|-----|--------|----------|-------|--------|
| **random_forest** | **0,673** | 0,705 | 0,660 | 0,675 | 0,645 | 0,226 | **0,653** |
| logistic | 0,671 | 0,703 | 0,640 | 0,623 | 0,659 | 0,227 | 0,644 |
| tree | 0,663 | 0,685 | 0,669 | 0,723 | 0,623 | 0,228 | 0,646 |
| knn | 0,643 | 0,672 | 0,663 | 0,714 | 0,618 | 0,235 | 0,622 |
| dummy | 0,500 | 0,536 | 0,698 | 1,000 | 0,536 | 0,464 | — |

**Escolhido: Random Forest** (`n_estimators=80`, `max_depth=16`, `min_samples_leaf=80`, `max_features='sqrt'`, `class_weight='balanced_subsample'`), por ter o maior ROC AUC na validação cruzada agrupada (0,653 contra 0,644 da logística). A logística fica como modelo interpretável de referência, com odds ratio em `reports/logistic_coefficients.csv`.

Holdout agrupado do campeão: ROC AUC **0,673**, PR AUC **0,705**, F1 **0,660**, acurácia **0,626**, acurácia balanceada 0,623, Brier 0,226.

## Métricas de avaliação

ROC AUC, PR AUC, F1, precisão, recall, balanced accuracy, Brier, matriz de confusão, curvas ROC/PR, calibração, curva de aprendizado.

A base é quase balanceada (53,6% de alfabetizados no holdout), então **acurácia isolada engana**: o `DummyClassifier` que prevê sempre "alfabetizado" atinge 0,536 de acurácia e F1 0,698 com ROC AUC 0,500. O modelo entrega 0,626 de acurácia — 9 pontos percentuais acima desse baseline — e a seleção usa ROC AUC justamente para não premiar o degenerado.

Comparação split aleatório vs agrupado: 4.854 municípios aparecem nos dois lados num split por aluno (AUC 0,657); o split agrupado tem overlap 0 (AUC 0,671). O número reportado é o do split agrupado.

Números e tabelas: [`reports/modelagem.md`](reports/modelagem.md), [`reports/model_metrics.json`](reports/model_metrics.json).

## Interpretação e insights

Permutation importance no holdout (queda de ROC AUC): `lag1_media_portugues` (0,0114) lidera com folga, seguido de `nome_regiao` (0,0059), `sigla_uf` (0,0030), `populacao` (0,0025) e `lag1_taxa_alfabetizacao` (0,0023). IVS e PIB per capita ficam em torno de zero — o efeito socioeconômico já está mediado pelo território e pelo histórico. SHAP da floresta em `images/model_shap_summary.png`.

Ranking municipal de risco no holdout: `reports/risco_municipal.csv`, `reports/risco_regional.csv`. Risco médio previsto por região vai de **0,575 no Norte** a 0,374 no Centro-Oeste. **793 dos 1.099** municípios do holdout ficam abaixo da meta de 2024 e 918 abaixo da de 2025.

### Regiões com padrões semelhantes

KMeans sobre o perfil municipal (`taxa_prevista`, `lag1_taxa_alfabetizacao`, `lag1_media_portugues`, `ivs`, `pib_per_capita`, `populacao`), padronizado, com `k` escolhido por silhouette — **k=3** (silhouette 0,245). Nenhuma variável geográfica entra no agrupamento, então a composição regional de cada cluster é resultado, não premissa.

| Cluster | Municípios | Taxa prevista | IVS | PIB per capita | Região predominante |
|---------|-----------|---------------|-----|----------------|---------------------|
| 0 — alta vulnerabilidade | 330 | 0,414 | 0,438 | 18,6 mil | Nordeste (65%) |
| 1 — renda alta, resultado médio | 439 | 0,538 | 0,239 | 64,6 mil | Sul (39%) |
| 2 — pequenos e bem posicionados | 330 | 0,715 | 0,352 | 27,0 mil | Nordeste (36%) |

Norte (69%) e Nordeste (61%) concentram-se no cluster 0; Sul (72%), Centro-Oeste (59%) e Sudeste (55%) no cluster 1. O achado mais útil é que **o Nordeste não é um bloco homogêneo**: 61% dos seus municípios caem no pior cluster, mas 34% caem no melhor — uma política desenhada por região trataria os dois grupos como se fossem iguais. Detalhe em `reports/clusters_perfil.csv` e `reports/clusters_por_regiao.csv`; figura em `images/model_clusters.png`.

O teto de performance é o **risco municipal**: não há atributos individuais do aluno. Mesmo com AUC individual de 0,673, a taxa prevista por município correlaciona **0,635** com a observada — é como ferramenta de triagem territorial que o modelo se justifica.

## Limitações

- Gold sem features no grain do aluno (além de rede/série; série é constante = 2).
- `nivel_alfabetizacao` excluído do X por risco de leakage same-year.
- `lag1_proporcao_aluno_nivel_*` 100% nulo.
- Amostra de 300k, não os 2,1M da partição.
- KNN em no máximo 50k linhas.
- Preprocessamento deliberadamente simples (sem Target/Frequency Encoding).

## Aplicação prática para políticas públicas

Priorizar municípios com alta probabilidade de não alfabetização e gap negativo em relação à meta; comparar regiões; usar o modelo como **triagem**, não como substituto da avaliação in loco.

## Evoluções futuras

Enriquecer com Censo Escolar no grain da escola/aluno; usar a partição completa; calibrar por UF; monitoramento temporal 2025+.

## Estrutura

```
data/  notebooks/  src/{preprocessing,modeling,evaluation,visualization}
reports/  images/  models/  scripts/  requirements.txt
```
