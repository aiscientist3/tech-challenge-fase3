# Modelagem supervisionada — alfabetização

Base: `alunos_features` partição `ano=2024`. Amostra aleatória de **300.000** linhas (seed 42), cache em `data/processed/`.

## Problema

Classificação binária: prever se o aluno será considerado **alfabetizado** (`alfabetizado` ∈ {0, 1}).

## Pipeline

1. **Imputação** — numéricas: mediana + indicador de ausência; categóricas: `DESCONHECIDO`.
2. **Transformação** — logística/KNN: `StandardScaler` + One-Hot (rede, região, UF) + Target Encoding (meso/microrregião). Árvores: Ordinal + Frequency Encoding, sem scaling.
3. **Data leakage** — drop de IDs/metadados; split agrupado por `id_municipio`; pré-processamento só no treino; Target Encoding com CV interno; features de 2023 prevendo 2024; holdout tocado uma vez.
4. **Integração** — `Pipeline(prep, clf)` serializado em `models/best_model.joblib`.
5. **Treino/validação** — `RandomizedSearchCV` + `StratifiedGroupKFold` (5 folds, ROC AUC).
6. **Replicabilidade** — `RANDOM_STATE=42`, `requirements.txt`, amostra cacheada, `python scripts/run_modeling.py`.

Features no X: **22**. Municípios treino/teste: **4392 / 1099** (overlap = 0).

## Comparação (holdout agrupado)

```
        model  roc_auc   pr_auc      f1  recall  precision  brier  cv_roc_auc
random_forest   0.686    0.719   0.654   0.642      0.666  0.222      0.668
     logistic   0.684    0.717   0.656   0.650      0.663  0.223      0.667
         tree   0.684    0.700   0.644   0.616      0.674  0.223      0.663
          knn   0.652    0.682   0.649   0.674      0.626  0.233      0.630
        dummy   0.500    0.536   0.698   1.000      0.536  0.464        —
```

Logística e Random Forest empatam (< 0,5 pp na CV). KNN fica atrás mesmo com dados normalizados; Dummy confirma que acurácia/F1 sozinhos enganam (prever sempre “alfabetizado” dá F1 alto).

## Modelo escolhido: regressão logística

Empate técnico com a floresta na CV agrupada (0,667 vs 0,668). A logística vence o desempate: é o algoritmo citado no enunciado, tem odds ratio interpretáveis e o ganho da floresta não justifica a complexidade.

- Hiperparâmetros: `C=0.01`, `class_weight=balanced`
- Holdout: ROC AUC **0,684**, PR AUC **0,717**, F1 **0,656**, acurácia **0,635**, Brier **0,223**
- Ponto de operação: limiar 0,5 (um limiar “de recall” com precisão mínima 0,50 colapsa para prever todo mundo alfabetizado, porque a base positiva já é ~54%)

## Data leakage: split aleatório vs agrupado

A mesma logística, sem tuning:

| split | ROC AUC | municípios em treino **e** teste |
|-------|---------|----------------------------------|
| aleatório (aluno) | 0,674 | 4854 |
| agrupado (município) | 0,684 | 0 |

Neste recorte o AUC agrupado **não ficou menor** — a memorização municipal não inflou o número. Mesmo assim o split aleatório é inválido para a pergunta de negócio (“o modelo generaliza para municípios que nunca viu”): 4.854 municípios vazam para os dois lados. O número que reportamos é o do **split agrupado**.

## O que mais pesa na predição

Permutation importance no holdout (queda de ROC AUC):

1. `nivel_alfabetizacao` (contexto municipal, não o nível do aluno)
2. `rede`
3. `uf_meta_alfabetizacao_2024`
4. `lag1_uf_taxa_alfabetizacao` / `lag1_taxa_alfabetizacao`
5. `meta_alfabetizacao_2024`

IVS e PIB per capita, sozinhos, quase não movem o AUC depois que o histórico e a meta já estão no modelo (efeito mediado pelo território). SHAP da floresta em `images/model_shap_summary.png`.

## Aplicação estratégica (holdout, 1.099 municípios)

- Região com maior risco médio previsto: **Norte** (taxa prevista ~42,5%), depois Sul/Nordeste; Sudeste e Centro-Oeste mais altos.
- **690 / 1.099** municípios do holdout têm taxa prevista abaixo da `meta_alfabetizacao_2024`.
- Ranking completo: `reports/risco_municipal.csv`. Perfil regional: `reports/risco_regional.csv`.

O modelo é **triagem territorial**, não diagnóstico do aluno: não há feature individual além da rede.

## Limitações

- Teto de performance municipal (correlação máxima com o target ~0,31).
- `lag1_proporcao_aluno_nivel_*` 100% nulo na Gold.
- Amostra de 300 mil (partição 2024 tem ~2,1 milhões).
- KNN em no máximo 50 mil linhas.

## Como reproduzir

```bash
python scripts/run_modeling.py
```
