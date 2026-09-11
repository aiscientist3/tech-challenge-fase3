# Modelagem supervisionada — alfabetização

Base: `alunos_features` partição `ano=2024`. Amostra estratificada de **300,000** linhas (config `SAMPLE_N=300000`).

## Problema

Classificação binária: prever se o aluno será considerado **alfabetizado** (`alfabetizado` ∈ {0, 1}). Modelos de regressão contínua não se aplicam. Candidatos: Regressão Logística e Random Forest; comparativos: Dummy, Árvore de Decisão e KNN.

## Pipeline

1. Imputação numérica (`SimpleImputer` mediana) e categórica (`DESCONHECIDO`).
2. Transformação única: `StandardScaler` nas numéricas + One-Hot nas categóricas (`rede`, região, UF).
3. Data leakage: exclusão de IDs/metadados e de `nivel_alfabetizacao` (agregado municipal contemporâneo); split agrupado por `id_municipio`; pré-processamento dentro do `Pipeline`; features de 2023 prevendo 2024; holdout único.
4. `Pipeline(prep, clf)` serializado em `models/*.joblib`.
5. `RandomizedSearchCV` + `StratifiedGroupKFold` (ROC AUC).
6. Seed única, versões fixadas, amostra cacheada em `data/processed/`.

Features no X: **18**. Municípios treino/teste: 4392 / 1099 (sem overlap).

## Comparação de modelos (holdout agrupado)

```
        model  roc_auc   pr_auc       f1   recall  precision    brier  cv_roc_auc
random_forest 0.672617 0.705246 0.659866 0.675441   0.644993 0.225846    0.652687
     logistic 0.670652 0.703196 0.640264 0.622758   0.658783 0.226572    0.644275
         tree 0.663391 0.684570 0.669406 0.722875   0.623303 0.227725    0.645531
          knn 0.642724 0.671555 0.662683 0.714038   0.618219 0.235248    0.621505
        dummy 0.500000 0.536479 0.698323 1.000000   0.536479 0.463521         NaN
```

## Modelo escolhido: `random_forest`

Critério: maior ROC AUC na validação cruzada agrupada (desempate < 0,5 pp favorece a Regressão Logística).

Melhores hiperparâmetros:

```
{'clf__n_estimators': 80, 'clf__min_samples_leaf': 80, 'clf__max_features': 'sqrt', 'clf__max_depth': 16, 'clf__class_weight': 'balanced_subsample'}
```

Métricas no holdout (threshold de 0,5 e threshold orientado a recall):

```
{'threshold': 0.5, 'roc_auc': 0.6726174036524766, 'pr_auc': 0.7052457618691506, 'accuracy': 0.6264359665295702, 'balanced_accuracy': 0.6225792740513119, 'f1': 0.6598657024793388, 'precision': 0.6449926070179235, 'recall': 0.6754409154424261, 'brier': 0.22584634195488984, 'tn': 13034, 'fp': 9844, 'fn': 8594, 'tp': 17885, 'recall_oriented_threshold': 0.44386189763121936, 'recall_oriented_roc_auc': 0.6726174036524766, 'recall_oriented_pr_auc': 0.7052457618691506, 'recall_oriented_accuracy': 0.6223028141904897, 'recall_oriented_balanced_accuracy': 0.6114869613103104, 'recall_oriented_f1': 0.6833684353556627, 'recall_oriented_precision': 0.6209525573355558, 'recall_oriented_recall': 0.7597341289323615, 'recall_oriented_brier': 0.22584634195488984, 'recall_oriented_tn': 10598, 'recall_oriented_fp': 12280, 'recall_oriented_fn': 6362, 'recall_oriented_tp': 20117, 'champion': 'random_forest'}
```

## Data leakage: split aleatório vs agrupado

A mesma Regressão Logística, sem busca de hiperparâmetros, nos dois esquemas de split:

```
{'random_split_roc_auc': 0.6571178813590098, 'grouped_split_roc_auc': 0.6706461961762696, 'auc_inflation': -0.013528314817259801, 'random_municipio_overlap': 4854, 'grouped_municipio_overlap': 0, 'n_municipios_train_grouped': 4392, 'n_municipios_test_grouped': 1099}
```

Overlap de municípios no split aleatório mostra a memorização da média municipal. O split agrupado é o número que generaliza para municípios nunca vistos.

## Interpretação

Importâncias (permutação no holdout), coeficientes/odds da logística e SHAP da floresta estão em `reports/*.csv` e `images/model_*.png`.

## Aplicação estratégica

Ranking municipal de risco em `reports/risco_municipal.csv` (probabilidade média prevista, gap vs `meta_alfabetizacao_2024`/`2025`). Perfil regional em `reports/risco_regional.csv`.

### Regiões com padrões semelhantes (KMeans)

Agrupamento de municípios por perfil educacional e socioeconômico em **k=3** clusters (silhouette 0.2451; testados {3: 0.2451, 4: 0.2353, 5: 0.205, 6: 0.2122}). Variáveis: taxa_prevista, lag1_taxa_alfabetizacao, lag1_media_portugues, ivs, pib_per_capita, populacao — nenhuma delas é geográfica, portanto a composição regional de cada cluster é resultado, não premissa.

```
 cluster  n_municipios  taxa_prevista  lag1_taxa_alfabetizacao  lag1_media_portugues      ivs  pib_per_capita    populacao regiao_predominante  share_regiao_predominante
       0           330       0.414138                41.119724            729.020650 0.438196    18563.285827 29094.163636            Nordeste                     0.6515
       1           439       0.537732                61.230185            749.640233 0.238667    64624.034374 45962.407745                 Sul                     0.3872
       2           330       0.714539                81.006170            777.723591 0.351786    26979.593507 13778.384848            Nordeste                     0.3606
```

Perfil por cluster em `reports/clusters_perfil.csv`, distribuição por região em `reports/clusters_por_regiao.csv`, figura em `images/model_clusters.png`.

## Limitações

- Não há atributos individuais do aluno além da rede/série; o teto de performance é o do risco municipal.
- `nivel_alfabetizacao` foi excluído do X por risco de leakage same-year.
- `lag1_proporcao_aluno_nivel_*` veio 100% nulo na Gold e foi descartado.
- Amostra de 300 mil linhas (não a partição completa de 2,1 milhões).
- KNN treinado em no máximo 50 mil linhas por custo computacional.
- Preprocessamento deliberadamente simples (sem Target/Frequency Encoding).
