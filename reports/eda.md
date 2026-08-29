# Análise Exploratória — Alfabetização

> Amostra local (~50 linhas/tabela). Padrões e correlações são **provisórios** e devem ser revalidados em amostra maior / partição completa antes do treino.

## Objetivo

Compreender o comportamento da Gold, apoiar seleção de features e evitar data leakage na modelagem supervisionada (`alfabetizado`).

## Tabelas analisadas

| Tabela | Linhas (amostra) | Colunas |
|--------|------------------|---------|
| `alunos_features` | 50 | 78 |
| `contexto_territorio` | 50 | 81 |
| `indicador_crianca_alfabetizada_municipio` | 50 | 44 |
| `indicador_crianca_alfabetizada_uf` | 50 | 37 |

## `alunos_features` (tabela de ML)

- **Target:** `alfabetizado`
- **Features candidatas:** 67 (57 numéricas, 10 categóricas)
- **Excluídas (leakage/ID):** `id_aluno`, `nivel_alfabetizacao`, `_ingestion_timestamp`, `_silver_processed_at`, `_silver_batch_id`, `_gold_processed_at`, `_gold_batch_id`, `_source_table`, `_batch_id`, `_join_match`

### Qualidade dos dados (S3)

Colunas 100% missing na partição (`lag1_*`, `uf_*`, `brasil_*`, etc.) são removidas em `build_eda_frame()`. Demais gaps refletem a Gold da Fase 2.

### Distribuição do target

```
              count  proportion
alfabetizado                   
1.0              29        0.58
0.0              21        0.42
```

### Maior % missing (top 10)

```
                 column  pct_missing   dtype
meta_alfabetizacao_2029        100.0 float64
meta_alfabetizacao_2028        100.0 float64
meta_alfabetizacao_2026        100.0 float64
meta_alfabetizacao_2027        100.0 float64
meta_alfabetizacao_2024        100.0 float64
meta_alfabetizacao_2025        100.0 float64
       regiao_municipio        100.0  object
                nome_uf        100.0  object
               sigla_uf        100.0  object
         nome_municipio        100.0  object
```

### Correlação com o target (amostra)

```
peso_aluno    0.110892
```

## Outras tabelas (contexto)

### `contexto_territorio`

- Shape amostra: `(50, 81)`
- Colunas (amostra): `['id_municipio', 'rede', 'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027', 'meta_alfabetizacao_2028', 'meta_alfabetizacao_2029', 'meta_alfabetizacao_2030', 'nivel_alfabetizacao', 'percentual_participacao']...`

Missing (top 5):

```
                        column  pct_missing
brasil_meta_alfabetizacao_2030        100.0
    uf_meta_alfabetizacao_2029        100.0
    uf_meta_alfabetizacao_2030        100.0
                    uf_nome_uf        100.0
    uf_percentual_participacao        100.0
```

### `indicador_crianca_alfabetizada_municipio`

- Shape amostra: `(50, 44)`
- Colunas (amostra): `['id_municipio', 'rede', 'total_alunos', 'total_peso', 'total_alfabetizados_ponderado', 'proficiencia_media_ponderada', 'taxa_crianca_alfabetizada', 'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027']...`

Missing (top 5):

```
                 column  pct_missing
meta_alfabetizacao_2024        100.0
meta_alfabetizacao_2025        100.0
meta_alfabetizacao_2026        100.0
     taxa_alfabetizacao        100.0
meta_alfabetizacao_2027        100.0
```

### `indicador_crianca_alfabetizada_uf`

- Shape amostra: `(50, 37)`
- Colunas (amostra): `['sigla_uf', 'rede', 'total_alunos', 'total_peso', 'total_alfabetizados_ponderado', 'proficiencia_media_ponderada', 'taxa_crianca_alfabetizada', 'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027']...`

Missing (top 5):

```
                 column  pct_missing
meta_alfabetizacao_2027        100.0
meta_alfabetizacao_2025        100.0
meta_alfabetizacao_2026        100.0
     taxa_alfabetizacao        100.0
meta_alfabetizacao_2024        100.0
```

## Figuras

- `images/eda_correlation_heatmap.png`
- `images/eda_nome_regiao_vs_target.png`
- `images/eda_numeric_distributions.png`
- `images/eda_rede_vs_target.png`
- `images/eda_sigla_uf_vs_target.png`
- `images/eda_target_distribution.png`
- `images/eda_uf_taxa.png`

## Hipóteses analíticas

| # | Hipótese | Variáveis | Implicação para modelagem |
|---|----------|-----------|---------------------------|
| H1 | Contexto socioeconômico influencia a chance de alfabetização | `ivs*`, `pib_per_capita`, `populacao` | Incluir no pipeline; imputar missing |
| H2 | Histórico municipal (lag) é preditivo sem vazamento do mesmo evento | `lag1_*` | Preferir lags a taxas do mesmo ano |
| H3 | Região e rede escolar alteram a taxa de alfabetização | `nome_regiao`, `sigla_uf`, `rede`, `amazonia_legal` | Encoding categórico + análise de risco territorial |
| H4 | Metas futuras sozinhas não explicam o aluno; gaps/contexto sim | `meta_alfabetizacao_*` | Usar com cautela; evitar targets derivados |

## Decisões de modelagem (a partir da EDA)

1. Unidade de análise: **aluno** (`alunos_features`).
2. Target: **`alfabetizado`** (binário 0/1).
3. Remover `id_aluno` e metadados `_silver_*` / `_gold_*`; usar `nivel_alfabetizacao` como feature se a Gold expuser.
4. Pipeline Scikit-learn: imputação numérica + scaling; imputação + one-hot em categóricas.
5. Revalidar correlações e balanceamento em amostra maior antes do treino final.
6. Tabelas `contexto_territorio` e indicadores município/UF: enriquecimento / perguntas de negócio (risco municipal, metas), não como grain do classificador.
7. Consumir a Gold enriquecida diretamente do S3 (`GOLD_TABLE`, `GOLD_YEAR`).
