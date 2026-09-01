# Análise Exploratória — Alfabetização

> Fonte: **S3** `s3://tech-challenge-2-datalake-prod/gold/br_inep_alfabetizacao/alunos_features/ano=2024/` (`EDA_N_ROWS=5000` para `alunos_features`).

Gold entrega fato + contexto já integrados (Fase 2). Sem join/normalização na Fase 3.

## Objetivo

Compreender o comportamento da Gold, apoiar seleção de features e evitar data leakage na modelagem supervisionada (`alfabetizado`).

## Tabelas analisadas

| Tabela | Linhas | Colunas |
|--------|--------|---------|
| `alunos_features` | 5000 | 79 |
| `contexto_territorio` | 6543 | 81 |
| `indicador_crianca_alfabetizada_municipio` | 6543 | 44 |
| `indicador_crianca_alfabetizada_uf` | 51 | 37 |

**Tabela de modelagem (`alunos_features`):** `(5000, 61)` — `5000` linhas com `sigla_uf` preenchida.


## `alunos_features` (modelagem)

- **Target:** `alfabetizado`
- **Features candidatas:** 51 (39 numéricas, 12 categóricas)
- **Excluídas (leakage/ID):** `id_aluno`, `_ingestion_timestamp`, `_silver_processed_at`, `_silver_batch_id`, `_gold_processed_at`, `_gold_batch_id`, `_source_table`, `_batch_id`, `_join_match`

### Distribuição do target

```
              count  proportion
alfabetizado                   
1.0            2686      0.5372
0.0            2314      0.4628
```

### Maior % missing (top 10)

```
                 column  pct_missing   dtype
                    ivs        21.30 float64
meta_alfabetizacao_2024        16.32 float64
meta_alfabetizacao_2026        15.34 float64
meta_alfabetizacao_2027        15.34 float64
meta_alfabetizacao_2025        15.34 float64
    nivel_alfabetizacao        15.34 float64
       regiao_municipio        15.34     str
              _batch_id        15.34     str
          _source_table        15.34     str
meta_alfabetizacao_2029        15.34 float64
```

### Correlação com o target

```
nivel_alfabetizacao           0.296944
meta_alfabetizacao_2029       0.248107
meta_alfabetizacao_2028       0.247254
meta_alfabetizacao_2025       0.247247
meta_alfabetizacao_2026       0.246833
meta_alfabetizacao_2027       0.246826
lag1_media_portugues          0.245630
meta_alfabetizacao_2024       0.243208
lag1_taxa_alfabetizacao       0.240862
lag1_uf_media_portugues       0.201523
lag1_uf_taxa_alfabetizacao    0.185445
uf_meta_alfabetizacao_2024    0.181993
uf_meta_alfabetizacao_2027    0.181152
uf_meta_alfabetizacao_2026    0.181004
uf_meta_alfabetizacao_2028    0.180968
```

## Outras tabelas (contexto)

### `contexto_territorio`

- Shape amostra: `(6543, 81)`
- Colunas (amostra): `['id_municipio', 'rede', 'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027', 'meta_alfabetizacao_2028', 'meta_alfabetizacao_2029', 'meta_alfabetizacao_2030', 'nivel_alfabetizacao', 'percentual_participacao']...`

Missing (top 5):

```
                         column  pct_missing
   lag1_proporcao_aluno_nivel_3        100.0
lag1_uf_proporcao_aluno_nivel_5        100.0
lag1_uf_proporcao_aluno_nivel_6        100.0
   lag1_proporcao_aluno_nivel_5        100.0
   lag1_proporcao_aluno_nivel_4        100.0
```

### `indicador_crianca_alfabetizada_municipio`

- Shape amostra: `(6543, 44)`
- Colunas (amostra): `['id_municipio', 'rede', 'total_alunos', 'total_peso', 'total_alfabetizados_ponderado', 'proficiencia_media_ponderada', 'taxa_crianca_alfabetizada', 'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027']...`

Missing (top 5):

```
                 column  pct_missing
meta_alfabetizacao_2024        20.04
          gap_meta_2024        20.04
     taxa_alfabetizacao        18.20
meta_alfabetizacao_2026        18.20
meta_alfabetizacao_2027        18.20
```

### `indicador_crianca_alfabetizada_uf`

- Shape amostra: `(51, 37)`
- Colunas (amostra): `['sigla_uf', 'rede', 'total_alunos', 'total_peso', 'total_alfabetizados_ponderado', 'proficiencia_media_ponderada', 'taxa_crianca_alfabetizada', 'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027']...`

Missing (top 5):

```
                 column  pct_missing
meta_alfabetizacao_2024         5.88
          gap_meta_2024         5.88
           total_alunos         0.00
                   rede         0.00
               sigla_uf         0.00
```

## Figuras

- `images/eda_target_distribution.png`
- `images/eda_numeric_distributions.png`
- `images/eda_correlation_heatmap.png`
- `images/eda_rede_vs_target.png`
- `images/eda_nome_regiao_vs_target.png`
- `images/eda_sigla_uf_vs_target.png`

## Hipóteses analíticas

| # | Hipótese | Variáveis | Implicação para modelagem |
|---|----------|-----------|---------------------------|
| H1 | Contexto socioeconômico influencia a chance de alfabetização | `ivs*`, `pib_per_capita`, `populacao` | Incluir no pipeline; imputar missing |
| H2 | Histórico municipal (lag) é preditivo sem vazamento do mesmo evento | `lag1_*` | Preferir lags a taxas do mesmo ano |
| H3 | Região e rede escolar alteram a taxa de alfabetização | `nome_regiao`, `sigla_uf`, `rede`, `amazonia_legal` | Encoding categórico + análise de risco territorial |
| H4 | Metas futuras sozinhas não explicam o aluno; gaps/contexto sim | `meta_alfabetizacao_*` | Usar com cautela; evitar targets derivados |

## Decisões de modelagem (a partir da EDA)

1. Unidade de análise: **aluno** (`GOLD_TABLE`).
2. Target: **`alfabetizado`** (binário 0/1).
3. Remover `id_aluno` e metadados `_silver_*` / `_gold_*`; usar `nivel_alfabetizacao` como feature se a Gold expuser.
4. Pipeline Scikit-learn: imputação numérica + scaling; imputação + one-hot em categóricas.
5. Revalidar correlações e balanceamento em amostra maior antes do treino final.
6. Indicadores município/UF: análise agregada e perguntas de negócio.
