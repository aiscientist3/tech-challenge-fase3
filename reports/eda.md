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
1.0            2592      0.5184
0.0            2408      0.4816
```

### Maior % missing (top 10)

```
                   column  pct_missing   dtype
                      ivs        21.50 float64
  meta_alfabetizacao_2024        16.94 float64
ivs_infraestrutura_urbana        15.82 float64
  meta_alfabetizacao_2026        15.60 float64
  meta_alfabetizacao_2025        15.60 float64
      nivel_alfabetizacao        15.60 float64
         regiao_municipio        15.60     str
                _batch_id        15.60     str
            _source_table        15.60     str
  meta_alfabetizacao_2029        15.60 float64
```

### Correlação com o target

```
nivel_alfabetizacao           0.308271
meta_alfabetizacao_2029       0.249869
meta_alfabetizacao_2025       0.249098
meta_alfabetizacao_2028       0.249082
meta_alfabetizacao_2026       0.248654
meta_alfabetizacao_2027       0.248634
meta_alfabetizacao_2024       0.246881
lag1_media_portugues          0.246826
lag1_taxa_alfabetizacao       0.243055
lag1_uf_media_portugues       0.197272
lag1_uf_taxa_alfabetizacao    0.188007
uf_meta_alfabetizacao_2028    0.182436
uf_meta_alfabetizacao_2029    0.182341
uf_meta_alfabetizacao_2027    0.181620
uf_meta_alfabetizacao_2026    0.181358
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

## Revisão da EDA para a modelagem

Amostra **aleatória** entre row groups (seed fixa). Partição de modelagem: `ano=2024`.

- UFs na amostra: **26**; redes: `['estadual', 'municipal']`
- Colunas constantes (nunique ≤ 1): `['serie', 'meta_alfabetizacao_2030', 'uf_meta_alfabetizacao_2030', 'brasil_meta_alfabetizacao_2024', 'brasil_meta_alfabetizacao_2025', 'brasil_meta_alfabetizacao_2026', 'brasil_meta_alfabetizacao_2027', 'brasil_meta_alfabetizacao_2028', 'brasil_meta_alfabetizacao_2029', 'brasil_meta_alfabetizacao_2030', 'populacao_ano_ref', 'pib_ano_ref', 'socio_ano_ref']`
- Features efetivas após limpeza: 17 numéricas, 3 categóricas de baixa cardinalidade, 2 de alta cardinalidade

### `nivel_alfabetizacao` e leakage

`nivel_alfabetizacao` tem no máximo **1** valor distinto por `id_municipio` nesta amostra — é contexto municipal, não o nível do aluno.

O vazamento relevante para o classificador **não** é essa coluna: é o **split aleatório por aluno**. Quase todas as features preditivas são constantes dentro do município. Se o mesmo `id_municipio` aparecer em treino e teste, o modelo memoriza a média municipal.

### Cardinalidade das categóricas

| coluna | n_unique |
|--------|----------|
| `rede` | 2 |
| `nome_regiao` | 5 |
| `sigla_uf` | 26 |
| `nome_mesorregiao` | 135 |
| `nome_microrregiao` | 524 |
| `nome_municipio` | 1840 |

### Missing em `lag1_*`

```
lag1_taxa_alfabetizacao       2.2
lag1_media_portugues          2.2
lag1_uf_taxa_alfabetizacao    2.0
lag1_uf_media_portugues       2.0
```

### Correlação das metas municipais com o target

```
meta_alfabetizacao_2024    0.247
meta_alfabetizacao_2025    0.249
meta_alfabetizacao_2026    0.249
meta_alfabetizacao_2027    0.249
meta_alfabetizacao_2028    0.249
meta_alfabetizacao_2029    0.250
meta_alfabetizacao_2030      NaN
```

Metas 2024–2029 tendem a ser colineares (são interpolações da mesma linha de base). No X usamos `meta_alfabetizacao_2024`.


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
3. Remover `id_aluno` e metadados `_silver_*` / `_gold_*`. `nivel_alfabetizacao` é atributo **municipal** (não do aluno) e pode entrar no X sem leakage direto do target individual.
4. Amostragem **aleatória** entre row groups (não os primeiros registros do parquet).
5. Split **agrupado por `id_municipio`**: as features preditivas são constantes no município; split aleatório infla a métrica.
6. Metas 2025–2030 são colineares com 2024 — manter só `meta_alfabetizacao_2024` no X.
7. Preferir `ano=2024` para ter `lag1_*` preenchidos e coerência temporal (contexto 2023 → resultado 2024).
8. Indicadores município/UF: análise agregada e perguntas de negócio.
