# Gold inventory — `br_inep_alfabetizacao`

Path: `s3://{DATALAKE_BUCKET}/gold/br_inep_alfabetizacao/`

Each table is Delta Lake style: `_delta_log/`, `ano=2023/`, `ano=2024/`.

Diagnóstico as-is (S3) vs contrato to-be (samples): ver `reports/gold_as_is_vs_to_be.md`.

| Table | Grain | Role | S3 (aprox.) |
|-------|-------|------|-------------|
| `alunos_features` | aluno | **ML fact** — target `alfabetizado` | ~2.1M rows, 78 cols |
| `contexto_territorio` | município × rede | contexto socioeconômico / lags | ~5.3k rows, 81 cols |
| `indicador_crianca_alfabetizada_municipio` | município × rede | indicador + metas municipais | ~6.5k rows, 44 cols |
| `indicador_crianca_alfabetizada_uf` | UF × rede | indicador + metas estaduais | ~51 rows, 37 cols |

## Target (modelagem)

- **Tabela:** `alunos_features`
- **Coluna:** `alfabetizado` (`0.0` / `1.0`)
- **Granularidade:** aluno (`id_aluno`)

## Leakage candidates (não usar como feature)

- `nivel_alfabetizacao` — derivado da mesma avaliação
- Agregados do **mesmo ano** que reproduzem o resultado (`taxa_*` no nível já conhecido do aluno)
- IDs e metadados de pipeline: `id_aluno`, `_ingestion_*`, `_silver_*`, `_gold_*`, `_batch_id`, `_source_table`, `_join_match`

## Features úteis (candidatas)

- Território: `sigla_uf`, `nome_regiao`, `rede`, `capital_uf`, `amazonia_legal`
- Metas: `meta_alfabetizacao_2024`… (cuidado com ano da predição)
- Socioeconomia: `populacao`, `pib_per_capita`, `ivs*`
- Histórico: `lag1_*` (preferível — menor risco de leakage temporal)

## Env

```env
GOLD_TABLE=alunos_features
GOLD_YEAR=2024
```
