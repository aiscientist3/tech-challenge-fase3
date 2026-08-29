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

- **Coluna:** `alfabetizado` (`0.0` / `1.0`)
- **Env:** `GOLD_TABLE=alunos_features` (ou `alunos_analytic`)

## Leakage (excluir do modelo)

- `id_aluno`, metadados `_silver_*` / `_gold_*`

## Feature opcional (Gold Fase 2)

- `nivel_alfabetizacao` — incluir em X quando disponível na tabela ML

## Env

```env
GOLD_TABLE=alunos_features
GOLD_YEAR=2024
EDA_N_ROWS=5000
```
