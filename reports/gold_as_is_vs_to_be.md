# Gold — as-is (S3) vs contrato alvo

Bucket: `tech-challenge-2-datalake-prod`
Prefix: `gold/br_inep_alfabetizacao/`
Região: `us-east-1`

Snapshot avaliado: `_gold_batch_id=1a06cc53-5440-40c5-91b1-c670bc3bbd5d` (`2026-09-01T22:45`), Delta version 9.

## Tabelas

| Tabela | Partições | Snapshot atual (`ano=2024`) |
|--------|-----------|------------------------------|
| `alunos_features` | `ano=2023`, `ano=2024` | fato ML, 79 colunas |
| `contexto_territorio` | `ano=2023`, `ano=2024` | 6.543 linhas × 81 |
| `indicador_crianca_alfabetizada_municipio` | `ano=2023`, `ano=2024` | 6.543 × 44 |
| `indicador_crianca_alfabetizada_uf` | `ano=2023`, `ano=2024` | 51 × 37 |

Coluna `ano` **não** vem no parquet (só no path da partição).

## Leitura: sempre pelo `_delta_log`

As tabelas são Delta. Quando a Fase 2 reescreve uma partição, o parquet antigo **continua no S3** como tombstone até um `VACUUM`. Em `ano=2024` chegaram a coexistir 5 arquivos de `alunos_features`, sendo 1 ativo.

Listar o prefixo com `list_objects_v2` e ler o primeiro arquivo devolve um batch obsoleto — foi a causa dos NaN reportados em `populacao`, `pib*`, `uf_*` e `brasil_meta_*`, e também de duplicatas aparentes em `contexto_territorio` (batches empilhados). O loader (`src/preprocessing/load_gold.py`) resolve os arquivos ativos via `DeltaTable.file_uris()`.

## Estado atual (validado no snapshot)

- `rede` em texto (`municipal`, `estadual`, `privada`) em todas as tabelas, inclusive `indicador_crianca_alfabetizada_uf`
- `id_municipio` string 7 dígitos
- `contexto_territorio` no grão `(id_municipio, rede)` — 6.543 linhas para 6.543 chaves, sem duplicata
- `_join_match = True` em 100% da amostra de 5.000 alunos
- Enriquecimento materializado no fato: `populacao`, `pib`, `pib_per_capita`, `nome_municipio`, `sigla_uf`, `uf_*`, `brasil_meta_*`, `lag1_*`
- Target `alfabetizado` (`0.0` / `1.0`) presente

## Missing remanescente em `alunos_features`

| Coluna | Missing | Natureza |
|--------|---------|----------|
| `lag1_proporcao_aluno_nivel_0..8`, `lag1_uf_proporcao_aluno_nivel_0..8` | 100% | fonte INEP não publicou proporções por nível em 2023 (ver abaixo) |
| `ivs` | 21,3% | cobertura IPEA/AVS por município |
| `meta_alfabetizacao_2024` | 16,3% | cobertura de metas municipais INEP |
| `meta_alfabetizacao_2025..2030`, `nivel_alfabetizacao`, `regiao_municipio` | 15,3% | idem |
| `ivs_infraestrutura_urbana` | 15,2% | cobertura IPEA |
| `peso_aluno` | 11,5% | microdado INEP |
| `ivs_capital_humano` | 9,2% | cobertura IPEA |
| `lag1_taxa_alfabetizacao`, `lag1_media_portugues` | 2,5% | sem ano anterior para o município |
| `uf_meta_alfabetizacao_2024`, `lag1_uf_*` | 2,3% | cobertura de metas por UF |
| `ivs_renda_trabalho` | 0,2% | cobertura IPEA |
| `socio_ano_ref` | 0,02% | cobertura IPEA |

`build_eda_frame` descarta as colunas 100% nulas; o restante é missing legítimo e vai para imputação no pipeline.

### Lags do indicador INEP

`lag1_X` no ano N vem do indicador de N-1. Cobertura na Silver:

| Coluna de origem | `ano=2023` | `ano=2024` |
|------------------|-----------|-----------|
| `taxa_alfabetizacao`, `media_portugues` | preenchidas | preenchidas |
| `proporcao_aluno_nivel_0..8` | **vazias** | preenchidas |

Por isso `lag1_taxa_alfabetizacao` funciona em 2024 e `lag1_proporcao_aluno_nivel_*` não: o INEP
só passou a publicar proporções por nível em 2024. Em `ano=2023` todo `lag1_*` é nulo porque 2022
não foi ingerido. Quando a Gold cobrir 2025, essas colunas passam a ser preenchidas.

## Pendências para a Fase 2

1. Documentar no dicionário que `proporcao_aluno_nivel_*` existe a partir de 2024 e que o lag
   correspondente só fica disponível em 2025.
2. Avaliar cobertura IVS/IPEA — confirmar se o gap de 21% é limitação da fonte e documentar no dicionário.
3. Limpar `rede='0'` na Silver dos indicadores (`municipio_indicadores`, `uf_indicadores`, `ano=2024`).
4. Rodar `VACUUM` nas tabelas Gold para eliminar os tombstones acumulados.

## Leakage (Fase 3 — modelagem)

Não usar como feature: `nivel_alfabetizacao`, taxas/agregados do **mesmo ano** que vazam o resultado, IDs e metadados `_ingestion_*`, `_silver_*`, `_gold_*`, `_batch_id`, `_source_table`, `_join_match`.
