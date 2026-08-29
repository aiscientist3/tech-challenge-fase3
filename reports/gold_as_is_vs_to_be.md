# Gold — as-is (S3) vs to-be (contrato alvo)

Bucket: `tech-challenge-2-datalake-prod`  
Prefix: `gold/br_inep_alfabetizacao/`  
Região: `us-east-1`

Os arquivos `*_sample.parquet` (Downloads) são o **contrato alvo** (como deve ficar), não o estado atual do S3.

## Tabelas

| Tabela | Partições | Observação |
|--------|-----------|------------|
| `alunos_features` | `ano=2023`, `ano=2024` | Fato ML (~2.1M × 78 em 2024) |
| `contexto_territorio` | `ano=2023`, `ano=2024` | ~5.3k × 81 |
| `indicador_crianca_alfabetizada_municipio` | `ano=2023`, `ano=2024` | várias parts |
| `indicador_crianca_alfabetizada_uf` | `ano=2023`, `ano=2024` | várias parts |

Coluna `ano` **não** vem no parquet (só no path da partição).

## Diagnóstico as-is (S3 prod)

### `rede` inconsistente (bloqueia o join)

| Tabela | Valores atuais |
|--------|----------------|
| `alunos_features` | códigos `"2"`, `"3"`, `"4"` |
| `contexto_territorio` | só texto `"municipal"` |
| indicadores | códigos `"2"`, `"3"` |

- Join cru `(id_municipio, rede)` aluno → contexto: **0%**
- Após mapear `3→municipal` / `2→estadual`: ~**83–85%** (contexto ainda não tem outras redes)
- Enriquecimento no fato (`populacao`, `pib_*`, `ivs*`, `lag1_*`, `nome_municipio`, `_join_match`): **~100% NaN**

### O que já está ok no S3

- `id_municipio`: string com 7 dígitos
- Target `alfabetizado` (`0.0` / `1.0`) presente
- Partições 2023/2024 existentes

## Contrato to-be (samples alvo)

Referência: `alunos_features_sample.parquet`, `contexto_territorio_sample.parquet`, indicadores `*_sample.parquet`.

| Regra | Esperado |
|-------|----------|
| `rede` | texto padronizado (`municipal`, `estadual`, …) em **todas** as tabelas |
| Join aluno→contexto | materializado na Gold **ou** fato sem colunas vazias + FK 100% coberta |
| `id_municipio` | sempre string 7 dígitos |
| Contexto | todas as redes necessárias (não só municipal) |
| Visão ML | `alunos_analytic` (ou fato limpo): target + features, sem cols de pipeline / leakage |
| Qualidade | dicionário + testes documentando o contrato |

No sample alvo, `rede` já está em texto, `id_municipio` ok e o enriquecimento aluno↔contexto bate nas chaves.

## Mínimo indispensável (Fase 2)

1. Padronizar `rede` em `alunos_features`, `contexto_territorio` e indicadores.
2. Garantir `contexto_territorio` com todas as redes usadas no fato.
3. Refazer join aluno→contexto (preencher NaNs) ou remover colunas vazias do fato + FK coberta.
4. Manter `id_municipio` como string 7 dígitos.
5. Dicionário + testes de qualidade do contrato.
6. (Ideal) Publicar `alunos_analytic` pronta para ML.

## Mapa sugerido de `rede` (códigos → texto)

Confirmar com o dicionário INEP / Silver antes de fixar em produção:

| Código (as-is) | Texto (to-be) |
|----------------|---------------|
| `2` | `estadual` |
| `3` | `municipal` |
| `4` | (confirmar — aparece no fato; ex.: `privada`) |

## Leakage (Fase 3 — modelagem)

Não usar como feature: `nivel_alfabetizacao`, taxas/agregados do **mesmo ano** que vazam o resultado, IDs e metadados `_ingestion_*`, `_silver_*`, `_gold_*`, `_batch_id`, `_source_table`, `_join_match`.
