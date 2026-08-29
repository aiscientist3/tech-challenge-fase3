# Tech Challenge – Fase 3

Predição e inteligência analítica para alfabetização no Brasil (IAST).

## Contexto

Na Fase 2 foi construída a pipeline de engenharia de dados do Indicador Criança Alfabetizada, materializada na **camada Gold** em S3. Nesta fase, esses dados alimentam análise exploratória e um modelo supervisionado para apoiar decisões de gestores públicos.

## Objetivo analítico

Prever se um aluno será **alfabetizado** ou **não alfabetizado** (`alunos_features.alfabetizado`), com pipeline Scikit-learn reprodutível.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Preencha credenciais AWS no `.env` (ou use `AWS_PROFILE`).

| Variável | Descrição |
|----------|-----------|
| `DATALAKE_BUCKET` | Bucket do datalake |
| `GOLD_PREFIX` | `gold/br_inep_alfabetizacao/` |
| `GOLD_TABLE` | Tabela padrão (`alunos_features`) |
| `GOLD_YEAR` | Partição `ano=` (`2024`) |
| `AWS_DEFAULT_REGION` | Região AWS |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | Credenciais |

## Gold (datalake)

```
gold/br_inep_alfabetizacao/
├── alunos_features/                         # ML — target alfabetizado
├── contexto_territorio/
├── indicador_crianca_alfabetizada_municipio/
└── indicador_crianca_alfabetizada_uf/
```

Cada tabela: `_delta_log/`, `ano=2023/`, `ano=2024/`.

Detalhes: `reports/gold_inventory.md`.

Samples locais `part-*.parquet` na raiz são ignorados pelo Git; o loader lê no máximo N linhas via `peek_parquet` / `load_gold_sample`.

## Como rodar a EDA

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

