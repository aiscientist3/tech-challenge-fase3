# Tech Challenge – Fase 3

Predição e inteligência analítica para alfabetização no Brasil (IAST).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Preencha credenciais AWS no `.env`.

## EDA (Gold no S3)

```powershell
python scripts/run_eda.py
jupyter notebook notebooks/01_eda.ipynb
```

A Gold (Fase 2) entrega fato + contexto integrados — **sem join/normalização na Fase 3**.

| Variável | Descrição |
|----------|-----------|
| `GOLD_TABLE` | Tabela ML (`alunos_features` ou `alunos_analytic`) |
| `EDA_N_ROWS` | Linhas lidas da tabela de alunos no S3 |

Saídas: `reports/eda.md`, `images/eda_*.png`
