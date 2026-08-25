# Tech Challenge – Fase 3

Predição e inteligência analítica para alfabetização no Brasil (IAST).

## Contexto

Na Fase 2 foi construída a pipeline de engenharia de dados do Indicador Criança Alfabetizada, materializada na **camada Gold** em S3. Nesta fase, esses dados alimentam análise exploratória e um modelo supervisionado para apoiar decisões de gestores públicos.

## Objetivo analítico

Desenvolver um modelo capaz de prever se um aluno será considerado **alfabetizado** ou **não alfabetizado**, com pipeline de Machine Learning reprodutível (Scikit-learn), interpretabilidade e insights aplicáveis a políticas públicas.

## Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/          # cache local da Gold (não versionado)
│   └── processed/    # datasets preparados para ML (não versionado)
├── notebooks/        # EDA e experimentação
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   └── visualization/
├── reports/          # documentação analítica
├── requirements.txt
└── README.md
```

## Base de dados

- **Origem:** camada Gold da Fase 2 no datalake S3  
- **Caminho lógico:** `s3://{DATALAKE_BUCKET}/{GOLD_PREFIX}`  
- **Domínio:** `br_inep_alfabetizacao`  
