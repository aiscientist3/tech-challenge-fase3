"""Write reports/modelagem.md from collected experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import GOLD_TABLE, GOLD_YEAR, REPORTS_DIR, SAMPLE_N


def write_modelagem_report(
    compare: pd.DataFrame,
    champion: str,
    champion_metrics: dict,
    leakage: dict,
    best_params: dict,
    n_rows: int,
    n_features: int,
    n_municipios_train: int,
    n_municipios_test: int,
    path: Path | None = None,
) -> Path:
    path = path or (REPORTS_DIR / "modelagem.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    table = compare.to_string(index=False)
    lines = [
        "# Modelagem supervisionada — alfabetização",
        "",
        f"Base: `{GOLD_TABLE}` partição `ano={GOLD_YEAR}`. Amostra estratificada de **{n_rows:,}** linhas (config `SAMPLE_N={SAMPLE_N}`).",
        "",
        "## Problema",
        "",
        "Classificação binária: prever se o aluno será considerado **alfabetizado** (`alfabetizado` ∈ {0, 1}). "
        "Modelos de regressão contínua não se aplicam. Candidatos: Regressão Logística e Random Forest; "
        "comparativos: Dummy, Árvore de Decisão e KNN.",
        "",
        "## Pipeline",
        "",
        "1. Imputação numérica (`SimpleImputer` mediana + indicador de ausência) e categórica (`DESCONHECIDO`).",
        "2. Transformação: `StandardScaler` + One-Hot + Target Encoding (modelos lineares/KNN); "
        "Ordinal + Frequency Encoding (árvores), sem scaling.",
        "3. Data leakage: exclusão de IDs/metadados; split agrupado por `id_municipio`; pré-processamento "
        "dentro do `Pipeline`; Target Encoding cross-fitted; features de 2023 prevendo 2024; holdout único.",
        "4. `Pipeline(prep, clf)` serializado em `models/*.joblib`.",
        "5. `RandomizedSearchCV` + `StratifiedGroupKFold` (ROC AUC).",
        "6. Seed única, versões fixadas, amostra cacheada em `data/processed/`.",
        "",
        f"Features no X: **{n_features}**. Municípios treino/teste: {n_municipios_train} / {n_municipios_test} (sem overlap).",
        "",
        "## Comparação de modelos (holdout agrupado)",
        "",
        "```",
        table,
        "```",
        "",
        f"## Modelo escolhido: `{champion}`",
        "",
        f"Critério: maior ROC AUC na validação cruzada agrupada (desempate < 0,5 pp favorece a Regressão Logística).",
        "",
        "Melhores hiperparâmetros:",
        "",
        "```",
        str(best_params if isinstance(best_params, str) else best_params),
        "```",
        "",
        "Métricas no holdout (threshold de 0,5 e threshold orientado a recall):",
        "",
        "```",
        str(champion_metrics),
        "```",
        "",
        "## Data leakage: split aleatório vs agrupado",
        "",
        "A mesma Regressão Logística, sem busca de hiperparâmetros, nos dois esquemas de split:",
        "",
        "```",
        str(leakage),
        "```",
        "",
        "Overlap de municípios no split aleatório mostra a memorização da média municipal. "
        "O split agrupado é o número que generaliza para municípios nunca vistos.",
        "",
        "## Interpretação",
        "",
        "Importâncias (permutação no holdout), coeficientes/odds da logística e SHAP da floresta "
        "estão em `reports/*.csv` e `images/model_*.png`.",
        "",
        "## Aplicação estratégica",
        "",
        "Ranking municipal de risco em `reports/risco_municipal.csv` (probabilidade média prevista, "
        "gap vs `meta_alfabetizacao_2024`/`2025`). Perfil regional em `reports/risco_regional.csv`.",
        "",
        "## Limitações",
        "",
        "- Não há atributos individuais do aluno além da rede/série; o teto de performance é o do risco municipal.",
        "- `lag1_proporcao_aluno_nivel_*` veio 100% nulo na Gold e foi descartado.",
        "- Amostra de 300 mil linhas (não a partição completa de 2,1 milhões).",
        "- KNN treinado em no máximo 50 mil linhas por custo computacional.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
