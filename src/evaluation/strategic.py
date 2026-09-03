"""Municipal risk ranking from predicted literacy probabilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import GROUP_COL, REPORTS_DIR


def municipal_risk_table(
    frame: pd.DataFrame,
    y_true,
    y_score,
    extra_cols: tuple[str, ...] = (
        "sigla_uf",
        "nome_regiao",
        "rede",
        "meta_alfabetizacao_2024",
        "meta_alfabetizacao_2025",
        "lag1_taxa_alfabetizacao",
    ),
) -> pd.DataFrame:
    work = frame[[GROUP_COL]].copy()
    work["y_true"] = pd.Series(y_true).to_numpy()
    work["y_score"] = pd.Series(y_score).to_numpy()
    for col in extra_cols:
        if col in frame.columns:
            work[col] = frame[col].to_numpy()

    agg = {
        "n_alunos": (GROUP_COL, "size"),
        "taxa_observada": ("y_true", "mean"),
        "taxa_prevista": ("y_score", "mean"),
    }
    grouped = work.groupby(GROUP_COL, as_index=False).agg(
        n_alunos=(GROUP_COL, "size"),
        taxa_observada=("y_true", "mean"),
        taxa_prevista=("y_score", "mean"),
    )
    for col in extra_cols:
        if col in work.columns:
            grouped[col] = work.groupby(GROUP_COL)[col].first().to_numpy()

    grouped["risco"] = 1.0 - grouped["taxa_prevista"]
    if "meta_alfabetizacao_2024" in grouped.columns:
        grouped["gap_meta_2024"] = grouped["taxa_prevista"] * 100 - grouped["meta_alfabetizacao_2024"]
        grouped["risco_nao_atingir_meta_2024"] = grouped["gap_meta_2024"] < 0
    if "meta_alfabetizacao_2025" in grouped.columns:
        grouped["gap_meta_2025"] = grouped["taxa_prevista"] * 100 - grouped["meta_alfabetizacao_2025"]
        grouped["risco_nao_atingir_meta_2025"] = grouped["gap_meta_2025"] < 0

    grouped = grouped.sort_values("risco", ascending=False).reset_index(drop=True)
    grouped["rank_risco"] = range(1, len(grouped) + 1)
    return grouped


def region_profile(risk: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("nome_regiao", "sigla_uf") if c in risk.columns]
    if not keys:
        return pd.DataFrame()
    key = keys[0]
    return (
        risk.groupby(key, as_index=False)
        .agg(
            n_municipios=(GROUP_COL, "nunique") if GROUP_COL in risk.columns else ("risco", "size"),
            taxa_prevista_media=("taxa_prevista", "mean"),
            risco_medio=("risco", "mean"),
            n_alunos=("n_alunos", "sum"),
        )
        .sort_values("risco_medio", ascending=False)
    )


def save_risk_tables(risk: pd.DataFrame, directory: Path | None = None) -> dict[str, Path]:
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = directory / "risco_municipal.csv"
    risk.to_csv(p, index=False)
    paths["municipal"] = p
    profile = region_profile(risk)
    if not profile.empty:
        p2 = directory / "risco_regional.csv"
        profile.to_csv(p2, index=False)
        paths["regional"] = p2
    return paths
