"""Municipal risk ranking and profile clustering from predicted probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import GROUP_COL, RANDOM_STATE, REPORTS_DIR


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
        "lag1_media_portugues",
        "populacao",
        "pib_per_capita",
        "ivs",
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


CLUSTER_FEATURES = (
    "taxa_prevista",
    "lag1_taxa_alfabetizacao",
    "lag1_media_portugues",
    "ivs",
    "pib_per_capita",
    "populacao",
)


def cluster_municipalities(
    risk: pd.DataFrame,
    n_clusters: int | None = None,
    k_range: tuple[int, ...] = (3, 4, 5, 6),
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Group municipalities by socioeconomic/educational profile (not by rank).

    Answers "which regions share similar patterns": clusters are built without
    any geographic feature, so the regional composition of each cluster is an
    empirical finding rather than an artefact of the input.

    Returns (risk + `cluster`, per-cluster profile, diagnostics).
    """
    feats = [c for c in CLUSTER_FEATURES if c in risk.columns]
    if len(feats) < 2 or len(risk) < max(k_range):
        return risk.assign(cluster=np.nan), pd.DataFrame(), {"skipped": True}

    raw = risk[feats].astype(float)
    # Population and income are heavy-tailed; cluster on their log scale.
    for col in ("populacao", "pib_per_capita"):
        if col in raw.columns:
            raw[col] = np.log1p(raw[col].clip(lower=0))

    prep = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    Z = prep.fit_transform(raw)

    candidates = [n_clusters] if n_clusters else list(k_range)
    scores: dict[int, float] = {}
    best_k, best_labels, best_score = candidates[0], None, -1.0
    sample = min(len(Z), 5000)
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Z)
        score = float(
            silhouette_score(Z, labels, sample_size=sample, random_state=random_state)
        )
        scores[k] = round(score, 4)
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score

    out = risk.copy()
    out["cluster"] = best_labels

    agg = {f: (f, "mean") for f in feats}
    profile = out.groupby("cluster", as_index=False).agg(
        n_municipios=("cluster", "size"), **agg
    )
    if "nome_regiao" in out.columns:
        # Dominant region and its share tell whether a cluster is territorial.
        mode = out.groupby("cluster")["nome_regiao"].agg(
            lambda s: s.value_counts().idxmax() if s.notna().any() else np.nan
        )
        share = out.groupby("cluster")["nome_regiao"].agg(
            lambda s: s.value_counts(normalize=True).max() if s.notna().any() else np.nan
        )
        profile["regiao_predominante"] = profile["cluster"].map(mode)
        profile["share_regiao_predominante"] = profile["cluster"].map(share).round(4)
    profile = profile.sort_values("taxa_prevista").reset_index(drop=True)

    diagnostics = {
        "features": feats,
        "k_escolhido": int(best_k),
        "silhouette_por_k": scores,
        "silhouette": round(best_score, 4),
    }
    return out, profile, diagnostics


def region_cluster_crosstab(risk: pd.DataFrame) -> pd.DataFrame:
    """Share of each region's municipalities falling in each cluster."""
    if "cluster" not in risk.columns or "nome_regiao" not in risk.columns:
        return pd.DataFrame()
    work = risk.dropna(subset=["cluster", "nome_regiao"])
    if work.empty:
        return pd.DataFrame()
    tab = pd.crosstab(work["nome_regiao"], work["cluster"].astype(int), normalize="index")
    tab.columns = [f"cluster_{c}" for c in tab.columns]
    return tab.round(4).reset_index()


def save_cluster_tables(
    risk: pd.DataFrame,
    profile: pd.DataFrame,
    directory: Path | None = None,
) -> dict[str, Path]:
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not profile.empty:
        p = directory / "clusters_perfil.csv"
        profile.to_csv(p, index=False)
        paths["perfil"] = p
    cross = region_cluster_crosstab(risk)
    if not cross.empty:
        p = directory / "clusters_por_regiao.csv"
        cross.to_csv(p, index=False)
        paths["regiao"] = p
    return paths


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
