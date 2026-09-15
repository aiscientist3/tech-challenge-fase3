"""Feature groups, leakage-safe drops and derived columns for modeling."""

from __future__ import annotations

import pandas as pd

from src.config import GROUP_COL, LEAKAGE_COLS, TARGET_COL

# Always dropped from X (IDs, pipeline metadata, sampling weight, identifiers).
ALWAYS_DROP = frozenset(LEAKAGE_COLS) | {
    "peso_aluno",  # INEP sampling weight, not a student attribute
    "nome_municipio",  # same grain as GROUP_COL; would reintroduce memorization
    # Same-year municipal aggregate — correlates strongly with the target and
    # risks outcome leakage at municipal grain. Prefer lag1_* instead.
    "nivel_alfabetizacao",
}

# Kept on the modeling frame for reports / ranking, never used as X.
CONTEXT_COLS = frozenset({"meta_alfabetizacao_2025", "nivel_alfabetizacao"})

# Known-constant or redundant Gold columns (also detected at runtime).
REDUNDANT_COLS = (
    "serie",
    "nome_uf",
    "uf_nome_uf",
    "regiao_municipio",
    "uf_regiao_uf",
    "populacao_ano_ref",
    "pib_ano_ref",
    "socio_ano_ref",
    "brasil_meta_alfabetizacao_2024",
    "brasil_meta_alfabetizacao_2025",
    "brasil_meta_alfabetizacao_2026",
    "brasil_meta_alfabetizacao_2027",
    "brasil_meta_alfabetizacao_2028",
    "brasil_meta_alfabetizacao_2029",
    "brasil_meta_alfabetizacao_2030",
    "meta_alfabetizacao_2026",
    "meta_alfabetizacao_2027",
    "meta_alfabetizacao_2028",
    "meta_alfabetizacao_2029",
    "meta_alfabetizacao_2030",
    "uf_meta_alfabetizacao_2025",
    "uf_meta_alfabetizacao_2026",
    "uf_meta_alfabetizacao_2027",
    "uf_meta_alfabetizacao_2028",
    "uf_meta_alfabetizacao_2029",
    "uf_meta_alfabetizacao_2030",
    "pib",  # collinear with populacao / pib_per_capita
    # High-cardinality geography — one-hot explodes; not needed for the baseline.
    "nome_mesorregiao",
    "nome_microrregiao",
)

CATEGORICAL_COLS = ("rede", "nome_regiao", "sigla_uf")

NUMERIC_CANDIDATES = (
    "lag1_taxa_alfabetizacao",
    "lag1_media_portugues",
    "lag1_uf_taxa_alfabetizacao",
    "lag1_uf_media_portugues",
    "meta_alfabetizacao_2024",
    "uf_meta_alfabetizacao_2024",
    "populacao",
    "pib_per_capita",
    "ivs",
    "ivs_infraestrutura_urbana",
    "ivs_capital_humano",
    "ivs_renda_trabalho",
    "capital_uf",
    "amazonia_legal",
    "gap_meta",
)


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    null_cols = [c for c in df.columns if df[c].isna().all()]
    return df.drop(columns=null_cols) if null_cols else df


def constant_columns(df: pd.DataFrame, extra_exclude: set[str] | None = None) -> list[str]:
    skip = set(LEAKAGE_COLS) | {TARGET_COL, GROUP_COL} | (extra_exclude or set())
    return [
        c
        for c in df.columns
        if c not in skip and df[c].nunique(dropna=True) <= 1
    ]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only gap_meta: distance between 2024 goal and prior-year literacy rate."""
    out = df.copy()
    if {"meta_alfabetizacao_2024", "lag1_taxa_alfabetizacao"}.issubset(out.columns):
        out["gap_meta"] = out["meta_alfabetizacao_2024"] - out["lag1_taxa_alfabetizacao"]
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Gold alunos table: drop all-null, add derived features, drop constants."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not in columns")
    out = drop_all_null_columns(df)
    out = add_derived_features(out)
    constants = constant_columns(out)
    drop = [c for c in (*REDUNDANT_COLS, *constants) if c in out.columns]
    drop = list(dict.fromkeys(drop))
    if drop:
        out = out.drop(columns=drop)
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Return (numeric, categorical, high_card).

    `high_card` is always empty in the simplified pipeline (kept for API stability).
    """
    drop = set(ALWAYS_DROP) | set(CONTEXT_COLS) | {TARGET_COL, GROUP_COL}
    available = [c for c in df.columns if c not in drop and not str(c).startswith("_")]
    numeric = [c for c in NUMERIC_CANDIDATES if c in available]
    extra_num = [
        c
        for c in available
        if c not in numeric
        and c not in CATEGORICAL_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = numeric + extra_num
    categorical = [c for c in CATEGORICAL_COLS if c in available]
    leftover_cat = [
        c
        for c in available
        if c not in numeric
        and c not in categorical
        and not pd.api.types.is_numeric_dtype(df[c])
        and int(df[c].nunique(dropna=True)) <= 30
    ]
    categorical = categorical + leftover_cat
    return numeric, categorical, []


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric, categorical, high = feature_lists(df)
    return numeric + categorical + high


def xy_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feats = model_feature_columns(df)
    X = df[feats].copy()
    y = df[TARGET_COL].copy()
    if GROUP_COL not in df.columns:
        raise KeyError(f"Group column '{GROUP_COL}' is required for grouped splits")
    groups = df[GROUP_COL].astype(str)
    return X, y, groups
