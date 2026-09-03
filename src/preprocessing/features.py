"""Feature groups, leakage-safe drops and derived columns for modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_COL, LEAKAGE_COLS, TARGET_COL

# Always dropped from X (IDs, pipeline metadata, sampling weight, identifiers).
ALWAYS_DROP = frozenset(LEAKAGE_COLS) | {
    "peso_aluno",  # INEP sampling weight, not a student attribute
    "nome_municipio",  # same grain as GROUP_COL; would reintroduce memorization
}

# Kept on the modeling frame for reports / ranking, never used as X.
CONTEXT_COLS = frozenset({"meta_alfabetizacao_2025"})

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
)

LOW_CARD_CATS = ("rede", "nome_regiao", "sigla_uf")
HIGH_CARD_CATS = ("nome_mesorregiao", "nome_microrregiao")

NUMERIC_CANDIDATES = (
    "lag1_taxa_alfabetizacao",
    "lag1_media_portugues",
    "lag1_uf_taxa_alfabetizacao",
    "lag1_uf_media_portugues",
    "meta_alfabetizacao_2024",
    "uf_meta_alfabetizacao_2024",
    "nivel_alfabetizacao",
    "populacao",
    "pib_per_capita",
    "ivs",
    "ivs_infraestrutura_urbana",
    "ivs_capital_humano",
    "ivs_renda_trabalho",
    "capital_uf",
    "amazonia_legal",
    "gap_meta",
    "dist_uf",
    "log_populacao",
    "log_pib_per_capita",
)

LOG_SOURCE = ("populacao", "pib_per_capita")


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
    out = df.copy()
    if {"meta_alfabetizacao_2024", "lag1_taxa_alfabetizacao"}.issubset(out.columns):
        out["gap_meta"] = out["meta_alfabetizacao_2024"] - out["lag1_taxa_alfabetizacao"]
    if {"lag1_taxa_alfabetizacao", "lag1_uf_taxa_alfabetizacao"}.issubset(out.columns):
        out["dist_uf"] = out["lag1_taxa_alfabetizacao"] - out["lag1_uf_taxa_alfabetizacao"]
    if "populacao" in out.columns:
        out["log_populacao"] = np.log1p(out["populacao"].clip(lower=0))
    if "pib_per_capita" in out.columns:
        out["log_pib_per_capita"] = np.log1p(out["pib_per_capita"].clip(lower=0))
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Gold alunos table: drop all-null, add derived features, drop constants."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not in columns")
    out = drop_all_null_columns(df)
    out = add_derived_features(out)
    constants = constant_columns(out)
    drop = [c for c in (*REDUNDANT_COLS, *constants) if c in out.columns]
    # Keep raw populacao / pib_per_capita out of X by dropping after logs exist
    for raw in LOG_SOURCE:
        log_name = f"log_{raw}"
        if raw in out.columns and log_name in out.columns:
            drop.append(raw)
    drop = list(dict.fromkeys(drop))
    if drop:
        out = out.drop(columns=drop)
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, low-card cat, high-card cat) present in the frame."""
    drop = set(ALWAYS_DROP) | set(CONTEXT_COLS) | {TARGET_COL, GROUP_COL}
    available = [c for c in df.columns if c not in drop and not str(c).startswith("_")]
    numeric = [c for c in NUMERIC_CANDIDATES if c in available]
    extra_num = [
        c
        for c in available
        if c not in numeric
        and c not in LOW_CARD_CATS
        and c not in HIGH_CARD_CATS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = numeric + extra_num
    low = [c for c in LOW_CARD_CATS if c in available]
    high = [c for c in HIGH_CARD_CATS if c in available]
    leftover_cat = [
        c
        for c in available
        if c not in numeric and c not in low and c not in high
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    # leftover cats of small cardinality join one-hot / ordinal; large go frequency
    for c in leftover_cat:
        nuniq = int(df[c].nunique(dropna=True))
        if nuniq <= 30:
            low.append(c)
        else:
            high.append(c)
    return numeric, low, high


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric, low, high = feature_lists(df)
    return numeric + low + high


def xy_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feats = model_feature_columns(df)
    X = df[feats].copy()
    y = df[TARGET_COL].copy()
    if GROUP_COL not in df.columns:
        raise KeyError(f"Group column '{GROUP_COL}' is required for grouped splits")
    groups = df[GROUP_COL].astype(str)
    return X, y, groups
