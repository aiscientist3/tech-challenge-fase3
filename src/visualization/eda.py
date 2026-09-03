"""Exploratory analysis helpers — clean, reusable, modeling-oriented."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    EDA_N_ROWS,
    GOLD_TABLE,
    GOLD_YEAR,
    IMAGES_DIR,
    LEAKAGE_COLS,
    REPORTS_DIR,
    TARGET_COL,
    gold_s3_uri,
)
from src.preprocessing.load_s3 import load_eda_tables, load_gold_for_eda


def load_eda_data(n_alunos: int | None = None) -> dict[str, pd.DataFrame]:
    """Load Gold tables for EDA from S3."""
    n = n_alunos or EDA_N_ROWS
    return load_eda_tables(n_alunos=n, year=GOLD_YEAR)


def build_eda_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the Gold modeling table (enriched alunos — fato + contexto joined in Gold)."""
    table = GOLD_TABLE if GOLD_TABLE in tables else "alunos_features"
    if table not in tables:
        raise KeyError(f"Table '{table}' not in loaded tables: {list(tables)}")
    df = tables[table].copy()
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        df = df.drop(columns=null_cols)
    return df


def load_sample(table: str, n: int | None = None) -> pd.DataFrame:
    """Load one Gold table from S3 (peek for large tables)."""
    return load_gold_for_eda(table, n=n or EDA_N_ROWS)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level summary: dtype, non-null count, missing %, unique count."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "pct_missing": round(float(s.isna().mean() * 100), 2),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def feature_columns(df: pd.DataFrame, target: str = TARGET_COL) -> list[str]:
    """Candidate features for X — excludes target, id_aluno and pipeline metadata."""
    drop = set(LEAKAGE_COLS) | {target}
    return [
        c
        for c in df.columns
        if c not in drop and not str(c).startswith("_")
    ]


def split_feature_types(
    df: pd.DataFrame, features: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) among features."""
    cols = features if features is not None else feature_columns(df)
    numeric = df[cols].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical


def target_balance(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Class counts and proportions for the supervised target."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in columns")
    counts = df[target].value_counts(dropna=False)
    props = df[target].value_counts(normalize=True, dropna=False)
    return pd.DataFrame({"count": counts, "proportion": props.round(4)})


def correlation_with_target(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    top_n: int = 15,
) -> pd.Series:
    """Pearson correlation of numeric features vs target (sample-level)."""
    numeric, _ = split_feature_types(df)
    if target not in df.columns or not numeric:
        return pd.Series(dtype=float)
    cols = [c for c in numeric if c != target and df[c].notna().any()]
    if not cols:
        return pd.Series(dtype=float)
    corr = (
        df[cols + [target]]
        .corr(numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
    )
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)


def plot_target_distribution(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Bar chart of alfabetizado classes."""
    path = path or (IMAGES_DIR / "eda_target_distribution.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    balance = target_balance(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    balance["count"].plot(kind="bar", ax=ax, color=["#c44e52", "#4c72b0"])
    ax.set_title("Distribuição do target (alfabetizado)")
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel("contagem (amostra)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    path: Path | None = None,
    max_cols: int = 6,
) -> Path | None:
    """Histograms for a small set of numeric features."""
    numeric, _ = split_feature_types(df)
    cols = (cols or numeric)[:max_cols]
    if not cols:
        return None

    path = path or (IMAGES_DIR / "eda_numeric_distributions.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        df[col].dropna().hist(bins=15, ax=ax, color="#4c72b0", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Distribuições numéricas (amostra)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    path: Path | None = None,
    max_cols: int = 12,
) -> Path | None:
    """Heatmap of numeric feature correlations (sample — exploratory only)."""
    numeric, _ = split_feature_types(df)
    # Prefer features most correlated with target when available.
    if TARGET_COL in df.columns and TARGET_COL in df.select_dtypes(include="number").columns:
        ranked = correlation_with_target(df, top_n=max_cols).index.tolist()
        cols = [c for c in ranked if c in numeric][:max_cols]
        if TARGET_COL not in cols:
            cols = cols + [TARGET_COL]
    else:
        cols = numeric[:max_cols]

    if len(cols) < 2:
        return None

    path = path or (IMAGES_DIR / "eda_correlation_heatmap.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Correlação (amostra — provisória)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_categorical_vs_target(
    df: pd.DataFrame,
    col: str,
    path: Path | None = None,
) -> Path | None:
    """Target rate by category for a key categorical feature."""
    if col not in df.columns or TARGET_COL not in df.columns:
        return None

    path = path or (IMAGES_DIR / f"eda_{col}_vs_target.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    rate = (
        df.groupby(col, dropna=False)[TARGET_COL]
        .mean()
        .sort_values(ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    rate.plot(kind="bar", ax=ax, color="#55a868")
    ax.set_title(f"Taxa média de alfabetizado por {col}")
    ax.set_ylabel(f"média({TARGET_COL})")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _modeling_review_section(df: pd.DataFrame) -> list[str]:
    """Review notes that feed the supervised pipeline (constants, collinearity, grain)."""
    from src.preprocessing.features import build_model_frame, constant_columns, feature_lists

    constants = constant_columns(df)
    model_df = build_model_frame(df)
    numeric, low, high = feature_lists(model_df)
    cat_card = []
    for col in ["rede", "nome_regiao", "sigla_uf", "nome_mesorregiao", "nome_microrregiao", "nome_municipio"]:
        if col in df.columns:
            cat_card.append(f"| `{col}` | {int(df[col].nunique(dropna=True))} |")

    nivel_note = ""
    if "nivel_alfabetizacao" in df.columns and "id_municipio" in df.columns:
        max_u = int(df.groupby("id_municipio")["nivel_alfabetizacao"].nunique(dropna=True).max())
        nivel_note = (
            f"`nivel_alfabetizacao` tem no máximo **{max_u}** valor distinto por "
            "`id_municipio` nesta amostra — é contexto municipal, não o nível do aluno."
        )

    lag_cols = [c for c in df.columns if str(c).startswith("lag1_")]
    lag_miss = (
        (df[lag_cols].isna().mean() * 100).round(1).to_string()
        if lag_cols
        else "(sem colunas lag1 nesta amostra)"
    )

    meta_cols = [c for c in df.columns if str(c).startswith("meta_alfabetizacao_")]
    meta_corr = ""
    if meta_cols and TARGET_COL in df.columns:
        meta_corr = (
            df[meta_cols + [TARGET_COL]]
            .corr(numeric_only=True)[TARGET_COL]
            .drop(labels=[TARGET_COL], errors="ignore")
            .round(3)
            .to_string()
        )

    n_uf = int(df["sigla_uf"].nunique(dropna=True)) if "sigla_uf" in df.columns else 0
    redes = sorted(df["rede"].dropna().astype(str).unique().tolist()) if "rede" in df.columns else []

    return [
        "",
        "## Revisão da EDA para a modelagem",
        "",
        "Amostra **aleatória** entre row groups (seed fixa). Partição de modelagem: `ano=2024`.",
        "",
        f"- UFs na amostra: **{n_uf}**; redes: `{redes}`",
        f"- Colunas constantes (nunique ≤ 1): `{constants}`",
        f"- Features efetivas após limpeza: {len(numeric)} numéricas, {len(low)} categóricas de baixa cardinalidade, {len(high)} de alta cardinalidade",
        "",
        "### `nivel_alfabetizacao` e leakage",
        "",
        nivel_note or "Coluna `nivel_alfabetizacao` ausente nesta amostra.",
        "",
        "O vazamento relevante para o classificador **não** é essa coluna: é o **split aleatório por aluno**. Quase todas as features preditivas são constantes dentro do município. Se o mesmo `id_municipio` aparecer em treino e teste, o modelo memoriza a média municipal.",
        "",
        "### Cardinalidade das categóricas",
        "",
        "| coluna | n_unique |",
        "|--------|----------|",
        *cat_card,
        "",
        "### Missing em `lag1_*`",
        "",
        "```",
        lag_miss,
        "```",
        "",
        "### Correlação das metas municipais com o target",
        "",
        "```",
        meta_corr or "(sem metas numéricas)",
        "```",
        "",
        "Metas 2024–2029 tendem a ser colineares (são interpolações da mesma linha de base). No X usamos `meta_alfabetizacao_2024`.",
        "",
    ]


def write_eda_report(
    tables: dict[str, pd.DataFrame],
    image_paths: list[Path],
    path: Path | None = None,
    enriched: pd.DataFrame | None = None,
) -> Path:
    """Write reports/eda.md with findings and modeling hypotheses."""
    from src.config import EDA_N_ROWS, GOLD_TABLE, GOLD_YEAR

    path = path or (REPORTS_DIR / "eda.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    eda_df = enriched if enriched is not None else tables.get(GOLD_TABLE) or tables.get("alunos_features")
    gold_uri = gold_s3_uri(GOLD_TABLE, GOLD_YEAR)
    lines: list[str] = [
        "# Análise Exploratória — Alfabetização",
        "",
        f"> Fonte: **S3** `{gold_uri}` (`EDA_N_ROWS={EDA_N_ROWS}` para `{GOLD_TABLE}`).",
        "",
        "Gold entrega fato + contexto já integrados (Fase 2). Sem join/normalização na Fase 3.",
        "",
        "## Objetivo",
        "",
        "Compreender o comportamento da Gold, apoiar seleção de features e "
        "evitar data leakage na modelagem supervisionada (`alfabetizado`).",
        "",
        "## Tabelas analisadas",
        "",
        "| Tabela | Linhas | Colunas |",
        "|--------|--------|---------|",
    ]

    for name, df in tables.items():
        lines.append(f"| `{name}` | {len(df)} | {df.shape[1]} |")

    if enriched is not None and "sigla_uf" in enriched.columns:
        filled = int(enriched["sigla_uf"].notna().sum())
        lines.extend(
            [
                "",
                f"**Tabela de modelagem (`{GOLD_TABLE}`):** `{enriched.shape}` — "
                f"`{filled}` linhas com `sigla_uf` preenchida.",
                "",
            ]
        )

    lines.extend(["", f"## `{GOLD_TABLE}` (modelagem)", ""])

    if eda_df is not None:
        feats = feature_columns(eda_df)
        num, cat = split_feature_types(eda_df, feats)
        bal = target_balance(eda_df)
        prof = profile(eda_df)
        top_missing = prof.head(10)
        corr = correlation_with_target(eda_df)

        lines.extend(
            [
                f"- **Target:** `{TARGET_COL}`",
                f"- **Features candidatas:** {len(feats)} "
                f"({len(num)} numéricas, {len(cat)} categóricas)",
                f"- **Excluídas (leakage/ID):** {', '.join(f'`{c}`' for c in LEAKAGE_COLS if c in eda_df.columns)}",
                "",
                "### Distribuição do target",
                "",
                "```",
                bal.to_string(),
                "```",
                "",
                "### Maior % missing (top 10)",
                "",
                "```",
                top_missing[["column", "pct_missing", "dtype"]].to_string(index=False),
                "```",
                "",
                "### Correlação com o target",
                "",
                "```",
                corr.to_string() if len(corr) else "(poucas colunas numéricas preenchidas)",
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Outras tabelas (contexto)",
            "",
        ]
    )
    for name, df in tables.items():
        if name == GOLD_TABLE or name == "alunos_features":
            continue
        miss = profile(df).head(5)
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- Shape amostra: `{df.shape}`",
                f"- Colunas (amostra): `{list(df.columns)[:12]}...`",
                "",
                "Missing (top 5):",
                "",
                "```",
                miss[["column", "pct_missing"]].to_string(index=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Figuras",
            "",
        ]
    )
    for img in image_paths:
        rel = img.as_posix().split("images/")[-1] if "images" in img.as_posix() else img.name
        lines.append(f"- `images/{rel}`")

    if eda_df is not None:
        lines.extend(_modeling_review_section(eda_df))

    lines.extend(
        [
            "",
            "## Hipóteses analíticas",
            "",
            "| # | Hipótese | Variáveis | Implicação para modelagem |",
            "|---|----------|-----------|---------------------------|",
            "| H1 | Contexto socioeconômico influencia a chance de alfabetização | `ivs*`, `pib_per_capita`, `populacao` | Incluir no pipeline; imputar missing |",
            "| H2 | Histórico municipal (lag) é preditivo sem vazamento do mesmo evento | `lag1_*` | Preferir lags a taxas do mesmo ano |",
            "| H3 | Região e rede escolar alteram a taxa de alfabetização | `nome_regiao`, `sigla_uf`, `rede`, `amazonia_legal` | Encoding categórico + análise de risco territorial |",
            "| H4 | Metas futuras sozinhas não explicam o aluno; gaps/contexto sim | `meta_alfabetizacao_*` | Usar com cautela; evitar targets derivados |",
            "",
            "## Decisões de modelagem (a partir da EDA)",
            "",
            "1. Unidade de análise: **aluno** (`GOLD_TABLE`).",
            "2. Target: **`alfabetizado`** (binário 0/1).",
            "3. Remover `id_aluno` e metadados `_silver_*` / `_gold_*`. `nivel_alfabetizacao` é atributo **municipal** (não do aluno) e pode entrar no X sem leakage direto do target individual.",
            "4. Amostragem **aleatória** entre row groups (não os primeiros registros do parquet).",
            "5. Split **agrupado por `id_municipio`**: as features preditivas são constantes no município; split aleatório infla a métrica.",
            "6. Metas 2025–2030 são colineares com 2024 — manter só `meta_alfabetizacao_2024` no X.",
            "7. Preferir `ano=2024` para ter `lag1_*` preenchidos e coerência temporal (contexto 2023 → resultado 2024).",
            "8. Indicadores município/UF: análise agregada e perguntas de negócio.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
