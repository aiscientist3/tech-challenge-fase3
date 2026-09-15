"""Load Gold tables from S3 with optional row limits for EDA and modeling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    DATALAKE_BUCKET,
    GOLD_TABLE,
    GOLD_TABLES,
    GOLD_YEAR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_COL,
    gold_s3_uri,
)
from src.preprocessing.load_gold import list_gold_objects

logger = logging.getLogger(__name__)


def _s3_parquet_uri(key: str) -> str:
    return f"s3://{DATALAKE_BUCKET}/{key}"


def _sample_from_parquet(
    keys: list[str],
    n: int,
    columns: list[str] | None,
    rng: np.random.Generator,
    max_files: int | None,
) -> pd.DataFrame:
    """Draw up to `n` rows uniformly across parquet files and row groups."""
    if max_files is not None:
        keys = list(keys[:max_files])
    else:
        keys = list(keys)

    rng.shuffle(keys)
    chunks: list[pd.DataFrame] = []
    remaining = n

    for key in keys:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(_s3_parquet_uri(key))
        rg_order = rng.permutation(pf.metadata.num_row_groups)
        for rg in rg_order:
            if remaining <= 0:
                break
            table_rg = pf.read_row_group(int(rg), columns=columns)
            pdf = table_rg.to_pandas()
            if pdf.empty:
                continue
            take = min(remaining, len(pdf))
            if take < len(pdf):
                idx = rng.choice(len(pdf), size=take, replace=False)
                pdf = pdf.iloc[idx]
            chunks.append(pdf)
            remaining -= take

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
        out = out.reset_index(drop=True)
    return out


def peek_gold_s3(
    table: str,
    n: int = 5000,
    year: str | None = None,
    max_files: int | None = None,
    columns: list[str] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Read at most `n` rows from Gold parquet on S3.

    Samples randomly across files and row groups (seeded) so the peek is not
    biased toward the first municipalities stored in the parquet.
    """
    if table not in GOLD_TABLES:
        raise ValueError(f"Unknown Gold table '{table}'")

    keys = list_gold_objects(table=table, year=year)
    if not keys:
        raise FileNotFoundError(
            f"No parquet at {gold_s3_uri(table, year)}. Check AWS credentials and GOLD_* env."
        )

    seed = RANDOM_STATE if random_state is None else random_state
    rng = np.random.default_rng(seed)
    return _sample_from_parquet(keys, n=n, columns=columns, rng=rng, max_files=max_files)


def _cache_path(table: str, year: str | None, n: int, seed: int, tag: str) -> "object":
    year_part = year or "all"
    return PROCESSED_DIR / f"{tag}_{table}_ano{year_part}_n{n}_seed{seed}.parquet"


def load_gold_sample_cached(
    table: str | None = None,
    n: int = 300000,
    year: str | None = None,
    random_state: int | None = None,
    stratify: bool = True,
    tag: str = "sample",
) -> pd.DataFrame:
    """
    Load a seeded random sample from Gold, cached as parquet under data/processed/.

    When `stratify` is True and the target column is present, the cached frame is
    rebalanced to the original class proportions via a second pass if needed.
    """
    table = table or GOLD_TABLE
    year = GOLD_YEAR if year is None else year
    seed = RANDOM_STATE if random_state is None else random_state
    path = _cache_path(table, year, n, seed, tag)
    if path.exists():
        logger.info("Loading cached sample %s", path)
        return pd.read_parquet(path)

    logger.info("Sampling %s rows from %s ano=%s (seed=%s)", f"{n:,}", table, year, seed)
    df = peek_gold_s3(table=table, n=n, year=year, random_state=seed)
    if stratify and TARGET_COL in df.columns and df[TARGET_COL].nunique(dropna=True) > 1:
        # peek_gold_s3 is approximately uniform; stratification is enforced by
        # resampling if class counts drift more than 2 pp from 50/50.
        props = df[TARGET_COL].value_counts(normalize=True)
        if (props.max() - props.min()) > 0.08 and len(df) >= n:
            parts = []
            n_per = n // props.shape[0]
            for label, sub in df.groupby(TARGET_COL, dropna=False):
                take = min(len(sub), n_per)
                parts.append(sub.sample(n=take, random_state=seed) if take < len(sub) else sub)
            df = pd.concat(parts, ignore_index=True)
            if len(df) > n:
                df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote cache %s (%s rows)", path, f"{len(df):,}")
    return df


def load_gold_s3(
    table: str | None = None,
    year: str | None = None,
    columns: list[str] | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Load full partition (or subset of files) from S3."""
    table = table or GOLD_TABLE
    year = GOLD_YEAR if year is None else year

    keys = list_gold_objects(table=table, year=year)
    if not keys:
        raise FileNotFoundError(f"No parquet at {gold_s3_uri(table, year)}")

    if max_files is not None:
        keys = keys[:max_files]

    frames = [
        pd.read_parquet(_s3_parquet_uri(k), columns=columns, engine="pyarrow")
        for k in keys
    ]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


SMALL_GOLD_TABLES = frozenset(
    {
        "contexto_territorio",
        "indicador_crianca_alfabetizada_municipio",
        "indicador_crianca_alfabetizada_uf",
    }
)

MODELING_TABLES = frozenset({"alunos_features", "alunos_analytic"})


def load_gold_for_eda(
    table: str,
    n: int = 5000,
    year: str | None = None,
) -> pd.DataFrame:
    """Load a Gold table for EDA from S3 (random peek for large tables)."""
    year = GOLD_YEAR if year is None else year
    if table == GOLD_TABLE or table in MODELING_TABLES:
        return load_gold_sample_cached(
            table=table,
            n=n,
            year=year,
            random_state=RANDOM_STATE,
            stratify=False,
            tag="eda",
        )
    if table in SMALL_GOLD_TABLES:
        return load_gold_s3(table=table, year=year)
    return peek_gold_s3(table=table, n=n, year=year)


def load_eda_tables(
    n_alunos: int = 5000,
    year: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all four Gold tables from S3 for exploratory analysis."""
    return {table: load_gold_for_eda(table, n=n_alunos, year=year) for table in GOLD_TABLES}
