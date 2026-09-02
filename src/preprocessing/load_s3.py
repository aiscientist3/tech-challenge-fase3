"""Load Gold tables from S3 with optional row limits for EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    DATALAKE_BUCKET,
    GOLD_TABLE,
    GOLD_TABLES,
    GOLD_YEAR,
    gold_s3_uri,
)
from src.preprocessing.load_gold import list_gold_objects


def _s3_parquet_uri(key: str) -> str:
    return f"s3://{DATALAKE_BUCKET}/{key}"


def peek_gold_s3(
    table: str,
    n: int = 5000,
    year: str | None = None,
    max_files: int = 1,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read at most `n` rows from Gold parquet on S3 (row-group aware).

    Does not download the full dataset into memory.
    """
    if table not in GOLD_TABLES:
        raise ValueError(f"Unknown Gold table '{table}'")

    keys = list_gold_objects(table=table, year=year)
    if not keys:
        raise FileNotFoundError(
            f"No parquet at {gold_s3_uri(table, year)}. Check AWS credentials and GOLD_* env."
        )

    chunks: list[pd.DataFrame] = []
    remaining = n

    for key in keys[:max_files]:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(_s3_parquet_uri(key))
        for rg in range(pf.metadata.num_row_groups):
            if remaining <= 0:
                break
            table_rg = pf.read_row_group(rg, columns=columns)
            take = min(remaining, table_rg.num_rows)
            chunks.append(table_rg.slice(0, take).to_pandas())
            remaining -= take

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]


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


# Small Gold tables — safe to load whole partition for joins / EDA.
SMALL_GOLD_TABLES = frozenset(
    {
        "contexto_territorio",
        "indicador_crianca_alfabetizada_municipio",
        "indicador_crianca_alfabetizada_uf",
    }
)


# Large modeling table — peek; small tables — full partition.
MODELING_TABLES = frozenset({"alunos_features", "alunos_analytic"})


def load_gold_for_eda(
    table: str,
    n: int = 5000,
    year: str | None = None,
) -> pd.DataFrame:
    """Load a Gold table for EDA from S3."""
    year = GOLD_YEAR if year is None else year
    if table == GOLD_TABLE or table in MODELING_TABLES:
        return peek_gold_s3(table=table, n=n, year=year)
    if table in SMALL_GOLD_TABLES:
        return load_gold_s3(table=table, year=year)
    return peek_gold_s3(table=table, n=n, year=year)


def load_eda_tables(
    n_alunos: int = 5000,
    year: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all four Gold tables from S3 for exploratory analysis."""
    return {table: load_gold_for_eda(table, n=n_alunos, year=year) for table in GOLD_TABLES}
