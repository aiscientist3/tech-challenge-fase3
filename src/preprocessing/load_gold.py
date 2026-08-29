"""Load Gold layer data from the S3 datalake."""

from __future__ import annotations

import boto3
import pandas as pd

from src.config import (
    AWS_DEFAULT_REGION,
    DATALAKE_BUCKET,
    GOLD_PREFIX,
    GOLD_TABLE,
    GOLD_TABLES,
    GOLD_YEAR,
    gold_s3_uri,
)


def _s3_client():
    return boto3.client("s3", region_name=AWS_DEFAULT_REGION)


def _table_prefix(table: str, year: str | None = None) -> str:
    table = table.strip("/")
    year = GOLD_YEAR if year is None else year
    prefix = f"{GOLD_PREFIX}/{table}/"
    if year:
        prefix = f"{prefix}ano={year}/"
    return prefix


def list_gold_objects(
    table: str | None = None,
    year: str | None = None,
    suffix: str = ".parquet",
) -> list[str]:
    """List parquet keys for a Gold table (skips _delta_log)."""
    table = table or GOLD_TABLE
    client = _s3_client()
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    prefix = _table_prefix(table, year)

    for page in paginator.paginate(Bucket=DATALAKE_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/_delta_log/" in key:
                continue
            if not suffix or key.endswith(suffix):
                keys.append(key)

    return keys


def load_gold(
    table: str | None = None,
    year: str | None = None,
    columns: list[str] | None = None,
    max_files: int | None = 1,
) -> pd.DataFrame:
    """Read Gold parquet files from S3 for a given table/year."""
    table = table or GOLD_TABLE
    if table not in GOLD_TABLES:
        raise ValueError(f"Unknown Gold table '{table}'. Expected one of {GOLD_TABLES}")

    keys = list_gold_objects(table=table, year=year)
    if not keys:
        raise FileNotFoundError(
            f"No parquet files at {gold_s3_uri(table, year)}. "
            "Check credentials and GOLD_* env vars."
        )

    if max_files is not None:
        keys = keys[:max_files]

    frames = [
        pd.read_parquet(f"s3://{DATALAKE_BUCKET}/{key}", columns=columns, engine="pyarrow")
        for key in keys
    ]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_gold_sample(
    table: str | None = None,
    n: int = 50,
    year: str | None = None,
) -> pd.DataFrame:
    """Load a bounded row sample from Gold on S3 (for EDA)."""
    from src.preprocessing.load_s3 import load_gold_for_eda

    return load_gold_for_eda(table or GOLD_TABLE, n=n, year=year)
