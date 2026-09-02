"""Load Gold layer data from the S3 datalake."""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)


def _s3_client():
    return boto3.client("s3", region_name=AWS_DEFAULT_REGION)


def _table_prefix(table: str, year: str | None = None) -> str:
    table = table.strip("/")
    year = GOLD_YEAR if year is None else year
    prefix = f"{GOLD_PREFIX}/{table}/"
    if year:
        prefix = f"{prefix}ano={year}/"
    return prefix


def _table_uri(table: str) -> str:
    return f"s3://{DATALAKE_BUCKET}/{GOLD_PREFIX}/{table.strip('/')}"


def _active_delta_keys(table: str, year: str | None) -> list[str] | None:
    """
    Keys of the files currently active in the Delta transaction log.

    Rewritten partitions leave the superseded parquet on S3 until a VACUUM runs,
    so a plain bucket listing mixes stale batches with the current one.
    Returns None when the table is not readable as Delta.
    """
    try:
        from deltalake import DeltaTable
    except ImportError:
        logger.warning("deltalake not installed — falling back to raw S3 listing.")
        return None

    year = GOLD_YEAR if year is None else year
    try:
        delta_table = DeltaTable(_table_uri(table))
        partitions = [("ano", "=", str(year))] if year else None
        uris = delta_table.file_uris(partitions)
    except Exception as exc:  # not a Delta table, or log unreadable
        logger.warning("Delta log unavailable for '%s' (%s).", table, exc)
        return None

    bucket_uri = f"s3://{DATALAKE_BUCKET}/"
    return sorted(uri.removeprefix(bucket_uri) for uri in uris)


def _list_s3_keys(table: str, year: str | None, suffix: str) -> list[str]:
    """Bucket listing ordered by recency (newest first)."""
    client = _s3_client()
    objects: list[tuple[str, object]] = []
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=DATALAKE_BUCKET, Prefix=_table_prefix(table, year)):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/_delta_log/" in key:
                continue
            if not suffix or key.endswith(suffix):
                objects.append((key, obj["LastModified"]))

    return [key for key, _ in sorted(objects, key=lambda item: item[1], reverse=True)]


def list_gold_objects(
    table: str | None = None,
    year: str | None = None,
    suffix: str = ".parquet",
) -> list[str]:
    """Parquet keys of the current Gold snapshot (Delta log aware)."""
    table = table or GOLD_TABLE
    keys = _active_delta_keys(table, year)
    if keys is not None:
        return keys
    return _list_s3_keys(table, year, suffix)


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
