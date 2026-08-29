"""Load and inventory Gold layer data from the S3 datalake or local samples."""

from __future__ import annotations

from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    AWS_DEFAULT_REGION,
    DATALAKE_BUCKET,
    GOLD_PREFIX,
    GOLD_TABLE,
    GOLD_TABLES,
    GOLD_YEAR,
    RAW_DIR,
    ROOT_DIR,
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


def peek_parquet(path: str | Path, n: int = 50) -> pd.DataFrame:
    """Read at most the first n rows of a parquet file (row-group aware)."""
    pf = pq.ParquetFile(path)
    chunks: list[pd.DataFrame] = []
    remaining = n

    for i in range(pf.metadata.num_row_groups):
        if remaining <= 0:
            break
        table = pf.read_row_group(i)
        take = min(remaining, table.num_rows)
        chunks.append(table.slice(0, take).to_pandas())
        remaining -= take

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]


def parquet_schema(path: str | Path) -> dict:
    """Return lightweight parquet metadata without loading all rows."""
    path = Path(path)
    pf = pq.ParquetFile(path)
    return {
        "path": str(path),
        "num_rows": pf.metadata.num_rows,
        "num_columns": pf.metadata.num_columns,
        "num_row_groups": pf.metadata.num_row_groups,
        "columns": list(pf.schema_arrow.names),
    }


def discover_local_parquets() -> list[Path]:
    """Find local part-*.parquet samples in project root or data/raw."""
    candidates = list(ROOT_DIR.glob("part-*.parquet"))
    candidates += list(RAW_DIR.glob("**/*.parquet"))
    # de-dupe preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def infer_table_name(columns: list[str]) -> str | None:
    """Map a local parquet sample to a Gold table by column signature."""
    cols = set(columns)
    if "id_aluno" in cols and "alfabetizado" in cols:
        return "alunos_features"
    if "id_municipio" in cols and "taxa_crianca_alfabetizada" in cols:
        return "indicador_crianca_alfabetizada_municipio"
    if "sigla_uf" in cols and "taxa_crianca_alfabetizada" in cols and "id_municipio" not in cols:
        return "indicador_crianca_alfabetizada_uf"
    if "id_municipio" in cols and "taxa_alfabetizacao" in cols and "populacao" in cols:
        return "contexto_territorio"
    return None


def inventory_local_samples(n: int = 50) -> pd.DataFrame:
    """Summarize local parquet samples (schema + inferred table)."""
    rows = []
    for path in discover_local_parquets():
        meta = parquet_schema(path)
        table = infer_table_name(meta["columns"])
        sample = peek_parquet(path, n=min(5, n))
        rows.append(
            {
                "inferred_table": table,
                "file": path.name,
                "path": str(path),
                "num_rows": meta["num_rows"],
                "num_columns": meta["num_columns"],
                "columns": meta["columns"],
                "sample_preview_cols": list(sample.columns[:12]),
            }
        )
    return pd.DataFrame(rows)


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
    path: Path | None = None,
) -> pd.DataFrame:
    """
    Load a small sample for EDA from a local parquet.

    Prefers an explicit path; otherwise uses the local file inferred as `table`.
    """
    table = table or GOLD_TABLE

    if path is not None:
        return peek_parquet(path, n=n)

    for local in discover_local_parquets():
        meta = parquet_schema(local)
        if infer_table_name(meta["columns"]) == table:
            return peek_parquet(local, n=n)

    raise FileNotFoundError(
        f"No local parquet sample for table '{table}'. "
        f"Place part-*.parquet in the project root or under {RAW_DIR}."
    )


def cache_gold_sample(
    table: str | None = None,
    year: str | None = None,
    max_files: int = 1,
    dest_dir: Path | None = None,
) -> Path:
    """Download a small Gold sample to data/raw for offline EDA."""
    table = table or GOLD_TABLE
    dest = dest_dir or (RAW_DIR / table)
    dest.mkdir(parents=True, exist_ok=True)

    keys = list_gold_objects(table=table, year=year)
    if not keys:
        raise FileNotFoundError(f"No parquet files at {gold_s3_uri(table, year)}.")

    client = _s3_client()
    saved: list[Path] = []
    for key in keys[:max_files]:
        local_path = dest / Path(key).name
        client.download_file(DATALAKE_BUCKET, key, str(local_path))
        saved.append(local_path)

    return saved[0] if len(saved) == 1 else dest
