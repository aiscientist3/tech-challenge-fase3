"""Project configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = ROOT_DIR / "images"
REPORTS_DIR = ROOT_DIR / "reports"

load_dotenv(ROOT_DIR / ".env")

for _key in (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
):
    if os.getenv(_key, "").strip() == "":
        os.environ.pop(_key, None)

DATALAKE_BUCKET = os.getenv("DATALAKE_BUCKET", "tech-challenge-2-datalake-prod")
GOLD_PREFIX = os.getenv("GOLD_PREFIX", "gold/br_inep_alfabetizacao/").strip("/")
GOLD_TABLE = os.getenv("GOLD_TABLE", "alunos_features").strip()
# When Gold is enriched by Fase 2, set GOLD_TABLE=alunos_analytic (or alunos_features if joined in place).
GOLD_YEAR = os.getenv("GOLD_YEAR", "2024").strip() or None

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
if AWS_DEFAULT_REGION:
    os.environ.setdefault("AWS_DEFAULT_REGION", AWS_DEFAULT_REGION)

# Delta/Gold tables under gold/br_inep_alfabetizacao/
GOLD_TABLES = (
    "alunos_features",
    "contexto_territorio",
    "indicador_crianca_alfabetizada_municipio",
    "indicador_crianca_alfabetizada_uf",
)

TARGET_COL = "alfabetizado"

# Excluded from model features (X) — kept in the dataframe for join/scoring.
ID_COLS = ("id_aluno",)
PIPELINE_META_COLS = (
    "_ingestion_timestamp",
    "_silver_processed_at",
    "_silver_batch_id",
    "_gold_processed_at",
    "_gold_batch_id",
    "_source_table",
    "_batch_id",
    "_join_match",
)

LEAKAGE_COLS = ID_COLS + PIPELINE_META_COLS

# Gold may expose this after Fase 2; when present, include in X via feature_columns().
OPTIONAL_MODEL_FEATURES = ("nivel_alfabetizacao",)

EDA_N_ROWS = int(os.getenv("EDA_N_ROWS", "5000"))


def gold_s3_uri(table: str | None = None, year: str | None = None) -> str:
    """S3 URI for a Gold table (optionally year-partitioned)."""
    table = table or GOLD_TABLE
    year = GOLD_YEAR if year is None else year
    base = f"s3://{DATALAKE_BUCKET}/{GOLD_PREFIX}/{table}"
    if year:
        return f"{base}/ano={year}/"
    return f"{base}/"
