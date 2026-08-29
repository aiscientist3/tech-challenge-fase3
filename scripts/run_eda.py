"""Run full EDA from Gold on S3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GOLD_TABLES, IMAGES_DIR
from src.visualization.eda import (
    build_eda_frame,
    correlation_with_target,
    feature_columns,
    load_eda_data,
    plot_categorical_vs_target,
    plot_correlation_heatmap,
    plot_numeric_distributions,
    plot_target_distribution,
    profile,
    split_feature_types,
    target_balance,
    write_eda_report,
)


def run() -> None:
    print("Loading Gold tables from S3...")
    tables = load_eda_data()
    for table in GOLD_TABLES:
        if table in tables:
            print(f"  {table}: {tables[table].shape}")

    df = build_eda_frame(tables)
    print(f"\nModeling frame: {df.shape}")

    print("\n--- target ---")
    print(target_balance(df))
    print("\n--- profile (top missing) ---")
    print(profile(df).head(10).to_string(index=False))
    print("\n--- corr vs target ---")
    print(correlation_with_target(df))

    feats = feature_columns(df)
    numeric, categorical = split_feature_types(df, feats)
    print(f"\nfeatures: {len(feats)} (num={len(numeric)}, cat={len(categorical)})")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []

    images.append(plot_target_distribution(df))
    p = plot_numeric_distributions(df)
    if p:
        images.append(p)
    p = plot_correlation_heatmap(df)
    if p:
        images.append(p)
    for col in ("rede", "nome_regiao", "sigla_uf"):
        if col in df.columns and df[col].notna().any():
            p = plot_categorical_vs_target(df, col)
            if p:
                images.append(p)

    report = write_eda_report(tables, images, enriched=df)
    print(f"\nreport: {report}")
    for img in images:
        print(f"image: {img}")


if __name__ == "__main__":
    run()
