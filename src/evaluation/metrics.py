"""Holdout metrics and threshold selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import REPORTS_DIR


def predict_scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    return estimator.predict(X).astype(float)


def classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    if prefix:
        return {f"{prefix}{k}": v for k, v in out.items()}
    return out


def recall_oriented_threshold(
    y_true,
    y_score,
    min_precision: float = 0.5,
) -> float:
    """
    Highest-recall threshold whose precision stays at least `min_precision`.
    Falls back to the F1-maximising threshold if none qualify.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    prec_t = precision[:-1]
    rec_t = recall[:-1]
    ok = prec_t >= min_precision
    if ok.any():
        best = int(np.argmax(np.where(ok, rec_t, -1)))
        return float(thresholds[best])
    f1 = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def report_table(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=3)


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    path = path or (REPORTS_DIR / "model_metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
