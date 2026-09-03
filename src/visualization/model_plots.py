"""Plots for model evaluation, interpretability and EDA-style diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import LearningCurveDisplay

from src.config import IMAGES_DIR, RANDOM_STATE


def _path(name: str, path: Path | None) -> Path:
    out = path or (IMAGES_DIR / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_confusion(y_true, y_pred, title: str = "Matriz de confusão", path: Path | None = None) -> Path:
    path = _path("model_confusion.png", path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["não alfabetizado", "alfabetizado"])
    ax.set_yticks([0, 1], labels=["não alfabetizado", "alfabetizado"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Observado")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc(y_true, y_score, path: Path | None = None) -> Path:
    path = _path("model_roc.png", path)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="modelo")
    ax.set_title("Curva ROC (holdout agrupado)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pr(y_true, y_score, path: Path | None = None) -> Path:
    path = _path("model_pr.png", path)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="modelo")
    ax.set_title("Curva Precision-Recall (holdout agrupado)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(y_true, y_score, path: Path | None = None) -> Path:
    path = _path("model_calibration.png", path)
    frac, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfeito")
    ax.plot(mean_pred, frac, "o-", label="modelo")
    ax.set_xlabel("Probabilidade média prevista")
    ax.set_ylabel("Fração positiva observada")
    ax.set_title("Calibração")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_metric_bars(frame: pd.DataFrame, metric: str = "roc_auc", path: Path | None = None) -> Path:
    path = _path("model_compare_auc.png", path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = frame.sort_values(metric)
    ax.barh(ordered["model"], ordered[metric], color="#4c72b0")
    ax.set_xlabel(metric)
    ax.set_title("Comparação de modelos (holdout agrupado)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_importance(frame: pd.DataFrame, title: str, path: Path | None = None, n: int = 15) -> Path:
    path = _path("model_importance.png", path)
    top = frame.head(n).iloc[::-1]
    col = "importance_mean" if "importance_mean" in top.columns else "importance"
    if col not in top.columns:
        col = "abs_coef" if "abs_coef" in top.columns else top.columns[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"].astype(str), top[col], color="#55a868")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_learning_curve(estimator, X, y, groups=None, path: Path | None = None) -> Path | None:
    path = _path("model_learning_curve.png", path)
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        LearningCurveDisplay.from_estimator(
            estimator,
            X,
            y,
            cv=3,
            scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 4),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            ax=ax,
        )
        ax.set_title("Curva de aprendizado (ROC AUC)")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception:
        plt.close("all")
        return None


def plot_leakage_compare(random_auc: float, grouped_auc: float, path: Path | None = None) -> Path:
    path = _path("model_leakage_compare.png", path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["split aleatório", "split agrupado\n(município)"], [random_auc, grouped_auc], color=["#c44e52", "#4c72b0"])
    ax.set_ylabel("ROC AUC (holdout)")
    ax.set_title("Data leakage: memorização municipal")
    ax.set_ylim(0.5, max(0.85, max(random_auc, grouped_auc) + 0.05))
    for i, v in enumerate([random_auc, grouped_auc]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
