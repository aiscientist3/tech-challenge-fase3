"""Feature importance, coefficients and SHAP values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import IMAGES_DIR, RANDOM_STATE


def transformed_feature_names(pipeline: Pipeline) -> np.ndarray:
    prep = pipeline.named_steps["prep"]
    return prep.get_feature_names_out()


def permutation_importances(
    pipeline: Pipeline,
    X,
    y,
    n_repeats: int = 5,
    n_samples: int = 15000,
) -> pd.DataFrame:
    if len(X) > n_samples:
        X = X.sample(n=n_samples, random_state=RANDOM_STATE)
        y = y.loc[X.index]
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tree_importances(pipeline: Pipeline) -> pd.DataFrame | None:
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    names = transformed_feature_names(pipeline)
    return (
        pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame | None:
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    names = transformed_feature_names(pipeline)
    coef = clf.coef_.ravel()
    return (
        pd.DataFrame(
            {
                "feature": names,
                "coef": coef,
                "odds_ratio": np.exp(coef),
                "abs_coef": np.abs(coef),
            }
        )
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )


def shap_values_tree(
    pipeline: Pipeline,
    X,
    n_samples: int = 2000,
):
    """TreeExplainer SHAP on a subsample. Returns (shap_values, X_trans, names)."""
    import shap

    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    if not hasattr(clf, "feature_importances_"):
        raise TypeError("SHAP TreeExplainer expects a tree model")
    if len(X) > n_samples:
        X = X.sample(n=n_samples, random_state=RANDOM_STATE)
    Xt = prep.transform(X)
    names = transformed_feature_names(pipeline)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[1]
    return values, Xt, names, explainer


def save_importance_table(frame: pd.DataFrame, stem: str) -> Path:
    path = IMAGES_DIR.parent / "reports" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
