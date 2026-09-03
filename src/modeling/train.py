"""Train candidate classifiers with grouped CV and persist pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    KNN_MAX_ROWS,
    MODELS_DIR,
    N_ITER_SEARCH,
    N_JOBS,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.modeling.preprocessing import build_preprocessor
from src.modeling.split import grouped_cv
from src.preprocessing.features import feature_lists

logger = logging.getLogger(__name__)

FAMILIES = {
    "dummy": "linear",
    "logistic": "linear",
    "tree": "tree",
    "random_forest": "tree",
    "knn": "linear",
}


def _estimator(name: str):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    if name == "logistic":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if name == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=80,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15)
    raise ValueError(name)


def param_distributions(name: str) -> dict:
    if name == "logistic":
        return {
            "clf__C": np.logspace(-2, 2, 12),
            "clf__class_weight": [None, "balanced"],
        }
    if name == "tree":
        return {
            "clf__max_depth": [4, 6, 8, 12, None],
            "clf__min_samples_leaf": [20, 50, 100, 200],
            "clf__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3],
        }
    if name == "random_forest":
        return {
            "clf__n_estimators": [50, 80],
            "clf__max_depth": [8, 12, 16, None],
            "clf__min_samples_leaf": [10, 30, 80],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced_subsample"],
        }
    if name == "knn":
        return {
            "clf__n_neighbors": [5, 11, 21, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        }
    return {}


def build_model_pipeline(
    name: str,
    numeric: list[str],
    low_card: list[str],
    high_card: list[str],
) -> Pipeline:
    family = FAMILIES[name]
    prep = build_preprocessor(numeric, low_card, high_card, family=family)
    return Pipeline([("prep", prep), ("clf", _estimator(name))])


def _maybe_subsample(name: str, X, y, groups):
    if name != "knn" or len(X) <= KNN_MAX_ROWS:
        return X, y, groups
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=KNN_MAX_ROWS, replace=False)
    idx.sort()
    return X.iloc[idx], y.iloc[idx], groups.iloc[idx]


def fit_candidate(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    numeric: list[str] | None = None,
    low_card: list[str] | None = None,
    high_card: list[str] | None = None,
) -> dict:
    if numeric is None:
        numeric, low_card, high_card = feature_lists(X_train)
    X_fit, y_fit, g_fit = _maybe_subsample(name, X_train, y_train, groups_train)
    pipe = build_model_pipeline(name, numeric, low_card, high_card)
    dist = param_distributions(name)
    cv = grouped_cv()

    if not dist:
        logger.info("Fitting %s without search", name)
        pipe.fit(X_fit, y_fit)
        return {
            "name": name,
            "estimator": pipe,
            "best_params": {},
            "cv_roc_auc": None,
            "n_train": int(len(X_fit)),
        }

    n_jobs_search = 1 if name in {"random_forest", "knn"} else N_JOBS
    n_iter = min(N_ITER_SEARCH, int(np.prod([len(v) for v in dist.values()])))
    search = RandomizedSearchCV(
        pipe,
        param_distributions=dist,
        n_iter=max(1, n_iter),
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs_search,
        random_state=RANDOM_STATE,
        refit=True,
        error_score="raise",
    )
    logger.info("RandomizedSearchCV %s n_iter=%s n=%s", name, n_iter, f"{len(X_fit):,}")
    search.fit(X_fit, y_fit, groups=g_fit)
    return {
        "name": name,
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
        "cv_results": _cv_summary(search),
        "n_train": int(len(X_fit)),
        "search": search,
    }


def _cv_summary(search: RandomizedSearchCV) -> dict:
    return {
        "mean_test_score": float(search.best_score_),
        "std_test_score": float(
            search.cv_results_["std_test_score"][search.best_index_]
        ),
    }


def save_pipeline(estimator: Pipeline, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(estimator, path)
    return path


def save_search_report(results: list[dict], path: Path | None = None) -> Path:
    path = path or (REPORTS_DIR / "model_search.json")
    payload = []
    for row in results:
        payload.append(
            {
                "name": row["name"],
                "best_params": _jsonable(row.get("best_params") or {}),
                "cv_roc_auc": row.get("cv_roc_auc"),
                "cv_results": row.get("cv_results"),
                "n_train": row.get("n_train"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def pick_champion(results: list[dict], holdout_auc: dict[str, float]) -> str:
    """Highest grouped-CV ROC AUC; logistic wins ties vs random forest."""
    scored = []
    for row in results:
        name = row["name"]
        if name == "dummy":
            continue
        cv = row.get("cv_roc_auc")
        ho = holdout_auc.get(name)
        metric = cv if cv is not None else ho
        if metric is None:
            continue
        scored.append((name, float(metric)))
    if not scored:
        return "logistic"
    best = max(s[1] for s in scored)
    winners = [n for n, m in scored if abs(m - best) < 0.005]
    if "logistic" in winners and "random_forest" in winners:
        return "logistic"
    return max(scored, key=lambda t: t[1])[0]
