"""Evidence that a random split inflates AUC via municipality memorization."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.evaluation.metrics import classification_metrics, predict_scores
from src.modeling.preprocessing import build_preprocessor
from src.modeling.split import grouped_holdout, overlap_municipalities, random_holdout
from src.preprocessing.features import feature_lists


def _logit_pipeline(numeric, low, high) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, low, high, family="linear")),
            (
                "clf",
                LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE),
            ),
        ]
    )


def compare_random_vs_grouped(X, y, groups) -> dict:
    numeric, low, high = feature_lists(X)
    Xtr_r, Xte_r, ytr_r, yte_r, gtr_r, gte_r = random_holdout(X, y, groups)
    Xtr_g, Xte_g, ytr_g, yte_g, gtr_g, gte_g = grouped_holdout(X, y, groups)

    pipe_r = _logit_pipeline(numeric, low, high)
    pipe_r.fit(Xtr_r, ytr_r)
    auc_r = classification_metrics(yte_r, predict_scores(pipe_r, Xte_r))["roc_auc"]

    pipe_g = _logit_pipeline(numeric, low, high)
    pipe_g.fit(Xtr_g, ytr_g)
    auc_g = classification_metrics(yte_g, predict_scores(pipe_g, Xte_g))["roc_auc"]

    return {
        "random_split_roc_auc": auc_r,
        "grouped_split_roc_auc": auc_g,
        "auc_inflation": auc_r - auc_g,
        "random_municipio_overlap": overlap_municipalities(gtr_r, gte_r),
        "grouped_municipio_overlap": overlap_municipalities(gtr_g, gte_g),
        "n_municipios_train_grouped": int(gtr_g.nunique()),
        "n_municipios_test_grouped": int(gte_g.nunique()),
    }
