"""ColumnTransformer builders per model family (linear / tree)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from src.config import RANDOM_STATE
from src.modeling.encoders import FrequencyEncoder


def _numeric_pipeline(*, scale: bool, add_indicator: bool = True) -> Pipeline:
    steps: list[tuple] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _onehot_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=30,
                    sparse_output=False,
                ),
            ),
        ]
    )


def _ordinal_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )


def _target_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            (
                "encode",
                TargetEncoder(
                    target_type="binary",
                    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                ),
            ),
        ]
    )


def _frequency_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            ("encode", FrequencyEncoder()),
        ]
    )


def build_preprocessor(
    numeric: list[str],
    low_card: list[str],
    high_card: list[str],
    family: str = "linear",
    high_card_encoding: str | None = None,
) -> ColumnTransformer:
    """
    family='linear' → scale + one-hot + target encoding (logistic, KNN).
    family='tree'   → no scale + ordinal + frequency encoding (trees, forests).
    """
    if family not in {"linear", "tree"}:
        raise ValueError(f"Unknown family '{family}'")

    high_mode = high_card_encoding or ("target" if family == "linear" else "frequency")
    transformers: list[tuple] = []
    if numeric:
        transformers.append(("num", _numeric_pipeline(scale=family == "linear"), numeric))
    if low_card:
        enc = _onehot_pipeline() if family == "linear" else _ordinal_pipeline()
        transformers.append(("low", enc, low_card))
    if high_card:
        if high_mode == "target":
            transformers.append(("high", _target_pipeline(), high_card))
        elif high_mode == "frequency":
            transformers.append(("high", _frequency_pipeline(), high_card))
        else:
            raise ValueError(f"Unknown high_card_encoding '{high_mode}'")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
