"""Custom encoders used inside Scikit-learn pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set relative frequency."""

    def __init__(self, unseen: float = 0.0):
        self.unseen = unseen

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.maps_: dict[str, dict[str, float]] = {}
        self.n_features_in_ = frame.shape[1]
        for col in frame.columns:
            series = frame[col].astype("string")
            freq = series.value_counts(normalize=True, dropna=False)
            self.maps_[str(col)] = {str(k): float(v) for k, v in freq.items()}
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        frame = pd.DataFrame(X, columns=self.feature_names_in_)
        out = np.empty((len(frame), frame.shape[1]), dtype=float)
        for i, col in enumerate(frame.columns):
            mapping = self.maps_[str(col)]
            values = frame[col].astype("string").astype(str)
            out[:, i] = values.map(mapping).fillna(self.unseen).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(input_features, dtype=object)
