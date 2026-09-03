"""Grouped and random train/test splits for leakage-aware validation."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit

from src.config import N_CV_SPLITS, RANDOM_STATE, TEST_SIZE


def grouped_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Holdout where no municipality appears in both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        groups.iloc[train_idx].reset_index(drop=True),
        groups.iloc[test_idx].reset_index(drop=True),
    )


def random_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Row-wise stratified split (leaky when features are municipal)."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y))
    g_train = groups.iloc[train_idx].reset_index(drop=True) if groups is not None else None
    g_test = groups.iloc[test_idx].reset_index(drop=True) if groups is not None else None
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        g_train,
        g_test,
    )


def grouped_cv(n_splits: int = N_CV_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def overlap_municipalities(groups_train: pd.Series, groups_test: pd.Series) -> int:
    return int(len(set(groups_train.astype(str)) & set(groups_test.astype(str))))
