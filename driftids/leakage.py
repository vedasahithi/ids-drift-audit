"""Leakage and look-ahead checks . Each raises LeakageError; runs must not catch it."""
from __future__ import annotations

import numpy as np

from .schema import is_excluded


class LeakageError(AssertionError):
    pass


def assert_no_excluded_features(feature_names) -> None:
    """: no identifier, absolute-time or label column among the features."""
    bad = [c for c in feature_names if is_excluded(c)]
    if bad:
        raise LeakageError(f"excluded columns used as features: {bad}")


def assert_no_raw_column_copies(X: np.ndarray, raw: dict[str, np.ndarray]) -> None:
    """: no feature column is an exact copy of an excluded raw column (e.g. a renamed timestamp)."""
    for name, col in raw.items():
        col = np.asarray(col, dtype=np.float64)
        for j in range(X.shape[1]):
            if np.array_equal(X[:, j].astype(np.float64), col):
                raise LeakageError(f"feature column {j} is an exact copy of excluded column {name}")


def assert_before(train_pos: np.ndarray, eval_pos: np.ndarray, start: np.ndarray, end: np.ndarray,
                  what: str = "") -> None:
    """: every training flow starts and ends strictly before the first evaluated flow starts."""
    if len(train_pos) == 0 or len(eval_pos) == 0:
        return
    t_eval = start[eval_pos].min()
    if start[train_pos].max() >= t_eval or end[train_pos].max() >= t_eval:
        raise LeakageError(f"look-ahead {what}: training data reaches t={max(start[train_pos].max(), end[train_pos].max())}"
                           f" but evaluation starts at t={t_eval}")


def assert_disjoint(a: np.ndarray, b: np.ndarray, what: str = "") -> None:
    if np.intersect1d(a, b).size:
        raise LeakageError(f"overlapping rows {what}")


def assert_fitted_on(rows_seen: np.ndarray, allowed: np.ndarray, what: str = "") -> None:
    """: a fitted object saw only rows from allowed blocks."""
    extra = np.setdiff1d(rows_seen, allowed)
    if extra.size:
        raise LeakageError(f"{what} was fitted on {extra.size} rows outside the allowed blocks")
