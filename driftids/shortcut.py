"""Shortcut screen: single-feature AUC on the source train block only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def single_feature_auc(X: pd.DataFrame, y_bin: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(X)
    for c in X.columns:
        v = X[c].to_numpy(dtype=np.float64)
        nan = np.isnan(v)
        if nan.any():
            v = np.where(nan, np.nanmin(v) - 1 if (~nan).any() else 0.0, v)
        nun = int(pd.Series(v).nunique())
        auc = 0.5 if nun < 2 or len(np.unique(y_bin)) < 2 else float(roc_auc_score(y_bin, v))
        rows.append({"feature": c, "auc": max(auc, 1 - auc), "direction": "+" if auc >= 0.5 else "-",
                     "n_unique": nun, "unique_share": nun / n, "nan_share": float(nan.mean())})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
