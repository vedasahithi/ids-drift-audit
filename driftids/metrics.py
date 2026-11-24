"""Per-window metrics (estimation targets)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BENIGN


def _codes(y_true, y_pred):
    """Integer codes for true and predicted class names over one sorted label list (hash-based: fast)."""
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    codes, uniq = pd.factorize(np.concatenate([yt, yp]), sort=False)
    uniq = np.asarray([str(u) for u in uniq])
    order = np.argsort(uniq, kind="stable")
    remap = np.empty(len(uniq), dtype=np.int64)
    remap[order] = np.arange(len(uniq))
    codes = remap[codes]
    return codes[:len(yt)], codes[len(yt):], uniq[order]


def present_classes(y_true, m: int) -> list[str]:
    """Classes with at least m true flows in the window (the macro-F1 class set)."""
    vc = pd.Series(np.asarray(y_true, dtype=object)).value_counts()
    return sorted(str(v) for v, c in vc.items() if c >= m)


def macro_f1(y_true, y_pred, m: int, restrict_to=None) -> tuple[float, list[str]]:
    """Macro-F1 over classes with >= m true flows (optionally only those also in `restrict_to`).

    Computed from one confusion matrix; equal to sklearn f1_score(labels=classes, average='macro',
    zero_division=0) (checked in the tests)."""
    t, p, labels = _codes(y_true, y_pred)
    K = len(labels)
    cm = np.bincount(t * K + p, minlength=K * K).reshape(K, K)
    true_counts = cm.sum(1)
    classes = sorted(str(labels[i]) for i in range(K) if true_counts[i] >= m)
    if restrict_to is not None:
        allowed = {c.lower() for c in restrict_to}
        classes = [c for c in classes if c.lower() in allowed]
    if not classes:
        return float("nan"), []
    return _f1_from_cm(cm, np.searchsorted(labels, np.asarray(classes, dtype=str))), classes


def _f1_from_cm(cm: np.ndarray, idx: np.ndarray) -> float:
    tp = np.diag(cm)[idx].astype(np.float64)
    fp = cm.sum(0)[idx] - tp
    fn = cm.sum(1)[idx] - tp
    den = 2 * tp + fp + fn
    return float(np.mean(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)))


def macro_f1_block_bootstrap(y_true, y_pred, classes, *, block: int = 1000, B: int = 200,
                             seed: int = 0) -> np.ndarray:
    """Macro-F1 over a fixed class set on B moving-block bootstrap resamples (blocks of consecutive flows).

    Blocks keep the burstiness of attack traffic, so the spread is not underestimated as with flow-level
    resampling. Used for the noise rule of the drift check.
    """
    if not len(classes):
        return np.full(B, np.nan)
    t, p, labels = _codes(y_true, y_pred)
    K, n = len(labels), len(t)
    idx = np.searchsorted(labels, np.asarray(classes, dtype=str))
    nb = -(-n // block)
    rng = np.random.default_rng(seed)
    # Vectorised (same result as resampling flows block by block): one confusion matrix per block, then each
    # resample is a weighted sum of block matrices, the weights being how often each block was drawn.
    blk = np.arange(n) // block
    cm_blocks = np.bincount(blk * K * K + t * K + p, minlength=nb * K * K).reshape(nb, K * K).astype(np.float64)
    draws = rng.integers(0, nb, (B, nb))
    mult = np.bincount((draws + (np.arange(B) * nb)[:, None]).ravel(), minlength=B * nb).reshape(B, nb)
    cms = (mult @ cm_blocks).reshape(B, K, K)
    tp = cms[:, idx, idx]
    fp = cms.sum(1)[:, idx] - tp
    fn = cms.sum(2)[:, idx] - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return f1.mean(1)


def _macro_f1_block_bootstrap_loop(y_true, y_pred, classes, *, block: int = 1000, B: int = 200,
                                   seed: int = 0) -> np.ndarray:
    """Reference (slow, loop) version, kept to test that the vectorised version gives the same numbers."""
    if not len(classes):
        return np.full(B, np.nan)
    t, p, labels = _codes(y_true, y_pred)
    K, n = len(labels), len(t)
    idx = np.searchsorted(labels, np.asarray(classes, dtype=str))
    nb = -(-n // block)
    rng = np.random.default_rng(seed)
    offs = np.arange(block)
    out = np.empty(B)
    for b in range(B):
        ii = (rng.integers(0, nb, nb)[:, None] * block + offs).ravel()
        ii = ii[ii < n]
        cm = np.bincount(t[ii] * K + p[ii], minlength=K * K).reshape(K, K)
        out[b] = _f1_from_cm(cm, idx)
    return out


def binary_metrics(y_true_cls, y_pred_cls) -> dict:
    """Attack-vs-benign F1, detection rate and false-alarm rate."""
    t = np.asarray(y_true_cls, dtype=object) != BENIGN
    p = np.asarray(y_pred_cls, dtype=object) != BENIGN
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    dr = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"bin_f1": float(f1), "detection_rate": float(dr), "false_alarm_rate": float(far),
            "n_attack": int(t.sum()), "n_benign": int((~t).sum())}
