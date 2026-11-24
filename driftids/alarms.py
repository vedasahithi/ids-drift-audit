"""Label-free alarm signals (the "detect" step). All use only the detector's outputs on the window, the
source calibration block, or unlabelled source traffic - never window labels.

- conformal (split, class-conditional "Mondrian"): nonconformity 1 - p_y on the source calibration block; per-class
  threshold q_k at level 1 - alpha (finite-sample corrected; classes with fewer than `min_class` calibration flows use
  the global threshold). Prediction set of a flow = {k : 1 - p_k <= q_k}.
    abstention rate  = share of flows whose set does not hold exactly one class (Ahmed et al. 2026);
    empty-set rate   = share of flows that fit no known class (novelty).
- domain classifier: AUC of a classifier separating a source sample from the window (2-fold cross-fitted, so the
  AUC is out-of-sample; 0.5 = indistinguishable).
- disagreement: share of flows where two detectors predict different classes.
- confidence: mean maximum class probability (and its drop from the source value).
Thresholds come from windows of unshifted source data only (see `null_thresholds`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


@dataclass
class Conformal:
    classes: np.ndarray       # model class order
    q: np.ndarray             # threshold per class (on 1 - p_k)
    alpha: float


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_conformal(proba_cal: np.ndarray, y_cal, classes, alpha: float = 0.1, min_class: int = 30,
                  mondrian: bool = False) -> Conformal:
    """Split conformal thresholds. Default: one marginal threshold (as Ahmed et al. 2026 for the abstention alarm).
    mondrian=True gives per-class thresholds (class-conditional coverage) - on UNSW this saturates the abstention
    rate at 1.0, because poorly calibrated classes get thresholds near 1 and then enter every set."""
    classes = np.asarray([str(c) for c in classes])
    y = np.asarray(y_cal, dtype=str)
    known = np.isin(y, classes)                         # calibration flows of classes the model knows
    idx = np.searchsorted(classes, y[known])
    s = 1.0 - proba_cal[known][np.arange(known.sum()), idx]
    q_global = _quantile(s, alpha)
    q = np.full(len(classes), q_global)
    if mondrian:
        for k in range(len(classes)):
            sk = s[idx == k]
            if len(sk) >= min_class:
                q[k] = _quantile(sk, alpha)
    return Conformal(classes=classes, q=q, alpha=alpha)


def conformal_rates(proba: np.ndarray, cp: Conformal) -> dict:
    sets = (1.0 - proba) <= cp.q[None, :]
    size = sets.sum(1)
    return {"abstention": float(np.mean(size != 1)), "empty_set": float(np.mean(size == 0)),
            "multi_set": float(np.mean(size > 1))}


def density_ratio(src: np.ndarray, win: np.ndarray, score_X: np.ndarray, seed: int = 0, max_iter: int = 60,
                  src_weight: np.ndarray | None = None) -> np.ndarray:
    """Window-vs-source density ratio p/(1-p) (clipped) for the rows of score_X (e.g. the labelled source
    calibration flows), from one domain classifier fitted on all of src and win (label-free)."""
    sw = np.ones(len(src)) if src_weight is None else np.asarray(src_weight, dtype=float) * len(src) / np.sum(src_weight)
    X = np.vstack([src, win])
    y = np.r_[np.zeros(len(src)), np.ones(len(win))]
    clf = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed).fit(
        X, y, sample_weight=np.r_[sw, np.ones(len(win))])
    p = np.clip(clf.predict_proba(score_X)[:, 1], 0.01, 0.99)
    return p / (1 - p) * (sw.sum() / len(win))


def domain_auc(src: np.ndarray, win: np.ndarray, seed: int = 0, max_iter: int = 60,
               src_weight: np.ndarray | None = None, win_weight: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Cross-fitted AUC of source-vs-window and the out-of-fold window probabilities (for novelty scores).
    src_weight (optional): weights of the source flows (e.g. label-shift correction), used in fitting and in the AUC;
    the source total weight is matched to the window size."""
    rng = np.random.default_rng(seed)
    fw, fs = rng.integers(0, 2, len(win)), rng.integers(0, 2, len(src))
    sw = np.ones(len(src)) if src_weight is None else np.asarray(src_weight, dtype=float) * len(src) / np.sum(src_weight)
    ww = np.ones(len(win)) if win_weight is None else np.asarray(win_weight, dtype=float) * len(win) / np.sum(win_weight)
    pw, ps = np.empty(len(win)), np.empty(len(src))
    for k in (0, 1):
        X = np.vstack([src[fs != k], win[fw != k]])
        y = np.r_[np.zeros((fs != k).sum()), np.ones((fw != k).sum())]
        wt = np.r_[sw[fs != k], ww[fw != k]]
        clf = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed).fit(X, y, sample_weight=wt)
        pw[fw == k] = clf.predict_proba(win[fw == k])[:, 1]
        ps[fs == k] = clf.predict_proba(src[fs == k])[:, 1]
    auc = float(roc_auc_score(np.r_[np.zeros(len(src)), np.ones(len(win))], np.r_[ps, pw],
                              sample_weight=np.r_[sw, ww]))
    return auc, pw


def disagreement(pred_a, pred_b) -> float:
    return float(np.mean(np.asarray(pred_a, dtype=str) != np.asarray(pred_b, dtype=str)))


def mean_confidence(proba: np.ndarray) -> float:
    return float(proba.max(1).mean())


def null_thresholds(null_signals: dict[str, np.ndarray], q: float = 0.99, higher_is_alarm=None) -> dict:
    """Alarm threshold per signal = the q-quantile of the signal over windows of UNSHIFTED source data (for signals
    where lower means trouble, e.g. confidence, the (1 - q)-quantile and the alarm fires below it)."""
    higher_is_alarm = higher_is_alarm or {}
    out = {}
    for name, vals in null_signals.items():
        vals = np.asarray(vals, dtype=float)
        if higher_is_alarm.get(name, True):
            out[name] = {"threshold": float(np.quantile(vals, q)), "direction": "above"}
        else:
            out[name] = {"threshold": float(np.quantile(vals, 1 - q)), "direction": "below"}
    return out


def fires(value: float, rule: dict) -> bool:
    return value > rule["threshold"] if rule["direction"] == "above" else value < rule["threshold"]
