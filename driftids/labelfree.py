"""Label-free performance estimators (baselines the audit is compared with).

- soft confusion ("average confidence"): the detector's own class probabilities stand in for the unknown labels:
  E[C(t, p)] = sum_i P(t | x_i) * 1[pred_i = p]. Macro-F1, binary F1 and false-alarm rate follow from it.
- ATC (Garg et al. 2022): a confidence threshold tau is fixed on held-out source data so that the share of flows
  above tau equals the source accuracy; in a window, flows above tau are counted as correct and the rest are spread
  by their probabilities. With tau from the source calibration block only.
Both are known to fail silently when the detector is confidently wrong (Ahmed et al. 2026); that is the point of
comparing them with the audit.
"""
from __future__ import annotations

import numpy as np

from .audit import _metrics_from_cms
from .schema import BENIGN


def _class_index(model_classes, pred_names):
    labels = np.asarray([str(c) for c in model_classes])
    p = np.searchsorted(labels, np.asarray(pred_names, dtype=str))
    if not np.array_equal(labels[p], np.asarray(pred_names, dtype=str)):
        raise ValueError("predictions contain a class the model does not have")
    return labels, p


def _metrics(cm: np.ndarray, labels: np.ndarray, m: int, restrict_to=None) -> dict:
    est_true = cm.sum(1)
    cls = [i for i in range(len(labels)) if est_true[i] >= m]
    if restrict_to is not None:
        allowed = {c.lower() for c in restrict_to}
        cls = [i for i in cls if labels[i].lower() in allowed]
    benign = int(np.searchsorted(labels, BENIGN)) if BENIGN in labels else None
    r = {k: float(v[0]) for k, v in _metrics_from_cms(cm[None], labels, np.array(cls, dtype=int), benign).items()}
    r["classes"] = [str(labels[i]) for i in cls]
    return r


def soft_confusion(proba: np.ndarray, model_classes, pred_names, *, m: int, restrict_to=None) -> dict:
    """proba: (N, K) class probabilities in the order of model_classes (sorted names, as sklearn/LightGBM give)."""
    labels, p = _class_index(model_classes, pred_names)
    K = len(labels)
    onehot = np.zeros((len(p), K))
    onehot[np.arange(len(p)), p] = 1.0
    cm = proba.T @ onehot                                   # (true K, predicted K)
    return _metrics(cm, labels, m, restrict_to)


def atc_threshold(proba_cal: np.ndarray, correct_cal: np.ndarray) -> float:
    """tau such that the share of calibration flows with max probability > tau equals the calibration accuracy."""
    conf = proba_cal.max(1)
    acc = float(np.mean(correct_cal))
    return float(np.quantile(conf, 1 - acc)) if 0 < acc < 1 else (0.0 if acc >= 1 else 1.0)


def atc(proba: np.ndarray, model_classes, pred_names, tau: float, *, m: int, restrict_to=None) -> dict:
    labels, p = _class_index(model_classes, pred_names)
    K = len(labels)
    conf = proba.max(1)
    sure = conf > tau
    cm = np.zeros((K, K))
    np.add.at(cm, (p[sure], p[sure]), 1.0)                 # confident flows counted as correct
    onehot = np.zeros(((~sure).sum(), K))
    onehot[np.arange((~sure).sum()), p[~sure]] = 1.0
    cm += proba[~sure].T @ onehot                           # the rest spread by probability
    r = _metrics(cm, labels, m, restrict_to)
    r["predicted_accuracy"] = float(sure.mean())
    return r
