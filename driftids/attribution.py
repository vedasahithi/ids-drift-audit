"""Drift-type attribution (H4): none / covariate / prior / concept / new_class for one window,
from label-free signals plus the audit.

Decision:
 1. new_class - the audit contains attack labels the detector was never trained on (design-weighted share above
    `new_share`; in a real audit the analyst names the new attack type);
 2. what the label-free evidence predicts for the window's macro-F1, given the shift it sees:
      no P(x) change (domain AUC <= threshold)        -> the source performance (calibration block);
      P(x) change explained by the class mix (residual AUC after BBSE re-weighting <= threshold)
                                                       -> source confusion P(pred | y) re-weighted to the BBSE mix;
      otherwise (covariate)                            -> calibration flows importance-weighted by the
                                                          domain-classifier density ratio;
 3. concept - the audited macro-F1 interval excludes that prediction (the label-free view cannot explain what the
    audit measures, so P(y | x) changed); otherwise the type from step 2 (none / prior / covariate).
Predictions are built from labelled source data, so the detector's own over-confidence does not enter them (the
first version compared with ATC and confused prior shift with concept shift on UNSW).
"""
from __future__ import annotations

import numpy as np

from .alarms import domain_auc
from .audit import _metrics_from_cms
from .schema import BENIGN


def _codes(classes, labels):
    classes = np.asarray([str(c) for c in classes])
    labels = np.asarray(labels, dtype=str)
    idx = np.searchsorted(classes, labels)
    ok = (idx < len(classes)) & (classes[np.minimum(idx, len(classes) - 1)] == labels)
    return classes, np.minimum(idx, len(classes) - 1), ok


def bbse_class_weights(pred_cal, y_cal, classes, pred_window) -> np.ndarray:
    """BBSE: solve C w = q, C[i, j] = P(pred = i, y = j) on the source calibration block, q = predicted class shares
    in the window; returns non-negative weights w_j = p_target(j) / p_source(j) per model class."""
    classes, yi, ok = _codes(classes, y_cal)
    _, pi, _ = _codes(classes, pred_cal)
    K = len(classes)
    C = np.zeros((K, K))
    np.add.at(C, (pi[ok], yi[ok]), 1.0)
    C /= max(ok.sum(), 1)
    _, pw, _ = _codes(classes, pred_window)
    q = np.bincount(pw, minlength=K) / len(pw)
    w, *_ = np.linalg.lstsq(C, q, rcond=None)
    return np.clip(w, 0, None)


def residual_auc(src_X, src_y, win_X, w, classes, seed: int = 0) -> float:
    """Domain AUC after re-weighting the source sample by the BBSE class weights."""
    _, idx, known = _codes(classes, src_y)
    sw = np.where(known, w[idx], 0.0)
    keep = sw > 0
    auc, _ = domain_auc(src_X[keep], win_X, seed=seed, src_weight=sw[keep])
    return auc



def _macro_from_cm(cm: np.ndarray, classes: np.ndarray, m: float) -> float:
    est_true = cm.sum(1)
    cls = np.array([i for i in range(len(classes)) if est_true[i] >= m], dtype=int)
    benign = int(np.searchsorted(classes, BENIGN)) if BENIGN in classes else None
    return float(_metrics_from_cms(cm[None], classes, cls, benign)["macro_f1"][0])


def predicted_macro_f1(pred_cal, y_cal, classes, *, N: int, m: float, class_weights=None, flow_weights=None) -> float:
    """Macro-F1 the label-free view predicts for a window of N flows, from the labelled calibration block:
    plain (no shift), with per-class weights (BBSE: prior shift) or with per-flow weights (importance weighting:
    covariate shift)."""
    classes, yi, ok = _codes(classes, y_cal)
    _, pi, _ = _codes(classes, pred_cal)
    wt = np.ones(len(yi))
    if class_weights is not None:
        wt = wt * np.asarray(class_weights)[yi]
    if flow_weights is not None:
        wt = wt * np.asarray(flow_weights)
    wt = np.where(ok, wt, 0.0)
    K = len(classes)
    cm = np.zeros((K, K))
    np.add.at(cm, (yi, pi), wt)
    if cm.sum() <= 0:
        return float("nan")
    cm *= N / cm.sum()
    return _macro_from_cm(cm, classes, m)


def predicted_accuracy(pred_cal, y_cal, classes, *, class_weights=None, flow_weights=None) -> float:
    """Accuracy the label-free view predicts for the window, from the labelled calibration block (plain, BBSE class
    weights, or importance weights). Accuracy is a simple proportion, so these predictions are well behaved; macro-F1
    predictions were fragile on UNSW (rare classes, class-set cut-off) and gave false concept diagnoses."""
    classes, yi, ok = _codes(classes, y_cal)
    _, pi, _ = _codes(classes, pred_cal)
    wt = np.ones(len(yi))
    if class_weights is not None:
        wt = wt * np.asarray(class_weights)[yi]
    if flow_weights is not None:
        wt = wt * np.asarray(flow_weights)
    wt = np.where(ok, wt, 0.0)
    return float((wt * (yi == pi)).sum() / wt.sum()) if wt.sum() > 0 else float("nan")


def attribute(*, audit_unseen_share: float, audit_lo: float, audit_hi: float, dom_auc: float, res_auc: float,
              auc_threshold: float, pred_none: float, pred_prior: float, pred_cov: float,
              new_share: float = 0.01) -> str:
    """ concept = no label-free explanation fits the audit: the audited interval
    excludes the plain, the label-shift-corrected (BBSE) AND the covariate-corrected (importance-weighted) prediction.
    Otherwise the type follows the domain-classifier evidence (none / prior / covariate)."""
    if audit_unseen_share > new_share:
        return "new_class"
    preds = [p for p in (pred_none, pred_prior, pred_cov) if np.isfinite(p)]
    if preds and not any(audit_lo <= p <= audit_hi for p in preds):
        return "concept"
    if dom_auc <= auc_threshold:
        return "none"
    if res_auc <= auc_threshold:
        return "prior"
    return "covariate"



def diagnose_drift(*, audit_unseen_share: float, audit_lo: float, audit_hi: float, dom_auc: float,
                 res_auc_audit: float, auc_threshold: float, pred_none: float, pred_prior: float, pred_cov: float,
                 conf_err_lo: float, conf_ref: float, conf_ref_prior: float, new_share: float = 0.01,
                 rel_drop_min: float = 0.08, conf_margin: float = 0.01, prior_factor: float = 2.0) -> str:
    """Drift diagnosis (thresholds set on UNSW-NB15 and CSE-CIC-IDS2018).
    1. new_class: the audit contains classes the detector never saw (design-weighted share > new_share);
    2. base type: none if the domain AUC <= the null threshold; prior if re-weighting the source to the class mix
       MEASURED by the audit removes a clear part of the shift, (dom_auc - res_auc_audit) / (dom_auc - 0.5) >=
       rel_drop_min; otherwise covariate;
    3. concept: the audit's confident-error share (max proba >= 0.9, lower 95% bound) exceeds the reference
       (conf_ref = max(source rate, 99th percentile on unshifted fitting windows); x prior_factor for a prior base)
       plus conf_margin, or - for a non-prior base - no label-free accuracy prediction fits the audited interval."""
    if audit_unseen_share > new_share:
        return "new_class"
    rel = (dom_auc - res_auc_audit) / max(dom_auc - 0.5, 1e-6) if np.isfinite(res_auc_audit) else -1.0
    base = "none" if dom_auc <= auc_threshold else ("prior" if rel >= rel_drop_min else "covariate")
    ref = prior_factor * conf_ref_prior if base == "prior" else conf_ref
    if np.isfinite(conf_err_lo) and np.isfinite(ref) and conf_err_lo > ref + conf_margin:
        return "concept"
    if base != "prior":
        preds = [p for p in (pred_none, pred_prior, pred_cov) if np.isfinite(p)]
        if preds and not any(audit_lo <= p <= audit_hi for p in preds):
            return "concept"
    return base
