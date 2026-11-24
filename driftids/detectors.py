"""Detectors.

LightGBM: one of three fixed settings, chosen per scenario by macro-F1 on the SOURCE VALIDATION
block only (ties -> defaults; never deployment data,):
  (i) defaults, (ii) balanced class weights, (iii) defaults with at least 100 flows per leaf.
HistGradientBoosting (defaults): second detector (also used where LightGBM is not installed).
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

LGBM_SETTINGS = {
    "defaults": {},
    "balanced": {"class_weight": "balanced"},
    "min_leaf_100": {"min_child_samples": 100},
}


def _threads() -> int:
    return int(os.environ.get("DRIFTIDS_THREADS", "-1"))


def make_lightgbm(setting: str = "defaults", seed: int = 0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(random_state=seed, n_jobs=_threads(), verbose=-1, **LGBM_SETTINGS[setting])


def make_hgb(seed: int = 0):
    return HistGradientBoostingClassifier(random_state=seed)


def choose_lightgbm(X_tr, y_tr, X_val, y_val, seed: int = 0):
    """Fit the three declared settings on the train block, score macro-F1 on the validation block, return the
    winner (ties go to the defaults, the first entry) and the score table."""
    scores, models = {}, {}
    for name in LGBM_SETTINGS:
        mdl = make_lightgbm(name, seed).fit(X_tr, y_tr)
        p = mdl.predict(X_val)
        labels = sorted(set(map(str, y_val)))
        scores[name] = float(f1_score(np.asarray(y_val, dtype=str), np.asarray(p, dtype=str), labels=labels,
                                      average="macro", zero_division=0))
        models[name] = mdl
    best = max(scores, key=lambda k: (round(scores[k], 6), k == "defaults"))
    return models[best], best, scores


def make_detector(kind: str, X_tr, y_tr, X_val=None, y_val=None, seed: int = 0):
    """kind 'hgb' -> HistGradientBoosting defaults; 'lightgbm' -> the setting chosen on the validation block;
    'extratrees' -> ExtraTrees (alternative second detector)."""
    if kind == "hgb":
        return make_hgb(seed).fit(X_tr, y_tr), {"detector": "hgb"}
    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        n_jobs = _threads() if _threads() > 0 else -1
        mdl = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed)
        return mdl.fit(X_tr, y_tr), {"detector": "extratrees"}
    mdl, best, scores = choose_lightgbm(X_tr, y_tr, X_val, y_val, seed)
    return mdl, {"detector": "lightgbm", "setting": best, "val_macro_f1": scores}
