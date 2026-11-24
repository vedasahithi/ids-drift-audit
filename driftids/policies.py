"""Stream simulation of the five monitoring policies (H5).

Every policy starts from the same deployed detector and sees the same windows. Each window is scored with the model
in use when the window arrives; a retrained model is used from the next window on. Labels revealed by a
policy go into its buffer, which is added to the source training data at every retraining. Total labels spent are
counted per policy.

  never      - no labels, no retraining;
  periodic   - every k windows: n random labels, retrain;
  alarm      - alarm (abstention OR domain AUC above its source threshold), at most once every `cap` windows:
               n random labels, retrain;
  dad        - detect-audit-decide: alarm-triggered active audit of n flows (at most once every `cap` windows) and a
               routine active audit of n_routine flows every k windows; decision rule after each audit: retrain if the
               audited macro-F1 upper bound < source macro-F1 - delta; otherwise, if the attribution says "prior",
               recalibrate (re-weight the class probabilities by the BBSE weights) until the next action;
  oracle     - retrain with n random labels whenever the TRUE macro-F1 is below source - delta (upper reference).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PolicyConfig:
    k: int = 5
    n: int = 200
    n_routine: int = 50
    cap: int = 3
    delta: float = 0.05
    lam: float = 0.5
    n_base: int = 13       # labels per window for the ddm / uncertainty baselines (matched to DAD, ~13 per window)
    # error-driven retraining for dad - also retrain when the audit shows the model
    # clearly makes errors (accuracy upper bound < 1 - err_tol) and new labels arrived since the last retraining.
    # None disables the rule. Set via DRIFTIDS_ERR_TOL.
    err_tol: float | None = float(os.environ.get("DRIFTIDS_ERR_TOL", "0.01"))


@dataclass
class PolicyState:
    name: str
    model: object
    buffer_idx: list = field(default_factory=list)      # pool positions with revealed labels
    labels: int = 0
    last_capped: int = -10**9                            # window of the last alarm-triggered action
    class_weights: np.ndarray | None = None              # active prior recalibration (dad)
    retrains: int = 0
    log: list = field(default_factory=list)
    # "audit on change": trigger on alarm onset or a clear rise since the last trigger
    prev_alarm: bool = False
    trig_auc: float | None = None
    trig_ab: float | None = None
    prev_audit: dict | None = None                       # last audit sample (dad), pooled if the model is unchanged
    # DDM baseline statistics (errors in the audited labels since the last retraining)
    ddm_n: int = 0
    ddm_err: int = 0
    ddm_min: float = float("inf")
    ddm_smin: float = float("inf")


def change_trigger(st: PolicyState, alarm: bool, auc: float, ab: float, auc_spread: float, ab_spread: float,
                   t: int, cap: int) -> bool:
    """Alarm-triggered action only when something CHANGED: the alarm switches on, or one of its signals rose by more
    than its normal range (spread on unshifted source windows) since the last trigger; and at most once every `cap`
    windows. A persistent, already-measured shift does not trigger new audits."""
    onset = alarm and not st.prev_alarm
    rise = alarm and st.trig_auc is not None and (auc - st.trig_auc > auc_spread or ab - st.trig_ab > ab_spread)
    if (onset or rise) and t - st.last_capped >= cap:
        st.last_capped, st.trig_auc, st.trig_ab = t, auc, ab
        return True
    return False


def apply_recalibration(proba: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        return proba
    p = proba * w[None, :]
    s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), proba)
