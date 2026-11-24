"""Controlled shifts S3: four types at strengths 0.25 / 0.50 / 0.75, built by resampling a
held-out pool (the source test block, never used for training, calibration or thresholds).

Each generator returns (idx, y): positions into the pool (with replacement) and the true labels of the window, which
differ from the pool labels only for concept shift. Rules use features or classes, never the domain classifier used
in the attribution (not circular), and quantiles come from the source TRAIN block (no deployment information).

(a) covariate: a share s of the window is drawn from the long-duration flows (FLOW_DURATION_MILLISECONDS above the
    source-train 70% quantile), the rest uniformly: P(x) changes, P(y | x) does not.
(b) prior:     a share s is drawn class-balanced over the attack classes present, the rest uniformly: P(y) changes,
    P(x | y) does not.
(c) concept:   uniform window; inside a fixed region (IN_BYTES above the source-train 90% quantile) a share s of the
    benign flows is relabelled as the most frequent attack class of the source train block: P(y | x) changes, x does
    not, so no label-free statistic can see it.
(d) new class: a detector trained WITHOUT one attack class is deployed on windows where that class makes up s x 10%
    of the flows; the rest is drawn from the other flows.
"""
from __future__ import annotations

import numpy as np

from .schema import BENIGN

STRENGTHS = (0.25, 0.50, 0.75)
TYPES = ("covariate", "prior", "concept", "new_class")


def _mix(base_pool: np.ndarray, special_pool: np.ndarray, size: int, s: float, rng) -> np.ndarray:
    k = int(round(s * size))
    return np.concatenate([rng.choice(special_pool, k, replace=True), rng.choice(base_pool, size - k, replace=True)])


def covariate(pool_X_duration: np.ndarray, y: np.ndarray, size: int, s: float, rng, q70: float):
    allpos = np.arange(len(y))
    idx = _mix(allpos, np.flatnonzero(pool_X_duration > q70), size, s, rng)
    return idx, y[idx]


def prior(y: np.ndarray, size: int, s: float, rng):
    allpos = np.arange(len(y))
    attacks = [c for c in np.unique(y) if c != BENIGN]
    k = int(round(s * size))
    per = rng.choice(len(attacks), k, replace=True)               # class-balanced over attack classes
    special = np.array([rng.choice(np.flatnonzero(y == attacks[j])) for j in per], dtype=np.int64) if k else \
        np.array([], dtype=np.int64)
    idx = np.concatenate([special, rng.choice(allpos, size - k, replace=True)])
    return idx, y[idx]


def concept(pool_in_bytes: np.ndarray, y: np.ndarray, size: int, s: float, rng, q90: float, new_label: str):
    idx = rng.choice(len(y), size, replace=True)
    yw = y[idx].astype(object).copy()
    region = (pool_in_bytes[idx] > q90) & (yw == BENIGN)
    flip = region & (rng.random(size) < s)
    yw[flip] = new_label
    return idx, yw.astype(str)


def new_class(y: np.ndarray, size: int, s: float, rng, held_out: str):
    others = np.flatnonzero(y != held_out)
    special = np.flatnonzero(y == held_out)
    idx = _mix(others, special, size, s * 0.10, rng)
    return idx, y[idx]
