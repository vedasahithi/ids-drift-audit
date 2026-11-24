"""Label-efficient audits: estimate a deployed detector's macro-F1, binary F1 and false-alarm rate in a window
from a small labelled sample (audit estimators).

Every estimator returns an AuditSample: the audited flows (true and predicted class codes) with design weights such
that sum(w * onehot(true, pred)) is an unbiased estimate of the window's confusion counts. Metrics are computed from
the estimated confusion matrix (ratio estimators, so approximately unbiased); 95% intervals by bootstrap of the
audited sample (within strata for the stratified design).

Designs:
- random:     simple random sample without replacement, weight N / n;
- stratified: strata = predicted class, proportional allocation (at least `min_per` per stratum when possible),
              sampling without replacement inside strata, weight N_h / n_h;
- active:     importance sampling with replacement (Kossen et al. 2021 style), probability
              q_i = lam / N + (1 - lam) * a_i / sum(a) for an acquisition score a_i >= 0 (the uniform part bounds the
              weights: the "defensive mixture"), weight 1 / (n q_i) (Hansen-Hurwitz).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .schema import BENIGN

# interval method used by all stage-2 drivers
INTERVAL = os.environ.get("DRIFTIDS_INTERVAL", "posterior3")
# results root
def results_root(code_root):
    from pathlib import Path
    r = os.environ.get("DRIFTIDS_RESULTS")
    return Path(r) if r else Path(code_root).parent / "results"


@dataclass
class AuditSample:
    idx: np.ndarray          # positions in the window of the audited draws (repeats possible for 'active')
    w: np.ndarray            # design weights (float)
    strata: np.ndarray | None = None   # stratum id per draw (stratified design), for the bootstrap
    design: str = ""


# ------------------------------------------------------------------ designs
def random_audit(N: int, n: int, rng) -> AuditSample:
    n = min(n, N)
    idx = rng.choice(N, size=n, replace=False)
    return AuditSample(idx=idx, w=np.full(n, N / n), design="random")


def stratified_audit(pred_codes: np.ndarray, n: int, rng, min_per: int = 2) -> AuditSample:
    N = len(pred_codes)
    n = min(n, N)
    groups = [np.flatnonzero(pred_codes == h) for h in np.unique(pred_codes)]
    sizes = np.array([len(g) for g in groups])
    base = np.minimum(sizes, min_per)
    if base.sum() > n:                                   # too many strata for the budget: plain proportional
        base = np.zeros_like(sizes)
    rest = n - base.sum()
    share = rest * sizes / sizes.sum()
    alloc = base + np.floor(share).astype(int)
    for j in np.argsort(-(share - np.floor(share)))[: n - alloc.sum()]:   # largest remainders
        alloc[j] += 1
    alloc = np.minimum(alloc, sizes)
    while alloc.sum() < n:                               # capacity left in other strata
        j = np.argmax(sizes - alloc)
        alloc[j] += 1
    idx, w, st = [], [], []
    for h, (g, a) in enumerate(zip(groups, alloc)):
        if a == 0:
            continue
        idx.append(rng.choice(g, size=a, replace=False))
        w.append(np.full(a, len(g) / a))
        st.append(np.full(a, h))
    return AuditSample(idx=np.concatenate(idx), w=np.concatenate(w), strata=np.concatenate(st), design="stratified")


def stratified_active_audit(pred_codes: np.ndarray, acq: np.ndarray, n: int, rng, lam: float = 0.5,
                            min_per: int = 2) -> AuditSample:
    """Hybrid: strata = predicted class with the same allocation as stratified_audit; inside
    stratum h, n_h draws with replacement with probability q_i = lam / N_h + (1 - lam) * a_i / sum_h(a); weight
    1 / (n_h q_i) (Hansen-Hurwitz per stratum), so stratum totals and hence confusion counts stay unbiased."""
    base = stratified_audit(pred_codes, n, rng, min_per=min_per)       # only used for the allocation
    acq = np.clip(np.asarray(acq, dtype=np.float64), 0, None)
    idx, w, st = [], [], []
    for h in np.unique(base.strata):
        n_h = int((base.strata == h).sum())
        g = np.flatnonzero(pred_codes == np.unique(pred_codes)[h])
        a = acq[g]
        q = np.full(len(g), 1.0 / len(g)) if a.sum() <= 0 else lam / len(g) + (1 - lam) * a / a.sum()
        q = q / q.sum()
        pick = rng.choice(len(g), size=n_h, replace=True, p=q)
        idx.append(g[pick])
        w.append(1.0 / (n_h * q[pick]))
        st.append(np.full(n_h, h))
    return AuditSample(idx=np.concatenate(idx), w=np.concatenate(w), strata=np.concatenate(st),
                       design="stratactive")


def active_audit(acq: np.ndarray, n: int, rng, lam: float = 0.5) -> AuditSample:
    acq = np.clip(np.asarray(acq, dtype=np.float64), 0, None)
    N = len(acq)
    tot = acq.sum()
    q = np.full(N, 1.0 / N) if tot <= 0 else lam / N + (1 - lam) * acq / tot
    q = q / q.sum()
    idx = rng.choice(N, size=n, replace=True, p=q)
    return AuditSample(idx=idx, w=1.0 / (n * q[idx]), design="active")


# ------------------------------------------------------------------ metrics from (weighted) confusion matrices
def _metrics_from_cms(cms: np.ndarray, labels: np.ndarray, cls_idx: np.ndarray, benign: int | None) -> dict:
    """cms: (B, K, K) estimated confusion counts (rows true, columns predicted)."""
    tp = cms[:, cls_idx, cls_idx]
    fp = cms.sum(1)[:, cls_idx] - tp
    fn = cms.sum(2)[:, cls_idx] - tp
    den = 2 * tp + fp + fn
    macro = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean(1) if len(cls_idx) else np.full(len(cms), np.nan)
    if benign is None:                                   # no benign class anywhere: every flow is an attack
        att_t = np.ones(cms.shape[1], bool)
    else:
        att_t = np.arange(cms.shape[1]) != benign
    att_p = att_t
    TP = cms[:, att_t][:, :, att_p].sum((1, 2))
    FN = cms[:, att_t][:, :, ~att_p].sum((1, 2))
    FP = cms[:, ~att_t][:, :, att_p].sum((1, 2))
    TN = cms[:, ~att_t][:, :, ~att_p].sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_f1 = np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), np.nan)
        far = np.where(FP + TN > 0, FP / (FP + TN), np.nan)
        det = np.where(TP + FN > 0, TP / (TP + FN), np.nan)
        tot = cms.sum((1, 2))
        acc = np.where(tot > 0, np.trace(cms, axis1=1, axis2=2) / np.where(tot > 0, tot, 1), np.nan)
    return {"macro_f1": macro, "bin_f1": bin_f1, "false_alarm_rate": far, "detection_rate": det, "accuracy": acc}


def _codes(true_names, pred_names):
    labels = np.array(sorted(set(map(str, true_names)) | set(map(str, pred_names))))
    return np.searchsorted(labels, np.asarray(true_names, dtype=str)), \
        np.searchsorted(labels, np.asarray(pred_names, dtype=str)), labels


def _posterior_cms(k: np.ndarray, w: np.ndarray, K: int, N_total: float, B: int, rng,
                   prior: float = 0.5) -> np.ndarray:
    """Posterior draws of the window's confusion counts: Dirichlet over the K x K cells with a Jeffreys prior
    (0.5 per cell) plus the audit's weighted counts rescaled to the Kish effective sample size
    n_eff = (sum w)^2 / sum w^2. Unlike the bootstrap, this gives a sensible upper tail when an event (e.g. a
    false alarm) was not seen in the audit at all."""
    tot = w.sum()
    n_eff = tot ** 2 / (w ** 2).sum()
    counts = np.bincount(k, weights=w, minlength=K * K) * (n_eff / tot)
    draws = rng.dirichlet(counts + prior, size=B)          # proportions per cell
    return (draws * N_total).reshape(B, K, K)


def _posterior2(k, w, K, labels, cls_idx, benign, point, m, restrict_to, B, rng, level, design,
                unseen_allowance=False) -> dict:
    """Intervals for rare events: weighted counts rescaled to the Kish effective sample size.
    - binary F1, detection rate, false-alarm rate: Dirichlet posterior of the collapsed 2 x 2 table (attack/benign x
      flagged/not) with a Jeffreys prior (0.5 per cell: 2 pseudo-flows in total), so a zero count still has an
      upper tail;
    - macro-F1: Dirichlet posterior of the K x K table with a weak prior (1 pseudo-flow in total), and the class set
      (estimated true count >= m) is re-decided in every draw, so the risk of having missed a rare class shows up."""
    tot = w.sum()
    n_eff = tot ** 2 / (w ** 2).sum()
    counts = np.bincount(k, weights=w, minlength=K * K) * (n_eff / tot)
    a = (1 - level) / 2
    out = {"classes": [str(labels[i]) for i in cls_idx], "n": int(len(k)), "design": design, **point}
    # binary 2 x 2: cells (attack, flagged), (attack, not), (benign, flagged), (benign, not)
    C = counts.reshape(K, K)
    att = np.ones(K, bool) if benign is None else np.arange(K) != benign
    c4 = np.array([C[att][:, att].sum(), C[att][:, ~att].sum(), C[~att][:, att].sum(), C[~att][:, ~att].sum()])
    d4 = rng.dirichlet(c4 + 0.5, size=B)
    TP, FN, FP, TN = d4.T
    sims = {"bin_f1": 2 * TP / (2 * TP + FP + FN), "detection_rate": TP / (TP + FN), "false_alarm_rate": FP / (FP + TN)}
    for kk, v in sims.items():
        if np.isfinite(point[kk]):
            out[kk + "_lo"], out[kk + "_hi"] = (float(x) for x in np.quantile(v, [a, 1 - a]))
        else:
            out[kk + "_lo"] = out[kk + "_hi"] = float("nan")
    # boundary convention (as Clopper-Pearson): an event type never seen in the audit cannot be ruled out as absent,
    # so the interval reaches the boundary (found on CIC, where the true false-alarm rate is exactly 0)
    if c4[2] == 0 and np.isfinite(point["false_alarm_rate"]):
        out["false_alarm_rate_lo"] = 0.0
    if c4[1] == 0 and np.isfinite(point["detection_rate"]):
        out["detection_rate_hi"] = 1.0
    if c4[1] == 0 and c4[2] == 0 and np.isfinite(point["bin_f1"]):
        out["bin_f1_hi"] = 1.0
    # accuracy (share of flows classified correctly; used by the drift attribution): Jeffreys posterior of the
    # collapsed correct / wrong counts, boundary convention when no error was seen
    corr, wrong = float(np.trace(C)), float(C.sum() - np.trace(C))
    acc_draws = rng.beta(corr + 0.5, wrong + 0.5, size=B)
    out["accuracy_lo"], out["accuracy_hi"] = (float(x) for x in np.quantile(acc_draws, [a, 1 - a]))
    if wrong == 0:
        out["accuracy_hi"] = 1.0
    # macro-F1: K x K with total prior mass 1, class set per draw
    draws = rng.dirichlet(counts + 1.0 / (K * K), size=B).reshape(B, K, K) * tot
    allowed = np.ones(K, bool) if restrict_to is None else \
        np.array([lab.lower() in {c.lower() for c in restrict_to} for lab in labels])
    est_true = draws.sum(2)                                   # (B, K)
    mask = (est_true >= m) & allowed                          # class set re-decided per draw
    tp = draws[:, np.arange(K), np.arange(K)]
    den = 2 * tp + (draws.sum(1) - tp) + (est_true - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    cnt = mask.sum(1)
    with np.errstate(invalid="ignore"):
        vals = np.where(cnt > 0, (f1 * mask).sum(1) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(vals)
    out["macro_f1_lo"], out["macro_f1_hi"] = ((float(x) for x in np.quantile(vals[ok], [a, 1 - a])) if ok.any()
                                              else (float("nan"), float("nan")))
    if np.isfinite(point["macro_f1"]) and C[~np.eye(K, dtype=bool)].sum() == 0:   # no error seen at all
        out["macro_f1_hi"] = 1.0
    if unseen_allowance and ok.any():
        # posterior3: a class with >= m flows in a window of N flows is missed by all
        # n_eff audited draws with probability (1 - m/N)^n_eff. If that is not negligible, the lower bound also allows
        # one unseen class with F1 = 0 (a silent failure the audit could not see).
        p_miss = (1 - m / max(tot, m)) ** n_eff
        if p_miss > a:
            # boundary correction: the unseen class's flows (share at most
            # 1 - a^(1/n_eff)) are predicted as some seen class and lower that class's F1 as well, so the true value
            # can sit just below sum/(cnt+1) (e.g. 0.49998 vs 0.5 on CIC total-miss windows)
            frac = 1 - a ** (1 / max(n_eff, 1.0))
            extra = np.where(cnt > 0, ((f1 * mask).sum(1) - frac) / (cnt + 1), 0.0)
            out["macro_f1_lo"] = min(out["macro_f1_lo"], float(np.quantile(extra, a)))
        out["p_unseen_class"] = float(p_miss)
    return out


def estimate(sample: AuditSample, true_names, pred_names, *, m: int, restrict_to=None, B: int = 500,
             seed: int = 0, level: float = 0.95, interval: str = "bootstrap") -> dict:
    """Point estimates and bootstrap intervals from an audit sample.

    true_names: labels revealed for the audited draws (same order as sample.idx); pred_names: detector predictions
    for the audited draws. The class set for macro-F1 is the set of classes whose ESTIMATED true count is >= m
    (the monitor cannot know the true class set), optionally restricted to `restrict_to`.
    """
    t, p, labels = _codes(true_names, pred_names)
    K = len(labels)
    k = t * K + p
    w = sample.w
    cm = np.bincount(k, weights=w, minlength=K * K).reshape(1, K, K)
    est_true = cm[0].sum(1)
    cls = [i for i in range(K) if est_true[i] >= m]
    if restrict_to is not None:
        allowed = {c.lower() for c in restrict_to}
        cls = [i for i in cls if labels[i].lower() in allowed]
    cls_idx = np.array(cls, dtype=int)
    benign = int(np.searchsorted(labels, BENIGN)) if BENIGN in labels else None
    point = {kk: float(v[0]) for kk, v in _metrics_from_cms(cm, labels, cls_idx, benign).items()}
    rng = np.random.default_rng(seed)
    n = len(k)
    if interval in ("posterior2", "posterior3"):
        return _posterior2(k, w, K, labels, cls_idx, benign, point, m, restrict_to, B, rng, level, sample.design,
                           unseen_allowance=(interval == "posterior3"))
    if interval == "posterior":
        cms = _posterior_cms(k, w, K, float(w.sum()), B, rng)
        boot = _metrics_from_cms(cms, labels, cls_idx, benign)
        a = (1 - level) / 2
        out = {"classes": [str(labels[i]) for i in cls_idx], "n": int(n), "design": sample.design}
        for kk, v in point.items():
            ok = np.isfinite(boot[kk])
            lo, hi = np.quantile(boot[kk][ok], [a, 1 - a]) if ok.any() else (np.nan, np.nan)
            out[kk], out[kk + "_lo"], out[kk + "_hi"] = v, float(lo), float(hi)
        return out
    # bootstrap: resample the audited draws (within strata if stratified)
    if sample.strata is None:
        draws = rng.integers(0, n, (B, n))
    else:
        draws = np.empty((B, n), dtype=np.int64)
        col = 0
        for h in np.unique(sample.strata):
            pos = np.flatnonzero(sample.strata == h)
            draws[:, col:col + len(pos)] = pos[rng.integers(0, len(pos), (B, len(pos)))]
            col += len(pos)
    flat = (np.arange(B)[:, None] * (K * K) + k[draws]).ravel()
    cms = np.bincount(flat, weights=w[draws].ravel(), minlength=B * K * K).reshape(B, K, K)
    boot = _metrics_from_cms(cms, labels, cls_idx, benign)
    a = (1 - level) / 2
    out = {"classes": [str(labels[i]) for i in cls_idx], "n": int(n), "design": sample.design}
    for kk, v in point.items():
        lo, hi = (np.nanquantile(boot[kk], [a, 1 - a]) if np.isfinite(boot[kk]).any() else (np.nan, np.nan))
        out[kk] = v
        out[kk + "_lo"] = float(lo)
        out[kk + "_hi"] = float(hi)
    return out


# ------------------------------------------------------------------ true values (for evaluation only)
def true_metrics(true_names, pred_names, *, m: int, restrict_to=None) -> dict:
    """The same metrics on the full window with all labels (used only to score the estimators)."""
    t, p, labels = _codes(true_names, pred_names)
    K = len(labels)
    cm = np.bincount(t * K + p, minlength=K * K).reshape(1, K, K).astype(float)
    true_counts = cm[0].sum(1)
    cls = [i for i in range(K) if true_counts[i] >= m]
    if restrict_to is not None:
        allowed = {c.lower() for c in restrict_to}
        cls = [i for i in cls if labels[i].lower() in allowed]
    benign = int(np.searchsorted(labels, BENIGN)) if BENIGN in labels else None
    r = {kk: float(v[0]) for kk, v in _metrics_from_cms(cm, labels, np.array(cls, dtype=int), benign).items()}
    r["classes"] = [str(labels[i]) for i in cls]
    return r
