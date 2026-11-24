"""Controlled shifts (S3): audit designs and label-free estimators on shifted windows.

Windows of 10,000 flows are drawn from the source test block with four shift types (covariate, prior, concept, new
class) at three strengths. For each window and budget, the audit designs (random, stratified, active, hybrid) are
applied repeatedly and the estimates and intervals of macro-F1 and the binary metrics are compared with the true
values; soft confusion and ATC give the label-free estimates.

Outputs: results/phase1/<name>_s3_<interval>_<detector>/{raw.csv.gz, labelfree.csv, summary.csv, meta.json}.
Usage: python phase1.py --config configs/phase0_cic_s1a.json --detector lightgbm --interval posterior3
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

import phase0
from driftids import shifts
from driftids.audit import (active_audit, estimate, random_audit, stratified_active_audit, stratified_audit,
                            true_metrics)
from driftids.detectors import make_detector
from driftids.labelfree import atc, atc_threshold, soft_confusion

from driftids.audit import results_root
ROOT = Path(__file__).resolve().parent
OUT = results_root(ROOT) / "phase1" / "unsw_s3"
BUDGETS = (50, 100, 200, 500, 1000)
LAMS = (0.3, 0.5, 0.7)
ACQS = ("uncert", "novel", "combo")
WINDOW = 10_000
M = 20
METRICS = ("macro_f1", "bin_f1", "false_alarm_rate", "detection_rate")


def prepare(config: str = "configs/phase0_unsw.json", detector: str = "hgb", second: str | None = None):
    """Pool, label-free source sample and settings for one dataset. The class held out for the new-class shift is
    the second most frequent attack class of the source train block (UNSW: Fuzzers; fixed rule, ."""
    cfg, f = phase0._read_cfg_flows(str(ROOT / config))
    sp = phase0._split(cfg, f)
    excl, _ = phase0._shortcut_exclusions(cfg, f.features)
    cols = [c for c in f.features if c not in excl]
    y = np.asarray(f.attack, dtype=object).astype(str)
    tr, cal, te = sp.blocks["train"], sp.blocks["calib"], sp.blocks["test"]
    X = f.X[cols]
    va = sp.blocks["val"]
    attack_rank = pd.Series(y[tr][y[tr] != "Benign"]).value_counts()
    # A source train block with a single attack class (ToN-IoT S1c: scanning only) has no class to hold out: the
    # new-class shift is then skipped (decision  and the "_d" detector is the main detector. Datasets with
    # two or more attack classes follow exactly the earlier code path.
    HELD_OUT = attack_rank.index[1] if len(attack_rank) >= 2 else None
    det, det_info = make_detector(detector, X.iloc[tr].to_numpy(), y[tr], X.iloc[va].to_numpy(), y[va])
    if HELD_OUT is None:
        tr_d, va_d, det_d, det_d_info = tr, va, det, det_info
    else:
        tr_d, va_d = tr[y[tr] != HELD_OUT], va[y[va] != HELD_OUT]
        det_d, det_d_info = make_detector(detector, X.iloc[tr_d].to_numpy(), y[tr_d], X.iloc[va_d].to_numpy(),
                                          y[va_d])
    Xp = X.iloc[te].to_numpy()
    pool = {"y": y[te], "X": Xp,
            "pred": det.predict(Xp).astype(str), "proba": det.predict_proba(Xp), "classes": det.classes_.astype(str),
            "pred_d": det_d.predict(Xp).astype(str), "proba_d": det_d.predict_proba(Xp),
            "classes_d": det_d.classes_.astype(str),
            "duration": f.X["FLOW_DURATION_MILLISECONDS"].to_numpy()[te], "in_bytes": f.X["IN_BYTES"].to_numpy()[te]}
    Xc = X.iloc[cal].to_numpy()
    pc = det.predict_proba(Xc)
    tau = atc_threshold(pc, det.predict(Xc).astype(str) == y[cal])
    keep = y[cal] != HELD_OUT
    pcd = det_d.predict_proba(Xc[keep])
    tau_d = atc_threshold(pcd, det_d.predict(Xc[keep]).astype(str) == y[cal][keep])
    # extra outputs for the stage-2a alarms (calibration probabilities, second detector); trained after everything
    #  uses, so  results are unchanged
    pool["proba_cal"], pool["y_cal"] = pc, y[cal]
    pool["proba_cal_d"], pool["y_cal_d"] = pcd, y[cal][keep]
    if second is not None:
        det2, info2 = make_detector(second, X.iloc[tr].to_numpy(), y[tr], X.iloc[va].to_numpy(), y[va])
        det2_d = det2 if HELD_OUT is None else \
            make_detector(second, X.iloc[tr_d].to_numpy(), y[tr_d], X.iloc[va_d].to_numpy(), y[va_d])[0]
        pool["pred2"], pool["pred2_d"] = det2.predict(Xp).astype(str), det2_d.predict(Xp).astype(str)
        pool["second_detector"] = info2
    trq = f.X.iloc[tr]
    rng = np.random.default_rng(0)
    src_idx = rng.choice(tr, size=min(WINDOW, len(tr)), replace=False)
    src_sample = X.iloc[src_idx].to_numpy()
    pool["src_y"] = y[src_idx]                       # labels of the source sample (source data; for BBSE re-weighting)
    pool["pred_cal"] = det.predict(Xc).astype(str)
    pool["pred_cal_d"] = det_d.predict(Xc[keep]).astype(str)
    pool["X_cal"], pool["X_cal_d"] = Xc, Xc[keep]
    pool["X_tr"], pool["y_tr"] = X.iloc[tr].to_numpy(), y[tr]          # for retraining in the policy simulation
    top_attack = pd.Series(y[tr][y[tr] != "Benign"]).value_counts().index[0]
    meta = {"q70_duration": float(np.quantile(trq["FLOW_DURATION_MILLISECONDS"], 0.70)),
            "q90_in_bytes": float(np.quantile(trq["IN_BYTES"], 0.90)), "concept_label": top_attack,
            "tau": tau, "tau_d": tau_d, "features": cols, "excluded": excl, "pool_size": int(len(te)),
            "pool_classes": pd.Series(y[te]).value_counts().to_dict(), "held_out": HELD_OUT, "config": config,
            "detector": det_info, "detector_new_class": det_d_info,
            "train_classes": pd.Series(y[tr]).value_counts().to_dict()}
    return pool, src_sample, meta


def s3_types(meta) -> tuple:
    """S3 shift types that exist for this dataset (no new-class shift without a held-out class)."""
    return tuple(t for t in shifts.TYPES if t != "new_class" or meta["held_out"] is not None)


def seed_for(*parts) -> int:
    """Deterministic seed from the setting (CRC32 of its text). Python's hash() of strings changes between runs
    (hash randomisation), which made earlier runs non-repeatable (fixed ."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def novelty(src: np.ndarray, win: np.ndarray, seed: int) -> np.ndarray:
    """Label-free density ratio of each window flow (source vs. window classifier, 2-fold cross-fitting)."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(win))
    sfold = rng.integers(0, 2, len(src))
    r = np.empty(len(win))
    for k in (0, 1):
        Xtr = np.vstack([src[sfold != k], win[fold != k]])
        ytr = np.r_[np.zeros((sfold != k).sum()), np.ones((fold != k).sum())]
        clf = HistGradientBoostingClassifier(max_iter=60, random_state=seed).fit(Xtr, ytr)
        p = np.clip(clf.predict_proba(win[fold == k])[:, 1], 0.01, 0.99)
        r[fold == k] = p / (1 - p)
    return r


def make_window(pool, meta, stype, s, wseed):
    """Positions and true labels of one S3 window (same seeds as every  run). stype 'none' = an unshifted
    window drawn uniformly from the pool (null windows for alarm thresholds)."""
    if stype == "none":
        rng = np.random.default_rng(seed_for("null", wseed))
        idx = rng.choice(len(pool["y"]), WINDOW, replace=True)
        return idx, pool["y"][idx]
    rng = np.random.default_rng(10_000 * (shifts.TYPES.index(stype) + 1) + int(100 * s) * 100 + wseed)
    if stype == "covariate":
        return shifts.covariate(pool["duration"], pool["y"], WINDOW, s, rng, meta["q70_duration"])
    if stype == "prior":
        return shifts.prior(pool["y"], WINDOW, s, rng)
    if stype == "concept":
        return shifts.concept(pool["in_bytes"], pool["y"], WINDOW, s, rng, meta["q90_in_bytes"],
                              meta["concept_label"])
    return shifts.new_class(pool["y"], WINDOW, s, rng, meta["held_out"])


def one_window(pool, src_sample, meta, stype, s, wseed, reps, budgets, interval="posterior2"):
    idx, yw = make_window(pool, meta, stype, s, wseed)
    d = "_d" if stype == "new_class" else ""
    pred, proba, classes = pool["pred" + d][idx], pool["proba" + d][idx], pool["classes" + d]
    tau = meta["tau" + d]
    key = {"type": stype, "strength": s, "window": wseed}
    truth = true_metrics(yw, pred, m=M)
    lf_rows = []
    for name, est in (("soft_confusion", soft_confusion(proba, classes, pred, m=M)),
                      ("atc", atc(proba, classes, pred, tau, m=M))):
        lf_rows.append({**key, "estimator": name, **{k: est[k] for k in METRICS},
                        **{"true_" + k: truth[k] for k in METRICS}})
    unc = 1 - proba.max(1)
    nov = novelty(src_sample, pool["X"][idx], seed=wseed)
    acqs = {"uncert": unc, "novel": nov, "combo": 0.5 * unc / max(unc.mean(), 1e-12) + 0.5 * nov / nov.mean()}
    pred_codes = np.searchsorted(np.unique(pred), pred)
    rows = []
    for n in budgets:
        for rep in range(reps):
            r = np.random.default_rng(seed_for(stype, s, wseed, n, rep))
            samples = [("random", None, None, random_audit(WINDOW, n, r)),
                       ("stratified", None, None, stratified_audit(pred_codes, n, r))]
            for a in ACQS:
                for lam in LAMS:
                    samples.append(("active", a, lam, active_audit(acqs[a], n, r, lam=lam)))
            # hybrid
            samples.append(("stratactive", "combo", 0.5, stratified_active_audit(pred_codes, acqs["combo"], n, r, lam=0.5)))
            for design, a, lam, smp in samples:
                e = estimate(smp, yw[smp.idx], pred[smp.idx], m=M, B=200, seed=rep, interval=interval)
                row = {**key, "n": n, "rep": rep, "design": design, "acq": a, "lam": lam}
                for k in METRICS:
                    row[k] = e[k]
                    row[k + "_lo"] = e[k + "_lo"]
                    row[k + "_hi"] = e[k + "_hi"]
                    row["true_" + k] = truth[k]
                rows.append(row)
    return rows, lf_rows


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.assign(acq=raw["acq"].fillna("-"), lam=raw["lam"].fillna(-1))
    out = []
    for k in METRICS:
        d = g.assign(err=g[k] - g["true_" + k],
                     cover=(g[k + "_lo"] <= g["true_" + k]) & (g["true_" + k] <= g[k + "_hi"]),
                     width=g[k + "_hi"] - g[k + "_lo"]).dropna(subset=["err"])
        s = d.groupby(["type", "strength", "design", "acq", "lam", "n"]).agg(
            bias=("err", "mean"), mae=("err", lambda e: e.abs().mean()), coverage=("cover", "mean"),
            width=("width", "mean"), runs=("err", "size")).reset_index()
        s.insert(0, "metric", k)
        s["passes"] = (s["bias"].abs() <= 0.2 * s["width"]) & ((s["coverage"] >= 0.90) | (s["n"] < 100))
        out.append(s)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 window, 2 reps, n in {100, 200}")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--interval", choices=["bootstrap", "posterior2", "posterior3"], default="posterior3")
    ap.add_argument("--config", default="configs/phase0_unsw.json")
    ap.add_argument("--detector", choices=["hgb", "lightgbm"], default="hgb")
    a = ap.parse_args()
    windows, reps, budgets = (1, 2, (100, 200)) if a.quick else (a.windows, a.reps, BUDGETS)
    dname = phase0._config(str(ROOT / a.config))["name"]
    name = f"{dname}_s3" + ("" if a.interval == "bootstrap" else f"_{a.interval}") +         ("" if a.detector == "hgb" else f"_{a.detector}")
    out = OUT.parent / (name + "_quick" if a.quick else name)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool, src, meta = prepare(a.config, a.detector)
    print(f"prepared in {time.time() - t0:.0f}s: pool {meta['pool_size']:,} flows {meta['pool_classes']}", flush=True)
    tasks = [(t, s, w) for t in s3_types(meta) for s in shifts.STRENGTHS for w in range(windows)]
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_window)(pool, src, meta, t, s, w, reps, budgets, a.interval)
                                             for t, s, w in tasks)
    raw = pd.DataFrame([r for rows, _ in res for r in rows])
    lf = pd.DataFrame([r for _, lfr in res for r in lfr])
    raw.to_csv(out / "raw.csv.gz", index=False)
    lf.to_csv(out / "labelfree.csv", index=False)
    summarise(raw).to_csv(out / "summary.csv", index=False)
    meta["minutes"] = round((time.time() - t0) / 60, 1)
    meta["windows"], meta["reps"], meta["budgets"], meta["interval"] = windows, reps, list(budgets), a.interval
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"written to {out} in {meta['minutes']} min")


if __name__ == "__main__":
    main()
