"""Policies on a real temporal stream (H5) and the post-retraining windows (H2).

Runs phase2a_policies.run_seed on the deployment windows of a stream. The starting model and every retraining use a
fixed class-stratified sample of 100,000 source train flows (classes with fewer than 1,000 flows kept whole) plus the
policy's labelled flows; the LightGBM setting is chosen once on the source validation block. Seeds are independent
label and audit draws. For seed 0, the five windows after each periodic retraining are re-audited with the retrained
model (post-retraining windows).

Outputs: results/phase2b/policies_<tag>/{trace.csv, summary.json}, results/phase2b/postretrain_<tag>/{windows.csv,
audits.csv.gz}. Usage: python phase2b_policies.py --config configs/phase0_cic_s1a.json --seeds 10 --jobs 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import phase0
import phase1
import phase2a_policies as P
from driftids.alarms import domain_auc
from driftids.audit import (INTERVAL, active_audit, estimate, random_audit, stratified_active_audit, stratified_audit,
                            true_metrics)
from driftids.detectors import choose_lightgbm
from driftids.policies import PolicyConfig

from driftids.audit import results_root
ROOT = Path(__file__).resolve().parent
M = phase1.M
SAMPLE_N, KEEP_SMALL = 100_000, 1_000
N_NULL, SRC_N = 40, 10_000
BUDGETS = (50, 100, 200, 500, 1000)
POST_REPS, POST_AFTER = 10, 5


def stratified_sample(idx, y, n=SAMPLE_N, keep_small=KEEP_SMALL, seed=0):
    rng = np.random.default_rng(seed)
    vals, counts = np.unique(y, return_counts=True)
    small = counts < keep_small
    budget = n - counts[small].sum()
    big_total = counts[~small].sum()
    out = []
    for v, c, sm in zip(vals, counts, small):
        pos = idx[y == v]
        k = c if sm else min(c, max(1, int(round(budget * c / big_total))))
        out.append(pos if k >= c else rng.choice(pos, k, replace=False))
    return np.sort(np.concatenate(out))


def post_window(j, rt, w, model, Xw, yw, src, tag, train_lower, reps):
    proba = model.predict_proba(Xw)
    classes = model.classes_.astype(str)
    pred = classes[proba.argmax(1)]
    truth = true_metrics(yw, pred, m=M)
    att = yw != "Benign"
    det = float((pred[att] != "Benign").mean()) if att.sum() else float("nan")
    level = ("benign_dominated" if att.sum() < M else "total_miss" if det < 0.05 else
             "full" if det > 0.95 else "partial")
    vals, counts = np.unique(yw, return_counts=True)
    new = any(str(v).lower() not in train_lower for v, c in zip(vals, counts) if c >= M)
    _, pw = domain_auc(src, Xw, seed=w)
    ratio = np.clip(pw, 0.01, 0.99) / (1 - np.clip(pw, 0.01, 0.99))
    unc = 1 - proba.max(1)
    acq = 0.5 * unc / max(unc.mean(), 1e-12) + 0.5 * ratio / ratio.mean()
    codes = np.searchsorted(np.unique(pred), pred)
    row = {"window": j, "retrain_t": rt, "stream_window": w, "true_macro_f1": truth["macro_f1"],
           "infiltration": int((yw == "Infiltration").sum()) >= M, "new_attack_window": new, "failure_level": level,
           "soft_macro_f1": np.nan, "atc_macro_f1": np.nan}
    audits = []
    for n in BUDGETS:
        for r in range(reps):
            g = np.random.default_rng(phase1.seed_for("post", tag, rt, w, n, r))
            for design in ("random", "stratified", "active", "stratactive"):
                s = (random_audit(len(yw), n, g) if design == "random" else
                     stratified_audit(codes, n, g) if design == "stratified" else
                     active_audit(acq, n, g, lam=0.5) if design == "active" else
                     stratified_active_audit(codes, acq, n, g, lam=0.5))
                e = estimate(s, yw[s.idx], pred[s.idx], m=M, B=200, seed=r, interval=INTERVAL)
                audits.append({"window": j, "n": n, "rep": r, "design": design, "macro_f1": e["macro_f1"],
                               "macro_f1_lo": e["macro_f1_lo"], "macro_f1_hi": e["macro_f1_hi"]})
    return row, audits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--detector", default="lightgbm", choices=["hgb", "lightgbm"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (extra seeds for H5 in the final run)")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-post", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg, f = phase0._read_cfg_flows(str(ROOT / a.config))
    y = np.asarray(f.attack, dtype=object).astype(str)
    sp = phase0._split(cfg, f)
    excl, _ = phase0._shortcut_exclusions(cfg, f.features)
    cols = [c for c in f.features if c not in excl]
    X = f.X[cols]
    tr, va, cal, te = (sp.blocks[b] for b in ("train", "val", "calib", "test"))
    tr_s = stratified_sample(tr, y[tr])
    setting, scores = None, None
    if a.detector == "lightgbm":
        _, setting, scores = choose_lightgbm(X.iloc[tr_s].to_numpy(), y[tr_s], X.iloc[va].to_numpy(), y[va])
    wins = sp.windows[: a.max_windows] if a.max_windows else sp.windows
    W = cfg["window_size"]
    stream_rows = np.concatenate(wins)
    pool = {"X": np.vstack([X.iloc[stream_rows].to_numpy(), X.iloc[te].to_numpy()]).astype(np.float32),
            "y": np.r_[y[stream_rows], y[te]]}
    n_stream = len(stream_rows)
    nulls = [n_stream + np.random.default_rng(phase1.seed_for("null2bpol", j)).choice(len(te), W, replace=True)
             for j in range(N_NULL)]
    src_idx = np.random.default_rng(0).choice(tr_s, size=min(SRC_N, len(tr_s)), replace=False)
    pool.update({"X_tr": X.iloc[tr_s].to_numpy(), "y_tr": y[tr_s], "src_y": y[src_idx],
                 "X_cal": X.iloc[cal].to_numpy(), "y_cal": y[cal]})
    meta = {"held_out": "__none__"}
    env = P.Env(pool, X.iloc[src_idx].to_numpy(), meta, a.detector, null_idx=nulls, lgbm_setting=setting)
    train_lower = {c.lower() for c in np.unique(y[tr])}
    stream = []
    for t in range(len(wins)):
        pos = np.arange(t * W, (t + 1) * W)
        vals, counts = np.unique(pool["y"][pos], return_counts=True)
        new = any(str(v).lower() not in train_lower for v, c in zip(vals, counts) if c >= M)
        stream.append((t, "new_attack" if new else "known", 0.0, pos))
    tag = cfg["name"] + ("" if a.detector == "lightgbm" else f"_{a.detector}")
    print(f"prepared {tag}: {len(wins)} windows, retrain sample {len(tr_s):,} flows, setting {setting}, "
          f"thresholds auc {env.auc_thr:.3f}, {time.time() - t0:.0f}s", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(P.run_seed)(env, s, PolicyConfig(), stream, keep_models=(s == 0 and not a.no_post))
                                             for s in range(a.seed_start, a.seed_start + a.seeds))
    trace = pd.DataFrame([r for out in res for r in out[0]])
    out = results_root(ROOT) / "phase2b" / f"policies_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    trace.to_csv(out / "trace.csv", index=False)
    agg = trace.groupby(["policy", "seed"]).agg(mean_macro_f1=("macro_f1", "mean"), labels=("labels_cum", "max"),
                                                retrains=("action", lambda x: int((x == "retrain").sum())))
    summ = agg.groupby("policy").mean().round(4)
    by_phase = trace.pivot_table(index="policy", columns="phase", values="macro_f1").round(4)
    summary = {"tag": tag, "summary": summ.to_dict(orient="index"), "by_phase": by_phase.to_dict(orient="index"),
               "reference_macro_f1": [o[1] for o in res], "lgbm_setting": setting, "lgbm_val_scores": scores,
               "retrain_sample": int(len(tr_s)), "seeds": a.seeds, "windows": len(wins),
               "auc_threshold": env.auc_thr, "minutes_policies": round((time.time() - t0) / 60, 1)}
    print(summ.to_string())
    print(by_phase.to_string())
    if not a.no_post:
        models = res[0][2]
        jobs = [(rt, w, mdl) for rt, mdl in models for w in range(rt + 1, min(rt + 1 + POST_AFTER, len(wins)))]
        src = X.iloc[src_idx].to_numpy()
        post = Parallel(n_jobs=a.jobs, verbose=3)(
            delayed(post_window)(j, rt, w, mdl, pool["X"][w * W:(w + 1) * W], pool["y"][w * W:(w + 1) * W], src, tag,
                                 train_lower, POST_REPS) for j, (rt, w, mdl) in enumerate(jobs))
        pdir = results_root(ROOT) / "phase2b" / f"postretrain_{tag}"
        pdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r for r, _ in post]).to_csv(pdir / "windows.csv", index=False)
        pd.DataFrame([x for _, au in post for x in au]).to_csv(pdir / "audits.csv.gz", index=False)
        summary["post_retraining_windows"] = len(jobs)
    summary["minutes_total"] = round((time.time() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"written to {out} in {summary['minutes_total']} min")


if __name__ == "__main__":
    main()
