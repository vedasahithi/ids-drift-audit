"""Real temporal (S1) and cross-network (S2) streams.

Per deployment window: the true metrics (all labels), the label-free estimates (soft confusion, ATC), the alarm
signals, a diagnosis from one active audit, and the audit estimates of every design x budget x repetition.
Alarm thresholds come from unshifted windows of the source test block.

Outputs: results/phase2b/streams_<tag>/{windows.csv, audits.csv.gz, meta.json}.
Usage: python phase2b_streams.py --config configs/phase0_cic_s1a.json --detector lightgbm [--deploy-config ...]
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
from driftids.alarms import (conformal_rates, density_ratio, disagreement, domain_auc, fit_conformal,
                             mean_confidence)
from driftids.attribution import attribute, bbse_class_weights, predicted_accuracy, residual_auc
from driftids.audit import (INTERVAL, active_audit, estimate, random_audit, stratified_active_audit, stratified_audit,
                            true_metrics)
from driftids.detectors import make_detector
from driftids.labelfree import atc, atc_threshold, soft_confusion

from driftids.audit import results_root
ROOT = Path(__file__).resolve().parent
BUDGETS = (50, 100, 200, 500, 1000)
DESIGNS = ("random", "stratified", "active", "stratactive")
METRICS = ("macro_f1", "bin_f1", "false_alarm_rate", "detection_rate")
M = 20
N_ATTR = 200
SRC_N = 10_000


def per_class_f1(y, p, classes):
    from sklearn.metrics import f1_score
    if not classes:
        return {}
    v = f1_score(np.asarray(y, str), np.asarray(p, str), labels=list(classes), average=None, zero_division=0)
    return {c: round(float(x), 4) for c, x in zip(classes, v)}


def window_job(k, w, ctx, reps, budgets, restrict):
    Xw, yw, pred, proba, pred2 = w["X"], w["y"], w["pred"], w["proba"], w["pred2"]
    classes = ctx["classes"]
    truth = true_metrics(yw, pred, m=M, restrict_to=restrict)
    acc_true = float(np.mean(yw == pred))
    vals, counts = np.unique(yw, return_counts=True)
    present = {str(v) for v, c in zip(vals, counts) if c >= M}
    new_classes = sorted(c for c in present if c.lower() not in ctx["train_lower"])
    att = yw != "Benign"
    n_att = int(att.sum())
    det_true = float((pred[att] != "Benign").mean()) if n_att else float("nan")
    level = ("benign_dominated" if n_att < M else "total_miss" if det_true < 0.05 else
             "full" if det_true > 0.95 else "partial")
    sc = soft_confusion(proba, classes, pred, m=M, restrict_to=restrict)
    at = atc(proba, classes, pred, ctx["tau"], m=M, restrict_to=restrict)
    rates = conformal_rates(proba, ctx["cp"])
    auc, pw = domain_auc(ctx["src"], Xw, seed=k)
    ratio = np.clip(pw, 0.01, 0.99) / (1 - np.clip(pw, 0.01, 0.99))
    unc = 1 - proba.max(1)
    acq = 0.5 * unc / max(unc.mean(), 1e-12) + 0.5 * ratio / ratio.mean()
    codes = np.searchsorted(np.unique(pred), pred)
    # attribution  from one active audit of N_ATTR flows
    rng = np.random.default_rng(phase1.seed_for("attr", ctx["tag"], k))
    smp = active_audit(acq, N_ATTR, rng, lam=0.5)
    ea = estimate(smp, yw[smp.idx], pred[smp.idx], m=M, B=300, seed=k, interval=INTERVAL)
    bw = bbse_class_weights(ctx["pred_cal"], ctx["y_cal"], classes, pred)
    res = residual_auc(ctx["src"], ctx["src_y"], Xw, bw, classes, seed=k)
    rc = density_ratio(ctx["X_cal"], Xw, ctx["X_cal"], seed=k)
    diag = attribute(audit_unseen_share=float(smp.w[~np.isin(yw[smp.idx], classes)].sum() / smp.w.sum()),
                     audit_lo=ea["accuracy_lo"], audit_hi=ea["accuracy_hi"], dom_auc=auc, res_auc=res,
                     auc_threshold=ctx["auc_thr"],
                     pred_none=predicted_accuracy(ctx["pred_cal"], ctx["y_cal"], classes),
                     pred_prior=predicted_accuracy(ctx["pred_cal"], ctx["y_cal"], classes, class_weights=bw),
                     pred_cov=predicted_accuracy(ctx["pred_cal"], ctx["y_cal"], classes, flow_weights=rc))
    row = {"window": k, "t_start": w["t_start"], **{"true_" + m: truth[m] for m in METRICS},
           "true_accuracy": acc_true, "classes": "|".join(truth["classes"]), "new_classes": "|".join(new_classes),
           "new_attack_window": bool(new_classes), "n_attack": n_att, "failure_level": level,
           "infiltration": int((yw == "Infiltration").sum()) >= M,
           "per_class_f1": json.dumps(per_class_f1(yw, pred, truth["classes"])),
           **{"soft_" + m: sc[m] for m in METRICS}, **{"atc_" + m: at[m] for m in METRICS},
           "abstention": rates["abstention"], "empty_set": rates["empty_set"], "domain_auc": auc,
           "disagreement": disagreement(pred, pred2), "neg_confidence": -mean_confidence(proba),
           "alarm": bool(rates["abstention"] > ctx["ab_thr"] or auc > ctx["auc_thr"]),
           "diagnosis": diag, "res_auc": res}
    audits = []
    for n in budgets:
        for r in range(reps):
            g = np.random.default_rng(phase1.seed_for("audit", ctx["tag"], k, n, r))
            for design in DESIGNS:
                if design == "random":
                    s = random_audit(len(yw), n, g)
                elif design == "stratified":
                    s = stratified_audit(codes, n, g)
                elif design == "active":
                    s = active_audit(acq, n, g, lam=0.5)
                else:
                    s = stratified_active_audit(codes, acq, n, g, lam=0.5)
                e = estimate(s, yw[s.idx], pred[s.idx], m=M, B=200, seed=r, interval=INTERVAL,
                             restrict_to=restrict)
                audits.append({"window": k, "n": n, "rep": r, "design": design,
                               **{m: e[m] for m in METRICS}, **{m + "_lo": e[m + "_lo"] for m in METRICS},
                               **{m + "_hi": e[m + "_hi"] for m in METRICS}})
    return row, audits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deploy-config", default=None)
    ap.add_argument("--detector", default="lightgbm", choices=["hgb", "lightgbm"])
    ap.add_argument("--second", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--null", type=int, default=100)
    ap.add_argument("--max-windows", type=int, default=None, help="smoke test: only the first windows")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    second = a.second or ("extratrees" if a.detector == "hgb" else "hgb")
    t0 = time.time()
    cfg, f = phase0._read_cfg_flows(str(ROOT / a.config))
    f_y = np.asarray(f.attack, dtype=object).astype(str)
    sp = phase0._split(cfg, f)
    excl, _ = phase0._shortcut_exclusions(cfg, f.features)
    cols = [c for c in f.features if c not in excl]
    X = f.X[cols]
    tr, va, cal, te = (sp.blocks[b] for b in ("train", "val", "calib", "test"))
    det, det_info = make_detector(a.detector, X.iloc[tr].to_numpy(), f_y[tr], X.iloc[va].to_numpy(), f_y[va])
    det2, _ = make_detector(second, X.iloc[tr].to_numpy(), f_y[tr], X.iloc[va].to_numpy(), f_y[va])
    classes = det.classes_.astype(str)
    Xc = X.iloc[cal].to_numpy()
    pc = det.predict_proba(Xc)
    rng = np.random.default_rng(0)
    src_idx = rng.choice(tr, size=min(SRC_N, len(tr)), replace=False)
    ctx = {"classes": classes, "train_lower": {c.lower() for c in np.unique(f_y[tr])},
           "tau": atc_threshold(pc, classes[pc.argmax(1)] == f_y[cal]),
           "cp": fit_conformal(pc, f_y[cal], classes), "src": X.iloc[src_idx].to_numpy(), "src_y": f_y[src_idx],
           "pred_cal": classes[pc.argmax(1)], "y_cal": f_y[cal], "X_cal": Xc}
    # deployment windows
    if a.deploy_config:
        dcfg, g = phase0._read_cfg_flows(str(ROOT / a.deploy_config))
        if g.features != f.features:
            raise SystemExit("feature columns differ")
        gX, g_y = g.X[cols], np.asarray(g.attack, dtype=object).astype(str)
        W = dcfg["window_size"]
        wins = [np.arange(s, s + W) for s in range(0, len(g_y) - W + 1, W)]
        restrict = sorted({c.lower() for c in np.unique(f_y[tr])} & {c.lower() for c in np.unique(g_y)})
        starts = g.start
        tag = f"{cfg['name']}_to_{dcfg['name']}"
    else:
        gX, g_y, wins, restrict, starts = X, f_y, sp.windows, None, f.start
        W = cfg["window_size"]
        tag = cfg["name"]
    if a.max_windows:
        wins = wins[: a.max_windows]
    tag += "" if a.detector == "lightgbm" else f"_{a.detector}"
    ctx["tag"] = tag
    # thresholds from unshifted source windows (same size as deployment windows)
    nulls = [te[np.random.default_rng(phase1.seed_for("null2b", j)).choice(len(te), W, replace=True)]
             for j in range(a.null)]
    auc_null = Parallel(n_jobs=a.jobs)(delayed(domain_auc)(ctx["src"], X.iloc[ix].to_numpy(), j)
                                       for j, ix in enumerate(nulls))
    ctx["auc_thr"] = float(np.quantile([v[0] for v in auc_null], 0.99))
    ab_null = [conformal_rates(det.predict_proba(X.iloc[ix].to_numpy()), ctx["cp"])["abstention"] for ix in nulls]
    ctx["ab_thr"] = float(np.quantile(ab_null, 0.99))
    print(f"prepared {tag}: {len(wins)} windows, detector {det_info}, thresholds auc {ctx['auc_thr']:.3f} "
          f"abstention {ctx['ab_thr']:.3f}, {time.time() - t0:.0f}s", flush=True)

    def win_data(ix):
        Xw = gX.iloc[ix].to_numpy()
        proba = det.predict_proba(Xw)
        return {"X": Xw, "y": g_y[ix], "proba": proba, "pred": classes[proba.argmax(1)],
                "pred2": det2.predict(Xw).astype(str), "t_start": int(starts[ix[0]])}

    res = Parallel(n_jobs=a.jobs, verbose=3)(delayed(window_job)(k, win_data(ix), ctx, a.reps, BUDGETS, restrict)
                                             for k, ix in enumerate(wins))
    out = results_root(ROOT) / "phase2b" / f"streams_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r, _ in res]).to_csv(out / "windows.csv", index=False)
    pd.DataFrame([x for _, au in res for x in au]).to_csv(out / "audits.csv.gz", index=False)
    meta = {"tag": tag, "detector": det_info, "second": second, "windows": len(wins), "window_size": W,
            "restrict_classes": restrict, "auc_threshold": ctx["auc_thr"], "abstention_threshold": ctx["ab_thr"],
            "train_classes": pd.Series(f_y[tr]).value_counts().to_dict(), "reps": a.reps, "budgets": list(BUDGETS),
            "minutes": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"written to {out} in {meta['minutes']} min")


if __name__ == "__main__":
    main()
