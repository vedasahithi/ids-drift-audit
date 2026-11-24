"""Drift diagnosis on controlled shifts (S3) and on unshifted null windows.

Per window: one active audit of 200 flows, the label-free signals (domain-classifier AUC, residual AUC after
re-weighting by the audited class mix, confident errors, label-free accuracy predictions) and the diagnosis with
diagnose_drift. Thresholds come from the first half of the null windows.

Outputs: results/phase2a/attribution_<name>_<detector>/{windows.csv, summary.json}.
Usage: python phase2a_attribution.py --config configs/phase0_cic_s1a.json --detector lightgbm --null 100 --windows 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta

import phase0
import phase1
from driftids import shifts
from driftids.alarms import density_ratio, domain_auc
from driftids.attribution import (_codes, diagnose_drift, bbse_class_weights, predicted_accuracy, predicted_macro_f1,
                                  residual_auc)
from driftids.audit import INTERVAL, active_audit, estimate, true_metrics
from driftids.labelfree import atc

from driftids.audit import results_root
ROOT = Path(__file__).resolve().parent
N_AUDIT = 200
CONF = 0.9          # confidence cut for the confident-error signal
CONF_Q = 0.99       # null-window quantile for its reference


def window_features(pool, src, meta, stype, s, w):
    idx, yw = phase1.make_window(pool, meta, stype, s, w)
    rows = []
    variants = ("", "_d") if stype == "none" else (("_d",) if stype == "new_class" else ("",))
    Xw = pool["X"][idx]
    auc, pw = domain_auc(src, Xw, seed=w)
    ratio = np.clip(pw, 0.01, 0.99) / (1 - np.clip(pw, 0.01, 0.99))
    for d in variants:
        pred, proba, classes = pool["pred" + d][idx], pool["proba" + d][idx], pool["classes" + d]
        truth = true_metrics(yw, pred, m=phase1.M)
        lf = atc(proba, classes, pred, meta["tau" + d], m=phase1.M)["macro_f1"]
        unc = 1 - proba.max(1)
        acq = 0.5 * unc / max(unc.mean(), 1e-12) + 0.5 * ratio / ratio.mean()
        rng = np.random.default_rng(phase1.seed_for("attr", stype, s, w, d))
        smp = active_audit(acq, N_AUDIT, rng, lam=0.5)
        e = estimate(smp, yw[smp.idx], pred[smp.idx], m=phase1.M, B=300, seed=w, interval=INTERVAL)
        unseen = ~np.isin(yw[smp.idx], classes)
        bw = bbse_class_weights(pool["pred_cal" + d], pool["y_cal" + d], classes, pred)
        res = residual_auc(src, pool["src_y"], Xw, bw, classes, seed=w)
        pc, yc = pool["pred_cal" + d], pool["y_cal" + d]
        n_w = len(idx)
        p_none = predicted_macro_f1(pc, yc, classes, N=n_w, m=phase1.M)
        p_prior = predicted_macro_f1(pc, yc, classes, N=n_w, m=phase1.M, class_weights=bw)
        # importance weights of the CALIBRATION flows towards the window (calibration vs window classifier), so the
        # correction also absorbs calibration-vs-pool differences (UNSW blocks differ in attack mix)
        ratio_cal = density_ratio(pool["X_cal" + d], Xw, pool["X_cal" + d], seed=w)
        p_cov = predicted_macro_f1(pc, yc, classes, N=n_w, m=phase1.M, flow_weights=ratio_cal)
        a_none = predicted_accuracy(pc, yc, classes)
        a_prior = predicted_accuracy(pc, yc, classes, class_weights=bw)
        a_cov = predicted_accuracy(pc, yc, classes, flow_weights=ratio_cal)
        # confident errors. Concept shift makes the model confidently wrong; prior and
        # covariate shift rarely do. Audit: design-weighted error share among audited flows with max proba >= 0.9;
        # source reference from the calibration block, plain and BBSE re-weighted to the window's class mix.
        conf_w = proba.max(1)[smp.idx] >= CONF
        wrong = yw[smp.idx] != pred[smp.idx]
        wc = smp.w[conf_w]
        ce = float((wc * wrong[conf_w]).sum() / wc.sum()) if wc.sum() > 0 else float("nan")
        ce_neff = float(wc.sum() ** 2 / (wc ** 2).sum()) if wc.sum() > 0 else 0.0
        ce_lo = float(beta.ppf(0.05, ce * ce_neff + 0.5, (1 - ce) * ce_neff + 0.5)) if wc.sum() > 0 else float("nan")
        pcal = pool["proba_cal" + d]
        conf_c = pcal.max(1) >= CONF
        wrong_c = (pc != yc)
        _, yi_c, ok_c = _codes(classes, yc)
        wb = np.where(ok_c, np.asarray(bw)[yi_c], 0.0)
        cal_ce = float(wrong_c[conf_c].mean()) if conf_c.any() else float("nan")
        cal_ce_prior = float((wb * (wrong_c & conf_c)).sum() / max((wb * conf_c).sum(), 1e-12))
        # re-weight the source to the class mix MEASURED by the audit (design-weighted label shares);
        # under pure prior shift the remaining domain AUC falls back to the unshifted level
        src_share = pd.Series(pool["src_y"]).value_counts(normalize=True)
        aud_share = pd.Series(smp.w, index=yw[smp.idx]).groupby(level=0).sum() / smp.w.sum()
        aw = np.array([aud_share.get(c, 0.0) / src_share[c] if c in src_share.index else 0.0 for c in classes])
        res_aud = residual_auc(src, pool["src_y"], Xw, aw, classes, seed=w) if aw.sum() > 0 else float("nan")
        rows.append({"res_auc_audit": res_aud, "conf_err": ce, "conf_err_lo": ce_lo, "conf_neff": ce_neff, "cal_conf_err": cal_ce,
                     "cal_conf_err_prior": cal_ce_prior,
                     "type": stype, "strength": s, "window": w, "variant": d or "main",
                     "auc_drop": auc - res,"type": stype, "strength": s, "window": w, "variant": d or "main",
                     "true_macro_f1": truth["macro_f1"], "labelfree": lf, "audit": e["macro_f1"],
                     "audit_lo": e["macro_f1_lo"], "audit_hi": e["macro_f1_hi"],
                     "unseen_share": float(smp.w[unseen].sum() / smp.w.sum()), "dom_auc": auc, "res_auc": res,
                     "pred_none": p_none, "pred_prior": p_prior, "pred_cov": p_cov,
                     "true_accuracy": truth["accuracy"], "audit_acc": e["accuracy"], "audit_acc_lo": e["accuracy_lo"],
                     "audit_acc_hi": e["accuracy_hi"], "acc_none": a_none, "acc_prior": a_prior, "acc_cov": a_cov})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_unsw.json")
    ap.add_argument("--detector", default="hgb", choices=["hgb", "lightgbm"])
    ap.add_argument("--null", type=int, default=100)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    t0 = time.time()
    pool, src, meta = phase1.prepare(a.config, a.detector)
    tasks = [("none", 0.0, w) for w in range(a.null)] + \
            [(t, s, w) for t in phase1.s3_types(meta) for s in shifts.STRENGTHS for w in range(a.windows)]
    res = Parallel(n_jobs=a.jobs, verbose=3)(delayed(window_features)(pool, src, meta, t, s, w) for t, s, w in tasks)
    df = pd.DataFrame([r for rows in res for r in rows])
    null = df[df.type == "none"]
    fit_w = sorted(null.window.unique())[: a.null // 2]
    fit = null[null.window.isin(fit_w)]
    auc_thr = float(np.quantile(fit[fit.variant == "main"].dom_auc, 0.99))
    offset = {v: float((g.labelfree - g.true_macro_f1).mean()) for v, g in fit.groupby("variant")}  # reported only
    # confident-error reference: at least the 99th percentile on the fitting null windows (per detector variant)
    conf_thr = {v: float(np.nanquantile(g.conf_err_lo, CONF_Q)) for v, g in fit.groupby("variant")}
    df["diagnosis"] = [diagnose_drift(audit_unseen_share=r.unseen_share, audit_lo=r.audit_acc_lo,
                                    audit_hi=r.audit_acc_hi, dom_auc=r.dom_auc, res_auc_audit=r.res_auc_audit,
                                    auc_threshold=auc_thr, pred_none=r.acc_none, pred_prior=r.acc_prior,
                                    pred_cov=r.acc_cov, conf_err_lo=r.conf_err_lo,
                                    conf_ref=max(r.cal_conf_err, conf_thr.get(r.variant, 0.0)),
                                    conf_ref_prior=max(r.cal_conf_err_prior, conf_thr.get(r.variant, 0.0)))
                       for r in df.itertuples()]
    held = df[(df.type == "none") & (~df.window.isin(fit_w)) & (df.variant == "main")]
    shifted = df[df.type != "none"]
    acc = shifted.assign(ok=shifted.diagnosis == shifted.type)
    by_type = acc.groupby("type").ok.mean().round(3).to_dict()
    by_ts = acc.groupby(["type", "strength"]).ok.mean().round(2)
    confusion = pd.crosstab(shifted.type, shifted.diagnosis)
    name = phase0._config(str(ROOT / a.config))["name"] + ("" if a.detector == "hgb" else f"_{a.detector}")
    out = results_root(ROOT) / "phase2a" / f"attribution_{name}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "windows.csv", index=False)
    summary = {"auc_threshold": auc_thr, "labelfree_offset": offset, "accuracy_overall": float(acc.ok.mean()),
               "accuracy_by_type": by_type, "none_correct_heldout_null": float((held.diagnosis == "none").mean()),
               "confusion": confusion.to_dict(), "accuracy_by_type_strength": {f"{k[0]} {k[1]}": v for k, v in by_ts.items()},
               "minutes": round((time.time() - t0) / 60, 1), "detector": meta["detector"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("auc_threshold", "labelfree_offset", "accuracy_overall",
                                              "accuracy_by_type", "none_correct_heldout_null", "minutes")}, indent=2))
    print(confusion.to_string())
    print(by_ts.to_string())


if __name__ == "__main__":
    main()
