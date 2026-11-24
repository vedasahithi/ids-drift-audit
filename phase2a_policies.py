"""Policy simulation core.

Env holds the data of one stream, trains and calibrates detectors and computes alarm thresholds; run_seed simulates
the policies never, periodic, alarm, oracle, detect-audit-decide (dad) and the baselines ddm, uncertainty and
conformal over a stream of windows for one seed. Used by phase2b_policies.py.
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import phase0
import phase1
from driftids import shifts
from driftids.alarms import conformal_rates, density_ratio, domain_auc, fit_conformal
from driftids.attribution import attribute, bbse_class_weights, predicted_accuracy, residual_auc
from driftids.audit import INTERVAL, active_audit, estimate, true_metrics
from driftids.detectors import make_detector
from driftids.audit import AuditSample
from driftids.policies import PolicyConfig, PolicyState, apply_recalibration, change_trigger

from driftids.audit import results_root
ROOT = Path(__file__).resolve().parent
PHASES = [("none", [0.0] * 8), ("prior", list(np.linspace(0.25, 0.75, 8))),
          ("covariate", list(np.linspace(0.25, 0.75, 8))), ("new_class", list(np.linspace(0.25, 0.75, 8)))]
N_NULL = 40
M = phase1.M
MIN_CLASS = 5


def build_stream(pool, meta, seed):
    held = meta["held_out"]
    keep = np.flatnonzero(pool["y"] != held)                    # pool rows without the held-out class
    sub = {"y": pool["y"][keep], "duration": pool["duration"][keep]}
    out, t = [], 0
    for phase, strengths in PHASES:
        for s in strengths:
            rng = np.random.default_rng(phase1.seed_for("stream", seed, t))
            if phase == "none":
                idx = keep[rng.choice(len(keep), phase1.WINDOW, replace=True)]
            elif phase == "prior":
                i, _ = shifts.prior(sub["y"], phase1.WINDOW, s, rng)
                idx = keep[i]
            elif phase == "covariate":
                i, _ = shifts.covariate(sub["duration"], sub["y"], phase1.WINDOW, s, rng, meta["q70_duration"])
                idx = keep[i]
            else:
                idx, _ = shifts.new_class(pool["y"], phase1.WINDOW, s, rng, held)
            out.append((t, phase, float(s), idx))
            t += 1
    return out


def null_windows(pool, meta, n=N_NULL):
    keep = np.flatnonzero(pool["y"] != meta["held_out"])
    return [keep[np.random.default_rng(phase1.seed_for("policynull", w)).choice(len(keep), phase1.WINDOW, replace=True)]
            for w in range(n)]


class Env:
    """Shared, model-independent data for one dataset."""

    def __init__(self, pool, src, meta, detector, null_idx=None, lgbm_setting=None):
        """null_idx: optional unshifted windows (positions into pool['X']); lgbm_setting: LightGBM setting chosen once
        on the source validation block and kept for every retraining ()."""
        self.pool, self.meta, self.detector, self.lgbm_setting = pool, meta, detector, lgbm_setting
        held = meta["held_out"]
        m_tr = pool["y_tr"] != held
        self.X_tr, self.y_tr = pool["X_tr"][m_tr], pool["y_tr"][m_tr]
        m_src = pool["src_y"] != held
        self.src, self.src_y = src[m_src], pool["src_y"][m_src]
        m_cal = pool["y_cal"] != held
        self.X_cal, self.y_cal = pool["X_cal"][m_cal], pool["y_cal"][m_cal]
        self.nulls = null_idx if null_idx is not None else null_windows(pool, meta)
        aucs = [domain_auc(self.src, pool["X"][ix], seed=w)[0] for w, ix in enumerate(self.nulls)]
        self.auc_thr = float(np.quantile(aucs, 0.99))
        self.auc_spread = float(np.quantile(aucs, 0.99) - np.quantile(aucs, 0.01))   # normal range (option B)

    def train(self, buffer_idx, seed=0):
        X, y = self.X_tr, self.y_tr
        if buffer_idx:
            b = np.array(sorted(set(buffer_idx)))
            X, y = np.vstack([X, self.pool["X"][b]]), np.r_[y, self.pool["y"][b]]
            # a class enters retraining only with >= MIN_CLASS flows (an audit may reveal a single flow of a new
            # attack type; one example cannot be learned and breaks HGB's internal stratified split). The flows stay
            # in the buffer and are used once enough have accumulated.
            vals, counts = np.unique(y, return_counts=True)
            ok = np.isin(y, vals[counts >= MIN_CLASS])
            X, y = X[ok], y[ok]
        if self.detector == "lightgbm":
            from driftids.detectors import make_lightgbm
            return make_lightgbm(self.lgbm_setting or "defaults", seed).fit(X, y)
        mdl, _ = make_detector(self.detector, X, y, seed=seed)
        return mdl

    def calibrate(self, mdl):
        """Conformal sets on the calibration block and the abstention threshold from the null windows."""
        cp = fit_conformal(mdl.predict_proba(self.X_cal), self.y_cal, mdl.classes_)
        ab = [conformal_rates(mdl.predict_proba(self.pool["X"][ix]), cp)["abstention"] for ix in self.nulls]
        return cp, float(np.quantile(ab, 0.99)), float(np.quantile(ab, 0.99) - np.quantile(ab, 0.01))


def run_seed(env: Env, seed: int, cfg: PolicyConfig, stream=None, keep_models: bool = False):
    """stream: optional list of (t, phase, strength, positions into pool['X']) - real streams (); default the
    synthetic development stream. keep_models: also return the periodic policy's retrained models (t, model), for the
    post-retraining side run of H2."""
    pool = env.pool
    stream = stream if stream is not None else build_stream(pool, env.meta, seed)
    periodic_models = []
    base = env.train([], seed=0)
    cal0 = env.calibrate(base)
    ref = float(np.mean([true_metrics(pool["y"][ix], base.predict(pool["X"][ix]).astype(str), m=M)["macro_f1"]
                         for ix in env.nulls]))
    # a subset of policies can be run (DRIFTIDS_POLICIES), e.g. to add seeds for some policies
    names = tuple(os.environ.get("DRIFTIDS_POLICIES", "never,periodic,alarm,dad,oracle,ddm,uncertainty,conformal").split(","))
    states = {name: PolicyState(name=name, model=base) for name in names}
    calib = {name: cal0 for name in states}
    rows = []
    for t, phase, s, idx in stream:
        Xw, yw = pool["X"][idx], pool["y"][idx]
        auc, pw = domain_auc(env.src, Xw, seed=1000 * seed + t)
        ratio = np.clip(pw, 0.01, 0.99) / (1 - np.clip(pw, 0.01, 0.99))
        for name, st in states.items():
            proba = apply_recalibration(st.model.predict_proba(Xw), st.class_weights)
            classes = st.model.classes_.astype(str)
            pred = classes[proba.argmax(1)]
            score = true_metrics(yw, pred, m=M)["macro_f1"]
            cp, ab_thr, ab_spread = calib[name]
            ab = conformal_rates(proba, cp)["abstention"]
            alarm = ab > ab_thr or auc > env.auc_thr
            trig = change_trigger(st, alarm, auc, ab, env.auc_spread, ab_spread, t, cfg.cap)                 if name in ("alarm", "dad") else False
            action, reveal = "", None
            rng = np.random.default_rng(phase1.seed_for("policy", name, seed, t))
            # literature baselines, label budget matched to DAD (cfg.n_base ~ DAD's labels per window)
            if name == "ddm":                    # DDM (Gama et al. 2004) on a small random audit every window
                reveal = rng.choice(len(idx), cfg.n_base, replace=False)
                st.ddm_n += len(reveal)
                st.ddm_err += int((yw[reveal] != pred[reveal]).sum())
                p_e = st.ddm_err / st.ddm_n
                s_e = float(np.sqrt(max(p_e * (1 - p_e), 1e-12) / st.ddm_n))
                if st.ddm_n >= 30:               # DDM warm-up of 30 examples
                    if p_e + s_e < st.ddm_min:
                        st.ddm_min, st.ddm_smin = p_e + s_e, s_e
                    if p_e + s_e > st.ddm_min + 3 * st.ddm_smin:
                        action = "retrain"
            elif name == "uncertainty":          # uncertainty-sampling active learning, retrain every k windows
                reveal = np.argsort(proba.max(1), kind="stable")[: cfg.n_base]
                if (t + 1) % cfg.k == 0:
                    action = "retrain"
            elif name == "conformal":            # Transcend-style (Jordaney et al. 2017): label rejected flows
                if ab > ab_thr and t - st.last_capped >= cfg.cap:
                    rej = np.flatnonzero(((1.0 - proba) <= cp.q[None, :]).sum(1) != 1)
                    reveal = rng.choice(rej, min(cfg.n, len(rej)), replace=False) if len(rej) else None
                    action, st.last_capped = "retrain", t
            if name == "periodic" and (t + 1) % cfg.k == 0:
                reveal, action = rng.choice(len(idx), cfg.n, replace=False), "retrain"
            elif name == "alarm" and trig:
                reveal, action = rng.choice(len(idx), cfg.n, replace=False), "retrain"
            elif name == "oracle" and score < ref - cfg.delta:
                reveal, action = rng.choice(len(idx), cfg.n, replace=False), "retrain"
            elif name == "dad":
                triggered = trig
                routine = (t + 1) % cfg.k == 0
                if triggered or routine:
                    n_a = cfg.n if triggered else cfg.n_routine
                    unc = 1 - proba.max(1)
                    acq = 0.5 * unc / max(unc.mean(), 1e-12) + 0.5 * ratio / ratio.mean()
                    smp = active_audit(acq, n_a, rng, lam=cfg.lam)
                    e = estimate(smp, yw[smp.idx], pred[smp.idx], m=M, B=300, seed=t, interval=INTERVAL)
                    reveal = np.unique(smp.idx)
                    cur = {"model": st.retrains, "y": yw[smp.idx], "p": pred[smp.idx], "w": smp.w}
                    pooled_hi = np.inf
                    if st.prev_audit is not None and st.prev_audit["model"] == st.retrains:
                        pv = st.prev_audit                 # pool with the previous audit of the same model
                        ps = AuditSample(idx=np.arange(len(pv["w"]) + len(cur["w"])), w=np.r_[pv["w"], cur["w"]])
                        e2 = estimate(ps, np.r_[pv["y"], cur["y"]], np.r_[pv["p"], cur["p"]], m=2 * M, B=300, seed=t,
                                      interval=INTERVAL)
                        pooled_hi = e2["macro_f1_hi"]
                    st.prev_audit = cur
                    if e["macro_f1_hi"] < ref - cfg.delta or pooled_hi < ref - cfg.delta:
                        action = "retrain"
                    elif cfg.err_tol is not None and e["accuracy_hi"] < 1 - cfg.err_tol:
                        action = "retrain"           # error-driven retraining
                    else:
                        pc = classes[st.model.predict_proba(env.X_cal).argmax(1)]
                        bw = bbse_class_weights(pc, env.y_cal, classes, pred)
                        res = residual_auc(env.src, env.src_y, Xw, bw, classes, seed=t)
                        rc = density_ratio(env.X_cal, Xw, env.X_cal, seed=t)
                        kind = attribute(audit_unseen_share=float(smp.w[~np.isin(yw[smp.idx], classes)].sum() / smp.w.sum()),
                                         audit_lo=e["accuracy_lo"], audit_hi=e["accuracy_hi"], dom_auc=auc, res_auc=res,
                                         auc_threshold=env.auc_thr,
                                         pred_none=predicted_accuracy(pc, env.y_cal, classes),
                                         pred_prior=predicted_accuracy(pc, env.y_cal, classes, class_weights=bw),
                                         pred_cov=predicted_accuracy(pc, env.y_cal, classes, flow_weights=rc))
                        if kind in ("new_class", "concept"):
                            action = "retrain"           # labels are the only source of the missing knowledge
                        elif kind == "prior":
                            st.class_weights, action = bw, "recalibrate"
                        else:
                            action = f"audit:{kind}"
            if reveal is not None:
                st.buffer_idx.extend(int(i) for i in idx[reveal])
                st.labels += len(reveal)
            st.prev_alarm = alarm
            if action == "retrain":
                st.prev_audit = None
                st.model, st.class_weights = env.train(st.buffer_idx, seed=0), None
                if keep_models and name == "periodic":
                    periodic_models.append((t, st.model))
                calib[name] = env.calibrate(st.model)
                st.retrains += 1
                if name == "ddm":                # DDM restarts its statistics after a detected drift
                    st.ddm_n, st.ddm_err, st.ddm_min, st.ddm_smin = 0, 0, np.inf, np.inf
            rows.append({"seed": seed, "t": t, "phase": phase, "strength": s, "policy": name, "macro_f1": score,
                         "alarm": bool(alarm), "action": action, "labels_cum": st.labels})
    return (rows, ref, periodic_models) if keep_models else (rows, ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_unsw.json")
    ap.add_argument("--detector", default="hgb", choices=["hgb", "lightgbm"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    pool, src, meta = phase1.prepare(a.config, a.detector)
    env = Env(pool, src, meta, a.detector)
    cfg = PolicyConfig()
    res = Parallel(n_jobs=a.jobs, verbose=3)(delayed(run_seed)(env, s, cfg) for s in range(a.seeds))
    trace = pd.DataFrame([r for rows, _ in res for r in rows])
    refs = [ref for _, ref in res]
    agg = trace.groupby(["policy", "seed"]).agg(mean_macro_f1=("macro_f1", "mean"), labels=("labels_cum", "max"),
                                                retrains=("action", lambda x: int((x == "retrain").sum())))
    summ = agg.groupby("policy").mean().round(4)
    by_phase = trace.pivot_table(index="policy", columns="phase", values="macro_f1").round(3)
    p, d = summ.loc["periodic"], summ.loc["dad"]
    h5 = {"dad_mean_minus_periodic": float(d.mean_macro_f1 - p.mean_macro_f1),
          "dad_labels_over_periodic": float(d.labels / max(p.labels, 1)),
          "h5_met": bool(d.mean_macro_f1 >= p.mean_macro_f1 - 0.01 and d.labels <= 0.5 * p.labels)}
    name = phase0._config(str(ROOT / a.config))["name"] + ("" if a.detector == "hgb" else f"_{a.detector}")
    out = results_root(ROOT) / "phase2a" / f"policies_{name}"
    out.mkdir(parents=True, exist_ok=True)
    trace.to_csv(out / "trace.csv", index=False)
    summary = {"reference_macro_f1": refs, "auc_threshold": env.auc_thr, "summary": summ.to_dict(orient="index"),
               "by_phase": by_phase.to_dict(orient="index"), "h5_preview": h5, "seeds": a.seeds,
               "minutes": round((time.time() - t0) / 60, 1), "config": cfg.__dict__}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(summ.to_string())
    print(by_phase.to_string())
    print(json.dumps(h5, indent=2), "minutes", summary["minutes"])


if __name__ == "__main__":
    main()
