"""Statistical checks of the main claims (H1-H5), computed from the outputs of the run scripts.

One-sided paired Wilcoxon signed-rank tests with Holm correction (H1a, H2), bootstrap non-inferiority bounds (H1b,
H5), interval coverage (H3) and diagnosis accuracy (H4). Writes results/analysis/<label>/hypotheses.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from driftids.audit import results_root
ROOT = Path(__file__).resolve().parent
RES = results_root(ROOT)
ALPHA = 0.05
NONINF = 0.01
MAIN = "active"


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def wilcoxon_less(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.all(d == 0):
        return {"n": int(len(d)), "p": None, "mean_diff": float(np.mean(d)) if len(d) else None}
    return {"n": int(len(d)), "p": float(wilcoxon(d, alternative="less").pvalue), "mean_diff": float(np.mean(d))}


def boot_upper(d, n_boot=10_000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(np.quantile(means, 0.95))


# ------------------------------------------------------------------ loaders (one row per window, errors per design)
S3_SUFFIX, ATTR_SUFFIX = "_s3_posterior3_lightgbm", "_lightgbm"


def s3_frame(ds):
    d = RES / "phase1" / f"{ds}{S3_SUFFIX}"
    if not (d / "raw.csv.gz").exists():
        return None
    raw = pd.read_csv(d / "raw.csv.gz")
    raw = raw[(raw.design != "active") | ((raw.acq == "combo") & (raw.lam == 0.5))]
    raw = raw.assign(err=(raw.macro_f1 - raw.true_macro_f1).abs(),
                     cover=(raw.macro_f1_lo <= raw.true_macro_f1) & (raw.true_macro_f1 <= raw.macro_f1_hi))
    lf = pd.read_csv(d / "labelfree.csv").assign(err=lambda x: (x.macro_f1 - x.true_macro_f1).abs())
    key = ["type", "strength", "window"]
    audit = raw.groupby(key + ["design", "n"]).agg(err=("err", "mean"), cover=("cover", "mean")).reset_index()
    lfw = lf.pivot_table(index=key, columns="estimator", values="err").reset_index()
    audit["wid"] = ds + ":" + audit.type + ":" + audit.strength.astype(str) + ":" + audit.window.astype(str)
    lfw["wid"] = ds + ":" + lfw.type + ":" + lfw.strength.astype(str) + ":" + lfw.window.astype(str)
    return audit, lfw


def stream_frame(tag, folder=None):
    # folder: the post-retraining side run is saved as postretrain_<tag>, not streams_<tag> (path fix, 
    d = RES / "phase2b" / (folder or f"streams_{tag}")
    if not (d / "windows.csv").exists():
        return None
    w = pd.read_csv(d / "windows.csv")
    w = w[~w.infiltration]                                          # excluded
    a = pd.read_csv(d / "audits.csv.gz").merge(w, on="window")
    a = a.assign(err=(a.macro_f1 - a.true_macro_f1).abs(),
                 cover=(a.macro_f1_lo <= a.true_macro_f1) & (a.true_macro_f1 <= a.macro_f1_hi))
    audit = a.groupby(["window", "design", "n"]).agg(err=("err", "mean"), cover=("cover", "mean")).reset_index()
    lfw = w.assign(soft_confusion=(w.soft_macro_f1 - w.true_macro_f1).abs(),
                   atc=(w.atc_macro_f1 - w.true_macro_f1).abs())
    audit["wid"] = tag + ":" + audit.window.astype(str)
    lfw["wid"] = tag + ":" + lfw.window.astype(str)
    return audit, lfw


def family_audit_vs_lf(audit, lfw, n=200):
    """Per window: active error at n and the best label-free error (estimator chosen per family)."""
    a = audit[(audit.design == MAIN) & (audit.n == n)].set_index("wid").err
    best = min(("soft_confusion", "atc"), key=lambda c: lfw[c].mean())
    l = lfw.set_index("wid")[best]
    j = pd.concat([a.rename("audit"), l.rename("lf")], axis=1).dropna()
    return j, best


def family_active_vs_random(audit, ns):
    a = audit[audit.n.isin(ns)].groupby(["wid", "design"]).err.mean().unstack()
    return a.dropna() if {MAIN, "random"} <= set(a.columns) else pd.DataFrame()


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", nargs="*", default=[])
    ap.add_argument("--s1", nargs="*", default=[])
    ap.add_argument("--s2", nargs="*", default=[])
    ap.add_argument("--policy", nargs="*", default=[])
    ap.add_argument("--label", default="final")
    ap.add_argument("--s3-suffix", default=S3_SUFFIX, help="suffix of the S3 result folders")
    ap.add_argument("--attr-suffix", default=ATTR_SUFFIX, help="suffix of the attribution result folders")
    ap.add_argument("--main-design", default=MAIN, help="audit design tested as the main audit")
    a = ap.parse_args()
    globals()["S3_SUFFIX"] = a.s3_suffix
    globals()["MAIN"] = a.main_design
    pending, out = [], {}
    s3 = {ds: s3_frame(ds) for ds in a.s3}
    s1 = {t: stream_frame(t) for t in a.s1}
    s2 = {t: stream_frame(t) for t in a.s2}
    for k, v in {**{f"S3 {k}": v for k, v in s3.items()}, **{f"stream {k}": v for k, v in {**s1, **s2}.items()}}.items():
        if v is None:
            pending.append(k)

    def cat(frames, sel_lf=None, sel_audit=None):
        au = pd.concat([f[0] for f in frames if f is not None], ignore_index=True) if any(f is not None for f in frames) else None
        lf = pd.concat([f[1] for f in frames if f is not None], ignore_index=True) if au is not None else None
        if au is None:
            return None, None
        if sel_lf is not None:
            lf = lf[sel_lf(lf)]
            au = au[au.wid.isin(lf.wid)]
        return au, lf

    # window families
    s3_cd = cat(list(s3.values()), lambda l: l.type.isin(["concept", "new_class"]))
    s3_ab = cat(list(s3.values()), lambda l: l.type.isin(["covariate", "prior"]))
    s3_all = cat(list(s3.values()))
    s1_new = cat([v for k, v in s1.items() if "ddos" not in k and "infil" not in k],   # S1b pairs: known-class
                 lambda l: l.new_attack_window)
    s1_known = cat([v for k, v in s1.items() if "infil" not in k], lambda l: ~l.new_attack_window)
    s2_all = cat(list(s2.values()))
    partial = cat([*s1.values(), *s2.values()], lambda l: l.failure_level == "partial")
    post = cat([stream_frame(f"postretrain_{t}", folder=f"postretrain_{t}") for t in a.policy])          # side run of phase2b_policies.py

    # H1a
    fam = {"S3 concept+new_class": s3_cd, "S1 new-attack windows": s1_new, "S2 windows": s2_all}
    res = {}
    for name, (au, lf) in fam.items():
        if au is None or au.empty:
            res[name] = {"status": "pending"}
            continue
        j, best = family_audit_vs_lf(au, lf)
        res[name] = {**wilcoxon_less(j.audit, j.lf), "best_labelfree": best, "audit_mae": float(j.audit.mean()),
                     "labelfree_mae": float(j.lf.mean())}
    adj = holm({k: v["p"] for k, v in res.items() if v.get("p") is not None})
    for k in adj:
        res[k]["p_holm"] = adj[k]
    won = sum(1 for v in res.values() if v.get("p_holm") is not None and v["p_holm"] < ALPHA)
    out["H1a"] = {"families": res, "families_won": won,
                  "met": (won >= 2) if all(v.get("status") != "pending" for v in res.values()) else "pending"}

    # H1b
    res = {}
    for name, (au, lf) in {"S3 covariate+prior": s3_ab, "S1 known-class windows": s1_known}.items():
        if au is None or au.empty:
            res[name] = {"status": "pending"}
            continue
        j, best = family_audit_vs_lf(au, lf)
        ub = boot_upper(j.audit - j.lf)
        res[name] = {"n": int(len(j)), "mean_diff": float((j.audit - j.lf).mean()), "upper95": ub,
                     "best_labelfree": best, "passes": ub is not None and ub < NONINF}
    out["H1b"] = {"families": res, "met": (all(v["passes"] for v in res.values())
                                           if all(v.get("status") != "pending" for v in res.values()) else "pending")}

    # H2-small / H2-large
    for label, ns in (("H2-small", [50, 100]), ("H2-large", [200, 500, 1000])):
        res = {}
        for name, (au, _) in {"S1+S2 partial-failure windows": partial, "post-retraining windows": post,
                              "S3 all": s3_all}.items():
            if au is None or au.empty:
                res[name] = {"status": "pending"}
                continue
            j = family_active_vs_random(au, ns)
            res[name] = {**wilcoxon_less(j[MAIN], j["random"]), "active_mae": float(j[MAIN].mean()),
                         "random_mae": float(j["random"].mean())}
        adj = holm({k: v["p"] for k, v in res.items() if v.get("p") is not None})
        for k in adj:
            res[k]["p_holm"] = adj[k]
        won = sum(1 for v in res.values() if v.get("p_holm") is not None and v["p_holm"] < ALPHA)
        out[label] = {"families": res, "families_won": won,
                      "met": (won >= 2) if all(v.get("status") != "pending" for v in res.values()) else "pending"}

    # H3: coverage of the active audit (macro-F1) per budget >= 100, all windows
    frames = [f for f in [*s3.values(), *s1.values(), *s2.values()] if f is not None]
    if frames:
        au = pd.concat([f[0] for f in frames], ignore_index=True)
        cov = au[(au.design == MAIN) & (au.n >= 100)].groupby("n").cover.mean().round(4).to_dict()
        out["H3"] = {"coverage_by_n": cov, "met": all(v >= 0.90 for v in cov.values()) and not pending}
    else:
        out["H3"] = {"status": "pending"}

    # H4: attribution on S3
    att = [RES / "phase2a" / f"attribution_{ds}{a.attr_suffix}" / "windows.csv" for ds in a.s3]
    if a.s3 and all(p.exists() for p in att):
        df = pd.concat([pd.read_csv(p) for p in att], ignore_index=True)
        df = df[df.type != "none"]
        acc = (df.diagnosis == df.type)
        by_type = acc.groupby(df.type).mean().round(3).to_dict()
        out["H4"] = {"overall": float(acc.mean()), "by_type": by_type,
                     "by_type_strength": {f"{t} {st}": round(float(v), 3) for (t, st), v in
                                          acc.groupby([df.type, df.strength]).mean().items()},
                     "met": bool(acc.mean() >= 0.80 and min(by_type.values()) >= 0.70)}
    else:
        out["H4"] = {"status": "pending"}

    # H5: policies
    res = {}
    for t in a.policy:
        p = RES / "phase2b" / f"policies_{t}" / "trace.csv"
        if not p.exists():
            res[t] = {"status": "pending"}
            continue
        tr = pd.read_csv(p)
        g = tr.groupby(["policy", "seed"]).agg(f1=("macro_f1", "mean"), labels=("labels_cum", "max")).reset_index()
        d = g.pivot(index="seed", columns="policy", values="f1")
        l = g.pivot(index="seed", columns="policy", values="labels")
        diff = d["periodic"] - d["dad"]                     # loss of DAD vs periodic
        ub = boot_upper(diff)
        ratio = float(l["dad"].mean() / max(l["periodic"].mean(), 1))
        res[t] = {"dad_f1": float(d["dad"].mean()), "periodic_f1": float(d["periodic"].mean()),
                  "loss_upper95": ub, "label_ratio": ratio,
                  "passes": ub is not None and ub <= 0.01 and ratio <= 0.5}
    out["H5"] = {"streams": res, "met": (all(v.get("passes") for v in res.values())
                                         if res and all(v.get("status") != "pending" for v in res.values()) else "pending")}
    out["pending_inputs"] = pending + ([] if any((RES / "phase2b" / f"postretrain_{t}").exists() for t in a.policy)
                                       else ["post-retraining side run (H2 family 2)"])
    dst = RES / "analysis" / a.label
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "hypotheses.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for h in ("H1a", "H1b", "H2-small", "H2-large", "H3", "H4", "H5"):
        print(f"{h:9s} met: {out[h].get('met', out[h].get('status'))}")
    print("pending:", out["pending_inputs"])
    print(f"written to {dst / 'hypotheses.json'}")


if __name__ == "__main__":
    main()
