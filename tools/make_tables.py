"""Result tables from the final run. Prints markdown tables.
Usage: python tools/make_tables.py [results root, default results/final]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "results" / "final"))
S1 = ["cic_s1a", "cic_s1b_ddos", "ton_s1c"]
S2 = ["unsw_to_cic_s1a", "unsw_to_ton_s1c", "cic_s1a_to_ton_s1c", "ton_s1c_to_cic_s1a"]
NAMES = {"cic_s1a": "CIC S1a", "cic_s1b_ddos": "CIC S1b-DDoS", "ton_s1c": "ToN S1c", "unsw_to_cic_s1a": "UNSW→CIC",
         "unsw_to_ton_s1c": "UNSW→ToN", "cic_s1a_to_ton_s1c": "CIC→ToN", "ton_s1c_to_cic_s1a": "ToN→CIC"}


def md(df, floatfmt=3):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                 ("-" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def stream(tag):
    d = R / "phase2b" / f"streams_{tag}"
    w = pd.read_csv(d / "windows.csv")
    w = w[~w.infiltration]
    a = pd.read_csv(d / "audits.csv.gz").merge(
        w, on="window")
    a["err"] = (a.macro_f1 - a.true_macro_f1).abs()
    a["cover"] = (a.macro_f1_lo <= a.true_macro_f1) & (a.true_macro_f1 <= a.macro_f1_hi)
    a["width"] = a.macro_f1_hi - a.macro_f1_lo
    w = w.assign(soft_err=(w.soft_macro_f1 - w.true_macro_f1).abs(), atc_err=(w.atc_macro_f1 - w.true_macro_f1).abs())
    return w, a


h = json.loads((R / "analysis" / "final" / "hypotheses.json").read_text())

# ---- T2: scenarios
rows = []
for t in S1 + S2:
    w, a = stream(t)
    rows.append({"scenario": NAMES[t], "windows": len(w), "new-attack": int(w.new_attack_window.sum()),
                 "total miss": int((w.failure_level == "total_miss").sum()),
                 "partial": int((w.failure_level == "partial").sum()),
                 "mean true macro-F1": w.true_macro_f1.mean()})
print("## T2 Scenarios (deployment windows, Infiltration excluded)\n")
print(md(pd.DataFrame(rows).set_index("scenario")), "\n")

# ---- T4: H1 per scenario (n = 200, active) and families
rows = []
for t in S1 + S2:
    w, a = stream(t)
    act = a[(a.design == "active") & (a.n == 200)].groupby("window").err.mean()
    rows.append({"scenario": NAMES[t], "audit n=200": act.mean(), "soft confusion": w.soft_err.mean(),
                 "ATC": w.atc_err.mean()})
print("## T4a Mean absolute error of macro-F1 per scenario (active audit, n = 200)\n")
print(md(pd.DataFrame(rows).set_index("scenario")), "\n")
fam = []
for f, v in h["H1a"]["families"].items():
    fam.append({"family": f, "windows": v["n"], "audit": v["audit_mae"], "best label-free": v["labelfree_mae"],
                "Holm p": f"{v['p_holm']:.1e}"})
for f, v in h["H1b"]["families"].items():
    fam.append({"family": f + " (H1b)", "windows": v["n"], "audit": np.nan, "best label-free": np.nan,
                "Holm p": f"bound {v['upper95']:+.3f}"})
print("## T4b H1 families\n")
print(md(pd.DataFrame(fam).set_index("family")), "\n")

# ---- H2 by budget, real streams pooled, and families
allw = pd.concat([stream(t)[1].assign(tag=t) for t in S1 + S2])
g = allw.groupby(["tag", "window", "design", "n"]).err.mean().reset_index()
t2 = g.groupby(["design", "n"]).err.mean().unstack("n")
t2.index.name = "design (real streams)"
print("## T5a MAE by design and budget (S1 + S2 windows)\n")
print(md(t2), "\n")
fam = []
for k in ("H2-small", "H2-large"):
    for f, v in h[k]["families"].items():
        fam.append({"test": f"{k}: {f}", "windows": v.get("n"), "active": v.get("active_mae"),
                    "random": v.get("random_mae"), "Holm p": f"{v['p_holm']:.2g}" if v.get("p_holm") is not None else "-"})
print("## T5b H2 families\n")
print(md(pd.DataFrame(fam).set_index("test")), "\n")

# ---- H3 coverage and width
act = allw[allw.design == "active"]
cov = act.groupby(["tag", "n"]).agg(cover=("cover", "mean")).unstack("n")["cover"]
cov.index = [NAMES[i] for i in cov.index]
cov.index.name = "coverage (active)"
print("## T6a Coverage of the 95% macro-F1 interval per scenario and budget\n")
print(md(cov), "\n")
wid = act.groupby("n").width.mean()
print("Mean width by n (real streams):", wid.round(3).to_dict())
print("H3 pooled (S1+S2+S3):", h["H3"]["coverage_by_n"], "\n")
s3 = []
for ds in ("cic_s1a", "ton_s1c"):
    r = pd.read_csv(R / "phase1" / f"{ds}_s3_posterior3_lightgbm" / "raw.csv.gz")
    r = r[(r.design == "active") & (r.acq == "combo") & (r.lam == 0.5)]
    for m in ("macro_f1", "bin_f1", "detection_rate", "false_alarm_rate"):
        c = ((r[m + "_lo"] <= r["true_" + m]) & (r["true_" + m] <= r[m + "_hi"]))
        wd = r[m + "_hi"] - r[m + "_lo"]
        s3.append({"dataset": ds, "metric": m, **{f"cov n={n}": c[r.n == n].mean() for n in (100, 200, 1000)},
                   **{f"width n={n}": wd[r.n == n].mean() for n in (100, 200, 1000)}})
print("## T6b S3: coverage and width per metric (active)\n")
print(md(pd.DataFrame(s3).set_index(["dataset"]).reset_index().set_index("metric")), "\n")
# binary metrics on real streams
rows = []
for m in ("bin_f1", "detection_rate", "false_alarm_rate"):
    c = ((act[m + "_lo"] <= act.groupby(["tag", "window"])[m].transform("size") * 0 + act[m + "_lo"]))  # placeholder
rows = None

# ---- T7 H4 confusion
for ds in ("cic_s1a", "ton_s1c"):
    w = pd.read_csv(R / "phase2a" / f"attribution_{ds}_lightgbm" / "windows.csv")
    sh = w[w.type != "none"]
    ct = pd.crosstab(sh.type, sh.diagnosis)
    ct.index.name = f"{ds}: true type"
    print(f"## T7 Attribution confusion {ds} (accuracy {(sh.type == sh.diagnosis).mean():.3f})\n")
    print(md(ct.astype(int), 0), "\n")
print("H4 pooled:", round(h["H4"]["overall"], 3), h["H4"]["by_type"], "\n")

# ---- T8 policies
for ds in ("cic_s1a", "ton_s1c"):
    tr = pd.read_csv(R / "phase2b" / f"policies_{ds}" / "trace.csv")
    tr10 = tr[tr.seed < 10]
    g = tr10.groupby(["policy", "seed"]).agg(f1=("macro_f1", "mean"), labels=("labels_cum", "max"),
                                             retrains=("action", lambda x: int((x == "retrain").sum()))).reset_index()
    s = g.groupby("policy").agg(f1=("f1", "mean"), sd=("f1", "std"), labels=("labels", "mean"),
                                retrains=("retrains", "mean"))
    ph = tr10.pivot_table(index="policy", columns="phase", values="macro_f1")
    s = s.join(ph).sort_values("f1", ascending=False)
    s.index.name = f"{ds} policy"
    print(f"## T8 Policies {ds} (seeds 0-9)\n")
    print(md(s, 3), "\n")
for s_, v in h["H5"]["streams"].items():
    print("H5", s_, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
