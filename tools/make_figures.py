"""Result figures from the final run. Writes PDF + PNG to figures/.
F2: silent failure on CIC S1a - true vs label-free vs audited macro-F1 per window.
F3: estimation error vs label budget per audit design (real streams), with the best label-free error.
F4: macro-F1 over windows for key policies on CIC S1a and ToN S1c.
Palette: validated reference categorical palette (dataviz skill), slots in fixed order."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

R = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "results" / "final"))
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # slots 1-5 (light)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 1.6, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def stream(tag):
    d = R / "phase2b" / f"streams_{tag}"
    w = pd.read_csv(d / "windows.csv")
    a = pd.read_csv(d / "audits.csv.gz")
    return w, a


# ---- F2 silent failure on CIC S1a
w, a = stream("cic_s1a")
w = w[~w.infiltration].sort_values("window")
au = a[(a.design == "active") & (a.n == 200)].groupby("window").agg(est=("macro_f1", "mean"), lo=("macro_f1_lo", "mean"),
                                                                     hi=("macro_f1_hi", "mean"))
w = w.merge(au, left_on="window", right_index=True)
fig, ax = plt.subplots(figsize=(7.2, 3.1))
x = np.arange(len(w))
ax.fill_between(x, w.lo, w.hi, color=C[0], alpha=0.15, linewidth=0, label="audit 95% interval (n = 200)")
ax.plot(x, w.true_macro_f1, color=INK, linewidth=1.4, label="true macro-F1 (all labels)")
ax.plot(x, w.soft_macro_f1, color=C[1], label="label-free estimate (soft confusion)")
ax.plot(x, w.est, color=C[0], label="audit estimate (n = 200)")
ax.set_xlabel("deployment window (50,000 flows each, time order)")
ax.set_ylabel("macro-F1")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
save(fig, "F2_silent_failure_cic_s1a")

# ---- F3 error vs budget (S1 + S2 windows pooled)
tags = ["cic_s1a", "cic_s1b_ddos", "ton_s1c", "unsw_to_cic_s1a", "unsw_to_ton_s1c", "cic_s1a_to_ton_s1c",
        "ton_s1c_to_cic_s1a"]
errs, lf = [], []
for t in tags:
    w, a = stream(t)
    w = w[~w.infiltration]
    m = a.merge(w, on="window")
    m["err"] = (m.macro_f1 - m.true_macro_f1).abs()
    errs.append(m.groupby(["window", "design", "n"]).err.mean().reset_index().assign(tag=t))
    lf.append(pd.DataFrame({"soft": (w.soft_macro_f1 - w.true_macro_f1).abs(), "atc": (w.atc_macro_f1 - w.true_macro_f1).abs()}))
e = pd.concat(errs).groupby(["design", "n"]).err.mean().unstack("design")
lfb = min(pd.concat(lf).soft.mean(), pd.concat(lf).atc.mean())
fig, ax = plt.subplots(figsize=(4.6, 3.0))
names = {"active": "active", "random": "random", "stratified": "stratified", "stratactive": "hybrid"}
for i, d in enumerate(["active", "random", "stratified", "stratactive"]):
    ax.plot(e.index, e[d], color=C[i], marker="o", markersize=4, label=names[d])
ax.axhline(lfb, color=INK2, linestyle="--", linewidth=1.2)
ax.text(1000, lfb * 1.06, f"best label-free ({lfb:.2f})", ha="right", va="bottom", color=INK2, fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([50, 100, 200, 500, 1000])
ax.set_xticklabels(["50", "100", "200", "500", "1000"])
ax.set_xlabel("labels per window (n)")
ax.set_ylabel("mean absolute error of macro-F1")
ax.legend(loc="lower left", fontsize=8)
save(fig, "F3_error_vs_budget")

# ---- F4 policies over windows
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, (ds, title, base) in zip(axes, [("cic_s1a", "CSE-CIC-IDS2018 S1a", "conformal"),
                                         ("ton_s1c", "ToN-IoT S1c", "uncertainty")]):
    tr = pd.read_csv(R / "phase2b" / f"policies_{ds}" / "trace.csv")
    tr = tr[tr.seed < 10]
    p = tr.pivot_table(index="t", columns="policy", values="macro_f1")
    show = [("dad", "detect-audit-decide"), ("periodic", "periodic"), (base, "best literature baseline"), ("never", "never")]
    for i, (k, lab) in enumerate(show):
        ax.plot(p.index, p[k].rolling(10, min_periods=1).mean(), color=C[i], label=lab)
    ax.set_title(f"{title}  (best baseline: {base})", fontsize=9, color=INK)
    ax.set_xlabel("deployment window")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("macro-F1 (10-window mean)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "F4_policies_over_windows.pdf")
fig.savefig(OUT / "F4_policies_over_windows.png", dpi=200)
plt.close(fig)
print("figures written to", OUT)
