"""Figure 1: the detect-audit-decide monitor (framework diagram). Writes PDF + PNG to figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]      # blue, orange, aqua, violet (slots 1, 2, 3, 7)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.set_xlim(0, 100)
ax.set_ylim(-9, 46)
ax.axis("off")


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=1.4,
                                edgecolor=color, facecolor=SURF))
    ax.add_patch(FancyBboxPatch((x, y + h - 5.2), w, 5.2, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=0,
                                facecolor=color, alpha=0.14))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
    ax.text(x + 1.0, y + h - 7.0, chr(10).join(lines), ha="left", va="top", fontsize=7.2, color=INK2,
            linespacing=1.35)


def arrow(x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=INK2))
    if text:
        ax.text((x0 + x1) / 2, y0 + 1.0, text, ha="center", va="bottom", fontsize=7, color=INK2)


# stream (caption to the right, arrow from the last window down to Detect)
ax.text(1, 43.5, "Deployment stream", fontsize=8.5, color=INK, weight="bold")
for i in range(9):
    ax.add_patch(FancyBboxPatch((1 + i * 2.6, 36.5), 2.0, 4.0, boxstyle="round,pad=0.1", linewidth=0.8,
                                edgecolor=INK2, facecolor="#e4e3df" if i < 8 else C[0], alpha=1 if i < 8 else 0.35))
ax.text(25.5, 38.5, "window W_t of N flows (unlabelled)", fontsize=7, color=INK2, va="center")
arrow(22.8, 36.0, 10.5, 30.6)

y, h, w = 6, 23, 19.5
xs = [1, 26, 51, 76]
box(xs[0], y, w, h, "Detect", ["label-free signals:", "- domain-classifier AUC", "- conformal abstention",
                               "thresholds from", "unshifted windows;", "audit on change", "or routine"], C[0])
box(xs[1], y, w, h, "Audit", ["analyst labels n flows", "(50 routine,", " 200 triggered)", "- active sampling",
                              "- weighted macro-F1", "- interval with an", "  unseen-class allowance"], C[1])
box(xs[2], y, w, h, "Diagnose", ["- new class:", "  unseen labels", "- prior vs covariate:", "  audited class mix",
                                 "- concept:", "  confident errors"], C[2])
box(xs[3], y, w + 3, h, "Decide", ["- retrain: upper bound", "  below target, clear", "  errors, new class",
                                   "  or concept", "- recalibrate: prior", "- otherwise: wait"], C[3])
arrow(21.4, 17.5, 25.6, 17.5, "trigger")
arrow(46.4, 17.5, 50.6, 17.5, "labels")
arrow(71.4, 17.5, 75.6, 17.5, "type")
# feedback below the boxes
ax.add_patch(FancyArrowPatch((88.5, 5.2), (10.5, 5.2), arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=INK2,
                             linestyle="--", connectionstyle="arc3,rad=-0.12"))
ax.text(50, -8.6, "retrained or recalibrated detector used from window t + 1 (labels reused, no extra cost)",
        ha="center", fontsize=7, color=INK2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "F1_framework.pdf")
fig.savefig(OUT / "F1_framework.png", dpi=220)
print("written")
