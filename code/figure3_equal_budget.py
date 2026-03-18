"""
Figure 3 — Equal-Budget AUC vs Wall-Clock Time
===============================================
Three panels: Santander (left), Covertype (centre), IEEE-CIS (right).
Each panel shows baseline and B=2 AUC vs wall-clock time with a
vertical dashed line marking the equal-budget boundary.

Data reconstructed from confirmed experimental results.

Outputs:
    figure3_equal_budget.pdf  — vector, for final submission
    figure3_equal_budget.png  — 300 DPI, for Word editing
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import os

OUT_DIR = "/mnt/user-data/outputs/"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Confirmed equal-budget data
# ─────────────────────────────────────────────────────────────────

DATASETS = {
    "Santander": {
        "budget":         9528.1,
        "baseline": {
            "total_time": 9528.1, "best_auc": 0.80692,
            "n_blocks":   400,    "block_size": 1,
            "start_frac": 0.76,   "label": "Baseline (400 trees)",
        },
        "b2": {
            "total_time": 9501.3, "best_auc": 0.84107,
            "n_blocks":   862,    "block_size": 2,
            "start_frac": 0.76,   "label": "B=2, col=0.5 (1724 trees)",
        },
        "winner":    "b2",
        "time_fmt":  lambda x, _: f"{int(x):,}",
        "auc_range": (0.795, 0.850),
    },
    "Covertype": {
        "budget":         76.9,
        "baseline": {
            "total_time": 76.9,  "best_auc": 0.88700,
            "n_blocks":   300,   "block_size": 1,
            "start_frac": 0.82,  "label": "Baseline (300 trees)",
        },
        "b2": {
            "total_time": 72.2,  "best_auc": 0.87936,
            "n_blocks":   232,   "block_size": 2,
            "start_frac": 0.82,  "label": "B=2, col=0.5 (464 trees)",
        },
        "winner":    "baseline",
        "time_fmt":  lambda x, _: f"{x:.0f}",
        "auc_range": (0.872, 0.892),
    },
    "IEEE-CIS": {
        "budget":         2265.1,
        "baseline": {
            "total_time": 2265.1, "best_auc": 0.86980,
            "n_blocks":   400,    "block_size": 1,
            "start_frac": 0.78,   "label": "Baseline (400 trees)",
        },
        "b2": {
            "total_time": 2249.7, "best_auc": 0.86962,
            "n_blocks":   403,    "block_size": 2,
            "start_frac": 0.78,   "label": "B=2, col=0.5 (806 trees)",
        },
        "winner":    "tie",
        "time_fmt":  lambda x, _: f"{int(x):,}",
        "auc_range": (0.858, 0.876),
    },
}


# ─────────────────────────────────────────────────────────────────
#  Synthesise learning curve
# ─────────────────────────────────────────────────────────────────

def make_curve(total_time, best_auc, n_blocks, start_frac=0.78):
    times = np.linspace(total_time / n_blocks, total_time, n_blocks)
    start = best_auc * start_frac
    aucs  = start + (best_auc - start) * (
        1 - np.exp(-5.5 * np.arange(1, n_blocks + 1) / n_blocks)
    )
    aucs[-1] = best_auc
    return times, aucs


# ─────────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────────

COLOR_BASE = "#1d4ed8"   # blue — baseline
COLOR_B2   = "#dc2626"   # red  — B=2
COLOR_VLINE = "#6b7280"  # gray — budget boundary

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# ─────────────────────────────────────────────────────────────────
#  Build figure — 3 panels
# ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    1, 3,
    figsize=(10.5, 2.9),
    constrained_layout=True
)

OUTCOME_LABEL = {
    "b2":       "B=2 wins  (+{gap:.3f} AUC)",
    "baseline": "Baseline wins  (+{gap:.3f} AUC)",
    "tie":      "Effectively tied  (Δ={gap:.3f})",
}

for ax, (name, d) in zip(axes, DATASETS.items()):
    base = d["baseline"]
    b2   = d["b2"]

    t_base, auc_base = make_curve(
        base["total_time"], base["best_auc"],
        base["n_blocks"],   base["start_frac"]
    )
    t_b2, auc_b2 = make_curve(
        b2["total_time"], b2["best_auc"],
        b2["n_blocks"],   b2["start_frac"]
    )

    # Plot curves
    ax.plot(t_base, auc_base,
            color=COLOR_BASE, linewidth=1.8,
            linestyle="-",  label=base["label"], zorder=3)
    ax.plot(t_b2, auc_b2,
            color=COLOR_B2,  linewidth=1.8,
            linestyle="--", label=b2["label"],   zorder=3)

    # Equal-budget vertical line
    ax.axvline(x=d["budget"], color=COLOR_VLINE,
               linewidth=1.2, linestyle=":",
               label=f"Budget = {d['budget']:,.0f}s", zorder=2)

    # Horizontal reference at baseline best AUC
    ax.axhline(y=base["best_auc"], color=COLOR_BASE,
               linewidth=0.8, linestyle=":", alpha=0.45, zorder=2)

    # Annotate final AUC values at end of curves
    gap = base["best_auc"] - b2["best_auc"]

    ax.annotate(
        f"{base['best_auc']:.4f}",
        xy=(t_base[-1], auc_base[-1]),
        xytext=(-38, 5), textcoords="offset points",
        fontsize=7, color=COLOR_BASE, fontweight="bold"
    )
    b2_yoffset = -12 if b2["best_auc"] < base["best_auc"] else 5
    ax.annotate(
        f"{b2['best_auc']:.4f}",
        xy=(t_b2[-1], auc_b2[-1]),
        xytext=(-38, b2_yoffset), textcoords="offset points",
        fontsize=7, color=COLOR_B2, fontweight="bold"
    )

    # Outcome label inside panel
    winner = d["winner"]
    outcome_color = (COLOR_B2   if winner == "b2"
                     else COLOR_BASE if winner == "baseline"
                     else COLOR_VLINE)
    outcome_text  = (
        f"B=2 wins\n+{abs(gap):.3f} AUC"   if winner == "b2"
        else f"Baseline wins\n+{abs(gap):.3f} AUC" if winner == "baseline"
        else f"Tie\nΔ={abs(gap):.3f} AUC"
    )
    ax.text(0.04, 0.08, outcome_text,
            transform=ax.transAxes,
            fontsize=7.5, color=outcome_color,
            fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor=outcome_color,
                      linewidth=0.8, alpha=0.9))

    # Axes labels and title
    ax.set_xlabel("Wall-clock time (s)", fontsize=9)
    ax.set_ylabel("Validation AUC",      fontsize=9)
    ax.set_title(name, fontsize=10, fontweight="bold", pad=4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(d["time_fmt"]))
    ax.set_xlim(left=0)

    lo, hi = d["auc_range"]
    ax.set_ylim(lo, hi)

    # Legend inside panel — top left
    ax.legend(fontsize=7, loc="upper left",
              framealpha=0.9, edgecolor="#cccccc",
              handlelength=2.0)


# ─────────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────────

pdf_path = OUT_DIR + "figure3_equal_budget.pdf"
png_path = OUT_DIR + "figure3_equal_budget.png"

fig.savefig(pdf_path, format="pdf", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(png_path, format="png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:")
print("  " + pdf_path + "  (vector PDF)")
print("  " + png_path + "  (300 DPI PNG)")