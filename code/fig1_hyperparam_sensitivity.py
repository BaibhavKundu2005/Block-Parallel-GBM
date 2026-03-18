"""
Fig. 1 — Hyperparameter Sensitivity Heatmaps (Covertype)
=========================================================
Generates:
    fig1_hparam_sensitivity.emf  — vector, insert into Word
    fig1_hparam_sensitivity.pdf  — vector, for final PDF submission
    fig1_hparam_sensitivity.png  — 300 DPI backup

Run locally on Windows. Requires matplotlib >= 3.5.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

OUT = "./"   # change to your output folder if needed

# ── Data ──────────────────────────────────────────────────────────
learning_rates = [0.05, 0.10, 0.20]
max_depths     = [3, 4, 6]

# AUC gap grid (baseline AUC - B=2 AUC), rows=depth, cols=lr
auc_gap = np.array([
    [0.011, 0.014, 0.014],   # depth=3
    [0.011, 0.015, 0.017],   # depth=4
    [0.017, 0.018, 0.015],   # depth=6
])

# Speedup grid (baseline_time / b2_time), rows=depth, cols=lr
speedup = np.array([
    [1.65, 1.51, 1.50],      # depth=3
    [1.72, 1.68, 1.71],      # depth=4
    [2.00, 2.00, 2.00],      # depth=6
])

# ── Colour maps ───────────────────────────────────────────────────
# AUC gap — white (low/good) to dark red (high/bad)
cmap_gap = LinearSegmentedColormap.from_list(
    "gap_cmap", ["#FFFFFF", "#FEE0D2", "#FC9272", "#DE2D26", "#A50F15"]
)
# Speedup — white (low) to dark green (high/good)
cmap_spd = LinearSegmentedColormap.from_list(
    "spd_cmap", ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"]
)

# ── Figure setup ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
fig.patch.set_facecolor("white")  # type: ignore

lr_labels    = ["0.05", "0.10", "0.20"]
depth_labels = ["3", "4", "6"]

def draw_heatmap(ax, data, cmap, vmin, vmax, title,
                 fmt, xlabel, ylabel, show_ylabel):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")

    # Tick labels
    ax.set_xticks(range(len(lr_labels)))
    ax.set_xticklabels(lr_labels, fontsize=10)
    ax.set_yticks(range(len(depth_labels)))
    if show_ylabel:
        ax.set_yticklabels(depth_labels, fontsize=10)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel(xlabel, fontsize=10)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Cell annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Choose text colour based on background darkness
            norm_val = (val - vmin) / (vmax - vmin)
            text_color = "white" if norm_val > 0.6 else "black"
            ax.text(j, i, fmt.format(val),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    return im

# ── Left panel: AUC gap ───────────────────────────────────────────
draw_heatmap(
    axes[0], auc_gap,
    cmap=cmap_gap, vmin=0.010, vmax=0.020,
    title="(a) AUC gap  (baseline \u2212 B=2)",
    fmt="{:.3f}",
    xlabel="Learning rate",
    ylabel="Max depth",
    show_ylabel=True
)

# ── Right panel: Speedup ──────────────────────────────────────────
draw_heatmap(
    axes[1], speedup,
    cmap=cmap_spd, vmin=1.45, vmax=2.05,
    title="(b) Speedup  (baseline time / B=2 time)",
    fmt="{:.2f}\u00d7",
    xlabel="Learning rate",
    ylabel="Max depth",
    show_ylabel=False
)


plt.tight_layout(rect=[0, 0.0, 1, 1])

# ── Save ──────────────────────────────────────────────────────────
for ext in ["emf", "pdf", "png"]:
    kw = {"dpi": 300} if ext == "png" else {}
    path = OUT + "fig1_hparam_sensitivity." + ext
    try:
        plt.savefig(path, format=ext, bbox_inches="tight",
                    facecolor="white", **kw)
        print("Saved: " + path)
    except Exception as e:
        print("Could not save " + ext + ": " + str(e))

plt.close()
print("Done.")