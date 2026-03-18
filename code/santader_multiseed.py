"""
Santander — Multi-Seed Equal-Budget Reruns
==========================================
Runs the equal-budget experiment on Santander across 4 random seeds
to establish mean and standard deviation for the headline claim.

Seed 42 result is already known:
    Baseline AUC = 0.80692  |  B=2 AUC = 0.84107  |  gap = -0.03415

This script runs seeds 123, 456, 789 fresh and produces a summary
table with mean ± std across all 4 seeds.

Each seed:
    1. Fit baseline (B=1, col=1.0, 400 trees) → records budget T
    2. Fit B=2 live (col=0.5, time_limit=T)   → records best_val_auc

Nothing from previous experiments is reused — fresh fits only.
Seed 42 numbers are hardcoded from previous run and included in
the final summary table without refitting.

Outputs (saved to /kaggle/working/):
    santander_seed_123_equal_budget.csv
    santander_seed_456_equal_budget.csv
    santander_seed_789_equal_budget.csv
    santander_multiseed_summary.csv
    santander_multiseed_summary.png

Runtime estimate (tau_tree ~24s per tree):
    Per seed: baseline ~400x24 = ~2.7 hrs + b2_live ~2.7 hrs = ~5.4 hrs
    3 seeds total = ~16 hrs
    Run across 2 Kaggle sessions (seeds 123+456 in session 1,
    seed 789 in session 2), or run one seed per session.

    To run a single seed only, set SEEDS = [123] at the top.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import warnings
import os
warnings.filterwarnings("ignore")

OUT_DIR   = "/kaggle/working/"
DATA_PATH = "/kaggle/input/santander-customer-transaction-prediction/train.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Configure which seeds to run in this session ──────────────────
# Change this to [123], [456], or [789] to run one seed per session.
# Run all three across sessions then combine using the summary at the end.
SEEDS        = [123, 456, 789]
N_ESTIMATORS = 400

# ── Seed 42 result from previous run — hardcoded, no refit ───────
SEED_42_RESULT = {
    "seed":           42,
    "baseline_time":  9528.1,
    "baseline_trees": 400,
    "baseline_auc":   0.80692,
    "b2_time":        9501.3,
    "b2_trees":       1724,
    "b2_auc":         0.84107,
    "auc_gap":        -0.03415,
    "trees_ratio":    4.31,
}


# ─────────────────────────────────────────────────────────────────
#  Core model
# ─────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def fit_single_tree(X, residuals, max_features, max_depth,
                    min_samples_leaf, seed):
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed
    )
    tree.fit(X, residuals)
    return tree


class BlockParallelGBM:
    def __init__(self, n_estimators=400, block_size=1, learning_rate=0.1,
                 max_depth=4, min_samples_leaf=20, colsample=1.0,
                 auto_scale_lr=True, n_jobs=-1, random_state=42,
                 verbose=True, time_limit_seconds=None):
        self.n_estimators       = n_estimators
        self.block_size         = block_size
        self.learning_rate      = learning_rate
        self.max_depth          = max_depth
        self.min_samples_leaf   = min_samples_leaf
        self.colsample          = colsample
        self.auto_scale_lr      = auto_scale_lr
        self.n_jobs             = n_jobs
        self.random_state       = random_state
        self.verbose            = verbose
        self.time_limit_seconds = time_limit_seconds
        self._effective_lr      = (learning_rate / block_size
                                   if auto_scale_lr else learning_rate)
        self.trees_             = []
        self.F0_                = None
        self.train_auc_         = []
        self.val_auc_           = []
        self.block_times_       = []
        self.cumulative_times_  = []

    def _initial_prediction(self, y):
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y, X_val=None, y_val=None):
        X        = np.asarray(X, dtype=np.float32)
        y        = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        rng      = np.random.RandomState(self.random_state)
        self.F0_ = self._initial_prediction(y)
        F        = np.full(n_samples, self.F0_, dtype=np.float64)

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            F_val = np.full(len(X_val), self.F0_, dtype=np.float64)

        n_blocks        = int(np.ceil(self.n_estimators / self.block_size))
        cumulative_time = 0.0
        train_start     = time.perf_counter()

        if self.verbose:
            n_feats = (int(n_features * self.colsample)
                       if isinstance(self.colsample, float)
                       else self.colsample)
            print("  block_size=" + str(self.block_size) +
                  " | colsample=" + str(self.colsample) +
                  " (~" + str(n_feats) + " feats)" +
                  " | lr=" + str(self.learning_rate) +
                  " | max_depth=" + str(self.max_depth) +
                  " | n_estimators=" + str(self.n_estimators) +
                  " | seed=" + str(self.random_state) +
                  " | time_limit=" + str(self.time_limit_seconds) + "s")
            print("  " + "-" * 65)

        for block_idx in range(n_blocks):

            if (self.time_limit_seconds is not None and
                    time.perf_counter() - train_start > self.time_limit_seconds):
                if self.verbose:
                    print("  Time limit reached after block " +
                          str(block_idx) + ".")
                break

            t0               = time.perf_counter()
            trees_this_block = min(self.block_size,
                                   self.n_estimators - block_idx * self.block_size)
            residuals = y - sigmoid(F)
            seeds     = rng.randint(0, 2**31, size=trees_this_block)

            if trees_this_block == 1:
                results = [fit_single_tree(
                    X, residuals, self.colsample,
                    self.max_depth, self.min_samples_leaf, seeds[0]
                )]
            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(fit_single_tree)(
                        X, residuals, self.colsample,
                        self.max_depth, self.min_samples_leaf, seeds[b]
                    )
                    for b in range(trees_this_block)
                )

            for tree in results:
                self.trees_.append(tree)
                F += self._effective_lr * tree.predict(X)
                if X_val is not None:
                    F_val += self._effective_lr * tree.predict(X_val)

            elapsed         = time.perf_counter() - t0
            cumulative_time += elapsed
            self.block_times_.append(elapsed)
            self.cumulative_times_.append(cumulative_time)

            train_auc = roc_auc_score(y, sigmoid(F))
            self.train_auc_.append(train_auc)

            if X_val is not None and y_val is not None:
                self.val_auc_.append(roc_auc_score(y_val, sigmoid(F_val)))

            if self.verbose and (block_idx % 20 == 0
                                  or block_idx == n_blocks - 1):
                val_str = (" | val_auc=" +
                           str(round(self.val_auc_[-1], 5))
                           if self.val_auc_ else "")
                print("  Block " + str(block_idx + 1).rjust(4) +
                      "/" + str(n_blocks) +
                      " | trees=" + str(len(self.trees_)).rjust(5) +
                      " | train_auc=" + str(round(train_auc, 5)) +
                      val_str +
                      " | elapsed=" + str(round(cumulative_time, 1)) + "s")

        return self

    @property
    def total_time(self):
        return sum(self.block_times_)

    @property
    def best_val_auc(self):
        return max(self.val_auc_) if self.val_auc_ else None


# ─────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────

def load_santander(path, seed):
    """
    Loads Santander and splits 80/20 stratified using the given seed.
    Different seeds produce different train/val splits — this is the
    primary source of variance across reruns.
    """
    print("\nLoading Santander (seed=" + str(seed) + ") ...")
    df    = pd.read_csv(path)
    y     = df["target"].values
    X     = df.drop(columns=["ID_code", "target"]).values.astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )
    print("  Train: " + str(X_tr.shape) +
          " | Val: " + str(X_val.shape) +
          " | Positive rate (train): " + str(round(y_tr.mean(), 4)))
    return X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────────────────────────
#  Single seed experiment
# ─────────────────────────────────────────────────────────────────

def run_seed(seed, path):
    SEP = "=" * 65
    print("\n" + SEP)
    print("SANTANDER EQUAL-BUDGET — SEED " + str(seed))
    print(SEP)

    X_tr, X_val, y_tr, y_val = load_santander(path, seed)

    # ── Baseline ─────────────────────────────────────────────────
    print("\n[baseline] B=1, col=1.0, n_estimators=" + str(N_ESTIMATORS))
    baseline = BlockParallelGBM(
        n_estimators=N_ESTIMATORS,
        block_size=1, learning_rate=0.1,
        max_depth=4, min_samples_leaf=20,
        colsample=1.0, auto_scale_lr=False,
        n_jobs=1, random_state=seed, verbose=True
    )
    baseline.fit(X_tr, y_tr, X_val, y_val)
    budget = baseline.total_time

    print("\n  Baseline complete:")
    print("  total_time   = " + str(round(budget, 1)) + "s")
    print("  trees        = " + str(len(baseline.trees_)))
    print("  best_val_auc = " + str(round(baseline.best_val_auc, 5)))
    print("\n  Budget for B=2: " + str(round(budget, 1)) + "s")

    # ── B=2 live ──────────────────────────────────────────────────
    print("\n[b2_live] B=2, col=0.5 | time_limit=" +
          str(round(budget, 1)) + "s")
    b2 = BlockParallelGBM(
        n_estimators=10_000,
        block_size=2, learning_rate=0.1,
        max_depth=4, min_samples_leaf=20,
        colsample=0.5, auto_scale_lr=True,
        n_jobs=-1, random_state=seed, verbose=True,
        time_limit_seconds=budget
    )
    b2.fit(X_tr, y_tr, X_val, y_val)

    print("\n  B=2 complete:")
    print("  total_time   = " + str(round(b2.total_time, 1)) + "s")
    print("  trees        = " + str(len(b2.trees_)))
    print("  best_val_auc = " + str(round(b2.best_val_auc, 5)))

    # ── Per-seed result ───────────────────────────────────────────
    auc_gap     = baseline.best_val_auc - b2.best_val_auc
    trees_ratio = len(b2.trees_) / len(baseline.trees_)

    if auc_gap < -0.002:
        verdict = "B=2 wins"
    elif auc_gap > 0.002:
        verdict = "baseline wins"
    else:
        verdict = "effectively tied"

    print("\n" + SEP)
    print("SEED " + str(seed) + " RESULT:")
    print("  Baseline AUC = " + str(round(baseline.best_val_auc, 5)))
    print("  B=2 AUC      = " + str(round(b2.best_val_auc, 5)))
    print("  AUC gap      = " + str(round(auc_gap, 5)) +
          "  (" + verdict + ")")
    print("  Trees ratio  = " + str(round(trees_ratio, 2)) + "x" +
          "  (" + str(len(b2.trees_)) + " vs " +
          str(len(baseline.trees_)) + ")")
    print(SEP)

    # ── Save per-seed CSV ─────────────────────────────────────────
    df = pd.DataFrame([
        {
            "seed":           seed,
            "config":         "Baseline (B=1, col=1.0)",
            "budget_s":       round(budget, 1),
            "time_used_s":    round(baseline.total_time, 1),
            "trees_built":    len(baseline.trees_),
            "best_val_auc":   round(baseline.best_val_auc, 5),
            "auc_gap":        0.0,
            "trees_ratio":    1.0,
        },
        {
            "seed":           seed,
            "config":         "Block B=2, col=0.5",
            "budget_s":       round(budget, 1),
            "time_used_s":    round(b2.total_time, 1),
            "trees_built":    len(b2.trees_),
            "best_val_auc":   round(b2.best_val_auc, 5),
            "auc_gap":        round(auc_gap, 5),
            "trees_ratio":    round(trees_ratio, 2),
        },
    ])
    fname = OUT_DIR + "santander_seed_" + str(seed) + "_equal_budget.csv"
    df.to_csv(fname, index=False)
    print("Saved: " + fname)

    # ── Per-seed plot ─────────────────────────────────────────────
    plot_seed(baseline, b2, budget, seed, auc_gap, trees_ratio)

    return {
        "seed":           seed,
        "baseline_time":  round(budget, 1),
        "baseline_trees": len(baseline.trees_),
        "baseline_auc":   round(baseline.best_val_auc, 5),
        "b2_time":        round(b2.total_time, 1),
        "b2_trees":       len(b2.trees_),
        "b2_auc":         round(b2.best_val_auc, 5),
        "auc_gap":        round(auc_gap, 5),
        "trees_ratio":    round(trees_ratio, 2),
    }


def plot_seed(baseline, b2, budget, seed, auc_gap, trees_ratio):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Santander Equal-Budget — Seed " + str(seed) +
        "  |  AUC gap=" + str(round(auc_gap, 5)) +
        "  |  Trees ratio=" + str(round(trees_ratio, 2)) + "x",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: AUC vs tree count
    baseline_counts = list(range(1, len(baseline.trees_) + 1))
    b2_counts       = list(range(2, len(b2.trees_) + 2, 2))
    # trim b2_counts to match val_auc_ length
    b2_counts       = [(i + 1) * 2 for i in range(len(b2.block_times_))]

    ax1.plot(baseline_counts, baseline.val_auc_,
             color="blue", linewidth=2,
             label="Baseline — " + str(len(baseline.trees_)) + " trees")
    ax1.plot(b2_counts, b2.val_auc_,
             color="red", linewidth=2, linestyle="--",
             label="B=2, col=0.5 — " + str(len(b2.trees_)) + " trees")
    ax1.set_xlabel("Number of trees", fontsize=11)
    ax1.set_ylabel("Validation AUC", fontsize=11)
    ax1.set_title("Val AUC vs Tree Count", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: AUC vs wall-clock time
    ax2.plot(baseline.cumulative_times_, baseline.val_auc_,
             color="blue", linewidth=2,
             label="Baseline  AUC=" +
             str(round(baseline.best_val_auc, 4)))
    ax2.plot(b2.cumulative_times_, b2.val_auc_,
             color="red", linewidth=2, linestyle="--",
             label="B=2  AUC=" + str(round(b2.best_val_auc, 4)))
    ax2.axvline(x=budget, color="gray", linewidth=1.5, linestyle=":",
                label="Equal budget = " + str(round(budget, 0)) + "s")
    ax2.axhline(y=baseline.best_val_auc, color="blue",
                linewidth=1, linestyle=":", alpha=0.4)
    ax2.annotate(str(round(baseline.best_val_auc, 4)),
                 xy=(baseline.cumulative_times_[-1],
                     baseline.val_auc_[-1]),
                 xytext=(8, 4), textcoords="offset points",
                 fontsize=9, color="blue", fontweight="bold")
    ax2.annotate(str(round(b2.best_val_auc, 4)),
                 xy=(b2.cumulative_times_[-1], b2.val_auc_[-1]),
                 xytext=(8, -12), textcoords="offset points",
                 fontsize=9, color="red", fontweight="bold")
    ax2.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax2.set_ylabel("Validation AUC", fontsize=11)
    ax2.set_title("Val AUC vs Wall-Clock Time", fontsize=11,
                  fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = OUT_DIR + "santander_seed_" + str(seed) + "_equal_budget.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: santander_seed_" + str(seed) + "_equal_budget.png")


# ─────────────────────────────────────────────────────────────────
#  Multi-seed summary
# ─────────────────────────────────────────────────────────────────

def summarise_all_seeds(results):
    """
    Combines seed 42 (hardcoded) with fresh results.
    Prints mean +- std for the headline AUC gap claim.
    """
    all_results = [SEED_42_RESULT] + results

    SEP = "=" * 65
    print("\n" + SEP)
    print("MULTI-SEED SUMMARY — SANTANDER EQUAL-BUDGET")
    print(SEP)

    rows = []
    for r in all_results:
        rows.append({
            "Seed":           r["seed"],
            "Baseline AUC":   r["baseline_auc"],
            "B=2 AUC":        r["b2_auc"],
            "AUC gap":        r["auc_gap"],
            "Trees ratio":    r["trees_ratio"],
            "Baseline trees": r["baseline_trees"],
            "B=2 trees":      r["b2_trees"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Stats on AUC gap
    gaps        = [r["auc_gap"] for r in all_results]
    ratios      = [r["trees_ratio"] for r in all_results]
    mean_gap    = round(float(np.mean(gaps)), 5)
    std_gap     = round(float(np.std(gaps)), 5)
    mean_ratio  = round(float(np.mean(ratios)), 2)
    min_gap     = round(float(np.min(gaps)), 5)
    max_gap     = round(float(np.max(gaps)), 5)

    print("\n  AUC gap (baseline - B=2):")
    print("    Mean  = " + str(mean_gap))
    print("    Std   = " + str(std_gap))
    print("    Min   = " + str(min_gap))
    print("    Max   = " + str(max_gap))
    print("    All negative (B=2 wins every seed): " +
          str(all(g < 0 for g in gaps)))
    print("\n  Trees ratio:")
    print("    Mean  = " + str(mean_ratio) + "x")

    # Paper-ready sentence
    print("\n  Paper-ready result:")
    print("  'Under equal wall-clock budget, B=2 improves over the")
    print("   sequential baseline by " + str(abs(mean_gap)) +
          " +- " + str(std_gap) + " AUC")
    print("   (mean +- std across " + str(len(all_results)) + " random seeds),")
    print("   building " + str(mean_ratio) + "x more trees in the same time.'")

    df.to_csv(OUT_DIR + "santander_multiseed_summary.csv", index=False)
    print("\nSaved: santander_multiseed_summary.csv")
    return df, mean_gap, std_gap


def plot_multiseed_summary(results):
    all_results = [SEED_42_RESULT] + results
    seeds       = [r["seed"] for r in all_results]
    gaps        = [-r["auc_gap"] for r in all_results]  # positive = B=2 wins
    baseline_aucs = [r["baseline_auc"] for r in all_results]
    b2_aucs       = [r["b2_auc"] for r in all_results]
    mean_gap      = float(np.mean(gaps))
    std_gap       = float(np.std(gaps))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Santander Multi-Seed Equal-Budget Results\n" +
        "Mean AUC gain = " + str(round(mean_gap, 4)) +
        " +- " + str(round(std_gap, 4)) +
        " across " + str(len(all_results)) + " seeds",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: AUC gain per seed (bar chart)
    colors = ["green" if g > 0.002 else "orange" if g > 0 else "red"
              for g in gaps]
    bars = ax1.bar([str(s) for s in seeds], gaps,
                   color=colors, edgecolor="white",
                   linewidth=1.2, width=0.5)
    ax1.axhline(y=0, color="black", linewidth=1.2, linestyle="-")
    ax1.axhline(y=mean_gap, color="blue", linewidth=1.5,
                linestyle="--",
                label="Mean = " + str(round(mean_gap, 4)))
    ax1.fill_between([-0.5, len(seeds) - 0.5],
                     mean_gap - std_gap, mean_gap + std_gap,
                     alpha=0.12, color="blue",
                     label="Mean +- std")

    for bar, g in zip(bars, gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 g + 0.0005,
                 str(round(g, 4)),
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax1.set_xlabel("Random seed", fontsize=11)
    ax1.set_ylabel("AUC gain of B=2 over baseline", fontsize=11)
    ax1.set_title("Equal-Budget AUC Gain per Seed\n(positive = B=2 wins)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Absolute AUC comparison per seed
    x      = np.arange(len(seeds))
    width  = 0.35
    ax2.bar(x - width/2, baseline_aucs, width,
            label="Baseline (B=1)", color="blue",
            alpha=0.8, edgecolor="white")
    ax2.bar(x + width/2, b2_aucs, width,
            label="B=2, col=0.5", color="red",
            alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Random seed", fontsize=11)
    ax2.set_ylabel("Best val AUC", fontsize=11)
    ax2.set_title("Baseline vs B=2 AUC per Seed",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    # Set y-axis to zoom in on the AUC range
    all_aucs = baseline_aucs + b2_aucs
    ax2.set_ylim(min(all_aucs) - 0.005, max(all_aucs) + 0.005)

    plt.tight_layout()
    p = OUT_DIR + "santander_multiseed_summary.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: santander_multiseed_summary.png")


# ─────────────────────────────────────────────────────────────────
#  Master runner
# ─────────────────────────────────────────────────────────────────

def run_all():
    SEP = "=" * 65
    print(SEP)
    print("SANTANDER MULTI-SEED EQUAL-BUDGET RERUNS")
    print("Seeds to run this session: " + str(SEEDS))
    print("Seed 42 result hardcoded from previous run (no refit)")
    print("Est. time per seed: ~5.4 hrs | Total: ~" +
          str(len(SEEDS) * 5.4) + " hrs")
    print(SEP)

    results = []
    for seed in SEEDS:
        result = run_seed(seed, DATA_PATH)
        results.append(result)
        # Save intermediate summary after each seed
        # in case session times out before all seeds finish
        interim_df = pd.DataFrame([SEED_42_RESULT] + results)
        interim_df.to_csv(
            OUT_DIR + "santander_multiseed_interim.csv",
            index=False
        )
        print("\nInterim summary saved after seed " + str(seed))

    # Final summary across all seeds including seed 42
    df_summary, mean_gap, std_gap = summarise_all_seeds(results)
    plot_multiseed_summary(results)

    print("\n" + SEP)
    print("ALL SEEDS COMPLETE")
    print("Outputs saved to: " + OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("santander"):
            print("  " + f)
    print(SEP)


if __name__ == "__main__":
    run_all()