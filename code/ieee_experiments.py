# MIT License — see LICENSE file in the root directory

"""
IEEE CIS Fraud Detection — Block-Parallel GBM Experiments
=====================================================================
Mirrors the Santander experiment structure exactly on a second large dataset.

Dataset:
    IEEE CIS Fraud Detection (Kaggle)
    Path: /kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv
    Original size: 595,212 rows, 59 columns (57 features + id + target)
    We subsample 200,000 rows stratified on the binary target.

Experiments:
    1. Ablation        — baseline, col_only, block_only, b2 (4 fits × 400 trees)
    2. Block sweep     — B=1,2,3,4 at col=0.5       (4 fits × 400 trees, B=1 reused)
    3. Equal budget    — baseline natural time → b2 live, b3 live (no tree cap)

Nothing from previous experiments is recomputed. All fits are fresh.

Outputs (all saved to /kaggle/working/):
    ieee_ablation.csv
    ieee_block_sweep.csv
    ieee_equal_budget.csv
    ieee_ablation.png
    ieee_block_sweep.png
    ieee_equal_budget.png
    ieee_learning_curves.png

Runtime estimate:
    Per-tree cost on IEEE CIS (200k rows, 57 features) ≈ 3-6s
    400-tree fit ≈ 25-40 min
    4 ablation fits  ≈ 2.0-2.5 hrs
    4 sweep fits     ≈ 2.0-2.5 hrs  (B=1 reused from ablation, 3 fresh)
    2 equal-budget   ≈ 1.0-1.5 hrs
    ─────────────────────────────────
    Total            ≈ 5-6 hrs
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
DATA_PATH = "/kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv"
N_SUBSAMPLE  = 200_000
N_ESTIMATORS = 400
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Core model (verbatim from main experiments)
# ─────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def compute_residuals(y, F):
    return y - sigmoid(F)

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
        self.n_estimators      = n_estimators
        self.block_size        = block_size
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.colsample         = colsample
        self.auto_scale_lr     = auto_scale_lr
        self.n_jobs            = n_jobs
        self.random_state      = random_state
        self.verbose           = verbose
        self.time_limit_seconds = time_limit_seconds
        self._effective_lr     = (learning_rate / block_size
                                  if auto_scale_lr else learning_rate)
        self.trees_            = []
        self.F0_               = None
        self.train_auc_        = []
        self.val_auc_          = []
        self.block_times_      = []
        self.cumulative_times_ = []

    def _initial_prediction(self, y):
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y, X_val=None, y_val=None):
        X   = np.asarray(X, dtype=np.float32)
        y   = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
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
            print(f"  block_size={self.block_size} | "
                  f"colsample={self.colsample} (~{n_feats} feats) | "
                  f"lr={self.learning_rate} | max_depth={self.max_depth} | "
                  f"n_estimators={self.n_estimators} | "
                  f"time_limit={self.time_limit_seconds}s")
            print("  " + "-" * 65)
        for block_idx in range(n_blocks):
            if (self.time_limit_seconds is not None and
                    time.perf_counter() - train_start > self.time_limit_seconds):
                if self.verbose:
                    print(f"  Time limit reached after block {block_idx}.")
                break
            t0               = time.perf_counter()
            trees_this_block = min(self.block_size,
                                   self.n_estimators - block_idx * self.block_size)
            residuals = compute_residuals(y, F)
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
                val_str = (f" | val_auc={self.val_auc_[-1]:.5f}"
                           if self.val_auc_ else "")
                print(f"  Block {block_idx+1:>4}/{n_blocks} | "
                      f"trees={len(self.trees_):>5} | "
                      f"train_auc={train_auc:.5f}{val_str} | "
                      f"elapsed={cumulative_time:.1f}s")
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

def load_ieee(path, n_samples=200_000, n_features=200, random_state=42):
    print(f"\nLoading IEEE-CIS from {path} ...")
    df = pd.read_csv(path)
    print(f"  Full dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    y_full = df["isFraud"].values
    X_full = df.drop(columns=["TransactionID", "isFraud"])

    # Label-encode all string columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in X_full.select_dtypes(include="object").columns:
        X_full[col] = X_full[col].fillna("missing")
        X_full[col] = le.fit_transform(X_full[col].astype(str))

    # Fill remaining numeric NaNs with -1
    X_full = X_full.fillna(-1).values.astype(np.float32)

    # Select top n_features by variance
    if n_features is not None and n_features < X_full.shape[1]:
        variances   = X_full.var(axis=0)
        top_indices = np.argsort(variances)[::-1][:n_features]
        top_indices = np.sort(top_indices)   # keep original column order
        X_full      = X_full[:, top_indices]
        print(f"  Selected top {n_features} features by variance")

    # Stratified subsample
    if n_samples is not None and n_samples < len(y_full):
        _, X_sub, _, y_sub = train_test_split(
            X_full, y_full,
            test_size=n_samples / len(y_full),
            random_state=random_state,
            stratify=y_full
        )
        print(f"  Subsampled: {X_sub.shape[0]:,} rows x {X_sub.shape[1]} features")
    else:
        X_sub, y_sub = X_full, y_full
        print(f"  Using full dataset: {X_sub.shape[0]:,} rows")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y_sub,
        test_size=0.2,
        random_state=random_state,
        stratify=y_sub
    )
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | "
          f"Positive rate (train): {y_tr.mean():.4f}")
    return X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────────────────────────
#  Shared fit helper
# ─────────────────────────────────────────────────────────────────

def fit_config(label, block_size, colsample, auto_scale_lr, n_jobs,
               X_tr, y_tr, X_val, y_val,
               n_estimators=N_ESTIMATORS,
               time_limit=None):
    print(f"\n[{label}]")
    m = BlockParallelGBM(
        n_estimators=n_estimators,
        block_size=block_size,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=20,
        colsample=colsample,
        auto_scale_lr=auto_scale_lr,
        n_jobs=n_jobs,
        random_state=42,
        verbose=True,
        time_limit_seconds=time_limit
    )
    m.fit(X_tr, y_tr, X_val, y_val)
    m.label = label
    print(f"  → total_time={m.total_time:.1f}s | "
          f"trees={len(m.trees_)} | "
          f"best_val_auc={m.best_val_auc:.5f}")
    return m


# ─────────────────────────────────────────────────────────────────
#  Experiment 1 — Ablation
# ─────────────────────────────────────────────────────────────────

def run_ablation(X_tr, y_tr, X_val, y_val):
    """
    Four configs: baseline, col_only, block_only, combined.
    Returns dict keyed by config name.
    """
    print(f"\n{'='*65}")
    print("EXPERIMENT 1: ABLATION — IEEE CIS")
    print(f"n_estimators={N_ESTIMATORS} | lr=0.1 | max_depth=4")
    print(f"{'='*65}")

    configs = [
        # label,        bs, col,  asl,   nj
        ("baseline",    1,  1.0,  False, 1),
        ("col_only",    1,  0.5,  False, 1),
        ("block_only",  2,  1.0,  True,  -1),
        ("combined",    2,  0.5,  True,  -1),
    ]

    cache = {}
    for label, bs, cs, asl, nj in configs:
        cache[label] = fit_config(label, bs, cs, asl, nj,
                                   X_tr, y_tr, X_val, y_val)
    return cache


def summarise_ablation(cache):
    ref_time = cache["baseline"].total_time
    ref_auc  = cache["baseline"].best_val_auc

    rows = []
    display = {
        "baseline":   "Baseline (B=1, col=1.0)",
        "col_only":   "Col only  (B=1, col=0.5)",
        "block_only": "Block only (B=2, col=1.0)",
        "combined":   "Combined  (B=2, col=0.5)",
    }
    bsizes = {"baseline": 1, "col_only": 1,
              "block_only": 2, "combined": 2}
    cols   = {"baseline": 1.0, "col_only": 0.5,
              "block_only": 1.0, "combined": 0.5}

    for key in ["baseline", "col_only", "block_only", "combined"]:
        m = cache[key]
        rows.append({
            "Config":         display[key],
            "Block size (B)": bsizes[key],
            "Colsample":      cols[key],
            "Trees":          len(m.trees_),
            "Total time (s)": round(m.total_time, 1),
            "Best val AUC":   round(m.best_val_auc, 5),
            "Speedup":        f"{ref_time / m.total_time:.2f}x",
            "AUC gap":        round(ref_auc - m.best_val_auc, 5),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*65}")
    print("ABLATION TABLE — IEEE CIS")
    print(df.to_string(index=False))

    # Additivity check
    d_col   = ref_auc - cache["col_only"].best_val_auc
    d_block = ref_auc - cache["block_only"].best_val_auc
    d_comb  = ref_auc - cache["combined"].best_val_auc
    print(f"\nAdditivity check:")
    print(f"  Δ_col   = {d_col:.5f}")
    print(f"  Δ_block = {d_block:.5f}")
    print(f"  Δ_col + Δ_block = {d_col + d_block:.5f}")
    print(f"  Actual combined gap = {d_comb:.5f}")
    print(f"  Residual = {abs((d_col + d_block) - d_comb):.5f}")

    df.to_csv(f"{OUT_DIR}ieee_ablation.csv", index=False)
    print(f"\nSaved: ieee_ablation.csv")
    return df


def plot_ablation(cache):
    colors  = {"baseline": "blue", "col_only": "green",
               "block_only": "red", "combined": "purple"}
    lstyles = {"baseline": "-", "col_only": "--",
               "block_only": "-.", "combined": ":"}
    labels  = {
        "baseline":   "Baseline (B=1, col=1.0)",
        "col_only":   "Col only (B=1, col=0.5)",
        "block_only": "Block only (B=2, col=1.0)",
        "combined":   "Combined (B=2, col=0.5)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation — IEEE CIS", fontsize=13,
                 fontweight="bold", y=1.01)

    for key, m in cache.items():
        tree_counts = [min((i + 1) * m.block_size, m.n_estimators)
                       for i in range(len(m.block_times_))]
        ax1.plot(tree_counts, m.val_auc_,
                 color=colors[key], linestyle=lstyles[key],
                 linewidth=2, label=labels[key])
        ax2.plot(m.cumulative_times_, m.val_auc_,
                 color=colors[key], linestyle=lstyles[key],
                 linewidth=2, label=labels[key])
        # annotate final AUC on time plot
        ax2.annotate(f"{m.best_val_auc:.4f}",
                     xy=(m.cumulative_times_[-1], m.val_auc_[-1]),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=8, color=colors[key])

    for ax, xlabel in [(ax1, "Number of trees"),
                       (ax2, "Wall-clock time (s)")]:
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Validation AUC", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax1.set_title("Val AUC vs Tree Count", fontsize=11, fontweight="bold")
    ax2.set_title("Val AUC vs Wall-Clock Time", fontsize=11, fontweight="bold")

    plt.tight_layout()
    p = f"{OUT_DIR}ieee_ablation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: ieee_ablation.png")

# ─────────────────────────────────────────────────────────────────
#  Experiment 2 — Block sweep
#  B=1 reference = col_only from ablation (already fitted, reused)
# ─────────────────────────────────────────────────────────────────

def run_block_sweep(ablation_cache, X_tr, y_tr, X_val, y_val):
    """
    B=1 (col=0.5) is reused from ablation cache — no refit.
    B=2 (col=0.5) is reused from ablation cache (combined) — no refit.
    B=3 and B=4 are fresh fits.
    """
    print(f"\n{'='*65}")
    print("EXPERIMENT 2: BLOCK SWEEP — IEEE CIS")
    print(f"col=0.5 throughout | n_estimators={N_ESTIMATORS}")
    print(f"{'='*65}")

    sweep_cache = {}

    # Reuse from ablation — no refit
    sweep_cache[1] = ablation_cache["col_only"]
    print(f"\n[B=1] Reused from ablation (col_only). "
          f"total_time={sweep_cache[1].total_time:.1f}s | "
          f"best_val_auc={sweep_cache[1].best_val_auc:.5f}")

    sweep_cache[2] = ablation_cache["combined"]
    print(f"[B=2] Reused from ablation (combined). "
          f"total_time={sweep_cache[2].total_time:.1f}s | "
          f"best_val_auc={sweep_cache[2].best_val_auc:.5f}")

    # Fresh fits for B=3 and B=4
    for bs in [3, 4]:
        sweep_cache[bs] = fit_config(
            f"B={bs}, col=0.5", bs, 0.5, True, -1,
            X_tr, y_tr, X_val, y_val
        )

    return sweep_cache


def summarise_block_sweep(sweep_cache):
    ref_time = sweep_cache[1].total_time
    ref_auc  = sweep_cache[1].best_val_auc

    rows = []
    for bs, m in sorted(sweep_cache.items()):
        rows.append({
            "Block size (B)": bs,
            "Trees":          len(m.trees_),
            "Total time (s)": round(m.total_time, 1),
            "Best val AUC":   round(m.best_val_auc, 5),
            "Speedup":        f"{ref_time / m.total_time:.2f}x",
            "AUC gap":        round(ref_auc - m.best_val_auc, 5),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*65}")
    print("BLOCK SWEEP TABLE — IEEE CIS")
    print(df.to_string(index=False))
    df.to_csv(f"{OUT_DIR}ieee_block_sweep.csv", index=False)
    print(f"\nSaved: ieee_block_sweep.csv")
    return df


def plot_block_sweep(sweep_cache, df):
    bsizes       = sorted(sweep_cache.keys())
    speedup_vals = [float(s.replace("x", "")) for s in df["Speedup"]]
    auc_gaps     = list(df["AUC gap"])

    colors  = {1: "blue", 2: "red", 3: "green", 4: "orange"}
    lstyles = {1: "-", 2: "--", 3: "-.", 4: ":"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Block Sweep — IEEE CIS  (col=0.5)",
                 fontsize=13, fontweight="bold", y=1.01)

    # Panel 1: Speedup vs B
    ax1 = axes[0]
    ax1.plot(bsizes, speedup_vals, "bo-", linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6)
    for x, y in zip(bsizes, speedup_vals):
        ax1.annotate(f"{y:.2f}x", (x, y),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=10)
    ax1.set_xlabel("Block size (B)", fontsize=11)
    ax1.set_ylabel("Speedup vs B=1", fontsize=11)
    ax1.set_title("Speedup vs Block Size", fontsize=11, fontweight="bold")
    ax1.set_xticks(bsizes)
    ax1.grid(True, alpha=0.3)

    # Panel 2: AUC gap vs B
    ax2 = axes[1]
    ax2.plot(bsizes, auc_gaps, "rs-", linewidth=2, markersize=8)
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Block size (B)", fontsize=11)
    ax2.set_ylabel("AUC gap (B=1 − B=x)", fontsize=11)
    ax2.set_title("AUC Gap vs Block Size", fontsize=11, fontweight="bold")
    ax2.set_xticks(bsizes)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Learning curves (AUC vs wall-clock time)
    ax3 = axes[2]
    for bs, m in sorted(sweep_cache.items()):
        ax3.plot(m.cumulative_times_, m.val_auc_,
                 color=colors[bs], linestyle=lstyles[bs],
                 linewidth=2,
                 label=f"B={bs}  (AUC={m.best_val_auc:.4f})")
    ax3.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax3.set_ylabel("Validation AUC", fontsize=11)
    ax3.set_title("Val AUC vs Wall-Clock Time", fontsize=11,
                  fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{OUT_DIR}ieee_block_sweep.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: ieee_block_sweep.png")

# ─────────────────────────────────────────────────────────────────
#  Experiment 3 — Equal budget
#  Baseline runs to completion (400 trees), records total time T.
#  B=2 and B=3 run with time_limit=T, no tree cap.
#  Baseline is reused from ablation — no refit.
# ─────────────────────────────────────────────────────────────────

def run_equal_budget(ablation_cache, X_tr, y_tr, X_val, y_val):
    print(f"\n{'='*65}")
    print("EXPERIMENT 3: EQUAL-BUDGET — IEEE CIS")
    print(f"{'='*65}")

    # Reuse baseline from ablation
    baseline = ablation_cache["baseline"]
    budget   = baseline.total_time
    print(f"\n[Baseline] Reused from ablation. "
          f"total_time={budget:.1f}s | "
          f"trees={len(baseline.trees_)} | "
          f"best_val_auc={baseline.best_val_auc:.5f}")
    print(f"\nTime budget for live runs: {budget:.1f}s")

    # B=2 live — no tree cap, time limited
    print(f"\n[B=2 live] col=0.5 | time_limit={budget:.1f}s")
    b2_live = fit_config(
        "B=2 live", 2, 0.5, True, -1,
        X_tr, y_tr, X_val, y_val,
        n_estimators=10_000,
        time_limit=budget
    )

    # B=3 live
    print(f"\n[B=3 live] col=0.5 | time_limit={budget:.1f}s")
    b3_live = fit_config(
        "B=3 live", 3, 0.5, True, -1,
        X_tr, y_tr, X_val, y_val,
        n_estimators=10_000,
        time_limit=budget
    )

    return baseline, b2_live, b3_live, budget


def summarise_equal_budget(baseline, b2_live, b3_live, budget):
    rows = []
    for label, m in [("Baseline (B=1, col=1.0)", baseline),
                     ("Block B=2, col=0.5",       b2_live),
                     ("Block B=3, col=0.5",       b3_live)]:
        rows.append({
            "Config":        label,
            "Budget (s)":    round(budget, 1),
            "Time used (s)": round(m.total_time, 1),
            "Trees built":   len(m.trees_),
            "Best val AUC":  round(m.best_val_auc, 5),
            "AUC gap":       round(baseline.best_val_auc - m.best_val_auc, 5),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*65}")
    print("EQUAL-BUDGET TABLE — IEEE CIS")
    print(df.to_string(index=False))

    # Trees ratio
    for _, m, lbl in [(None, b2_live, "B=2"),
                      (None, b3_live, "B=3")]:
        ratio = len(m.trees_) / len(baseline.trees_)
        gap   = baseline.best_val_auc - m.best_val_auc
        winner = ("baseline wins" if gap > 0.001
                  else "b_live wins" if gap < -0.001
                  else "effectively tied")
        print(f"\n  {lbl}: {len(m.trees_)} trees vs baseline "
              f"{len(baseline.trees_)} ({ratio:.2f}x) | "
              f"AUC gap={gap:+.5f} ({winner})")

    df.to_csv(f"{OUT_DIR}ieee_equal_budget.csv", index=False)
    print(f"\nSaved: ieee_equal_budget.csv")
    return df


def plot_equal_budget(baseline, b2_live, b3_live, budget):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(baseline.cumulative_times_, baseline.val_auc_,
            color="blue", linewidth=2,
            label=f"Baseline — {len(baseline.trees_)} trees")
    ax.plot(b2_live.cumulative_times_, b2_live.val_auc_,
            color="red", linewidth=2, linestyle="--",
            label=f"B=2, col=0.5 — {len(b2_live.trees_)} trees")
    ax.plot(b3_live.cumulative_times_, b3_live.val_auc_,
            color="green", linewidth=2, linestyle="-.",
            label=f"B=3, col=0.5 — {len(b3_live.trees_)} trees")

    ax.axvline(x=budget, color="gray", linewidth=1.5,
               linestyle=":", label=f"Equal budget = {budget:.0f}s")
    ax.axhline(y=baseline.best_val_auc, color="blue",
               linewidth=1, linestyle=":", alpha=0.4,
               label=f"Baseline AUC = {baseline.best_val_auc:.5f}")

    # Annotate final AUC values
    for m, color in [(baseline, "blue"),
                     (b2_live, "red"),
                     (b3_live, "green")]:
        ax.annotate(f"{m.best_val_auc:.4f}",
                    xy=(m.cumulative_times_[-1], m.val_auc_[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Equal-Budget Comparison — IEEE CIS\n"
                 "All configs given the same wall-clock budget as baseline",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{OUT_DIR}ieee_equal_budget.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: ieee_equal_budget.png")

# ─────────────────────────────────────────────────────────────────
#  Master runner
# ─────────────────────────────────────────────────────────────────

def profile_tree_cost(X_tr, y_tr, n_profile=3):
    """
    Fits n_profile single trees on the training data and reports
    average wall-clock time per tree. Uses the same settings as
    the actual experiments so the estimate is accurate.
    Prints a go/no-go recommendation before any experiment starts.
    """
    SEP = "=" * 65
    TAU_OVERHEAD = 0.3   # known joblib overhead on 4 cores

    print("")
    print(SEP)
    print("PRE-FLIGHT: PER-TREE COST PROFILING")
    print("Fitting " + str(n_profile) + " trees to estimate tau_tree ...")
    print(SEP)

    p     = np.clip(y_tr.mean(), 1e-6, 1 - 1e-6)
    F0    = np.log(p / (1 - p))
    F     = np.full(len(y_tr), F0, dtype=np.float64)
    r     = y_tr - sigmoid(F)
    X_arr = np.asarray(X_tr, dtype=np.float32)

    times = []
    for i in range(n_profile):
        t0   = time.perf_counter()
        tree = DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=20,
            max_features=1.0, random_state=i
        )
        tree.fit(X_arr, r)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print("  Tree " + str(i+1) + ": " + str(round(elapsed, 2)) + "s")

    tau_tree = float(np.mean(times))
    rho      = TAU_OVERHEAD / tau_tree

    print("")
    print("  Average tau_tree   = " + str(round(tau_tree, 2)) + "s")
    print("  tau_overhead       = " + str(TAU_OVERHEAD) + "s  (fixed, 4 cores)")
    print("  Overhead ratio rho = " + str(round(rho, 3)))
    print("")

    if rho < 0.05:
        verdict = ("STRONG GO — rho << 0.1. Block parallelism will give "
                   "clear equal-budget gains. Expect Santander-like pattern.")
    elif rho < 0.10:
        verdict = ("GO — rho < 0.1. Block parallelism likely to help. "
                   "Results may be less dramatic than Santander.")
    elif rho < 0.30:
        verdict = ("MARGINAL — rho 0.1-0.3. Equal-budget result could go "
                   "either way. Still worth running.")
    elif rho < 1.0:
        verdict = ("CAUTION — rho 0.3-1.0. Block parallelism unlikely to "
                   "help. Consider increasing N_SUBSAMPLE or max_depth.")
    else:
        verdict = ("NO-GO — rho >= 1.0. Overhead dominates. IEEE CIS at "
                   "200k rows is too small. Increase N_SUBSAMPLE to full "
                   "595k or choose a higher-dimensional dataset.")

    print("  Recommendation: " + verdict)
    print(SEP)
    print("")

    return tau_tree, rho

def run_all():
    print("=" * 65)
    print("IEEE CIS — BLOCK PARALLEL GBM EXPERIMENTS")
    print("=" * 65)

    # Load data once
    X_tr, X_val, y_tr, y_val = load_ieee(DATA_PATH, N_SUBSAMPLE)

    # Pre-flight: profile per-tree cost before committing
    tau_tree, rho = profile_tree_cost(X_tr, y_tr, n_profile=3)

    # Hard stop if overhead dominates
    if rho >= 1.0:
        print("STOPPING: rho >= 1.0. Block parallelism will not benefit ")
        print("this dataset at this subsample size. Options:")
        print("  1. Increase N_SUBSAMPLE toward 595212 (full dataset)")
        print("  2. Increase max_depth in fit_config to raise tau_tree")
        print("  3. Choose a different dataset")
        return

    if rho >= 0.10:
        print("WARNING: rho=" + str(round(rho,3)) + " above 0.1 threshold. "
              "Equal-budget result may not favour block parallelism. "
              "Continuing — result is scientifically valid either way.")
    # Experiment 1 — Ablation (4 fresh fits)
    ablation_cache = run_ablation(X_tr, y_tr, X_val, y_val)
    ablation_df    = summarise_ablation(ablation_cache)
    plot_ablation(ablation_cache)

    # Experiment 2 — Block sweep
    # B=1 and B=2 reused from ablation, B=3 and B=4 are fresh (2 new fits)
    sweep_cache = run_block_sweep(ablation_cache, X_tr, y_tr, X_val, y_val)
    sweep_df    = summarise_block_sweep(sweep_cache)
    plot_block_sweep(sweep_cache, sweep_df)

    # Experiment 3 — Equal budget
    # Baseline reused from ablation, B=2 live and B=3 live are fresh (2 new fits)
    baseline, b2_live, b3_live, budget = run_equal_budget(
        ablation_cache, X_tr, y_tr, X_val, y_val
    )
    budget_df = summarise_equal_budget(baseline, b2_live, b3_live, budget)
    plot_equal_budget(baseline, b2_live, b3_live, budget)

    # Summary
    print(f"\n{'='*65}")
    print("ALL IEEE CIS EXPERIMENTS COMPLETE")
    print(f"Outputs saved to: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("ieee") and any(
                f.endswith(ext) for ext in [".png", ".csv"]):
            print(f"  {f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_all()