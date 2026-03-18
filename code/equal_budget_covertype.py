"""
Covertype Equal-Budget Experiment
==================================
Mirrors the Santander equal-budget experiment on Covertype.

Workflow:
    1. Train baseline (B=1, col=1.0) for n_estimators=300 trees.
       Record total wall-clock time T.
    2. Train B=2 (col=0.5) with time_limit=T. No tree cap — let it
       build as many trees as it can within T seconds.
    3. Compare final val AUC at equal wall-clock budget.

Expected result (Claim 4):
    On Covertype, per-tree cost (~0.25s) is small relative to joblib
    overhead (~0.3s/block), so B=2 cannot build enough extra trees
    within budget T to compensate for the staleness AUC cost.
    Baseline should win or tie — the opposite of Santander.

Outputs:
    covertype_equal_budget.png
    covertype_equal_budget.csv
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import warnings
import os
warnings.filterwarnings("ignore")

OUT_DIR = "/kaggle/working/"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  BlockParallelGBM (verbatim)
# ─────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def compute_residuals(y, F):
    return y - sigmoid(F)

def fit_single_tree(X, residuals, max_features, max_depth, min_samples_leaf, seed):
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed
    )
    tree.fit(X, residuals)
    return tree

class BlockParallelGBM:
    def __init__(self, n_estimators=10_000, block_size=1, learning_rate=0.1,
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
                       if isinstance(self.colsample, float) else self.colsample)
            print(f"  block_size={self.block_size} | colsample={self.colsample} "
                  f"(~{n_feats} feats) | lr={self.learning_rate} "
                  f"| time_limit={self.time_limit_seconds}s")
            print("  " + "-" * 65)
        for block_idx in range(n_blocks):
            if (self.time_limit_seconds is not None and
                    time.perf_counter() - train_start > self.time_limit_seconds):
                if self.verbose:
                    print(f"  Time limit reached after block {block_idx}. Stopping.")
                break
            t0               = time.perf_counter()
            trees_this_block = min(self.block_size,
                                   self.n_estimators - block_idx * self.block_size)
            residuals = compute_residuals(y, F)
            seeds     = rng.randint(0, 2**31, size=trees_this_block)
            if trees_this_block == 1:
                results = [fit_single_tree(X, residuals, self.colsample,
                                           self.max_depth, self.min_samples_leaf,
                                           seeds[0])]
            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(fit_single_tree)(X, residuals, self.colsample,
                                            self.max_depth, self.min_samples_leaf,
                                            seeds[b])
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
            if self.verbose and (block_idx % 20 == 0 or block_idx == n_blocks - 1):
                val_str = (f" | val_auc={self.val_auc_[-1]:.5f}"
                           if self.val_auc_ else "")
                print(f"  Block {block_idx+1:>4} | "
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
#  Data
# ─────────────────────────────────────────────────────────────────

def load_covertype(n_samples=100_000, random_state=42):
    print("Loading Covertype...")
    data = fetch_covtype()
    X, y_multi = data.data.astype(np.float32), data.target
    _, X_sub, _, y_sub = train_test_split(
        X, y_multi,
        test_size=n_samples / len(y_multi),
        random_state=random_state,
        stratify=y_multi
    )
    y_bin = (y_sub == 1).astype(float)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y_bin, test_size=0.2, random_state=random_state, stratify=y_bin
    )
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | "
          f"Positive rate: {y_tr.mean():.3f}")
    return X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────────────────────────
#  Experiment
# ─────────────────────────────────────────────────────────────────

def run_equal_budget(X_tr, y_tr, X_val, y_val):
    print(f"\n{'='*65}")
    print("COVERTYPE EQUAL-BUDGET EXPERIMENT")
    print(f"{'='*65}")

    # Step 1 — baseline: fixed 300 trees, record total time
    print("\n[Step 1] Baseline (B=1, col=1.0) — 300 trees")
    baseline = BlockParallelGBM(
        n_estimators=300, block_size=1,
        learning_rate=0.1, max_depth=4, min_samples_leaf=20,
        colsample=1.0, auto_scale_lr=False, n_jobs=1,
        random_state=42, verbose=True, time_limit_seconds=None
    )
    baseline.fit(X_tr, y_tr, X_val, y_val)
    budget = baseline.total_time

    print(f"\n  → Baseline total_time = {budget:.1f}s | "
          f"trees = {len(baseline.trees_)} | "
          f"best_val_auc = {baseline.best_val_auc:.5f}")
    print(f"\n  Time budget for B=2: {budget:.1f}s")

    # Step 2 — B=2 with time limit = baseline total time, no tree cap
    print(f"\n[Step 2] B=2 (col=0.5) — time_limit={budget:.1f}s, no tree cap")
    b2 = BlockParallelGBM(
        n_estimators=10_000,       # effectively unlimited
        block_size=2,
        learning_rate=0.1, max_depth=4, min_samples_leaf=20,
        colsample=0.5, auto_scale_lr=True, n_jobs=-1,
        random_state=42, verbose=True, time_limit_seconds=budget
    )
    b2.fit(X_tr, y_tr, X_val, y_val)

    print(f"\n  → B=2 total_time = {b2.total_time:.1f}s | "
          f"trees = {len(b2.trees_)} | "
          f"best_val_auc = {b2.best_val_auc:.5f}")

    return baseline, b2, budget


# ─────────────────────────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────────────────────────

def summarise(baseline, b2, budget):
    auc_gap = baseline.best_val_auc - b2.best_val_auc
    trees_ratio = len(b2.trees_) / len(baseline.trees_)

    print(f"\n{'='*65}")
    print("EQUAL-BUDGET SUMMARY — COVERTYPE")
    print(f"{'='*65}")

    rows = [
        {
            "Config":        "Baseline (B=1, col=1.0)",
            "Budget (s)":    round(budget, 1),
            "Time used (s)": round(baseline.total_time, 1),
            "Trees built":   len(baseline.trees_),
            "Best val AUC":  round(baseline.best_val_auc, 5),
            "AUC gap":       0.0,
        },
        {
            "Config":        "Block B=2, col=0.5",
            "Budget (s)":    round(budget, 1),
            "Time used (s)": round(b2.total_time, 1),
            "Trees built":   len(b2.trees_),
            "Best val AUC":  round(b2.best_val_auc, 5),
            "AUC gap":       round(auc_gap, 5),
        },
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n  Trees built by B=2 vs baseline: "
          f"{len(b2.trees_)} vs {len(baseline.trees_)} "
          f"({trees_ratio:.2f}x)")
    print(f"  AUC gap (baseline − b2): {auc_gap:+.5f} "
          f"({'baseline wins' if auc_gap > 0.001 else 'b2 wins' if auc_gap < -0.001 else 'effectively tied'})")

    df.to_csv(f"{OUT_DIR}covertype_equal_budget.csv", index=False)
    print(f"\nSaved: covertype_equal_budget.csv")
    return df


# ─────────────────────────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────────────────────────

def plot_equal_budget(baseline, b2, budget):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(baseline.cumulative_times_, baseline.val_auc_,
            color="blue", linewidth=2, label="Baseline (B=1, col=1.0)")
    ax.plot(b2.cumulative_times_, b2.val_auc_,
            color="red", linewidth=2, linestyle="--",
            label=f"Block B=2, col=0.5 ({len(b2.trees_)} trees)")

    ax.axvline(x=budget, color="gray", linewidth=1.5, linestyle=":",
               label=f"Equal budget = {budget:.0f}s")

    # Mark final AUC of each at budget
    ax.scatter([baseline.cumulative_times_[-1]], [baseline.val_auc_[-1]],
               color="blue", zorder=5, s=80)
    ax.scatter([b2.cumulative_times_[-1]], [b2.val_auc_[-1]],
               color="red", zorder=5, s=80)

    ax.annotate(f"Baseline: {baseline.best_val_auc:.4f}",
                xy=(baseline.cumulative_times_[-1], baseline.val_auc_[-1]),
                xytext=(-90, 10), textcoords="offset points",
                fontsize=10, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1))
    ax.annotate(f"B=2: {b2.best_val_auc:.4f}",
                xy=(b2.cumulative_times_[-1], b2.val_auc_[-1]),
                xytext=(10, -20), textcoords="offset points",
                fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))

    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Equal-Budget Comparison — Covertype\n"
                 "B=2 builds more trees but per-tree cost is too low "
                 "for parallelism to overcome overhead",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{OUT_DIR}covertype_equal_budget.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val = load_covertype(n_samples=100_000)
    baseline, b2, budget = run_equal_budget(X_tr, y_tr, X_val, y_val)
    df = summarise(baseline, b2, budget)
    plot_equal_budget(baseline, b2, budget)

    print("\nDone. Output files:")
    for f in ["covertype_equal_budget.png", "covertype_equal_budget.csv"]:
        print(f"  {OUT_DIR}{f}")