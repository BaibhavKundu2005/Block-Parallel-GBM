"""
Covertype Ablation
==================
Mirrors the Santander four-config ablation on Covertype:

    baseline   (B=1, col=1.0)  — sequential GBM, no subsampling
    col_only   (B=1, col=0.5)  — colsample only, no block parallelism
    block_only (B=2, col=1.0)  — block parallelism only, no colsample
    combined   (B=2, col=0.5)  — both together

This lets us decompose the AUC gap and speedup into:
    - pure colsample effect       : baseline → col_only
    - pure block parallelism effect: baseline → block_only
    - interaction / combined effect: baseline → combined

Outputs:
    covertype_ablation.png
    covertype_ablation.csv

Runtime: ~10-15 min (4 fits x 300 trees on 80k train rows)
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
    def __init__(self, n_estimators=300, block_size=1, learning_rate=0.1,
                 max_depth=4, min_samples_leaf=20, colsample=1.0,
                 auto_scale_lr=True, n_jobs=-1, random_state=42,
                 verbose=True):
        self.n_estimators     = n_estimators
        self.block_size       = block_size
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.colsample        = colsample
        self.auto_scale_lr    = auto_scale_lr
        self.n_jobs           = n_jobs
        self.random_state     = random_state
        self.verbose          = verbose
        self._effective_lr    = (learning_rate / block_size
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
        if self.verbose:
            n_feats = (int(n_features * self.colsample)
                       if isinstance(self.colsample, float) else self.colsample)
            print(f"  block_size={self.block_size} | colsample={self.colsample} "
                  f"(~{n_feats} feats) | lr={self.learning_rate} "
                  f"| max_depth={self.max_depth} | n_estimators={self.n_estimators}")
            print("  " + "-" * 65)
        for block_idx in range(n_blocks):
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
                print(f"  Block {block_idx+1:>4}/{n_blocks} | "
                      f"trees={len(self.trees_):>4} | "
                      f"train_auc={train_auc:.5f}{val_str} | "
                      f"time={elapsed:.2f}s")
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
#  Ablation
# ─────────────────────────────────────────────────────────────────

def run_ablation(X_tr, y_tr, X_val, y_val, n_estimators=300):
    """
    Four configs mirroring the Santander ablation exactly.
    Fixed hyperparameters: lr=0.1, max_depth=4, min_samples_leaf=20.
    auto_scale_lr=False for B=1 configs (no scaling needed).
    auto_scale_lr=True  for B=2 configs (divides lr by 2).
    """
    print(f"\n{'='*65}")
    print("COVERTYPE ABLATION")
    print(f"n_estimators={n_estimators} | lr=0.1 | max_depth=4")
    print(f"{'='*65}")

    configs = [
        # label,        bs, col,  asl,   nj
        ("baseline",   1,  1.0,  False, 1),
        ("col_only",   1,  0.5,  False, 1),
        ("block_only", 2,  1.0,  True,  -1),
        ("combined",   2,  0.5,  True,  -1),
    ]

    models = {}
    for label, bs, cs, asl, nj in configs:
        print(f"\n[{label}]")
        m = BlockParallelGBM(
            n_estimators=n_estimators,
            block_size=bs,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=20,
            colsample=cs,
            auto_scale_lr=asl,
            n_jobs=nj,
            random_state=42,
            verbose=True
        )
        m.fit(X_tr, y_tr, X_val, y_val)
        m.label = label
        models[label] = m
        print(f"  → best_val_auc={m.best_val_auc:.5f} | "
              f"total_time={m.total_time:.1f}s")

    return models


# ─────────────────────────────────────────────────────────────────
#  Results table
# ─────────────────────────────────────────────────────────────────

def build_table(models):
    ref_time = models["baseline"].total_time
    ref_auc  = models["baseline"].best_val_auc

    rows = []
    for label, m in models.items():
        rows.append({
            "Config":         label,
            "Block size (B)": m.block_size,
            "Colsample":      m.colsample,
            "Trees":          len(m.trees_),
            "Total time (s)": round(m.total_time, 1),
            "Best val AUC":   round(m.best_val_auc, 5),
            "Speedup":        f"{ref_time / m.total_time:.2f}x",
            "AUC gap":        round(ref_auc - m.best_val_auc, 5),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*65}")
    print("ABLATION TABLE — COVERTYPE")
    print(df.to_string(index=False))
    print(f"{'='*65}")

    df.to_csv(f"{OUT_DIR}covertype_ablation.csv", index=False)
    print(f"Saved: covertype_ablation.csv")
    return df


# ─────────────────────────────────────────────────────────────────
#  Plot — two panels: AUC vs trees, AUC vs time
# ─────────────────────────────────────────────────────────────────

def plot_ablation(models, df):
    """
    Left panel:  val AUC vs number of trees (equal tree count view)
    Right panel: val AUC vs cumulative wall-clock time (equal budget view)

    Four lines: baseline, col_only, block_only, combined.
    Each logged per block, so x-axis points = number of blocks completed.
    For B=2 configs, trees = blocks * 2, so x-axis is already in trees.
    """
    colors  = {
        "baseline":   "blue",
        "col_only":   "green",
        "block_only": "red",
        "combined":   "purple",
    }
    lstyles = {
        "baseline":   "-",
        "col_only":   "--",
        "block_only": "-.",
        "combined":   ":",
    }
    display_labels = {
        "baseline":   "Baseline (B=1, col=1.0)",
        "col_only":   "Col only (B=1, col=0.5)",
        "block_only": "Block only (B=2, col=1.0)",
        "combined":   "Combined (B=2, col=0.5)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, m in models.items():
        # x-axis for trees: cumulative tree count per block
        tree_counts = []
        count = 0
        for i in range(len(m.block_times_)):
            count += m.block_size
            tree_counts.append(min(count, m.n_estimators))

        ax1.plot(tree_counts, m.val_auc_,
                 color=colors[label], linestyle=lstyles[label],
                 linewidth=2, label=display_labels[label])

        ax2.plot(m.cumulative_times_, m.val_auc_,
                 color=colors[label], linestyle=lstyles[label],
                 linewidth=2, label=display_labels[label])

    ax1.set_xlabel("Number of trees", fontsize=12)
    ax1.set_ylabel("Validation AUC", fontsize=12)
    ax1.set_title("Val AUC vs Tree Count\nCovertype Ablation", fontsize=12,
                  fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax2.set_ylabel("Validation AUC", fontsize=12)
    ax2.set_title("Val AUC vs Wall-Clock Time\nCovertype Ablation", fontsize=12,
                  fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Annotate final AUC values on time plot for readability
    for label, m in models.items():
        ax2.annotate(
            f"{m.best_val_auc:.4f}",
            xy=(m.cumulative_times_[-1], m.val_auc_[-1]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=8, color=colors[label]
        )

    plt.tight_layout()
    p = f"{OUT_DIR}covertype_ablation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val = load_covertype(n_samples=100_000)
    models = run_ablation(X_tr, y_tr, X_val, y_val, n_estimators=300)
    df     = build_table(models)
    plot_ablation(models, df)

    print("\nDone. Output files:")
    for f in ["covertype_ablation.png", "covertype_ablation.csv"]:
        print(f"  {OUT_DIR}{f}")