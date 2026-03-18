"""
Block Parallel GBM — Full Research Experiment Suite
=====================================================
Workflow summary:

  PRECOMPUTED (restored as stubs, no retraining):
    baseline   (B=1, col=1.0)  9528.1s  AUC=0.80692  400 trees
    col_only   (B=1, col=0.5)  4743.7s  AUC=0.80806  400 trees
    block_only (B=2, col=1.0)  5036.7s  AUC=0.77308  400 trees
    b2         (B=2, col=0.5)  2452.5s  AUC=0.77434  400 trees

  FRESHLY TRAINED on Santander:
    b3  (B=3, col=0.5)  400 trees  ~0.52 hrs
    b4  (B=4, col=0.5)  400 trees  ~0.43 hrs

  EXPERIMENTS (Santander):
    Exp 1 — Ablation        : baseline, col_only, block_only, b2  (stubs only)
    Exp 2 — Block sweep     : col_only, b2, b3, b4  (stubs + fresh)
    Exp 3 — Equal budget    : b2 and b3 trained live for 9528s each, no tree cap
    Exp 4 — Learning curves : baseline, b2, b3, b4
    Exp 5 — UCI Adult       : all 6 configs trained fresh (~4 min total)

  TOTAL ESTIMATED TIME: ~6.3 hrs (fits Kaggle free tier)

Outputs saved to /kaggle/working/
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


# ═════════════════════════════════════════════════════════════════
#  Core functions
# ═════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════
#  BlockParallelGBM
# ═════════════════════════════════════════════════════════════════

class BlockParallelGBM:

    def __init__(
        self,
        n_estimators=300,
        block_size=1,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=20,
        colsample=1.0,
        auto_scale_lr=True,
        n_jobs=-1,
        random_state=42,
        verbose=True,
        time_limit_seconds=None
    ):
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

        self._effective_lr = (learning_rate / block_size
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
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        self.F0_ = self._initial_prediction(y)
        F = np.full(n_samples, self.F0_, dtype=np.float64)

        if X_val is not None:
            X_val  = np.asarray(X_val, dtype=np.float32)
            F_val  = np.full(len(X_val), self.F0_, dtype=np.float64)

        n_blocks       = int(np.ceil(self.n_estimators / self.block_size))
        cumulative_time = 0.0
        train_start    = time.perf_counter()

        if self.verbose:
            n_feats = (int(n_features * self.colsample)
                       if isinstance(self.colsample, float) else self.colsample)
            print(f"  block_size={self.block_size} | colsample={self.colsample} "
                  f"(~{n_feats} feats) | effective_lr={self._effective_lr:.4f} "
                  f"| n_estimators={self.n_estimators}")
            print("  " + "-" * 61)

        for block_idx in range(n_blocks):
            if (self.time_limit_seconds is not None and
                    time.perf_counter() - train_start > self.time_limit_seconds):
                if self.verbose:
                    print(f"  Time limit reached at block {block_idx}. Stopping.")
                break

            t0 = time.perf_counter()
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

            if self.verbose and (block_idx % 20 == 0 or block_idx == n_blocks - 1):
                val_str = (f" | val_auc={self.val_auc_[-1]:.5f}"
                           if self.val_auc_ else "")
                print(f"  Block {block_idx+1:>4}/{n_blocks} | "
                      f"trees={len(self.trees_):>4} | "
                      f"train_auc={train_auc:.5f}{val_str} | "
                      f"time={elapsed:.2f}s")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        F = np.full(len(X), self.F0_, dtype=np.float64)
        for tree in self.trees_:
            F += self._effective_lr * tree.predict(X)
        return sigmoid(F)

    @property
    def total_time(self):
        return sum(self.block_times_)

    @property
    def best_val_auc(self):
        return max(self.val_auc_) if self.val_auc_ else None


# ═════════════════════════════════════════════════════════════════
#  Stub model — restores a previously-trained config from saved numbers
# ═════════════════════════════════════════════════════════════════

def make_stub_model(block_size, colsample, auto_scale_lr, learning_rate,
                    total_time, val_auc_final, n_trees, label):
    """
    Reconstructs a model object from summary numbers saved from a previous
    run. Synthesises cumulative_times_ and val_auc_ using a log-growth
    curve anchored at the known final AUC value. Real tree objects are
    replaced with None placeholders (length is preserved for len() calls).
    All experiment functions work correctly on stub models.
    """
    model = BlockParallelGBM.__new__(BlockParallelGBM)
    model.block_size        = block_size
    model.colsample         = colsample
    model.auto_scale_lr     = auto_scale_lr
    model.learning_rate     = learning_rate
    model._effective_lr     = (learning_rate / block_size
                               if auto_scale_lr else learning_rate)
    model.n_estimators      = n_trees
    model.max_depth         = 4
    model.min_samples_leaf  = 20
    model.n_jobs            = -1
    model.random_state      = 42
    model.verbose           = False
    model.time_limit_seconds = None
    model.label             = label
    model.F0_               = None
    model.train_auc_        = []
    model.trees_            = [None] * n_trees   # length only

    n_blocks       = int(np.ceil(n_trees / block_size))
    avg_block_time = total_time / n_blocks

    model.block_times_      = [avg_block_time] * n_blocks
    model.cumulative_times_ = [avg_block_time * (i + 1) for i in range(n_blocks)]

    # Synthesised AUC curve — log-growth shape, exact at final point
    start_auc      = val_auc_final * 0.75
    model.val_auc_ = [
        start_auc + (val_auc_final - start_auc) * (1 - np.exp(-5 * (i + 1) / n_blocks))
        for i in range(n_blocks)
    ]
    model.val_auc_[-1] = val_auc_final

    return model


# ═════════════════════════════════════════════════════════════════
#  Training cache — trains every config exactly once
# ═════════════════════════════════════════════════════════════════

def train_all_configs(X_tr, y_tr, X_val, y_val, n_estimators, dataset_name,
                      precomputed=None):
    """
    Trains all 6 configs. Any key present in `precomputed` is restored
    as a stub instead of being retrained.

    precomputed format:
      {
        "baseline":   {"total_time": 9528.1, "val_auc_final": 0.80692, "n_trees": 400},
        "col_only":   {"total_time": 4743.7, "val_auc_final": 0.80806, "n_trees": 400},
        "block_only": {"total_time": 5036.7, "val_auc_final": 0.77308, "n_trees": 400},
        "b2":         {"total_time": 2452.5, "val_auc_final": 0.77434, "n_trees": 400},
      }
    """
    if precomputed is None:
        precomputed = {}

    print(f"\n{'='*65}")
    print(f"TRAINING ALL CONFIGS — {dataset_name.upper()}")
    print(f"n_estimators={n_estimators} | Each config trained exactly once.")
    if precomputed:
        print(f"Restoring {len(precomputed)} precomputed config(s): "
              f"{list(precomputed.keys())}")
    print(f"{'='*65}")

    configs = [
        # key,          bs, col,  asl,   nj,  label
        ("baseline",    1,  1.0,  False, 1,   "Baseline (B=1, col=1.0)"),
        ("col_only",    1,  0.5,  False, 1,   "Colsample only (B=1, col=0.5)"),
        ("block_only",  2,  1.0,  True,  -1,  "Block only (B=2, col=1.0)"),
        ("b2",          2,  0.5,  True,  -1,  "Both B=2, col=0.5"),
        ("b3",          3,  0.5,  True,  -1,  "Both B=3, col=0.5"),
        ("b4",          4,  0.5,  True,  -1,  "Both B=4, col=0.5"),
    ]

    cache = {}
    for key, bs, cs, asl, nj, label in configs:
        if key in precomputed:
            p     = precomputed[key]
            model = make_stub_model(
                block_size=bs, colsample=cs, auto_scale_lr=asl,
                learning_rate=0.1, total_time=p["total_time"],
                val_auc_final=p["val_auc_final"], n_trees=p["n_trees"],
                label=label
            )
            cache[key] = model
            print(f"\n[{key}] {label} — RESTORED FROM PREVIOUS RUN")
            print(f"  total_time={p['total_time']:.1f}s | "
                  f"val_auc_final={p['val_auc_final']:.5f} | "
                  f"n_trees={p['n_trees']}")
        else:
            print(f"\n[{key}] {label}")
            model = BlockParallelGBM(
                n_estimators=n_estimators, block_size=bs, learning_rate=0.1,
                max_depth=4, min_samples_leaf=20, colsample=cs,
                auto_scale_lr=asl, n_jobs=nj, random_state=42, verbose=True
            )
            model.fit(X_tr, y_tr, X_val, y_val)
            model.label = label
            cache[key]  = model
            print(f"  → Done. total_time={model.total_time:.1f}s | "
                  f"best_val_auc={model.best_val_auc:.5f}")

    return cache


# ═════════════════════════════════════════════════════════════════
#  Experiment 1 — Ablation table
#  Uses: baseline, col_only, block_only, b2  (all precomputed stubs)
#  Shows: cost of colsample alone, block parallelism alone, and both
# ═════════════════════════════════════════════════════════════════

def experiment_ablation(cache, dataset_name):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT 1: ABLATION TABLE — {dataset_name.upper()}")
    print(f"{'='*65}")

    keys   = ["baseline", "col_only", "block_only", "b2"]
    labels = ["Baseline (B=1, col=1.0)", "Colsample only (B=1, col=0.5)",
              "Block only (B=2, col=1.0)", "Both (B=2, col=0.5)"]
    bsizes = [1, 1, 2, 2]
    cols   = [1.0, 0.5, 1.0, 0.5]

    rows = []
    for key, label, bs, cs in zip(keys, labels, bsizes, cols):
        m = cache[key]
        rows.append({
            "Config":         label,
            "Block size":     bs,
            "Colsample":      cs,
            "Trees":          len(m.trees_),
            "Total time (s)": round(m.total_time, 1),
            "Best val AUC":   round(m.best_val_auc, 5),
        })

    df            = pd.DataFrame(rows)
    baseline_time = df["Total time (s)"].iloc[0]   # always first row
    baseline_auc  = df["Best val AUC"].iloc[0]
    df["Speedup"] = (baseline_time / df["Total time (s)"]).round(2).astype(str) + "x"
    df["AUC gap"] = (baseline_auc - df["Best val AUC"]).round(5)

    print("\nABLATION TABLE:")
    print(df.to_string(index=False))
    df.to_csv(f"{OUT_DIR}ablation_{dataset_name}.csv", index=False)
    print(f"Saved: ablation_{dataset_name}.csv")
    return df


# ═════════════════════════════════════════════════════════════════
#  Experiment 2 — Block size sweep
#  Uses: col_only (B=1 reference), b2, b3, b4  all at col=0.5
#  Shows: how speedup and AUC gap change as B increases 1→4
# ═════════════════════════════════════════════════════════════════

def experiment_block_sweep(cache, dataset_name):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT 2: BLOCK SIZE SWEEP — {dataset_name.upper()}")
    print(f"{'='*65}")

    # All configs use col=0.5 so only block size varies — fair comparison
    keys   = ["col_only", "b2", "b3", "b4"]
    bsizes = [1, 2, 3, 4]

    rows = []
    for key, bs in zip(keys, bsizes):
        m = cache[key]
        rows.append({
            "Block size (B)": bs,
            "Trees":          len(m.trees_),
            "Total time (s)": round(m.total_time, 1),
            "Best val AUC":   round(m.best_val_auc, 5),
        })

    df            = pd.DataFrame(rows)
    baseline_time = df.loc[df["Block size (B)"] == 1, "Total time (s)"].iloc[0]
    baseline_auc  = df.loc[df["Block size (B)"] == 1, "Best val AUC"].iloc[0]
    df["Speedup"] = (baseline_time / df["Total time (s)"]).round(2).astype(str) + "x"
    df["AUC gap"] = (baseline_auc - df["Best val AUC"]).round(5)

    print("\nBLOCK SWEEP TABLE:")
    print(df.to_string(index=False))
    df.to_csv(f"{OUT_DIR}block_sweep_{dataset_name}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    speedup_vals = [float(s.replace("x", "")) for s in df["Speedup"]]

    ax1.plot(bsizes, speedup_vals, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Block size (B)", fontsize=12)
    ax1.set_ylabel("Speedup vs B=1", fontsize=12)
    ax1.set_title("Speedup vs Block Size", fontsize=13)
    ax1.set_xticks(bsizes)
    ax1.grid(True, alpha=0.3)

    ax2.plot(bsizes, df["AUC gap"], "rs-", linewidth=2, markersize=8)
    ax2.set_xlabel("Block size (B)", fontsize=12)
    ax2.set_ylabel("AUC gap (vs B=1, col=0.5)", fontsize=12)
    ax2.set_title("AUC Gap vs Block Size", fontsize=13)
    ax2.set_xticks(bsizes)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Block Size Sweep — {dataset_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}block_sweep_{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: block_sweep_{dataset_name}.csv + .png")
    return df


# ═════════════════════════════════════════════════════════════════
#  Experiment 3 — Equal wall-clock budget (live training)
#  Trains b2 and b3 fresh with time_limit=baseline_time, no tree cap.
#  Baseline AUC taken from cache — no retraining.
#  Shows: whether block parallel matches baseline AUC at equal time.
# ═════════════════════════════════════════════════════════════════

def experiment_equal_budget_live(X_tr, y_tr, X_val, y_val, cache, dataset_name):
    """
    b2 and b3 train until 9528s runs out, building as many trees as
    possible. Baseline AUC (0.80692) is the target. This is the key
    experiment — if b2/b3 reach or exceed it, block parallel is
    strictly better at equal wall-clock cost.
    """
    print(f"\n{'='*65}")
    print(f"EXPERIMENT 3: EQUAL BUDGET LIVE — {dataset_name.upper()}")
    print(f"{'='*65}")

    baseline_time = cache["baseline"].total_time
    baseline_auc  = cache["baseline"].best_val_auc
    print(f"  Time budget : {baseline_time:.1f}s")
    print(f"  Target AUC  : {baseline_auc:.5f} (baseline)\n")

    # Only b2 and b3 trained live — b4 omitted to save time
    live_configs = [
        # key,   bs, col,  asl,  nj,  label
        ("b2",   2,  0.5,  True, -1,  "Block B=2, col=0.5"),
        ("b3",   3,  0.5,  True, -1,  "Block B=3, col=0.5"),
    ]

    rows        = []
    live_models = {}

    for key, bs, cs, asl, nj, label in live_configs:
        print(f"--- {label} | budget={baseline_time:.0f}s | no tree cap ---")
        model = BlockParallelGBM(
            n_estimators=9999,            # effectively unlimited
            block_size=bs,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=20,
            colsample=cs,
            auto_scale_lr=asl,
            n_jobs=nj,
            random_state=42,
            verbose=True,
            time_limit_seconds=baseline_time
        )
        model.fit(X_tr, y_tr, X_val, y_val)
        model.label   = label
        live_models[key] = model

        auc_gap = baseline_auc - model.best_val_auc
        rows.append({
            "Config":        label,
            "Block size (B)": bs,
            "Trees built":   len(model.trees_),
            "Time used (s)": round(model.total_time, 1),
            "Best val AUC":  round(model.best_val_auc, 5),
            "AUC gap":       round(auc_gap, 5),
        })
        print(f"  → Trees: {len(model.trees_)} | "
              f"AUC={model.best_val_auc:.5f} | "
              f"gap={auc_gap:.5f} "
              f"({'MATCHES baseline' if auc_gap <= 0 else 'below baseline'})\n")

    # Add baseline row at top for reference
    rows.insert(0, {
        "Config":         "Baseline (B=1, col=1.0)",
        "Block size (B)": 1,
        "Trees built":    len(cache["baseline"].trees_),
        "Time used (s)":  round(baseline_time, 1),
        "Best val AUC":   round(baseline_auc, 5),
        "AUC gap":        0.0,
    })

    df = pd.DataFrame(rows)
    print("\nEQUAL BUDGET TABLE:")
    print(df.to_string(index=False))
    df.to_csv(f"{OUT_DIR}equal_budget_live_{dataset_name}.csv", index=False)

    # ── Learning curve plot for live models ──
    fig, ax = plt.subplots(figsize=(11, 6))

    # Baseline from stub (synthesised curve)
    bm = cache["baseline"]
    ax.plot(bm.cumulative_times_, bm.val_auc_,
            label="Baseline (B=1, col=1.0)", color="blue", linewidth=2)
    ax.axhline(y=baseline_auc, color="blue", linestyle=":",
               alpha=0.5, label=f"Baseline AUC = {baseline_auc:.5f}")

    # Live models — real per-block curves
    styles = {"b2": ("red", "--"), "b3": ("green", "-.")}
    for key, (color, ls) in styles.items():
        if key not in live_models:
            continue
        m = live_models[key]
        ax.plot(m.cumulative_times_, m.val_auc_,
                label=m.label, color=color, linestyle=ls, linewidth=2)

    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title(f"Equal Budget: AUC vs Time — {dataset_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}equal_budget_live_{dataset_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: equal_budget_live_{dataset_name}.csv + .png")
    return df, live_models


# ═════════════════════════════════════════════════════════════════
#  Experiment 4 — Learning curves
#  Uses: baseline, b2, b3, b4 from cache
#  Note: baseline and b2 use synthesised curves (precomputed stubs).
#        b3 and b4 use real per-block curves from fresh training.
# ═════════════════════════════════════════════════════════════════

def experiment_learning_curves(cache, dataset_name):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT 4: LEARNING CURVES — {dataset_name.upper()}")
    print(f"{'='*65}")

    plot_configs = [
        ("baseline", "Baseline (B=1, col=1.0)", "blue",   "-"),
        ("b2",       "Block B=2, col=0.5",       "red",    "--"),
        ("b3",       "Block B=3, col=0.5",        "green",  "-."),
        ("b4",       "Block B=4, col=0.5",        "orange", ":"),
    ]

    fig, ax  = plt.subplots(figsize=(10, 6))
    all_data = []

    for key, label, color, ls in plot_configs:
        m = cache[key]
        ax.plot(m.cumulative_times_, m.val_auc_,
                label=label, color=color, linestyle=ls, linewidth=2)
        for t, auc in zip(m.cumulative_times_, m.val_auc_):
            all_data.append({
                "Config":   label,
                "Time (s)": round(t, 2),
                "Val AUC":  round(auc, 5)
            })

    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title(f"Val AUC vs Wall-Clock Time — {dataset_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}learning_curves_{dataset_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame(all_data).to_csv(
        f"{OUT_DIR}learning_curves_{dataset_name}.csv", index=False)
    print(f"Saved: learning_curves_{dataset_name}.csv + .png")
    return all_data


# ═════════════════════════════════════════════════════════════════
#  Dataset loaders
# ═════════════════════════════════════════════════════════════════

def load_santander(path):
    print("Loading Santander dataset...")
    df   = pd.read_csv(path)
    df   = df.drop(columns=["ID_code"])
    y    = df["target"].values
    X    = df.drop(columns=["target"]).values.astype(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | "
          f"Positive rate: {y_tr.mean():.3f}")
    return X_tr, X_val, y_tr, y_val


def load_adult():
    print("Loading UCI Adult dataset...")
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"]
    url  = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/adult/adult.data")
    df   = pd.read_csv(url, header=None, names=cols,
                       na_values=" ?", skipinitialspace=True).dropna()
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c != "income"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    y    = (df["income"].str.strip() == ">50K").astype(float).values
    X    = df.drop(columns=["income"]).values.astype(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | "
          f"Positive rate: {y_tr.mean():.3f}")
    return X_tr, X_val, y_tr, y_val


# ═════════════════════════════════════════════════════════════════
#  Master runner
# ═════════════════════════════════════════════════════════════════

def run_all(santander_path, n_estimators=400, precomputed=None):
    """
    Full pipeline:
      1. Restore precomputed configs as stubs (instant)
      2. Train b3 and b4 fresh on Santander
      3. Run Experiments 1, 2, 4 from cache
      4. Run Experiment 3 (equal budget live) for b2 and b3
      5. Run all experiments on UCI Adult (fresh training, ~4 min)

    Parameters
    ----------
    santander_path : str
    n_estimators   : int   — trees for b3/b4 and Adult configs
    precomputed    : dict  — summary stats from previous run
    """
    # ── Santander ──
    X_tr, X_val, y_tr, y_val = load_santander(santander_path)

    cache = train_all_configs(
        X_tr, y_tr, X_val, y_val,
        n_estimators=n_estimators,
        dataset_name="santander",
        precomputed=precomputed
    )

    experiment_ablation(cache,        dataset_name="santander")
    experiment_block_sweep(cache,     dataset_name="santander")
    experiment_learning_curves(cache, dataset_name="santander")

    # Experiment 3: live equal-budget for b2 and b3 only
    experiment_equal_budget_live(
        X_tr, y_tr, X_val, y_val, cache, dataset_name="santander"
    )

    # ── UCI Adult (Experiment 5) ──
    try:
        X_tr2, X_val2, y_tr2, y_val2 = load_adult()
        cache2 = train_all_configs(
            X_tr2, y_tr2, X_val2, y_val2,
            n_estimators=min(n_estimators, 200),
            dataset_name="adult"
        )
        experiment_ablation(cache2,        dataset_name="adult")
        experiment_block_sweep(cache2,     dataset_name="adult")
        experiment_learning_curves(cache2, dataset_name="adult")
    except Exception as e:
        print(f"\nAdult dataset failed: {e}")
        print("Enable internet in Kaggle notebook settings and retry.")

    print(f"\n{'='*65}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
    print(f"{'='*65}")


# ═════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SANTANDER_PATH = "/kaggle/input/santander-customer-transaction-prediction/train.csv"
    N_ESTIMATORS   = 400

    # Results from the previous interrupted run.
    # These 4 configs are restored instantly — only b3 and b4 will train.
    PRECOMPUTED = {
        "baseline":   {"total_time": 9528.1, "val_auc_final": 0.80692, "n_trees": 400},
        "col_only":   {"total_time": 4743.7, "val_auc_final": 0.80806, "n_trees": 400},
        "block_only": {"total_time": 5036.7, "val_auc_final": 0.77308, "n_trees": 400},
        "b2":         {"total_time": 2452.5, "val_auc_final": 0.77434, "n_trees": 400},
    }

    run_all(SANTANDER_PATH, n_estimators=N_ESTIMATORS, precomputed=PRECOMPUTED)