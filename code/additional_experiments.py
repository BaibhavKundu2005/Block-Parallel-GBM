"""
Additional Experiments
======================
Part 1 — XGBoost / LightGBM comparison on Santander
    Fits XGBoost and LightGBM with comparable settings, records per-round
    AUC and cumulative time, and plots alongside the precomputed Santander
    baseline and b2_live results from logs.

    Outputs:
        xgb_lgbm_vs_block_trees.png   — AUC vs tree count
        xgb_lgbm_vs_block_time.png    — AUC vs wall-clock time

Part 2 — Covertype hyperparameter sensitivity
    Loads Covertype via sklearn, subsamples 100k rows stratified,
    binarises to class 1 vs rest. Trains baseline and b2 across a
    3 x 3 grid of (learning_rate x max_depth). Reports AUC gap and
    speedup at equal tree counts. Also runs block sweep B=1,2,3,4
    at the default hyperparameter setting.

    Outputs:
        covertype_hparam_auc_gap_heatmap.png
        covertype_hparam_speedup_heatmap.png
        covertype_block_sweep.png
        covertype_learning_curves.png
        covertype_hparam_results.csv
        covertype_block_sweep.csv

Workflow:
    1. Load Santander data (for XGB/LGB comparison)
    2. Fit XGBoost and LightGBM with eval callbacks
    3. Plot Part 1 figures using log data + freshly timed XGB/LGB
    4. Load and prepare Covertype
    5. Run hyperparameter grid (9 pairs x 2 models = 18 fits)
    6. Run block sweep at default setting (4 fits)
    7. Plot all Part 2 figures

Runtime estimate:
    Part 1: ~5-10 min (XGBoost + LightGBM on Santander)
    Part 2: ~20-40 min (Covertype is fast at 100k x 54 features)
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
import seaborn as sns
import time
import warnings
import os
warnings.filterwarnings("ignore")

OUT_DIR = "/kaggle/working/"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
#  BlockParallelGBM (copied verbatim from main experiments file)
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
                  f"| max_depth={self.max_depth} | n_estimators={self.n_estimators}")
            print("  " + "-" * 65)
        for block_idx in range(n_blocks):
            if (self.time_limit_seconds is not None and
                    time.perf_counter() - train_start > self.time_limit_seconds):
                if self.verbose:
                    print(f"  Time limit reached at block {block_idx}. Stopping.")
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
#  PART 1 — XGBoost / LightGBM comparison on Santander
# ═════════════════════════════════════════════════════════════════

# Santander results from logs — precomputed, no retraining needed
# baseline: 400 trees, logged every 20 blocks
SANT_BASELINE_TREES = [1,21,41,61,81,101,121,141,161,181,201,
                        221,241,261,281,301,321,341,361,381,400]
SANT_BASELINE_AUCS  = [0.60457,0.66146,0.68086,0.70135,0.71587,0.73117,
                        0.74291,0.75279,0.76090,0.76780,0.77329,0.77800,
                        0.78219,0.78651,0.78990,0.79346,0.79670,0.79949,
                        0.80212,0.80467,0.80692]
SANT_BASELINE_TIME  = 9528.1

# b2_live: logged every 20 blocks, 862 blocks total
SANT_B2_LIVE_BLOCKS = [1,21,41,61,81,101,121,141,161,181,201,221,241,
                        261,281,301,321,341,361,381,401,421,441,461,481,
                        501,521,541,561,581,601,621,641,661,681,701,721,
                        741,761,781,801,821,841,861]
SANT_B2_LIVE_AUCS   = [0.62788,0.67587,0.68558,0.70198,0.71708,0.73175,
                        0.74378,0.75404,0.76204,0.76943,0.77457,0.77947,
                        0.78362,0.78763,0.79122,0.79431,0.79771,0.80049,
                        0.80329,0.80576,0.80824,0.81040,0.81254,0.81455,
                        0.81635,0.81845,0.82017,0.82187,0.82342,0.82487,
                        0.82635,0.82767,0.82914,0.83037,0.83163,0.83276,
                        0.83403,0.83512,0.83621,0.83715,0.83819,0.83920,
                        0.84016,0.84105]
SANT_B2_LIVE_TREES  = [b * 2 for b in SANT_B2_LIVE_BLOCKS]
SANT_B2_LIVE_TIME   = 9501.3


def fit_xgb_lgbm_santander(santander_path):
    """
    Fit XGBoost and LightGBM on Santander with matched hyperparameters.
    Records per-round val AUC and cumulative wall-clock time using
    custom callbacks.
    Returns dicts with keys: val_auc, trees, cumulative_times, total_time
    """
    import xgboost as xgb
    import lightgbm as lgb

    print("=" * 65)
    print("PART 1: XGBoost / LightGBM — SANTANDER")
    print("=" * 65)

    print("\nLoading Santander...")
    df  = pd.read_csv(santander_path)
    y   = df["target"].values
    X   = df.drop(columns=["ID_code", "target"]).values.astype(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape}")

    N_ROUNDS = 400
    LR       = 0.1
    DEPTH    = 4

    # ── XGBoost ──
    print(f"\n--- XGBoost | n_rounds={N_ROUNDS} | lr={LR} | max_depth={DEPTH} ---")
    xgb_aucs  = []
    xgb_times = []
    xgb_start = time.perf_counter()
    dval_xgb  = xgb.DMatrix(X_val)

    class XGBTimedCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            t   = time.perf_counter() - xgb_start
            auc = roc_auc_score(y_val,
                                model.predict(dval_xgb,
                                              iteration_range=(0, epoch + 1)))
            xgb_aucs.append(auc)
            xgb_times.append(t)
            if epoch % 20 == 0 or epoch == N_ROUNDS - 1:
                print(f"  Round {epoch+1:>4}/{N_ROUNDS} | "
                      f"val_auc={auc:.5f} | time={t:.1f}s")
            return False   # False = do not stop early

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    xgb.train(
        params={
            "objective":        "binary:logistic",
            "eta":              LR,
            "max_depth":        DEPTH,
            "subsample":        1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 20,
            "nthread":          -1,
            "seed":             42,
            "verbosity":        0,
        },
        dtrain=dtrain,
        num_boost_round=N_ROUNDS,
        callbacks=[XGBTimedCallback()]
    )
    xgb_total = time.perf_counter() - xgb_start
    print(f"  → Done. total_time={xgb_total:.1f}s | "
          f"best_val_auc={max(xgb_aucs):.5f}")

    xgb_results = {
        "val_auc":          xgb_aucs,
        "trees":            list(range(1, N_ROUNDS + 1)),
        "cumulative_times": xgb_times,
        "total_time":       xgb_total,
    }

    # ── LightGBM ──
    print(f"\n--- LightGBM | n_rounds={N_ROUNDS} | lr={LR} | max_depth={DEPTH} ---")
    lgb_aucs  = []
    lgb_times = []
    lgb_start = time.perf_counter()

    def lgb_callback(env):
        iteration = env.iteration
        t   = time.perf_counter() - lgb_start
        auc = roc_auc_score(y_val,
                            env.model.predict(X_val,
                                              num_iteration=iteration + 1))
        lgb_aucs.append(auc)
        lgb_times.append(t)
        if iteration % 20 == 0 or iteration == N_ROUNDS - 1:
            print(f"  Round {iteration+1:>4}/{N_ROUNDS} | "
                  f"val_auc={auc:.5f} | time={t:.1f}s")

    lgb_callback.before_iteration = False
    lgb_callback.order            = 10

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb.train(
        params={
            "objective":       "binary",
            "metric":          "auc",
            "learning_rate":   LR,
            "max_depth":       DEPTH,
            "num_leaves":      2**DEPTH - 1,
            "min_child_samples": 20,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "n_jobs":          -1,
            "seed":            42,
            "verbosity":       -1,
        },
        train_set=lgb_train,
        num_boost_round=N_ROUNDS,
        callbacks=[lgb_callback]
    )
    lgb_total = time.perf_counter() - lgb_start
    print(f"  → Done. total_time={lgb_total:.1f}s | "
          f"best_val_auc={max(lgb_aucs):.5f}")

    lgb_results = {
        "val_auc":          lgb_aucs,
        "trees":            list(range(1, N_ROUNDS + 1)),
        "cumulative_times": lgb_times,
        "total_time":       lgb_total,
    }

    return xgb_results, lgb_results


def plot_xgb_lgbm_comparison(xgb_res, lgb_res):
    """
    Two plots:
      1. AUC vs tree count  (equal tree count comparison)
      2. AUC vs wall-clock time  (equal budget comparison)

    Baseline and b2_live taken from Santander logs.
    XGBoost and LightGBM from fresh fits above.
    """

    # Interpolate baseline and b2_live to per-tree resolution for plot 1
    baseline_auc_interp = np.interp(
        np.arange(1, 401),
        SANT_BASELINE_TREES,
        SANT_BASELINE_AUCS
    )
    b2_live_auc_interp = np.interp(
        np.arange(2, 1725, 2),
        SANT_B2_LIVE_TREES,
        SANT_B2_LIVE_AUCS
    )
    b2_live_tree_axis = np.arange(2, 1725, 2)

    # Cumulative times for baseline (uniform from log total)
    baseline_cum_times = np.linspace(0, SANT_BASELINE_TIME, 400)
    b2_live_cum_times  = np.linspace(0, SANT_B2_LIVE_TIME, len(SANT_B2_LIVE_BLOCKS))
    b2_live_time_aucs  = SANT_B2_LIVE_AUCS

    # ── Plot 1: AUC vs tree count ──
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(np.arange(1, 401), baseline_auc_interp,
            color="blue", linewidth=2, label="Baseline GBM (B=1, col=1.0)")
    ax.plot(b2_live_tree_axis, b2_live_auc_interp,
            color="red", linewidth=2, linestyle="--",
            label="Block B=2, col=0.5 (our method)")
    ax.plot(xgb_res["trees"], xgb_res["val_auc"],
            color="green", linewidth=2, linestyle="-.",
            label="XGBoost (within-tree parallel)")
    ax.plot(lgb_res["trees"], lgb_res["val_auc"],
            color="orange", linewidth=2, linestyle=":",
            label="LightGBM (within-tree parallel)")

    ax.axvline(x=400, color="gray", linestyle=":", alpha=0.5,
               label="400 trees (baseline budget)")
    ax.set_xlabel("Number of trees", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("AUC vs Tree Count — Santander\n"
                 "(XGBoost/LightGBM address within-tree parallelism; "
                 "our method addresses across-iteration parallelism)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{OUT_DIR}xgb_lgbm_vs_block_trees.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # ── Plot 2: AUC vs wall-clock time ──
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(baseline_cum_times, baseline_auc_interp,
            color="blue", linewidth=2, label="Baseline GBM (B=1, col=1.0)")
    ax.plot(np.linspace(0, SANT_B2_LIVE_TIME, len(b2_live_time_aucs)),
            b2_live_time_aucs,
            color="red", linewidth=2, linestyle="--",
            label=f"Block B=2, col=0.5 — {len(SANT_B2_LIVE_BLOCKS)*2} trees")
    ax.plot(xgb_res["cumulative_times"], xgb_res["val_auc"],
            color="green", linewidth=2, linestyle="-.",
            label=f"XGBoost — {len(xgb_res['trees'])} trees")
    ax.plot(lgb_res["cumulative_times"], lgb_res["val_auc"],
            color="orange", linewidth=2, linestyle=":",
            label=f"LightGBM — {len(lgb_res['trees'])} trees")

    ax.axhline(y=0.80692, color="blue", linestyle=":", alpha=0.4,
               label="Baseline final AUC = 0.80692")
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("AUC vs Wall-Clock Time — Santander\n"
                 "Note: XGBoost/LightGBM finish in far less time at 400 trees; "
                 "block method uses remaining time to build more trees",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{OUT_DIR}xgb_lgbm_vs_block_time.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ═════════════════════════════════════════════════════════════════
#  PART 2 — Covertype hyperparameter sensitivity
# ═════════════════════════════════════════════════════════════════

def load_covertype(n_samples=100_000, random_state=42):
    """
    Loads Covertype, subsamples n_samples rows stratified on the
    7-class label, then binarises to class 1 vs rest.
    Returns X_tr, X_val, y_tr, y_val.
    """
    print("\nLoading Covertype dataset...")
    data = fetch_covtype()
    X, y_multi = data.data.astype(np.float32), data.target

    # Stratified subsample from the full 581k rows
    _, X_sub, _, y_sub = train_test_split(
        X, y_multi,
        test_size=n_samples / len(y_multi),
        random_state=random_state,
        stratify=y_multi
    )

    # Binarise: class 1 (Spruce/Fir) vs rest — largest class, ~37%
    y_bin = (y_sub == 1).astype(float)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y_bin,
        test_size=0.2,
        random_state=random_state,
        stratify=y_bin
    )
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | "
          f"Positive rate: {y_tr.mean():.3f}")
    return X_tr, X_val, y_tr, y_val


def run_hparam_sensitivity(X_tr, y_tr, X_val, y_val, n_estimators=200):
    """
    Trains baseline (B=1, col=1.0) and b2 (B=2, col=0.5) across a
    3 x 3 grid of learning_rate x max_depth.
    Returns a DataFrame with one row per (lr, depth, model) combination.
    """
    print(f"\n{'='*65}")
    print(f"PART 2A: HYPERPARAMETER SENSITIVITY — COVERTYPE")
    print(f"n_estimators={n_estimators}")
    print(f"{'='*65}")

    learning_rates = [0.05, 0.1, 0.2]
    max_depths     = [3, 4, 6]

    rows = []
    for lr in learning_rates:
        for depth in max_depths:
            print(f"\n--- lr={lr} | max_depth={depth} ---")

            # Baseline
            print(f"  [baseline]")
            base = BlockParallelGBM(
                n_estimators=n_estimators, block_size=1,
                learning_rate=lr, max_depth=depth,
                min_samples_leaf=20, colsample=1.0,
                auto_scale_lr=False, n_jobs=1,
                random_state=42, verbose=True
            )
            base.fit(X_tr, y_tr, X_val, y_val)

            # b2
            print(f"  [b2]")
            b2 = BlockParallelGBM(
                n_estimators=n_estimators, block_size=2,
                learning_rate=lr, max_depth=depth,
                min_samples_leaf=20, colsample=0.5,
                auto_scale_lr=True, n_jobs=-1,
                random_state=42, verbose=True
            )
            b2.fit(X_tr, y_tr, X_val, y_val)

            auc_gap = base.best_val_auc - b2.best_val_auc
            speedup = base.total_time / b2.total_time

            print(f"  → baseline AUC={base.best_val_auc:.5f} "
                  f"time={base.total_time:.1f}s")
            print(f"  → b2      AUC={b2.best_val_auc:.5f}   "
                  f"time={b2.total_time:.1f}s")
            print(f"  → AUC gap={auc_gap:.5f} | speedup={speedup:.2f}x")

            rows.append({
                "learning_rate":    lr,
                "max_depth":        depth,
                "baseline_auc":     round(base.best_val_auc, 5),
                "b2_auc":           round(b2.best_val_auc, 5),
                "auc_gap":          round(auc_gap, 5),
                "speedup":          round(speedup, 3),
                "baseline_time_s":  round(base.total_time, 1),
                "b2_time_s":        round(b2.total_time, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}covertype_hparam_results.csv", index=False)
    print(f"\nSaved: covertype_hparam_results.csv")
    return df


def run_block_sweep_covertype(X_tr, y_tr, X_val, y_val,
                               n_estimators=200,
                               lr=0.1, depth=4):
    """
    Block sweep B=1,2,3,4 on Covertype at default hyperparameters.
    All configs use col=0.5 (col_only as B=1 reference for fair comparison).
    """
    print(f"\n{'='*65}")
    print(f"PART 2B: BLOCK SWEEP — COVERTYPE")
    print(f"lr={lr} | max_depth={depth} | n_estimators={n_estimators}")
    print(f"{'='*65}")

    configs = [
        # bs, col,  asl,   nj,  label
        (1,   0.5,  False, 1,   "B=1, col=0.5 (reference)"),
        (2,   0.5,  True,  -1,  "B=2, col=0.5"),
        (3,   0.5,  True,  -1,  "B=3, col=0.5"),
        (4,   0.5,  True,  -1,  "B=4, col=0.5"),
    ]

    models = []
    rows   = []
    for bs, cs, asl, nj, label in configs:
        print(f"\n[{label}]")
        m = BlockParallelGBM(
            n_estimators=n_estimators, block_size=bs,
            learning_rate=lr, max_depth=depth,
            min_samples_leaf=20, colsample=cs,
            auto_scale_lr=asl, n_jobs=nj,
            random_state=42, verbose=True
        )
        m.fit(X_tr, y_tr, X_val, y_val)
        m.label = label
        models.append(m)
        rows.append({
            "Block size (B)": bs,
            "Trees":          len(m.trees_),
            "Total time (s)": round(m.total_time, 1),
            "Best val AUC":   round(m.best_val_auc, 5),
        })

    df = pd.DataFrame(rows)
    ref_time = df["Total time (s)"].iloc[0]
    ref_auc  = df["Best val AUC"].iloc[0]
    df["Speedup"] = (ref_time / df["Total time (s)"]).round(2).astype(str) + "x"
    df["AUC gap"] = (ref_auc - df["Best val AUC"]).round(5)

    print("\nBLOCK SWEEP TABLE — COVERTYPE:")
    print(df.to_string(index=False))
    df.to_csv(f"{OUT_DIR}covertype_block_sweep.csv", index=False)
    print(f"Saved: covertype_block_sweep.csv")
    return df, models


def plot_hparam_heatmaps(df):
    """AUC gap and speedup as heatmaps over lr x max_depth grid."""
    for metric, title, fmt in [
        ("auc_gap",  "AUC Gap (baseline − b2) across Hyperparameters\n"
                     "Covertype | positive = baseline better | "
                     "negative = b2 better", ".4f"),
        ("speedup",  "Speedup (baseline_time / b2_time) across Hyperparameters\n"
                     "Covertype", ".2f"),
    ]:
        pivot = df.pivot(index="max_depth", columns="learning_rate",
                         values=metric)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn_r" if metric == "auc_gap"
                    else "YlGn", ax=ax, linewidths=0.5,
                    cbar_kws={"label": metric})
        ax.set_xlabel("Learning rate", fontsize=12)
        ax.set_ylabel("Max depth", fontsize=12)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        fname = f"{OUT_DIR}covertype_hparam_{metric}_heatmap.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_block_sweep_covertype(df, models):
    """Speedup and AUC gap vs B, plus learning curves."""
    bsizes       = [1, 2, 3, 4]
    speedup_vals = [float(s.replace("x", "")) for s in df["Speedup"]]

    # ── Sweep plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(bsizes, speedup_vals, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Block size (B)", fontsize=12)
    ax1.set_ylabel("Speedup vs B=1", fontsize=12)
    ax1.set_title("Speedup vs Block Size — Covertype", fontsize=12,
                  fontweight="bold")
    ax1.set_xticks(bsizes)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(bsizes, speedup_vals):
        ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=10)

    ax2.plot(bsizes, df["AUC gap"], "rs-", linewidth=2, markersize=8)
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Block size (B)", fontsize=12)
    ax2.set_ylabel("AUC gap (B=1 − B=x)", fontsize=12)
    ax2.set_title("AUC Gap vs Block Size — Covertype", fontsize=12,
                  fontweight="bold")
    ax2.set_xticks(bsizes)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{OUT_DIR}covertype_block_sweep.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # ── Learning curves ──
    colors  = ["blue", "red", "green", "orange"]
    lstyles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, color, ls in zip(models, colors, lstyles):
        ax.plot(m.cumulative_times_, m.val_auc_,
                label=m.label, color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Val AUC vs Wall-Clock Time — Covertype", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = f"{OUT_DIR}covertype_learning_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ═════════════════════════════════════════════════════════════════
#  Master runner
# ═════════════════════════════════════════════════════════════════

def run_all(santander_path, n_estimators_covertype=200):
    """
    Parameters
    ----------
    santander_path : str
        Path to Santander train.csv — used only for XGB/LGB fits.
    n_estimators_covertype : int
        Trees per model on Covertype. 200 is sufficient and fast.
    """

    # ── Part 1: XGBoost / LightGBM on Santander ──
    xgb_res, lgb_res = fit_xgb_lgbm_santander(santander_path)
    plot_xgb_lgbm_comparison(xgb_res, lgb_res)

    # ── Part 2: Covertype ──
    X_tr, X_val, y_tr, y_val = load_covertype(n_samples=100_000)

    # Hyperparameter sensitivity
    hparam_df = run_hparam_sensitivity(
        X_tr, y_tr, X_val, y_val,
        n_estimators=n_estimators_covertype
    )
    plot_hparam_heatmaps(hparam_df)

    # Block sweep at default setting (lr=0.1, depth=4)
    sweep_df, sweep_models = run_block_sweep_covertype(
        X_tr, y_tr, X_val, y_val,
        n_estimators=n_estimators_covertype,
        lr=0.1, depth=4
    )
    plot_block_sweep_covertype(sweep_df, sweep_models)

    # Summary
    print(f"\n{'='*65}")
    print("ALL ADDITIONAL EXPERIMENTS COMPLETE")
    print(f"Outputs saved to: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if any(f.endswith(ext) for ext in [".png", ".csv"]):
            print(f"  {f}")
    print(f"{'='*65}")


# ═════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SANTANDER_PATH = "/kaggle/input/santander-customer-transaction-prediction/train.csv"
    run_all(SANTANDER_PATH, n_estimators_covertype=200)