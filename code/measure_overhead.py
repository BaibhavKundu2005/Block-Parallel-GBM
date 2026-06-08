"""
measure_overhead.py
--------------------
Measures tau_tree (sequential single-tree fitting time) and
tau_block_B2 (wall-clock time per block under B=2 parallelism)
across four datasets, and computes the implied joblib process-level
overhead as:
    tau_overhead = tau_block_B2 - tau_tree_col05

where tau_tree_col05 is the per-tree time at colsample=0.5
(the configuration used inside each parallel block).

Results are saved to overhead_measurements.csv.

Usage:
    Add necessary datasets and replace the file paths if needed.

Requirements:
    pip install scikit-learn pandas numpy joblib kaggle
    (Datasets are loaded from Kaggle or local paths — see DATASET PATHS below)

Note: Run on the same hardware (Kaggle free-tier 4-core CPU) to reproduce
the values reported in the paper. Results will differ on other hardware.
"""

import time
import csv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Number of timing repetitions to average over for stability
N_TIMING_REPS_TREE  = 5   # single-tree fits averaged
N_TIMING_REPS_BLOCK = 10   # parallel block fits averaged

# Shared hyperparameters (must match paper)
MAX_DEPTH         = 4
MIN_SAMPLES_LEAF  = 20
COLSAMPLE         = 0.5   # feature subsampling ratio used inside blocks
RANDOM_STATE      = 42

# ─────────────────────────────────────────────
# DATASET PATHS — edit these to match your env
# ─────────────────────────────────────────────
# On Kaggle notebooks these are the standard input paths.
# Locally, download from Kaggle and point to the CSV files.

DATASET_PATHS = {
    "Santander": {
        "train": "/kaggle/input/competitions/santander-customer-transaction-prediction/train.csv",
        "target": "target",
        "n_rows": 200_000,
    },
    "Covertype": {
        # UCI Covertype — download via sklearn or from UCI
        "source": "sklearn",   # use sklearn.datasets.fetch_covtype
        "n_rows": 100_000,
    },
    "UCI_Adult": {
        "train": "/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv",
        "target": "income",
        "n_rows": None,        # use all rows
    },
    "IEEE_CIS": {
        "train": "/kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv",
        "target": "isFraud",
        "n_rows": 200_000,
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_santander(cfg):
    df = pd.read_csv(cfg["train"], nrows=cfg["n_rows"])
    y  = df[cfg["target"]].values
    X  = df.drop(columns=["ID_code", cfg["target"]]).values.astype(np.float32)
    return X, y

def load_covertype(cfg):
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    X, y = data.data, data.target
    rng  = np.random.RandomState(RANDOM_STATE)
    idx  = rng.choice(len(y), size=cfg["n_rows"], replace=False)
    X, y = X[idx], y[idx]
    y    = (y == 1).astype(int)   # binarise: class 1 vs rest
    return X.astype(np.float32), y

def load_adult(cfg):
    df = pd.read_csv(cfg["train"])
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    target_col = cfg["target"]
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    return X, y

def load_ieee_cis(cfg):
    df = pd.read_csv(cfg["train"], nrows=cfg["n_rows"])
    y  = df[cfg["target"]].values
    df = df.drop(columns=[cfg["target"], "TransactionID"], errors="ignore")
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(-1)
    X  = df.values.astype(np.float32)
    return X, y

def subsample_features(X, colsample, seed):
    """Return a randomly subsampled column view of X."""
    rng     = np.random.RandomState(seed)
    n_feats = max(1, int(X.shape[1] * colsample))
    cols    = rng.choice(X.shape[1], size=n_feats, replace=False)
    return X[:, cols]

def fit_single_tree(X, r, seed, colsample):
    """Fit one decision tree and return it (used inside parallel blocks)."""
    X_sub = subsample_features(X, colsample, seed)
    tree  = DecisionTreeRegressor(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=seed,
    )
    tree.fit(X_sub, r)
    return tree

# ─────────────────────────────────────────────
# TIMING FUNCTIONS
# ─────────────────────────────────────────────

def measure_tau_tree(X_train, r, colsample, n_reps):
    """
    Average wall-clock time to fit a single tree at given colsample.
    Uses sequential fitting (B=1) with fresh random seeds each rep.
    """
    times = []
    for rep in range(n_reps):
        seed = RANDOM_STATE + rep
        t0   = time.perf_counter()
        fit_single_tree(X_train, r, seed=seed, colsample=colsample)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))

def measure_tau_block_B2(X_train, r, colsample, n_reps):
    """
    Average wall-clock time for one B=2 parallel block.
    This includes: joblib worker initialisation/reuse, data
    serialisation to workers, parallel tree fitting, and
    result synchronisation.
    """
    times = []
    for rep in range(n_reps):
        seeds = [RANDOM_STATE + rep * 2, RANDOM_STATE + rep * 2 + 1]
        t0    = time.perf_counter()
        Parallel(n_jobs=2, prefer="processes")(
            delayed(fit_single_tree)(X_train, r, seed=s, colsample=colsample)
            for s in seeds
        )
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_dataset(name, X, y):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  |  X: {X.shape}  |  y mean: {y.mean():.3f}")
    print(f"{'='*60}")

    # Use 80% for training, consistent with paper
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Use a fixed residual vector (log-odds init, no ensemble yet)
    y_mean = y_train.mean()
    F0     = np.log(y_mean / (1 - y_mean + 1e-9))
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    r      = y_train - sigmoid(np.full(len(y_train), F0))

    # ── tau_tree at colsample=1.0 (paper reports this value) ──
    print(f"  Timing tau_tree (col=1.0, {N_TIMING_REPS_TREE} reps)...")
    tau_tree_col10, std_col10 = measure_tau_tree(
        X_train, r, colsample=1.0, n_reps=N_TIMING_REPS_TREE
    )
    print(f"    tau_tree (col=1.0): {tau_tree_col10:.4f}s ± {std_col10:.4f}s")

    # ── tau_tree at colsample=0.5 (used inside parallel blocks) ──
    print(f"  Timing tau_tree (col=0.5, {N_TIMING_REPS_TREE} reps)...")
    tau_tree_col05, std_col05 = measure_tau_tree(
        X_train, r, colsample=COLSAMPLE, n_reps=N_TIMING_REPS_TREE
    )
    print(f"    tau_tree (col=0.5): {tau_tree_col05:.4f}s ± {std_col05:.4f}s")

    # ── tau_block_B2 at colsample=0.5 ──
    print(f"  Timing tau_block_B2 (col=0.5, {N_TIMING_REPS_BLOCK} reps)...")
    tau_block, std_block = measure_tau_block_B2(
        X_train, r, colsample=COLSAMPLE, n_reps=N_TIMING_REPS_BLOCK
    )
    print(f"    tau_block_B2 (col=0.5): {tau_block:.4f}s ± {std_block:.4f}s")

    # ── Implied overhead ──
    # tau_block_B2 = tau_tree_col05 + tau_overhead  (since trees run in parallel,
    # block time ≈ max(tree times) + overhead ≈ tau_tree_col05 + tau_overhead)
    implied_overhead = tau_block - tau_tree_col05
    rho_col10        = (implied_overhead / tau_tree_col10
                        if tau_tree_col10 > 0 else float("inf"))

    print(f"    Implied overhead: {implied_overhead:.4f}s")
    print(f"    rho (overhead/tau_tree_col10): {rho_col10:.4f}")

    return {
        "Dataset":             name,
        "n_samples":           X.shape[0],
        "n_features":          X.shape[1],
        "tau_tree_col10_mean": round(tau_tree_col10, 4),
        "tau_tree_col10_std":  round(std_col10, 4),
        "tau_tree_col05_mean": round(tau_tree_col05, 4),
        "tau_tree_col05_std":  round(std_col05, 4),
        "tau_block_B2_mean":   round(tau_block, 4),
        "tau_block_B2_std":    round(std_block, 4),
        "implied_overhead":    round(implied_overhead, 4),
        "rho":                 round(rho_col10, 4),
    }


def main():
    results = []

    # ── Santander ──
    try:
        cfg  = DATASET_PATHS["Santander"]
        X, y = load_santander(cfg)
        results.append(run_dataset("Santander", X, y))
    except Exception as e:
        print(f"[SKIP] Santander: {e}")

    # ── Covertype ──
    try:
        cfg  = DATASET_PATHS["Covertype"]
        X, y = load_covertype(cfg)
        results.append(run_dataset("Covertype", X, y))
    except Exception as e:
        print(f"[SKIP] Covertype: {e}")

    # ── UCI Adult ──
    try:
        cfg  = DATASET_PATHS["UCI_Adult"]
        X, y = load_adult(cfg)
        results.append(run_dataset("UCI_Adult", X, y))
    except Exception as e:
        print(f"[SKIP] UCI_Adult: {e}")

    # ── IEEE-CIS ──
    try:
        cfg  = DATASET_PATHS["IEEE_CIS"]
        X, y = load_ieee_cis(cfg)
        results.append(run_dataset("IEEE_CIS", X, y))
    except Exception as e:
        print(f"[SKIP] IEEE_CIS: {e}")

    # ── Write CSV ──
    if results:
        out_path = "overhead_measurements.csv"
        keys     = results[0].keys()
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{'='*60}")
        print(f"Results saved to: {out_path}")
        print(f"{'='*60}")
        df = pd.DataFrame(results)
        print(df[[
            "Dataset",
            "tau_tree_col10_mean",
            "tau_tree_col05_mean",
            "tau_block_B2_mean",
            "implied_overhead",
            "rho"
        ]].to_string(index=False))
    else:
        print("No datasets loaded successfully.")


if __name__ == "__main__":
    main()
