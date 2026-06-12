"""
verify_alpha_integrated.py
--------------------------
Verifies the weak learner assumption (alpha > 0, bounded away from zero)
by subclassing BlockParallelGBM directly from the project repo.

Alpha is defined as:
    alpha^(k,b) = <h_b(X), r^(k)> / ||r^(k)||^2

where r^(k) are the residuals at the start of block k and h_b is the
b-th tree in that block. The assumption requires alpha > 0 at every
block, for every tree, across all datasets and block sizes.

Setup on Kaggle:
    Download "block_parallel_gbm_kaggle.py" from GitHub and upload it as a dataset on Kaggle.
    Replace the system file path with yours.
    Then run this script.

Results saved to: alpha_verification.csv
"""

import sys
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ── Import from your repo ──
# Download "block_parallel_gbm_kaggle.py" from this repository and upload it as a dataset in kaggle, then replace the path"
sys.path.append("/kaggle/input/datasets/.../blockpgbm")
from block_parallel_gbm_kaggle import BlockParallelGBM, fit_single_tree, compute_residuals, sigmoid
# Adjust the import path/module name to match your repo structure

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANDOM_STATE = 42
N_BLOCKS     = 30       # blocks to run — enough to see trend
BLOCK_SIZES  = [1, 2, 4]

DATASET_PATHS = {
    "Santander": {
        "train":  "/kaggle/input/competitions/santander-customer-transaction-prediction/train.csv",
        "target": "target",
        "n_rows": 200_000,
    },
    "Covertype": {
        "source": "sklearn",
        "n_rows": 100_000,
    },
    "UCI_Adult": {
        "train":  "/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv",
        "target": "income",
        "n_rows": None,
    },
    "IEEE_CIS": {
        "train":  "/kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv",
        "target": "isFraud",
        "n_rows": 200_000,
    },
}

# ─────────────────────────────────────────────
# SUBCLASS — adds alpha logging, zero other changes
# ─────────────────────────────────────────────

class AlphaLoggingGBM(BlockParallelGBM):
    """
    Identical to BlockParallelGBM but logs alpha^(k,b) at every block.

    After fit(), access:
        self.alpha_records_  — list of dicts, one per (block, tree) pair
    """

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

        n_blocks = int(np.ceil(self.n_estimators / self.block_size))

        # ── Storage for alpha records ──
        self.alpha_records_ = []

        if self.verbose:
            print(f"AlphaLoggingGBM | B={self.block_size} | "
                  f"n_blocks={n_blocks} | colsample={self.colsample}")
            print("-" * 60)

        for block_idx in range(n_blocks):

            trees_this_block = min(
                self.block_size,
                self.n_estimators - block_idx * self.block_size
            )

            # ── Residuals at start of block ──
            residuals = compute_residuals(y, F)

            # Pre-compute quantities needed for alpha
            r_norm_sq = float(np.dot(residuals, residuals))
            r_norm    = float(np.sqrt(r_norm_sq))

            seeds = rng.randint(0, 2**31, size=trees_this_block)

            # ── Fit trees (same logic as original) ──
            if trees_this_block == 1:
                results = [
                    fit_single_tree(
                        X, residuals, self.colsample,
                        self.max_depth, self.min_samples_leaf, seeds[0]
                    )
                ]
            else:
                results = Parallel(
                    n_jobs=self.n_jobs, prefer="processes"
                )(
                    delayed(fit_single_tree)(
                        X, residuals, self.colsample,
                        self.max_depth, self.min_samples_leaf, seeds[b]
                    )
                    for b in range(trees_this_block)
                )

            # ── Compute alpha for each tree, then update F ──
            for b, tree in enumerate(results):
                self.trees_.append(tree)

                h_pred = tree.predict(X)   # full training set predictions

                # alpha = <h, r> / ||r||^2
                # Measures directional alignment of tree with residuals
                if r_norm_sq > 1e-10:
                    alpha = float(np.dot(h_pred, residuals) / r_norm_sq)
                else:
                    alpha = None   # residuals essentially zero — converged

                self.alpha_records_.append({
                    "block":          block_idx,
                    "tree_in_block":  b,
                    "B":              self.block_size,
                    "alpha":          round(alpha, 8) if alpha is not None else None,
                    "r_norm":         round(r_norm, 6),
                    "alpha_positive": bool(alpha > 0) if alpha is not None else None,
                    "converged":      alpha is None,
                })

                F += self._effective_lr * h_pred
                if X_val is not None:
                    F_val += self._effective_lr * tree.predict(X_val)

            # Track AUC
            train_auc = roc_auc_score(y, sigmoid(F))
            self.train_auc_.append(train_auc)
            self.block_times_.append(0.0)   # timing not needed here

            if X_val is not None and y_val is not None:
                self.val_auc_.append(
                    roc_auc_score(y_val, sigmoid(F_val))
                )

            if self.verbose and (block_idx % 10 == 0 or
                                 block_idx == n_blocks - 1):
                alphas_this_block = [
                    rec["alpha"] for rec in self.alpha_records_
                    if rec["block"] == block_idx
                    and rec["alpha"] is not None
                ]
                mean_alpha = np.mean(alphas_this_block) if alphas_this_block else float("nan")
                print(f"Block {block_idx+1:>4}/{n_blocks} | "
                      f"r_norm={r_norm:.4f} | "
                      f"mean_alpha={mean_alpha:.5f} | "
                      f"train_auc={train_auc:.5f}")

        return self


# ─────────────────────────────────────────────
# DATA LOADERS
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
    X, y = X[idx], (y[idx] == 1).astype(int)
    return X.astype(np.float32), y

def load_adult(cfg):
    df = pd.read_csv(cfg["train"])
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    y = df[cfg["target"]].values
    X = df.drop(columns=[cfg["target"]]).values.astype(np.float32)
    return X, y

def load_ieee_cis(cfg):
    df = pd.read_csv(cfg["train"], nrows=cfg["n_rows"])
    y  = df[cfg["target"]].values
    df = df.drop(columns=[cfg["target"], "TransactionID"], errors="ignore")
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(-1)
    return df.values.astype(np.float32), y

# ─────────────────────────────────────────────
# SUMMARY + VERDICT
# ─────────────────────────────────────────────

def summarise(all_records, dataset, B):
    """Compute summary statistics and print verdict."""
    recs   = [r for r in all_records
              if r["B"] == B and r["alpha"] is not None]
    if not recs:
        return None

    alphas = np.array([r["alpha"] for r in recs])

    # Split into first and second half to check for trend
    mid         = len(alphas) // 2
    first_half  = alphas[:mid]
    second_half = alphas[mid:]
    trend       = float(np.mean(second_half) - np.mean(first_half))

    summary = {
        "dataset":          dataset,
        "B":                B,
        "n_observations":   len(alphas),
        "alpha_mean":       round(float(np.mean(alphas)), 6),
        "alpha_std":        round(float(np.std(alphas)),  6),
        "alpha_min":        round(float(np.min(alphas)),  6),
        "alpha_max":        round(float(np.max(alphas)),  6),
        "pct_positive":     round(100.0 * np.mean(alphas > 0), 2),
        "trend":            round(trend, 6),
        "trend_direction":  "stable"   if abs(trend) < 0.01
                            else "declining" if trend < 0
                            else "improving",
        "assumption_holds": bool(
            np.all(alphas > 0) and float(np.min(alphas)) > 1e-4
        ),
    }
    return summary


def print_verdict(summaries):
    print(f"\n{'='*65}")
    print("WEAK LEARNER ASSUMPTION VERDICT")
    print(f"{'='*65}")
    print(f"{'Dataset':12s} {'B':>3} {'alpha_min':>10} "
          f"{'alpha_mean':>11} {'100% pos':>9} {'Trend':>10} {'Result':>8}")
    print("-" * 65)
    for s in summaries:
        if s is None:
            continue
        result = "HOLDS" if s["assumption_holds"] else "VIOLATED"
        print(f"{s['dataset']:12s} {s['B']:>3} "
              f"{s['alpha_min']:>10.5f} "
              f"{s['alpha_mean']:>11.5f} "
              f"{s['pct_positive']:>8.1f}% "
              f"{s['trend_direction']:>10} "
              f"{result:>8}")
    print(f"{'='*65}")
    print("\nAssumption HOLDS requires:")
    print("  1. alpha > 0 at every block (pct_positive = 100%)")
    print("  2. alpha_min > 1e-4 (bounded away from zero)")
    print("  3. No strong declining trend (alpha not collapsing)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_dataset(name, X, y):
    print(f"\n{'='*65}")
    print(f"DATASET: {name}  |  shape: {X.shape}")
    print(f"{'='*65}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    all_records = []
    for B in BLOCK_SIZES:
        print(f"\n--- B={B} ---")
        model = AlphaLoggingGBM(
            n_estimators=N_BLOCKS * B,  # exactly N_BLOCKS blocks
            block_size=B,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=20,
            colsample=0.5,
            auto_scale_lr=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=True,
        )
        model.fit(X_train, y_train, X_val, y_val)
        all_records.extend(model.alpha_records_)

    return all_records


def main():
    all_records  = []
    all_summaries = []

    loaders = {
        "Santander": (load_santander, DATASET_PATHS["Santander"]),
        "Covertype": (load_covertype, DATASET_PATHS["Covertype"]),
        "UCI_Adult": (load_adult,     DATASET_PATHS["UCI_Adult"]),
        "IEEE_CIS":  (load_ieee_cis,  DATASET_PATHS["IEEE_CIS"]),
    }

    for name, (loader, cfg) in loaders.items():
        try:
            X, y    = loader(cfg)
            records = run_dataset(name, X, y)

            # Tag records with dataset name
            for r in records:
                r["dataset"] = name
            all_records.extend(records)

            for B in BLOCK_SIZES:
                s = summarise(records, name, B)
                if s:
                    all_summaries.append(s)

        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    # ── Save detailed records ──
    if all_records:
        keys = ["dataset", "block", "tree_in_block", "B",
                "alpha", "r_norm", "alpha_positive", "converged"]
        with open("alpha_records_detailed.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_records)
        print("\nDetailed records saved to: alpha_records_detailed.csv")

    # ── Save summary ──
    if all_summaries:
        with open("alpha_verification.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        print("Summary saved to: alpha_verification.csv")

    print_verdict(all_summaries)


if __name__ == "__main__":
    main()
