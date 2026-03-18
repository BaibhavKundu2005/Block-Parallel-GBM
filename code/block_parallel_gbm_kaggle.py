"""
Block Parallel GBM with Feature Subsampling — Kaggle Version
=============================================================
KEY FIX: sklearn's DecisionTreeRegressor with exact splitting is O(n x p)
per node — 20s per tree at Santander scale (160k rows, 200 features).

The fix is to pass max_features directly into DecisionTreeRegressor so
sklearn handles feature subsampling internally using its optimised C code,
rather than us slicing X in Python before fitting. This brings tree fit
time from ~20s down to ~2-4s at colsample=0.5 (100 features).

Usage: run all cells top to bottom in a Kaggle notebook.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Utility: sigmoid and log-loss gradient
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def compute_residuals(y, F):
    """
    Negative gradient of binary log-loss w.r.t. F (raw log-odds).
    gradient_i = y_i - sigmoid(F_i)
    """
    return y - sigmoid(F)


# ─────────────────────────────────────────────
#  Single tree fitter (called in parallel)
# ─────────────────────────────────────────────

def fit_single_tree(X, residuals, max_features, max_depth, min_samples_leaf, seed):
    """
    Fit one regression tree on the full X, letting sklearn subsample
    features internally via max_features. This is much faster than
    slicing X in Python because sklearn's splitter does it in C.

    Returns the fitted tree. No feature index tracking needed since
    the tree was trained on the full feature space.
    """
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,   # ← sklearn handles subsampling in C
        random_state=seed
    )
    tree.fit(X, residuals)
    return tree


# ─────────────────────────────────────────────
#  Block Parallel GBM
# ─────────────────────────────────────────────

class BlockParallelGBM:
    """
    Gradient Boosting Machine with:
      - Block parallelism: B trees trained simultaneously per round on shared residuals
      - Feature subsampling: each tree sees max_features randomly chosen features
        (handled internally by sklearn's optimised C splitter)
      - Exact residual sync after every block

    Parameters
    ----------
    n_estimators : int
        Total number of trees
    block_size : int
        Number of trees trained in parallel per boosting round (B)
    learning_rate : float
        Shrinkage applied to each tree's contribution.
        Automatically scaled by 1/block_size to compensate for effective LR inflation.
    max_depth : int
        Maximum depth of each tree
    min_samples_leaf : int
        Minimum samples per leaf
    colsample : float or int or "sqrt" or "log2"
        Features per tree. Float = fraction (e.g. 0.5 = 50%).
        Passed directly to DecisionTreeRegressor's max_features.
    auto_scale_lr : bool
        If True, divides learning_rate by block_size to compensate for
        the block's collective update magnitude. Recommended.
    n_jobs : int
        Number of parallel workers. -1 = all cores (4 on Kaggle).
    random_state : int
        Master seed for reproducibility
    verbose : bool
        Print progress per block
    """

    def __init__(
        self,
        n_estimators=300,
        block_size=2,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=20,
        colsample=0.5,
        auto_scale_lr=True,
        n_jobs=-1,
        random_state=42,
        verbose=True
    ):
        self.n_estimators = n_estimators
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.colsample = colsample
        self.auto_scale_lr = auto_scale_lr
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Effective learning rate accounts for block size
        self._effective_lr = learning_rate / block_size if auto_scale_lr else learning_rate

        self.trees_ = []        # list of fitted DecisionTreeRegressors
        self.F0_ = None         # initial log-odds
        self.train_auc_ = []    # AUC per block on training set
        self.val_auc_ = []      # AUC per block on validation set (if provided)
        self.block_times_ = []  # wall-clock time per block

    def _initial_prediction(self, y):
        """Log-odds of the positive class as the initial prediction."""
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary 0/1
        X_val, y_val : optional validation set for tracking AUC
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initial prediction: log-odds
        self.F0_ = self._initial_prediction(y)
        F = np.full(n_samples, self.F0_, dtype=np.float64)

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            F_val = np.full(len(X_val), self.F0_, dtype=np.float64)

        n_blocks = int(np.ceil(self.n_estimators / self.block_size))

        if self.verbose:
            n_feats_used = (
                int(n_features * self.colsample)
                if isinstance(self.colsample, float)
                else self.colsample
            )
            print(f"BlockParallelGBM | trees={self.n_estimators} | "
                  f"block_size={self.block_size} | n_blocks={n_blocks}")
            print(f"effective_lr={self._effective_lr:.4f} | "
                  f"colsample={self.colsample} (~{n_feats_used} features) | "
                  f"max_depth={self.max_depth}")
            print("-" * 65)

        for block_idx in range(n_blocks):
            t0 = time.perf_counter()

            # How many trees in this block (last block may be smaller)
            trees_this_block = min(
                self.block_size,
                self.n_estimators - block_idx * self.block_size
            )

            # ── EXACT residual computation (sync point) ──
            residuals = compute_residuals(y, F)

            # Generate one seed per tree for reproducibility
            seeds = rng.randint(0, 2**31, size=trees_this_block)

            # ── Tree fitting ──
            # Single tree: skip joblib entirely (no process-spawn overhead).
            # Multiple trees: launch in parallel processes.
            if trees_this_block == 1:
                results = [
                    fit_single_tree(
                        X, residuals, self.colsample,
                        self.max_depth, self.min_samples_leaf, seeds[0]
                    )
                ]
            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(fit_single_tree)(
                        X, residuals, self.colsample,
                        self.max_depth, self.min_samples_leaf, seeds[b]
                    )
                    for b in range(trees_this_block)
                )

            # ── Update F with all trees in the block ──
            for tree in results:
                self.trees_.append(tree)
                F += self._effective_lr * tree.predict(X)
                if X_val is not None:
                    F_val += self._effective_lr * tree.predict(X_val)

            elapsed = time.perf_counter() - t0
            self.block_times_.append(elapsed)

            # ── Track AUC ──
            train_auc = roc_auc_score(y, sigmoid(F))
            self.train_auc_.append(train_auc)

            if X_val is not None and y_val is not None:
                val_auc = roc_auc_score(y_val, sigmoid(F_val))
                self.val_auc_.append(val_auc)

            if self.verbose and (block_idx % 10 == 0 or block_idx == n_blocks - 1):
                val_str = f" | val_auc={self.val_auc_[-1]:.5f}" if self.val_auc_ else ""
                print(f"Block {block_idx+1:>4}/{n_blocks} | "
                      f"trees={len(self.trees_):>4} | "
                      f"train_auc={train_auc:.5f}{val_str} | "
                      f"time={elapsed:.3f}s")

        return self

    def predict_proba(self, X):
        """Return probability of positive class."""
        X = np.asarray(X, dtype=np.float32)
        F = np.full(len(X), self.F0_, dtype=np.float64)
        for tree in self.trees_:
            F += self._effective_lr * tree.predict(X)
        return sigmoid(F)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def summary(self):
        print("\n=== Training Summary ===")
        print(f"Total trees built : {len(self.trees_)}")
        print(f"Total blocks      : {len(self.block_times_)}")
        print(f"Total train time  : {sum(self.block_times_):.2f}s")
        print(f"Avg time/block    : {np.mean(self.block_times_):.3f}s")
        if self.train_auc_:
            print(f"Final train AUC   : {self.train_auc_[-1]:.5f}")
        if self.val_auc_:
            print(f"Final val AUC     : {self.val_auc_[-1]:.5f}")
            print(f"Best val AUC      : {max(self.val_auc_):.5f} "
                  f"(block {np.argmax(self.val_auc_)+1})")


# ─────────────────────────────────────────────
#  Baseline: Sequential GBM (B=1, no colsample)
# ─────────────────────────────────────────────

class SequentialGBM(BlockParallelGBM):
    """
    Standard sequential GBM. Identical to BlockParallelGBM but with
    block_size=1 and colsample=1.0 for a clean apples-to-apples baseline.
    """
    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=20, random_state=42, verbose=True):
        super().__init__(
            n_estimators=n_estimators,
            block_size=1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            colsample=1.0,       # all features, no subsampling
            auto_scale_lr=False,
            n_jobs=1,
            random_state=random_state,
            verbose=verbose
        )


# ─────────────────────────────────────────────
#  Main: Santander on Kaggle
# ─────────────────────────────────────────────

if __name__ == "__main__":

    TRAIN_PATH = "/kaggle/input/santander-customer-transaction-prediction/train.csv"
    N_ESTIMATORS = 300   # use 50 for a quick sanity check (~3 min)
    BLOCK_SIZE = 2       # safe default; try 3 or 4 for more speedup

    print("Loading Santander dataset...")
    df = pd.read_csv(TRAIN_PATH)
    df = df.drop(columns=["ID_code"])

    y = df["target"].values
    X = df.drop(columns=["target"]).values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | "
          f"Positive rate: {y_train.mean():.3f}\n")

    # ── Baseline: Sequential GBM ──
    print("=" * 65)
    print("BASELINE: Sequential GBM (B=1, all features)")
    print("=" * 65)
    baseline = SequentialGBM(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
        verbose=True
    )
    t0 = time.perf_counter()
    baseline.fit(X_train, y_train, X_val, y_val)
    baseline_time = time.perf_counter() - t0
    baseline.summary()

    # ── Block Parallel GBM ──
    print("\n" + "=" * 65)
    print(f"BLOCK PARALLEL GBM (B={BLOCK_SIZE}, colsample=0.5)")
    print("=" * 65)
    bp_model = BlockParallelGBM(
        n_estimators=N_ESTIMATORS,
        block_size=BLOCK_SIZE,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=20,
        colsample=0.5,
        auto_scale_lr=True,
        n_jobs=-1,
        random_state=42,
        verbose=True
    )
    t0 = time.perf_counter()
    bp_model.fit(X_train, y_train, X_val, y_val)
    bp_time = time.perf_counter() - t0
    bp_model.summary()

    # ── Comparison Table ──
    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<35} {'Baseline':>12} {f'Block (B={BLOCK_SIZE})':>12}")
    print("-" * 65)
    print(f"{'Total wall-clock time (s)':<35} {baseline_time:>12.2f} {bp_time:>12.2f}")
    print(f"{'Speedup':<35} {'1.00x':>12} {f'{baseline_time/bp_time:.2f}x':>12}")
    print(f"{'Best val AUC':<35} {max(baseline.val_auc_):>12.5f} {max(bp_model.val_auc_):>12.5f}")
    print(f"{'AUC gap (baseline - block)':<35} {'—':>12} "
          f"{max(baseline.val_auc_) - max(bp_model.val_auc_):>12.5f}")