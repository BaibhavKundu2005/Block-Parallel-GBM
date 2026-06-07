"""
CPU utilisation proof using the ACTUAL BlockParallelGBM implementation.

Kaggle-ready version.
Runtime: ~120-160 seconds on Kaggle free-tier CPUs.
"""

import os
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from joblib import Parallel, delayed

# ============================================================
# IMPORT YOUR MODEL
# ============================================================
# You must download the "block_parallel_gbm_kaggle.py" notebook and upload it as a dataset in Kaggle, replace the path accordingly.
import sys
sys.path.append("/kaggle/input/datasets/baibhavkundu/blockpgbm")

# If block_parallel_gbm_kaggle.py is uploaded directly
from block_parallel_gbm_kaggle import BlockParallelGBM

# ============================================================
# Shared worker log file
# ============================================================

LOG_FILE = "/kaggle/working/worker_log.txt"

open(LOG_FILE, "w").close()

# ============================================================
# Monkey-patch DecisionTreeRegressor.fit
# ============================================================

_original_fit = DecisionTreeRegressor.fit

def patched_fit(self, X, y, *args, **kwargs):

    pid = os.getpid()

    t0 = time.perf_counter()

    result = _original_fit(self, X, y, *args, **kwargs)

    elapsed = time.perf_counter() - t0

    with open(LOG_FILE, "a") as f:
        f.write(f"{pid},{round(elapsed,3)}\n")

    return result

DecisionTreeRegressor.fit = patched_fit

# ============================================================
# Dataset
# ============================================================

X, y = make_classification(
    n_samples=50000,
    n_features=200,
    n_informative=50,
    random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("=" * 60)
print("CPU UTILISATION PROOF — BlockParallelGBM")
print("=" * 60)

print(f"Available logical CPU cores: {os.cpu_count()}")

# ============================================================
# Helper
# ============================================================

def summarize_logs(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    records = []

    for line in lines:
        pid, elapsed = line.strip().split(",")

        records.append({
            "pid": int(pid),
            "time": float(elapsed)
        })

    for i, r in enumerate(records[:12]):
        print(
            f"Tree {i+1:2d} | "
            f"PID={r['pid']} | "
            f"time={r['time']}s"
        )

    pids = [r["pid"] for r in records]

    print("\nSummary")
    print(f"Total tree fits: {len(records)}")
    print(f"Unique PIDs:     {len(set(pids))}")

    print(f"PIDs observed:   {sorted(set(pids))}")

# ============================================================
# Sequential baseline
# ============================================================

open(LOG_FILE, "w").close()

baseline = BlockParallelGBM(
    n_estimators=16,
    block_size=1,
    colsample=1.0,
    auto_scale_lr=False,
    n_jobs=1,
    random_state=42,
    verbose=False
)

print("Training Sequential Baseline (B=1)")
t0 = time.perf_counter()

baseline.fit(X_tr, y_tr, X_val, y_val)

seq_time = time.perf_counter() - t0

summarize_logs("SEQUENTIAL BASELINE (B=1)")

print(f"\nWall-clock time: {seq_time:.3f}s")

# ============================================================
# Block-parallel
# ============================================================

open(LOG_FILE, "w").close()

parallel_model = BlockParallelGBM(
    n_estimators=16,
    block_size=4,
    colsample=0.5,
    auto_scale_lr=True,
    n_jobs=-1,
    random_state=42,
    verbose=False
)

print("Training Block Parallel (B=4)")
t0 = time.perf_counter()

parallel_model.fit(X_tr, y_tr, X_val, y_val)

par_time = time.perf_counter() - t0

summarize_logs("BLOCK-PARALLEL (B=4)")

print(f"\nWall-clock time: {par_time:.3f}s")

# ============================================================
# Final comparison
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"Sequential time:     {seq_time:.3f}s")
print(f"Block-parallel time: {par_time:.3f}s")

print(f"\nSpeedup: {seq_time / par_time:.2f}x")

print("\nInterpretation:")
print("- B=1 should show exactly one PID")
print("- B=4 should show multiple worker PIDs")
print("- Multiple PIDs confirms process-level parallelism")
print("- Reduced wall-clock time confirms simultaneous execution")
