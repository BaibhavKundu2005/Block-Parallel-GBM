# Block-Synchronous Gradient Boosting — Supplemental Material

Supplemental code and results for the paper:

> **Block-Synchronous Gradient Boosting: Characterising Across-Iteration
> Parallelism on Multi-Core CPUs**
> Anonymous submission — AutoML 2026

---

## Overview

This repository contains:
- The complete implementation of Block-Synchronous GBM
- All experiment scripts used to produce the results in the paper
- Raw CSV outputs from every experiment
- Figure generation scripts

---

## Requirements

Python 3.9 or later. Install all dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
joblib>=1.2.0
matplotlib>=3.6.0
```

All experiments were run on **Kaggle free tier** (4-core CPU, ~16GB RAM,
Python 3.12). No GPU is required. Total compute across all experiments:
approximately 60 CPU-hours.

---

## Repository Structure

```
supplemental/
├── LICENSE
├── README.md
├── requirements.txt
│
├── code/
│   ├── block_parallel_gbm.py              # Core model implementation
│   ├── block_parallel_gbm_experiments.py  # Santander + Covertype + Adult
│   ├── ieee_experiments.py               # IEEE-CIS experiments
│   ├── santander_multiseed.py             # Multi-seed stability analysis
│   ├── figure2_block_sweep.py             # Figure 2 generation
│   ├── figure3_equal_budget.py            # Figure 3 generation
│   ├── fig1_hparam_sensitivity.py         # Figure 1 generation
│   └── generate_result_csvs.py            # Reconstruct CSVs from raw numbers
│
└── results/
    ├── ablation_santander.csv
    ├── block_sweep_santander.csv
    ├── equal_budget_santander.csv
    ├── santander_multiseed_summary.csv
    ├── ablation_covertype.csv
    ├── block_sweep_covertype.csv
    ├── equal_budget_covertype.csv
    ├── hparam_sensitivity_covertype.csv
    ├── ablation_adult.csv
    ├── block_sweep_adult.csv
    ├── ablation_ieee_cis.csv
    ├── block_sweep_ieee_cis.csv
    └── equal_budget_ieee_cis.csv
```

---

## Reproducing Experiments

All experiments are run as Kaggle notebooks. Each script requires the
corresponding dataset to be added as a Kaggle input dataset before running.

### Minimal smoke test (no dataset required, runs in ~30 seconds)

```python
from code.block_parallel_gbm import BlockParallelGBM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                              random_state=42)

# Sequential baseline
baseline = BlockParallelGBM(n_estimators=50, block_size=1,
                             colsample=1.0, auto_scale_lr=False,
                             n_jobs=1, random_state=42)
baseline.fit(X_tr, y_tr, X_val, y_val)
print(f"Baseline AUC: {baseline.best_val_auc:.5f}")

# Block-synchronous B=2
b2 = BlockParallelGBM(n_estimators=50, block_size=2,
                       colsample=0.5, auto_scale_lr=True,
                       n_jobs=-1, random_state=42)
b2.fit(X_tr, y_tr, X_val, y_val)
print(f"B=2 AUC:      {b2.best_val_auc:.5f}")
```

---

### Santander — ablation, block sweep, equal budget

**Dataset:** Santander Customer Transaction Prediction
(kaggle.com/c/santander-customer-transaction-prediction)

1. Create a Kaggle notebook
2. Add the Santander dataset as input
3. Upload `block_parallel_gbm_experiments.py`
4. Run — outputs saved to `/kaggle/working/`

**Estimated runtime:** ~18 hours total across all configs.
Run in two sessions: ablation + block sweep in session 1,
equal budget in session 2.

---

### Santander — multi-seed stability (Table 4 in paper)

**Dataset:** Same Santander dataset as above.

1. Upload `santander_multiseed.py`
2. Set `SEEDS = [123]` at the top to run one seed per session
3. Run three separate sessions for seeds 123, 456, 789
4. Seed 42 result is hardcoded from the main experiment run

**Estimated runtime:** ~5.5 hours per seed.

---

### Covertype — ablation, block sweep, equal budget, hyperparameter sensitivity

**Dataset:** Downloaded automatically via `sklearn.datasets.fetch_covtype()`
— no Kaggle dataset needed.

1. Upload `block_parallel_gbm_experiments.py`
2. Run — Covertype experiments are included in the same script
3. Outputs saved to `/kaggle/working/`

**Estimated runtime:** ~2 hours total.

---

### UCI Adult — ablation, block sweep

**Dataset:** Downloaded automatically via `sklearn.datasets` — no Kaggle
dataset needed.

1. Same script as Covertype above
2. Adult experiments are included in `block_parallel_gbm_experiments.py`

**Estimated runtime:** ~5 minutes total.

---

### IEEE-CIS — ablation, block sweep, equal budget

**Dataset:** IEEE-CIS Fraud Detection
(kaggle.com/c/ieee-fraud-detection)

1. Create a Kaggle notebook
2. Add the IEEE-CIS dataset as input (transaction table only)
3. Upload `ieee_experiments.py`
4. Run — outputs saved to `/kaggle/working/`

**Note:** The script is named `ieee_experiments.py` for historical reasons
from the development process but runs on the IEEE-CIS dataset.
Set `DATA_PATH = "/kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv"`
at the top of the script.

**Estimated runtime:** ~4 hours total.

---

### Pre-flight profiler

Every experiment script includes a pre-flight profiling step that estimates
per-tree cost before committing to full training. The profiler fits 3 trees
and prints the overhead ratio rho and a go/no-go recommendation:

```
PRE-FLIGHT: PER-TREE COST PROFILING
Fitting 3 trees to estimate tau_tree ...
  Tree 1: 23.8s
  Tree 2: 24.1s
  Tree 3: 24.3s

  Average tau_tree   = 24.1s
  tau_overhead       = 0.3s  (fixed, 4 cores)
  Overhead ratio rho = 0.012

  Recommendation: STRONG GO — rho << 0.1.
```

If rho >= 1.0, the script hard-stops before running any expensive fits.

---

## Raw Results

All CSV files in `results/` correspond directly to tables in the paper:

| File | Paper table |
|---|---|
| `ablation_santander.csv` | Table 2 (main paper) |
| `ablation_covertype.csv` | Appendix A |
| `ablation_adult.csv` | Appendix A |
| `ablation_ieee_cis.csv` | Appendix A |
| `block_sweep_santander.csv` | Appendix B |
| `block_sweep_covertype.csv` | Appendix B |
| `block_sweep_adult.csv` | Appendix B |
| `block_sweep_ieee_cis.csv` | Appendix B |
| `equal_budget_santander.csv` | Table 3 (main paper) |
| `equal_budget_covertype.csv` | Table 3 (main paper) |
| `equal_budget_ieee_cis.csv` | Table 3 (main paper) |
| `hparam_sensitivity_covertype.csv` | Appendix C |
| `santander_multiseed_summary.csv` | Table 4 (main paper) |

---

## Generating Figures

All figures can be regenerated from the confirmed result numbers
without rerunning the full experiments:

```bash
python3 code/figure2_block_sweep.py       # Fig 2 — block sweep curves
python3 code/figure3_equal_budget.py      # Fig 3 — equal budget curves
python3 code/fig1_hparam_sensitivity.py   # Fig 1 — heatmaps (Appendix C)
```

Each script saves both a vector PDF and a 300 DPI PNG to the working directory.

---

## License

MIT License — see LICENSE file.

---

## Notes on Reproducibility

- All random seeds are fixed (`random_state=42` throughout unless varied
  intentionally in the multi-seed experiment)
- Wall-clock times will vary by hardware — Kaggle free tier CPUs vary
  in speed between sessions. The overhead ratio rho scales accordingly
  and the profiler will reflect the actual hardware speed
- The equal-budget experiment uses the baseline's actual measured total
  time as the budget for B=2, so timing variation across sessions does
  not affect the fairness of the comparison — both configs always get
  exactly the same wall-clock time on the same hardware in the same session
- AUC values are deterministic given fixed random seeds and will reproduce
  exactly regardless of hardware
