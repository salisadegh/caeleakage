# caeleakage

**Causal Adjacency Examination (CAE)** — a formalised leakage-detection procedure for clinical machine learning pipelines.

[![PyPI version](https://img.shields.io/pypi/v/caeleakage.svg)](https://pypi.org/project/caeleakage/)
[![Python](https://img.shields.io/pypi/pyversions/caeleakage.svg)](https://pypi.org/project/caeleakage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is CAE?

Data leakage is one of the most common causes of inflated performance in clinical ML. CAE is a **three-question decision algorithm** that classifies every feature in your pipeline by its causal and temporal relationship to the outcome, assigning a leakage-risk score λ ∈ {0, 1, 2, 3}.

| Score | Meaning | Action |
|---|---|---|
| λ=0 | Pre-T₀, non-causal | Retain unconditionally |
| λ=1 | Available at T₀ (discharge-time) | Retain + sensitivity analysis |
| λ=2 | Indirect proxy | Flag + sensitivity analysis |
| λ=3 | Direct causal edge to outcome | **Remove unconditionally** |

The algorithm runs in **O(d × |E|)** time and is deterministic given the same DAG and prediction timepoint T₀.

---

## Installation

```bash
pip install caeleakage
```

---

## Quick Start

```python
from caeleakage import CAEClassifier, CAEPipeline

# 1. Define your causal DAG and prediction timepoint T₀
cae = CAEClassifier(
    dag_edges={
        ('In_Hospital_Mortality', 'One_Year_Survival'),  # λ=3 → remove
    },
    t0_features={
        'ECMO', 'CRRT', 'Furosemide', 'ICU_Time_Day',   # λ=1 → annotate
        'CVP_Third_Day', 'RV_Dysfunction',
    },
    target='One_Year_Survival'
)

# 2. Classify features and print report
import pandas as pd
X = pd.read_csv('your_data.csv')
y = X.pop('One_Year_Survival').values

cae.fit(X)
cae.report()
# ═════════════════════════════════════════════════════════════════
# CAE CLASSIFICATION REPORT
# Target: One_Year_Survival
# ═════════════════════════════════════════════════════════════════
# Feature                             λ    Decision
# -----------------------------------------------------------------
# Age                                 0    retain
# Albumin                             0    retain
# ...
# ECMO                                1    retain_annotated
# Furosemide                          1    retain_annotated
# In_Hospital_Mortality               3    remove
# ...

# 3. Run corrected nested CV pipeline (SMOTE inside each inner fold)
pipeline = CAEPipeline(
    cae_classifier=cae,
    n_outer=5,
    n_inner=3,
    n_bootstrap=2000,
)
results = pipeline.fit_evaluate(X, y)
print(f"AUC = {results['AUC']}  95% CI {results['AUC_CI']}")
print(f"Features removed: {results['features_removed']}")
```

---

## CAE on UCI Heart Failure Dataset

```python
from caeleakage import CAEClassifier
import pandas as pd

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
y  = df.pop('DEATH_EVENT').values

# "time" = follow-up duration → directly encodes survival → λ=3
cae = CAEClassifier(
    dag_edges={('time', 'DEATH_EVENT')},
    target='DEATH_EVENT'
)
cae.fit(df)
cae.report()
# time → λ=3 → REMOVE
# All other 11 features → λ=0 → retain

# Naive AUC (time included):    ~0.865
# CAE-cleaned AUC (time removed): ~0.742
# Inflation detected: +0.118 AUC points
```

---

## Why corrected SMOTE matters

A common error is applying SMOTE **before** cross-validation splits:

```python
# ❌ WRONG — contaminates evaluation
X_res, y_res = SMOTE().fit_resample(X, y)
cross_val_score(clf, X_res, y_res, cv=5)  # inflated by 1.7–3.3 AUC pts

# ✅ CORRECT — CAEPipeline applies SMOTE inside each inner fold only
pipeline = CAEPipeline(apply_smote=True)
results  = pipeline.fit_evaluate(X_df, y)
```

---

## Citation

If you use `caeleakage` in your research, please cite:

```bibtex
@article{sadeghzadeh2025cae,
  title   = {Causal Adjacency Examination (CAE): A Formalised Leakage-Detection
             Procedure for Clinical Machine Learning Pipelines},
  author  = {Sadegh-Zadeh, Seyed-Ali and others},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/salisadegh/caeleakage}
}
```

---

## Author

**Dr. Ali Sadegh-Zadeh**  
School of Digital Technologies and Arts, Staffordshire University, UK  
ali.sadegh-zadeh@staffs.ac.uk  
https://github.com/salisadegh

---

## License

MIT License.
