# caeleakage

**Causal Adjacency Examination (CAE)** — a reproducible, DAG-guided leakage-risk classification procedure for clinical ML pipelines.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/salisadegh/caeleakage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Installation

> **Note:** PyPI publication is pending journal acceptance.
> Install directly from GitHub:

```bash
pip install git+https://github.com/salisadegh/caeleakage.git
```

Once published (after journal acceptance):
```bash
pip install caeleakage   # coming soon
```

---

## Two repositories

| Repository | Purpose |
|---|---|
| **[caeleakage](https://github.com/salisadegh/caeleakage)** | This repo — reusable Python package |
| **[cardiac-cae-ml](https://github.com/salisadegh/cardiac-cae-ml)** | Paper analysis code, figures, results |

---

## What is CAE?

CAE assigns each feature a leakage-risk score λ ∈ {0, 1, 2, 3}:

| λ | Class | Action |
|---|---|---|
| 0 | Safe baseline feature | Retain unconditionally |
| 1 | Deployment-dependent | Retain in Tier-1; exclude in conservative Tier-2 |
| 2 | Proxy feature | Flag for expert adjudication |
| 3 | Direct causal leakage | **Remove unconditionally** |

---

## Quick start

```python
from caeleakage import CAEClassifier, CAEPipeline

cae = CAEClassifier(
    dag_edges   = {('In_Hospital_Mortality', 'One_Year_Survival')},
    t0_features = {'ECMO', 'CRRT', 'Furosemide'},
    target      = 'One_Year_Survival'
)
cae.fit(X); cae.report()

pipeline = CAEPipeline(cae_classifier=cae)
results  = pipeline.fit_evaluate(X, y)
print(results['AUC'], results['AUC_CI'])
```

---

## Citation

```bibtex
@article{sadeghzadeh2025cae,
  title  = {CAE: A Reproducible, DAG-Guided Leakage-Risk Classification Procedure for Clinical ML Pipelines},
  author = {Sadegh-Zadeh, Seyed-Ali and others},
  journal= {Computer Methods and Programs in Biomedicine},
  year   = {2025},
  note   = {Under review},
  url    = {https://github.com/salisadegh/cardiac-cae-ml}
}
```

## License
MIT — see [LICENSE](LICENSE).
