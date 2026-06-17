"""
caeleakage
==========
Causal Adjacency Examination (CAE): a formalised leakage-detection
procedure for clinical machine learning pipelines.

Quick start
-----------
>>> from caeleakage import CAEClassifier, CAEPipeline
>>>
>>> # 1. Define your causal DAG and prediction timepoint
>>> cae = CAEClassifier(
...     dag_edges={('In_Hospital_Mortality', 'One_Year_Survival')},
...     t0_features={'ECMO', 'CRRT', 'Furosemide', 'ICU_Time_Day'},
...     target='One_Year_Survival'
... )
>>>
>>> # 2. Classify all features
>>> cae.fit(X_df)
>>> cae.report()
>>>
>>> # 3. Run corrected nested CV pipeline
>>> pipeline = CAEPipeline(cae_classifier=cae)
>>> results = pipeline.fit_evaluate(X_df, y)
>>> print(results['AUC'], results['AUC_CI'])

Reference
---------
Sadegh-Zadeh, A. et al. (2025). Causal Adjacency Examination (CAE):
A Formalised Leakage-Detection Procedure for Clinical Machine Learning
Pipelines. Computer Methods and Programs in Biomedicine [under review].
https://github.com/salisadegh/caeleakage

Author
------
Dr. Ali Sadegh-Zadeh
Staffordshire University, UK
ali.sadegh-zadeh@staffs.ac.uk
"""

from .classifier import CAEClassifier, CAEResult
from .pipeline   import CAEPipeline, smote_oversample

__version__   = '0.1.0'
__author__    = 'Ali Sadegh-Zadeh'
__email__     = 'ali.sadegh-zadeh@staffs.ac.uk'
__license__   = 'MIT'

__all__ = [
    'CAEClassifier',
    'CAEResult',
    'CAEPipeline',
    'smote_oversample',
]
