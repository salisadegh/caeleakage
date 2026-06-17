"""
caeleakage.pipeline
--------------------
Corrected clinical ML pipeline: CAE + nested CV + SMOTE-inside-fold.

Author: Dr. Ali Sadegh-Zadeh, Staffordshire University
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.svm import SVC

from .classifier import CAEClassifier


# ─────────────────────────────────────────────────────────────────
# SMOTE — corrected implementation
# ─────────────────────────────────────────────────────────────────

def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Minority Over-sampling Technique (Chawla et al., 2002).

    CRITICAL: This function must ONLY be called on inner-training folds.
    Applying SMOTE before cross-validation splits contaminates the
    evaluation and inflates reported AUC by 1.7–3.3 points
    (see Sadegh-Zadeh et al., 2025, Section 3.5).

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    k : int — nearest neighbours for synthesis
    random_state : int

    Returns
    -------
    X_resampled, y_resampled
    """
    rng = np.random.RandomState(random_state)
    X, y = np.array(X, dtype=float), np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    X_min = X[y == minority_class]
    n_synthetic = counts.max() - counts.min()
    synthetic = []
    for _ in range(n_synthetic):
        idx = rng.randint(0, len(X_min))
        sample = X_min[idx]
        dists = np.sqrt(((X_min - sample) ** 2).sum(axis=1))
        dists[idx] = np.inf
        k_eff = min(k, len(X_min) - 1)
        neighbours = X_min[np.argsort(dists)[:k_eff]]
        neighbour = neighbours[rng.randint(0, len(neighbours))]
        synthetic.append(sample + rng.rand() * (neighbour - sample))
    X_out = np.vstack([X, np.array(synthetic)])
    y_out = np.concatenate([y, np.full(n_synthetic, minority_class)])
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────
# Corrected nested CV pipeline
# ─────────────────────────────────────────────────────────────────

class CAEPipeline:
    """
    Corrected clinical ML pipeline combining CAE leakage detection
    with nested cross-validation and SMOTE-inside-fold oversampling.

    The pipeline enforces the CAE protocol:
    1. Features are classified and leaky features (λ=3) removed before any
       CV split.
    2. SMOTE is applied strictly inside each inner-training fold.
    3. Outer folds provide unbiased performance estimates.
    4. Bootstrap CIs quantify uncertainty.

    Parameters
    ----------
    estimator : sklearn estimator
        Any scikit-learn classifier with predict_proba support.
        Defaults to SVM (Linear) with balanced class weights.
    cae_classifier : CAEClassifier, optional
        Pre-fitted CAEClassifier instance. If None, no CAE filtering is applied
        (useful for comparing naive vs CAE-cleaned pipelines).
    n_outer : int
        Number of outer CV folds (default: 5).
    n_inner : int
        Number of inner CV folds for hyperparameter selection (default: 3).
    param_grid : list of dict, optional
        Hyperparameter grid for inner-fold selection.
    apply_smote : bool
        Whether to apply SMOTE inside inner training folds (default: True).
    smote_k : int
        SMOTE k nearest neighbours (default: 3).
    n_bootstrap : int
        Bootstrap resamples for 95% CI (default: 2000).
    random_state : int
        Random seed (default: 42).
    include_annotated : bool
        If True (default), λ=1 features are retained (full model).
        If False, only λ=0 features are used (conservative Tier-0).
    verbose : bool
        Print progress (default: True).

    Examples
    --------
    >>> from caeleakage import CAEClassifier, CAEPipeline
    >>> import pandas as pd
    >>>
    >>> cae = CAEClassifier(
    ...     dag_edges={('In_Hospital_Mortality', 'survived_1yr')},
    ...     t0_features={'ECMO', 'CRRT', 'Furosemide'},
    ...     target='survived_1yr'
    ... )
    >>> pipeline = CAEPipeline(cae_classifier=cae, n_outer=5, n_inner=3)
    >>> results = pipeline.fit_evaluate(X_df, y)
    >>> print(results['AUC'], results['AUC_CI'])
    """

    def __init__(
        self,
        estimator=None,
        cae_classifier: Optional[CAEClassifier] = None,
        n_outer: int = 5,
        n_inner: int = 3,
        param_grid: Optional[List[dict]] = None,
        apply_smote: bool = True,
        smote_k: int = 3,
        n_bootstrap: int = 2000,
        random_state: int = 42,
        include_annotated: bool = True,
        verbose: bool = True,
    ):
        if estimator is None:
            estimator = SVC(
                kernel='linear', C=1.0, probability=True,
                class_weight='balanced', random_state=random_state
            )
        self.estimator       = estimator
        self.cae_classifier  = cae_classifier
        self.n_outer         = n_outer
        self.n_inner         = n_inner
        self.param_grid      = param_grid or [{}]
        self.apply_smote     = apply_smote
        self.smote_k         = smote_k
        self.n_bootstrap     = n_bootstrap
        self.random_state    = random_state
        self.include_annotated = include_annotated
        self.verbose         = verbose
        self._y_true = None
        self._y_prob = None

    def _get_prob(self, clf, X):
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X)[:, 1]
        d = clf.decision_function(X)
        return (d - d.min()) / (d.max() - d.min() + 1e-8)

    def _inner_loop(self, X_tr, y_tr, fold_seed):
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner, shuffle=True, random_state=fold_seed
        )
        best_auc, best_params = -1, self.param_grid[0]
        for params in self.param_grid:
            aucs = []
            for itr, ival in inner_cv.split(X_tr, y_tr):
                Xi, yi = X_tr[itr], y_tr[itr]
                if self.apply_smote and yi.sum() >= 3 and (yi == 0).sum() >= 3:
                    k = min(self.smote_k, max(1, int((yi == 0).sum()) - 1))
                    Xi, yi = smote_oversample(Xi, yi, k=k, random_state=fold_seed)
                sc = StandardScaler()
                clf = clone(self.estimator)
                try:
                    clf.set_params(**params)
                except Exception:
                    pass
                clf.fit(sc.fit_transform(Xi), yi)
                Xv = sc.transform(X_tr[ival])
                if len(np.unique(y_tr[ival])) > 1:
                    aucs.append(roc_auc_score(y_tr[ival], self._get_prob(clf, Xv)))
            if aucs and np.mean(aucs) > best_auc:
                best_auc = np.mean(aucs)
                best_params = params
        return best_params

    def fit_evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
    ) -> Dict:
        """
        Run full CAE + nested CV pipeline and return evaluation metrics.

        Parameters
        ----------
        X : DataFrame or ndarray
        y : array-like binary labels

        Returns
        -------
        dict with keys: AUC, AUC_CI, Brier, y_true, y_prob,
                        features_used, features_removed
        """
        # Apply CAE filtering
        features_removed = []
        features_used = None
        if self.cae_classifier is not None:
            if isinstance(X, pd.DataFrame):
                if not self.cae_classifier._results:
                    self.cae_classifier.fit(X)
                X_filtered = self.cae_classifier.transform(
                    X, include_annotated=self.include_annotated
                )
                features_removed = self.cae_classifier.removed_features_
                features_used = list(X_filtered.columns)
                X_arr = X_filtered.values.astype(float)
            else:
                warnings.warn(
                    "Pass a DataFrame for CAE filtering to work correctly.",
                    UserWarning
                )
                X_arr = np.array(X, dtype=float)
        else:
            X_arr = np.array(X, dtype=float) if not isinstance(X, np.ndarray) \
                    else X.astype(float)

        y_arr = np.array(y)
        outer_cv = StratifiedKFold(
            n_splits=self.n_outer, shuffle=True, random_state=self.random_state
        )
        y_true_all, y_prob_all = [], []

        for fold_idx, (tr, te) in enumerate(outer_cv.split(X_arr, y_arr)):
            X_tr, X_te = X_arr[tr], X_arr[te]
            y_tr, y_te = y_arr[tr], y_arr[te]

            best_params = self._inner_loop(X_tr, y_tr, fold_seed=fold_idx)

            # Retrain on full outer training fold
            if self.apply_smote and y_tr.sum() >= 3 and (y_tr == 0).sum() >= 3:
                k = min(self.smote_k, max(1, int((y_tr == 0).sum()) - 1))
                X_sm, y_sm = smote_oversample(
                    X_tr, y_tr, k=k, random_state=fold_idx
                )
            else:
                X_sm, y_sm = X_tr, y_tr

            sc = StandardScaler()
            clf = clone(self.estimator)
            try:
                clf.set_params(**best_params)
            except Exception:
                pass
            clf.fit(sc.fit_transform(X_sm), y_sm)
            y_prob = self._get_prob(clf, sc.transform(X_te))
            y_true_all.extend(y_te)
            y_prob_all.extend(y_prob)
            if self.verbose:
                fold_auc = roc_auc_score(y_te, y_prob) \
                    if len(np.unique(y_te)) > 1 else float('nan')
                print(f"  Fold {fold_idx+1}/{self.n_outer}  AUC={fold_auc:.4f}")

        self._y_true = np.array(y_true_all)
        self._y_prob = np.array(y_prob_all)

        auc   = round(roc_auc_score(self._y_true, self._y_prob), 4)
        brier = round(brier_score_loss(self._y_true, self._y_prob), 4)
        ci_lo, ci_hi = self._bootstrap_ci()

        if self.verbose:
            print(f"\n  AUC = {auc:.4f}  95% CI [{ci_lo}–{ci_hi}]")
            print(f"  Brier = {brier:.4f}")
            if features_removed:
                print(f"  CAE removed: {features_removed}")

        return {
            'AUC':              auc,
            'AUC_CI':           (ci_lo, ci_hi),
            'AUC_CI_lo':        ci_lo,
            'AUC_CI_hi':        ci_hi,
            'Brier':            brier,
            'y_true':           self._y_true,
            'y_prob':           self._y_prob,
            'features_used':    features_used,
            'features_removed': features_removed,
        }

    def _bootstrap_ci(self, alpha: float = 0.05) -> Tuple[float, float]:
        rng = np.random.RandomState(self.random_state)
        scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(self._y_true), len(self._y_true), replace=True)
            if len(np.unique(self._y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(self._y_true[idx], self._y_prob[idx]))
        return (round(np.percentile(scores, 100 * alpha / 2), 4),
                round(np.percentile(scores, 100 * (1 - alpha / 2)), 4))
