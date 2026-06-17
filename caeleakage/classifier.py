"""
caeleakage.classifier
---------------------
Causal Adjacency Examination (CAE) — core feature classification algorithm.

Reference:
    Sadegh-Zadeh, A. et al. (2025). Causal Adjacency Examination (CAE):
    A Formalised Leakage-Detection Procedure for Clinical Machine Learning
    Pipelines. Computer Methods and Programs in Biomedicine [under review].

Author: Dr. Ali Sadegh-Zadeh, Staffordshire University
GitHub: https://github.com/salisadegh/caeleakage
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import warnings


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class CAEResult:
    """
    CAE classification result for a single feature.

    Attributes
    ----------
    feature : str
        Feature name.
    lambda_score : int
        Leakage-risk score: 0 (safe), 1 (discharge-time), 2 (indirect proxy), 3 (direct cause).
    decision : str
        One of 'retain', 'retain_annotated', 'flag_proxy', 'remove'.
    q1_direct_cause : bool
        True if a direct causal edge feature → target exists in the DAG.
    q2_at_t0 : bool
        True if the feature is first observed at or after prediction timepoint T₀.
    q3_indirect_proxy : bool
        True if the feature is an indirect proxy for the outcome.
    reason : str
        Human-readable justification.
    """
    feature: str
    lambda_score: int
    decision: str
    q1_direct_cause: bool
    q2_at_t0: bool
    q3_indirect_proxy: bool = False
    reason: str = ""

    @property
    def is_leaky(self) -> bool:
        """True if the feature should be removed (λ=3)."""
        return self.lambda_score == 3

    @property
    def needs_sensitivity(self) -> bool:
        """True if the feature requires sensitivity analysis annotation (λ=1 or λ=2)."""
        return self.lambda_score in (1, 2)

    def __repr__(self) -> str:
        return (f"CAEResult(feature={self.feature!r}, "
                f"λ={self.lambda_score}, decision={self.decision!r})")


# ─────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────

class CAEClassifier:
    """
    Causal Adjacency Examination (CAE) procedure.

    Classifies each feature in a supervised clinical ML pipeline
    according to its temporal and causal relationship to the outcome,
    assigning a leakage-risk score λ ∈ {0, 1, 2, 3}.

    Decision rules
    --------------
    Q1 — Direct causal edge exists (X → Y in DAG)?
         Yes → λ=3, decision='remove'

    Q2 — Feature observed at or after prediction timepoint T₀?
         Yes → λ=1, decision='retain_annotated'

    Q3 — Feature is an indirect causal proxy (user-specified)?
         Yes → λ=2, decision='flag_proxy'

    Default → λ=0, decision='retain'

    Complexity: O(d × |E|) time, deterministic given G and T₀.

    Parameters
    ----------
    dag_edges : set of (str, str), optional
        Directed edges (cause, effect) in the causal DAG.
        Include ALL domain-knowledge edges, not only those to the target.
    t0_features : set of str, optional
        Features whose first observation time is at or after T₀
        (e.g. discharge-time variables in a post-operative study).
    proxy_features : set of str, optional
        Features that are indirect causal proxies for the outcome
        (assigned λ=2). User must specify these explicitly.
    target : str
        Name of the outcome variable Y.

    Examples
    --------
    >>> cae = CAEClassifier(
    ...     dag_edges={('In_Hospital_Mortality', 'One_Year_Survival')},
    ...     t0_features={'ECMO', 'CRRT', 'Furosemide', 'ICU_Time_Day'},
    ...     target='One_Year_Survival'
    ... )
    >>> result = cae.classify('In_Hospital_Mortality')
    >>> result.lambda_score
    3
    >>> result = cae.classify('Age')
    >>> result.lambda_score
    0
    """

    def __init__(
        self,
        dag_edges: Optional[Set[Tuple[str, str]]] = None,
        t0_features: Optional[Set[str]] = None,
        proxy_features: Optional[Set[str]] = None,
        target: str = 'outcome',
    ):
        self.dag_edges      = set(dag_edges)      if dag_edges      else set()
        self.t0_features    = set(t0_features)    if t0_features    else set()
        self.proxy_features = set(proxy_features) if proxy_features else set()
        self.target         = target
        self._results: Dict[str, CAEResult] = {}

    # ── internal helpers ──────────────────────────────────────────

    def _q1(self, feature: str) -> bool:
        return (feature, self.target) in self.dag_edges

    def _q2(self, feature: str) -> bool:
        return feature in self.t0_features

    def _q3(self, feature: str) -> bool:
        return feature in self.proxy_features

    # ── public API ────────────────────────────────────────────────

    def classify(self, feature: str) -> CAEResult:
        """
        Classify a single feature.

        Parameters
        ----------
        feature : str
            Feature name to classify.

        Returns
        -------
        CAEResult
        """
        if feature == self.target:
            raise ValueError(
                f"Cannot classify the target variable: {feature!r}"
            )

        q1 = self._q1(feature)
        q2 = self._q2(feature)
        q3 = self._q3(feature)

        if q1:
            result = CAEResult(
                feature=feature, lambda_score=3, decision='remove',
                q1_direct_cause=True, q2_at_t0=q2, q3_indirect_proxy=q3,
                reason=(
                    f"Direct causal edge {feature!r} → {self.target!r} "
                    f"exists in DAG. Unconditional removal to prevent "
                    f"outcome leakage."
                )
            )
        elif q2:
            result = CAEResult(
                feature=feature, lambda_score=1, decision='retain_annotated',
                q1_direct_cause=False, q2_at_t0=True, q3_indirect_proxy=q3,
                reason=(
                    f"{feature!r} is first observed at or after T₀. "
                    f"Retained but must be included in sensitivity analysis."
                )
            )
        elif q3:
            result = CAEResult(
                feature=feature, lambda_score=2, decision='flag_proxy',
                q1_direct_cause=False, q2_at_t0=False, q3_indirect_proxy=True,
                reason=(
                    f"{feature!r} is flagged as an indirect proxy. "
                    f"Retained with mandatory sensitivity annotation."
                )
            )
        else:
            result = CAEResult(
                feature=feature, lambda_score=0, decision='retain',
                q1_direct_cause=False, q2_at_t0=False, q3_indirect_proxy=False,
                reason=(
                    f"{feature!r} is a pre-T₀ non-causal feature. "
                    f"Retained unconditionally."
                )
            )

        self._results[feature] = result
        return result

    def classify_all(self, features: List[str]) -> Dict[str, CAEResult]:
        """
        Classify a list of features.

        Parameters
        ----------
        features : list of str

        Returns
        -------
        dict mapping feature name → CAEResult
        """
        for f in features:
            self.classify(f)
        return dict(self._results)

    def fit(self, X: pd.DataFrame) -> 'CAEClassifier':
        """
        Classify all columns in a DataFrame (excluding the target if present).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        self
        """
        features = [c for c in X.columns if c != self.target]
        self.classify_all(features)
        return self

    def transform(self, X: pd.DataFrame,
                  include_annotated: bool = True) -> pd.DataFrame:
        """
        Remove λ=3 features and optionally drop λ=1/λ=2 features.

        Parameters
        ----------
        X : pd.DataFrame
        include_annotated : bool
            If True (default), λ=1 and λ=2 features are retained.
            Set to False to also remove λ=1 and λ=2 features (conservative).

        Returns
        -------
        pd.DataFrame — filtered feature matrix
        """
        if not self._results:
            warnings.warn(
                "CAEClassifier has not been fitted. Call fit() or "
                "classify_all() first.", UserWarning
            )
            return X

        drop = self.removed_features_
        if not include_annotated:
            drop = drop + self.annotated_features_

        cols_to_keep = [c for c in X.columns if c not in drop]
        return X[cols_to_keep]

    def fit_transform(self, X: pd.DataFrame,
                      include_annotated: bool = True) -> pd.DataFrame:
        """Fit then transform in one step."""
        return self.fit(X).transform(X, include_annotated=include_annotated)

    # ── properties ────────────────────────────────────────────────

    @property
    def retained_features_(self) -> List[str]:
        """Features with λ=0 (unconditional retain)."""
        return [f for f, r in self._results.items() if r.lambda_score == 0]

    @property
    def annotated_features_(self) -> List[str]:
        """Features with λ=1 or λ=2 (retain with sensitivity annotation)."""
        return [f for f, r in self._results.items() if r.lambda_score in (1, 2)]

    @property
    def removed_features_(self) -> List[str]:
        """Features with λ=3 (unconditional removal)."""
        return [f for f, r in self._results.items() if r.lambda_score == 3]

    # ── reporting ─────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of all classified features.

        Returns
        -------
        pd.DataFrame with columns:
            Feature, lambda, decision, Q1_direct_cause, Q2_at_T0,
            Q3_proxy, reason
        """
        rows = []
        for f, r in self._results.items():
            rows.append({
                'Feature':         f,
                'lambda':          r.lambda_score,
                'decision':        r.decision,
                'Q1_direct_cause': r.q1_direct_cause,
                'Q2_at_T0':        r.q2_at_t0,
                'Q3_proxy':        r.q3_indirect_proxy,
                'reason':          r.reason,
            })
        df = pd.DataFrame(rows).sort_values('lambda')
        return df.reset_index(drop=True)

    def report(self) -> str:
        """
        Print a human-readable CAE classification report.

        Returns
        -------
        str
        """
        if not self._results:
            return "CAEClassifier: no features classified yet."

        lines = [
            "=" * 65,
            "CAE CLASSIFICATION REPORT",
            f"Target: {self.target}",
            "=" * 65,
            f"{'Feature':<35} {'λ':<4} {'Decision'}",
            "-" * 65,
        ]
        for _, r in sorted(self._results.items(),
                            key=lambda x: x[1].lambda_score):
            lines.append(f"{r.feature:<35} {r.lambda_score:<4} {r.decision}")

        lines += [
            "-" * 65,
            f"λ=0 retain:            {len(self.retained_features_):>3} features",
            f"λ=1/2 annotated:       {len(self.annotated_features_):>3} features",
            f"λ=3 removed (leaky):   {len(self.removed_features_):>3} features",
            "=" * 65,
        ]
        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    def __repr__(self) -> str:
        n = len(self._results)
        return (f"CAEClassifier(target={self.target!r}, "
                f"classified={n}, "
                f"removed={len(self.removed_features_)})")
