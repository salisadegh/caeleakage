"""
Tests for caeleakage package.
Run: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from caeleakage import CAEClassifier, CAEPipeline, CAEResult


# ─────────────────────────────────────────────────────────────────
# CAEClassifier tests
# ─────────────────────────────────────────────────────────────────

def test_classify_lambda3():
    cae = CAEClassifier(
        dag_edges={('IHM', 'survival')},
        target='survival'
    )
    r = cae.classify('IHM')
    assert r.lambda_score == 3
    assert r.decision == 'remove'
    assert r.is_leaky is True


def test_classify_lambda1():
    cae = CAEClassifier(
        t0_features={'ECMO', 'CRRT'},
        target='survival'
    )
    r = cae.classify('ECMO')
    assert r.lambda_score == 1
    assert r.decision == 'retain_annotated'
    assert r.needs_sensitivity is True


def test_classify_lambda0():
    cae = CAEClassifier(target='survival')
    r = cae.classify('age')
    assert r.lambda_score == 0
    assert r.decision == 'retain'
    assert r.is_leaky is False


def test_classify_target_raises():
    cae = CAEClassifier(target='survival')
    with pytest.raises(ValueError):
        cae.classify('survival')


def test_classify_all():
    cae = CAEClassifier(
        dag_edges={('IHM', 'Y')},
        t0_features={'ECMO'},
        target='Y'
    )
    results = cae.classify_all(['age', 'IHM', 'ECMO'])
    assert results['IHM'].lambda_score == 3
    assert results['ECMO'].lambda_score == 1
    assert results['age'].lambda_score == 0


def test_fit_transform():
    cae = CAEClassifier(
        dag_edges={('leaky_feat', 'outcome')},
        target='outcome'
    )
    df = pd.DataFrame({
        'age':        [25, 30, 35],
        'leaky_feat': [1, 0, 1],
    })
    result = cae.fit_transform(df)
    assert 'leaky_feat' not in result.columns
    assert 'age' in result.columns


def test_summary_dataframe():
    cae = CAEClassifier(
        dag_edges={('IHM', 'Y')},
        t0_features={'ECMO'},
        target='Y'
    )
    cae.classify_all(['age', 'IHM', 'ECMO'])
    df = cae.summary()
    assert 'lambda' in df.columns
    assert len(df) == 3


def test_properties():
    cae = CAEClassifier(
        dag_edges={('IHM', 'Y')},
        t0_features={'ECMO'},
        target='Y'
    )
    cae.classify_all(['age', 'IHM', 'ECMO'])
    assert 'IHM' in cae.removed_features_
    assert 'ECMO' in cae.annotated_features_
    assert 'age' in cae.retained_features_


# ─────────────────────────────────────────────────────────────────
# CAEPipeline tests
# ─────────────────────────────────────────────────────────────────

def make_synthetic_data(n=100, n_features=10, random_state=42):
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y = (rng.randn(n) > 0).astype(int)
    return X, y


def test_pipeline_runs():
    X, y = make_synthetic_data()
    pipeline = CAEPipeline(n_outer=3, n_inner=2, n_bootstrap=100, verbose=False)
    results = pipeline.fit_evaluate(X.values, y)
    assert 0 <= results['AUC'] <= 1
    assert results['AUC_CI'][0] <= results['AUC'] <= results['AUC_CI'][1]


def test_pipeline_with_cae():
    X, y = make_synthetic_data()
    X['leaky'] = y.astype(float) + np.random.randn(len(y)) * 0.1
    cae = CAEClassifier(
        dag_edges={('leaky', 'outcome')},
        target='outcome'
    )
    pipeline = CAEPipeline(
        cae_classifier=cae,
        n_outer=3, n_inner=2, n_bootstrap=50, verbose=False
    )
    results = pipeline.fit_evaluate(X, y)
    assert 'leaky' in results['features_removed']


def test_smote_oversample():
    from caeleakage import smote_oversample
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = np.array([0]*40 + [1]*10)
    X_res, y_res = smote_oversample(X, y, k=3, random_state=42)
    assert (y_res == 0).sum() == (y_res == 1).sum()  # balanced
    assert len(X_res) == len(y_res)
