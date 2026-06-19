"""Tests for caeleakage. Run: pytest tests/ -v"""
import numpy as np
import pandas as pd
import pytest
from caeleakage import CAEClassifier, CAEPipeline, CAEResult, smote_oversample


# ── CAEClassifier ───────────────────────────────────────────────
def test_lambda3_direct_edge():
    cae = CAEClassifier(dag_edges={('IHM', 'Y')}, target='Y')
    r = cae.classify('IHM')
    assert r.lambda_score == 3
    assert r.decision == 'remove'
    assert r.is_leaky is True

def test_lambda1_deployment_dependent():
    cae = CAEClassifier(t0_features={'ECMO'}, target='Y')
    r = cae.classify('ECMO')
    assert r.lambda_score == 1
    assert r.decision == 'retain_annotated'
    assert r.needs_sensitivity is True

def test_lambda2_proxy():
    cae = CAEClassifier(proxy_features={'Furosemide'}, target='Y')
    r = cae.classify('Furosemide')
    assert r.lambda_score == 2
    assert r.decision == 'flag'

def test_lambda0_safe():
    cae = CAEClassifier(target='Y')
    r = cae.classify('Age')
    assert r.lambda_score == 0
    assert r.decision == 'retain'
    assert r.is_leaky is False

def test_target_raises():
    cae = CAEClassifier(target='Y')
    with pytest.raises(ValueError):
        cae.classify('Y')

def test_classify_all():
    cae = CAEClassifier(dag_edges={('IHM', 'Y')}, t0_features={'ECMO'}, target='Y')
    res = cae.classify_all(['Age', 'IHM', 'ECMO'])
    assert res['IHM'].lambda_score == 3
    assert res['ECMO'].lambda_score == 1
    assert res['Age'].lambda_score == 0

def test_fit_transform_removes_leaky():
    cae = CAEClassifier(dag_edges={('leaky', 'Y')}, target='Y')
    df = pd.DataFrame({'Age': [1, 2, 3], 'leaky': [0, 1, 0]})
    out = cae.fit_transform(df)
    assert 'leaky' not in out.columns
    assert 'Age' in out.columns

def test_summary_dataframe():
    cae = CAEClassifier(dag_edges={('IHM', 'Y')}, t0_features={'ECMO'}, target='Y')
    cae.classify_all(['Age', 'IHM', 'ECMO'])
    df = cae.summary()
    assert 'lambda' in df.columns
    assert len(df) == 3

def test_properties():
    cae = CAEClassifier(dag_edges={('IHM', 'Y')}, t0_features={'ECMO'}, target='Y')
    cae.classify_all(['Age', 'IHM', 'ECMO'])
    assert 'IHM' in cae.removed_features_
    assert 'ECMO' in cae.annotated_features_
    assert 'Age' in cae.retained_features_

# ── CAEPipeline ─────────────────────────────────────────────────
def make_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 5), columns=[f'f{i}' for i in range(5)])
    y = (rng.randn(n) > 0).astype(int)
    return X, y

def test_pipeline_runs():
    X, y = make_data()
    pipeline = CAEPipeline(n_outer=3, n_inner=2, n_bootstrap=50, verbose=False)
    results = pipeline.fit_evaluate(X.values, y)
    assert 0 <= results['AUC'] <= 1
    assert results['AUC_CI'][0] <= results['AUC'] <= results['AUC_CI'][1]

def test_pipeline_removes_leaky():
    X, y = make_data()
    X['leaky'] = y.astype(float) + np.random.randn(len(y)) * 0.1
    cae = CAEClassifier(dag_edges={('leaky', 'outcome')}, target='outcome')
    pipeline = CAEPipeline(cae_classifier=cae, n_outer=3, n_inner=2, n_bootstrap=50, verbose=False)
    results = pipeline.fit_evaluate(X, y)
    assert 'leaky' in results['features_removed']

def test_smote_balances():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = np.array([0]*40 + [1]*10)
    X_res, y_res = smote_oversample(X, y, k=3, random_state=42)
    assert (y_res == 0).sum() == (y_res == 1).sum()
