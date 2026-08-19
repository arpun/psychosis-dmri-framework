"""
Unit tests for diffusion model ablation analysis module (src/ablation.py).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.ablation import (
    get_ablation_subsets,
    compute_ece,
    bootstrap_metric_ci,
    build_nn_classifier,
    run_nested_cv_ablation,
    compare_ablation_models,
    set_seed
)


@pytest.fixture
def mock_feature_list():
    """Create a realistic list of 413 mock feature column names."""
    regions = [f"region_{i}" for i in range(68)]
    features = []
    # Add 68 of each
    for suffix in ['_FA', '_MD', '_MK', '_KFA', '_FW', '_PF']:
        for r in regions:
            features.append(f"harm_lh_{r}{suffix}")
    # Add the extra 5 regions (rh_insula) matching the dataset
    for suffix in ['_FA', '_MK', '_KFA', '_FW', '_PF']:
        features.append(f"harm_rh_insula{suffix}")
    return features


@pytest.fixture
def mock_dataset(mock_feature_list):
    """Create a synthetic dataset mimicking final data."""
    set_seed(41)
    n_samples = 120
    data = {}

    # Synthetic features
    for f in mock_feature_list:
        data[f] = np.random.randn(n_samples)

    # Diagnostic columns
    # 20 patients, 100 controls
    diag_pvc = ['Patient'] * 30 + ['Control'] * 90
    diag_scz = ['SCZ'] * 15 + ['Non SCZ'] * 15 + ['Remove'] * 90
    sites = (['SiteA', 'SiteB', 'SiteC'] * 40)[:n_samples]

    data['diag_pvc'] = diag_pvc
    data['diag_scz'] = diag_scz
    data['scan_site_text'] = sites

    # Add signal to MK and MD
    for f in mock_feature_list:
        if f.endswith('_MK') or f.endswith('_MD'):
            # Patient shift
            data[f][:30] += 0.8

    df = pd.DataFrame(data)
    return df


def test_feature_subsets_partitioning(mock_feature_list):
    """Verify that all 7 ablation configurations partition features correctly."""
    subsets = get_ablation_subsets(mock_feature_list)
    assert 'DTI-only' in subsets
    assert 'DKI-only' in subsets
    assert 'DTI+DKI' in subsets
    assert 'IVIM-only' in subsets
    assert 'FWI-only' in subsets
    assert 'IVIM+FWI' in subsets
    assert 'Full Integrated' in subsets

    # Check that DTI features only have FA and MD
    for f in subsets['DTI-only']:
        assert f.endswith(('_FA', '_MD'))

    # Check that DKI features only have MK and KFA
    for f in subsets['DKI-only']:
        assert f.endswith(('_MK', '_KFA'))

    # Check that IVIM features only have PF
    for f in subsets['IVIM-only']:
        assert f.endswith('_PF')

    # Check that FWI features only have FW
    for f in subsets['FWI-only']:
        assert f.endswith('_FW')

    # Check that Full Integrated has all features
    assert len(subsets['Full Integrated']) == len(mock_feature_list)

    # Check union sizes
    assert len(subsets['DTI+DKI']) == len(subsets['DTI-only']) + len(subsets['DKI-only'])
    assert len(subsets['IVIM+FWI']) == len(subsets['IVIM-only']) + len(subsets['FWI-only'])


def test_compute_ece():
    """Test Expected Calibration Error calculation."""
    # Perfectly calibrated case
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert isinstance(ece, float)
    assert ece >= 0.0
    assert ece <= 1.0


def test_bootstrap_metric_ci():
    """Test bootstrap confidence interval function."""
    y_true = np.array([0]*50 + [1]*50)
    y_pred = np.array([0.1]*45 + [0.8]*5 + [0.2]*5 + [0.9]*45)

    point, low, high, samples = bootstrap_metric_ci(y_true, y_pred, roc_auc_score, n_iter=100)
    assert isinstance(point, float)
    assert isinstance(low, float)
    assert isinstance(high, float)
    assert low <= point <= high
    assert len(samples) > 0


def test_build_nn_classifier():
    """Test NN model creation for both tasks."""
    model_pvc = build_nn_classifier(input_dim=15, task='pvc')
    assert model_pvc.input_shape[-1] == 15
    assert model_pvc.output_shape[-1] == 1

    model_scz = build_nn_classifier(input_dim=10, task='scz')
    assert model_scz.input_shape[-1] == 10
    assert model_scz.output_shape[-1] == 1


def test_nested_cv_execution(mock_dataset, mock_feature_list):
    """Test end-to-end nested CV execution on synthetic dataset."""
    subsets = get_ablation_subsets(mock_feature_list)

    res = run_nested_cv_ablation(
        df=mock_dataset,
        feature_subset=subsets['DTI-only'],
        diag_col='diag_pvc',
        pos_label='Patient',
        task_name='pvc',
        top_k=5,
        k_folds=3,
        seed=42
    )

    assert 'nn' in res
    assert 'rf' in res
    assert len(res['y_true']) == len(mock_dataset)
    assert len(res['y_pred_nn']) == len(mock_dataset)
    assert 0.0 <= res['nn']['pooled_auc'] <= 1.0
    assert 0.0 <= res['nn']['pooled_ap'] <= 1.0
    assert res['nn']['auc_ci_lower'] <= res['nn']['pooled_auc'] <= res['nn']['auc_ci_upper']
    assert len(res['selected_features']) > 0


def test_statistical_model_comparison(mock_dataset, mock_feature_list):
    """Test statistical comparison function across ablation results."""
    subsets = get_ablation_subsets(mock_feature_list)
    results = {}

    for name in ['DTI-only', 'Full Integrated']:
        results[name] = run_nested_cv_ablation(
            df=mock_dataset,
            feature_subset=subsets[name],
            diag_col='diag_pvc',
            pos_label='Patient',
            task_name='pvc',
            top_k=5,
            k_folds=3,
            seed=42
        )

    summary_df = compare_ablation_models(results, reference_name='Full Integrated')
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 2
    assert 'Model Configuration' in summary_df.columns
    assert 'AUC-ROC (Pooled [95% CI])' in summary_df.columns
    assert 'Delta AUC vs Full' in summary_df.columns
