"""
Diffusion Model Component Ablation Analysis Module

Provides utilities for:
1. Feature subset extraction by diffusion compartment (DTI, DKI, IVIM, FWI, combinations).
2. Nested cross-validation evaluation with within-fold feature selection.
3. Neural network and Random Forest model fitting with class balancing.
4. Comprehensive performance metrics (AUC-ROC, AP, Sensitivity, Specificity, Brier, ECE).
5. Statistical significance testing for model comparison (paired t-test, bootstrap delta CI).
6. Visualization functions for multi-model ROC/PR curves and performance comparisons.
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight, resample

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def set_seed(seed=41):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def get_ablation_subsets(features):
    """
    Partition feature list into constituent diffusion model subsets.

    Parameters
    ----------
    features : list of str
        List of all feature column names (e.g. starting with 'harm_').

    Returns
    -------
    dict
        Dictionary mapping configuration name to list of feature column names.
    """
    subsets = {
        'DTI-only': [f for f in features if f.endswith(('_FA', '_MD'))],
        'DKI-only': [f for f in features if f.endswith(('_MK', '_KFA'))],
        'DTI+DKI': [f for f in features if f.endswith(('_FA', '_MD', '_MK', '_KFA'))],
        'IVIM-only': [f for f in features if f.endswith('_PF')],
        'FWI-only': [f for f in features if f.endswith('_FW')],
        'IVIM+FWI': [f for f in features if f.endswith(('_PF', '_FW'))],
        'Full Integrated': list(features),
    }
    return subsets


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        Expected Calibration Error.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_iter=2000, ci=95, seed=41):
    """
    Compute bootstrap confidence interval for a metric function.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted continuous scores/probabilities.
    metric_fn : callable
        Metric function taking (y_true, y_pred).
    n_iter : int
        Number of bootstrap iterations.
    ci : float
        Confidence interval percentage (e.g. 95).
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (point_estimate, ci_lower, ci_upper, boot_samples)
    """
    rng = np.random.RandomState(seed)
    point_est = float(metric_fn(y_true, y_pred))
    boot_samples = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        y_t_boot = y_true[idx]
        if len(np.unique(y_t_boot)) < 2:
            continue
        boot_samples.append(metric_fn(y_t_boot, y_pred[idx]))
    boot_samples = np.array(boot_samples)
    alpha = (100 - ci) / 2.0
    ci_lower = float(np.percentile(boot_samples, alpha))
    ci_upper = float(np.percentile(boot_samples, 100 - alpha))
    return point_est, ci_lower, ci_upper, boot_samples


def build_nn_classifier(input_dim, task='pvc'):
    """
    Build standard sequential neural network matching manuscript specifications.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    task : str
        'pvc' (Patient vs Control) or 'scz' (SCZ vs Non-SCZ).

    Returns
    -------
    keras.Model
    """
    if task == 'pvc':
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
            layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
    else:  # SCZ vs Non-SCZ
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
            layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_nested_cv_ablation(
    df,
    feature_subset,
    diag_col,
    pos_label,
    task_name='pvc',
    top_k=20,
    k_folds=5,
    seed=14,
    site_col='scan_site_text'
):
    """
    Execute site-stratified K-fold nested CV with within-fold feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing harmonized features and diagnostic/site metadata.
    feature_subset : list of str
        Features available for this model configuration.
    diag_col : str
        Target column ('diag_pvc' or 'diag_scz').
    pos_label : str
        Positive class label ('Patient' or 'SCZ').
    task_name : str
        'pvc' or 'scz'.
    top_k : int
        Maximum number of features to select within fold.
    k_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for fold splitting.
    site_col : str
        Column containing scan site text.

    Returns
    -------
    dict
        Dictionary containing all predictions, fold metrics, pooled metrics, and feature selections.
    """
    set_seed(41)
    df_task = df[df[diag_col] != 'Remove'].copy()
    y = (df_task[diag_col] == pos_label).astype(int).values
    sites = df_task[site_col].astype(str).values

    # Strata for site-stratified splitting
    strata = sites + '_' + y.astype(str)
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    all_y_true = []
    all_y_nn = []
    all_y_rf = []
    fold_indices = []
    selected_features_counter = Counter()

    fold_aucs_nn = []
    fold_aucs_rf = []
    fold_aps_nn = []
    fold_aps_rf = []

    k_to_select = min(top_k, len(feature_subset))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df_task[feature_subset], strata), 1):
        X_train_full = df_task[feature_subset].iloc[train_idx]
        y_train = y[train_idx]
        X_test_full = df_task[feature_subset].iloc[test_idx]
        y_test = y[test_idx]

        # Nested within-fold feature selection
        fold_feat_aucs = []
        for feat in feature_subset:
            try:
                auc_val = roc_auc_score(y_train, X_train_full[feat])
                fold_feat_aucs.append({'feature': feat, 'auc': auc_val})
            except Exception:
                pass

        if len(fold_feat_aucs) == 0:
            # Fallback if AUC fails
            top_feats = feature_subset[:k_to_select]
        else:
            auc_df = pd.DataFrame(fold_feat_aucs)
            auc_df['disc'] = np.abs(auc_df['auc'] - 0.5)
            top_feats = auc_df.sort_values('disc', ascending=False).head(k_to_select)['feature'].tolist()

        selected_features_counter.update(top_feats)

        X_train = X_train_full[top_feats].values
        X_test = X_test_full[top_feats].values

        # Scaling strictly on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---- Neural Network ----
        model = build_nn_classifier(input_dim=X_train_scaled.shape[1], task=task_name)
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), cw))

        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        X_t, X_v, y_t, y_v = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        model.fit(
            X_t, y_t,
            validation_data=(X_v, y_v),
            epochs=150,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop],
            class_weight=class_weight_dict
        )
        y_pred_nn = model.predict(X_test_scaled, verbose=0).flatten()

        # ---- Random Forest Baseline ----
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight='balanced',
            random_state=41,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict_proba(X_test_scaled)[:, 1]

        # Record fold metrics
        all_y_true.extend(y_test)
        all_y_nn.extend(y_pred_nn)
        all_y_rf.extend(y_pred_rf)
        fold_indices.extend([fold] * len(y_test))

        fold_aucs_nn.append(roc_auc_score(y_test, y_pred_nn))
        fold_aucs_rf.append(roc_auc_score(y_test, y_pred_rf))
        fold_aps_nn.append(average_precision_score(y_test, y_pred_nn))
        fold_aps_rf.append(average_precision_score(y_test, y_pred_rf))

    all_y_true = np.array(all_y_true)
    all_y_nn = np.array(all_y_nn)
    all_y_rf = np.array(all_y_rf)
    fold_indices = np.array(fold_indices)

    # Compute Bootstrap CIs
    auc_nn, auc_nn_low, auc_nn_high, boot_auc_nn = bootstrap_metric_ci(all_y_true, all_y_nn, roc_auc_score)
    auc_rf, auc_rf_low, auc_rf_high, boot_auc_rf = bootstrap_metric_ci(all_y_true, all_y_rf, roc_auc_score)
    ap_nn, ap_nn_low, ap_nn_high, boot_ap_nn = bootstrap_metric_ci(all_y_true, all_y_nn, average_precision_score)
    ap_rf, ap_rf_low, ap_rf_high, boot_ap_rf = bootstrap_metric_ci(all_y_true, all_y_rf, average_precision_score)

    # Calibration
    brier_nn = brier_score_loss(all_y_true, all_y_nn)
    brier_rf = brier_score_loss(all_y_true, all_y_rf)
    ece_nn = compute_ece(all_y_true, all_y_nn)
    ece_rf = compute_ece(all_y_true, all_y_rf)

    # Operating point at Youden's J
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_nn)
    youden_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[youden_idx]
    binary_preds_nn = (all_y_nn >= optimal_thresh).astype(int)
    cm = confusion_matrix(all_y_true, binary_preds_nn)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        'y_true': all_y_true,
        'y_pred_nn': all_y_nn,
        'y_pred_rf': all_y_rf,
        'fold_indices': fold_indices,
        'selected_features': selected_features_counter,
        'n_features_pool': len(feature_subset),
        'n_features_selected': k_to_select,
        'nn': {
            'fold_aucs': fold_aucs_nn,
            'mean_fold_auc': np.mean(fold_aucs_nn),
            'std_fold_auc': np.std(fold_aucs_nn),
            'pooled_auc': auc_nn,
            'auc_ci_lower': auc_nn_low,
            'auc_ci_upper': auc_nn_high,
            'boot_aucs': boot_auc_nn,
            'fold_aps': fold_aps_nn,
            'mean_fold_ap': np.mean(fold_aps_nn),
            'std_fold_ap': np.std(fold_aps_nn),
            'pooled_ap': ap_nn,
            'ap_ci_lower': ap_nn_low,
            'ap_ci_upper': ap_nn_high,
            'boot_aps': boot_ap_nn,
            'brier': brier_nn,
            'ece': ece_nn,
            'optimal_threshold': float(optimal_thresh),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'balanced_accuracy': float(bal_acc),
            'f1': float(f1),
        },
        'rf': {
            'fold_aucs': fold_aucs_rf,
            'mean_fold_auc': np.mean(fold_aucs_rf),
            'std_fold_auc': np.std(fold_aucs_rf),
            'pooled_auc': auc_rf,
            'auc_ci_lower': auc_rf_low,
            'auc_ci_upper': auc_rf_high,
            'boot_aucs': boot_auc_rf,
            'fold_aps': fold_aps_rf,
            'mean_fold_ap': np.mean(fold_aps_rf),
            'std_fold_ap': np.std(fold_aps_rf),
            'pooled_ap': ap_rf,
            'ap_ci_lower': ap_rf_low,
            'ap_ci_upper': ap_rf_high,
            'boot_aps': boot_ap_rf,
            'brier': brier_rf,
            'ece': ece_rf,
        }
    }


def compare_ablation_models(ablation_results, reference_name='Full Integrated'):
    """
    Perform statistical significance testing of each ablation model against the reference model.

    Parameters
    ----------
    ablation_results : dict
        Dictionary of results from `run_nested_cv_ablation` keyed by model configuration name.
    reference_name : str
        Name of reference model (default: 'Full Integrated').

    Returns
    -------
    pd.DataFrame
        Summary table comparing metrics and statistical difference (paired t-test and bootstrap Delta AUC).
    """
    ref_res = ablation_results[reference_name]
    ref_fold_aucs = ref_res['nn']['fold_aucs']
    ref_y_pred = ref_res['y_pred_nn']
    y_true = ref_res['y_true']

    rows = []
    for model_name, res in ablation_results.items():
        nn = res['nn']
        fold_aucs = nn['fold_aucs']
        y_pred = res['y_pred_nn']

        # Delta AUC
        delta_auc = nn['pooled_auc'] - ref_res['nn']['pooled_auc']

        # Paired t-test across folds
        if model_name == reference_name:
            t_pval = 1.0
            p_boot = 1.0
            delta_ci = (0.0, 0.0)
        else:
            # Paired t-test on 5 fold AUCs
            t_stat, t_pval = stats.ttest_rel(fold_aucs, ref_fold_aucs)

            # Paired bootstrap for Delta AUC
            rng = np.random.RandomState(41)
            delta_boots = []
            n = len(y_true)
            for _ in range(2000):
                idx = rng.randint(0, n, size=n)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                a_model = roc_auc_score(y_true[idx], y_pred[idx])
                a_ref = roc_auc_score(y_true[idx], ref_y_pred[idx])
                delta_boots.append(a_model - a_ref)
            delta_boots = np.array(delta_boots)
            delta_ci = (float(np.percentile(delta_boots, 2.5)), float(np.percentile(delta_boots, 97.5)))
            # Two-tailed bootstrap p-value against 0
            p_boot = 2.0 * min(np.mean(delta_boots <= 0), np.mean(delta_boots >= 0))
            p_boot = min(p_boot, 1.0)

        rows.append({
            'Model Configuration': model_name,
            'Features in Pool': res['n_features_pool'],
            'Features Selected': res['n_features_selected'],
            'AUC-ROC (Fold Mean±SD)': f"{nn['mean_fold_auc']:.3f} ± {nn['std_fold_auc']:.3f}",
            'AUC-ROC (Pooled [95% CI])': f"{nn['pooled_auc']:.3f} [{nn['auc_ci_lower']:.3f}-{nn['auc_ci_upper']:.3f}]",
            'Avg Precision (Pooled [95% CI])': f"{nn['pooled_ap']:.3f} [{nn['ap_ci_lower']:.3f}-{nn['ap_ci_upper']:.3f}]",
            'Delta AUC vs Full': f"{delta_auc:+.3f}" if model_name != reference_name else "Reference",
            'Delta AUC 95% CI': f"[{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}]" if model_name != reference_name else "--",
            'Paired Fold t-p': f"{t_pval:.4f}" if model_name != reference_name else "--",
            'Bootstrap p-val': f"{p_boot:.4f}" if model_name != reference_name else "--",
            'Brier Score': f"{nn['brier']:.4f}",
            'ECE': f"{nn['ece']:.4f}",
            'Sensitivity': f"{nn['sensitivity']:.3f}",
            'Specificity': f"{nn['specificity']:.3f}",
            'RF AUC-ROC': f"{res['rf']['pooled_auc']:.3f} [{res['rf']['auc_ci_lower']:.3f}-{res['rf']['auc_ci_upper']:.3f}]"
        })

    return pd.DataFrame(rows)


def plot_ablation_roc_curves(ablation_results, task_title, save_path=None):
    """Plot multi-model ROC curve overlay."""
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab10", len(ablation_results))

    for i, (model_name, res) in enumerate(ablation_results.items()):
        y_true = res['y_true']
        y_pred = res['y_pred_nn']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = res['nn']['pooled_auc']
        auc_ci_low = res['nn']['auc_ci_lower']
        auc_ci_high = res['nn']['auc_ci_upper']
        lw = 3.5 if 'Full' in model_name else 2.0
        alpha = 1.0 if 'Full' in model_name else 0.85
        ax.plot(
            fpr, tpr,
            label=f"{model_name} (AUC = {auc_val:.3f} [{auc_ci_low:.3f}-{auc_ci_high:.3f}])",
            color=palette[i],
            lw=lw,
            alpha=alpha
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Chance (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'Model Component Ablation: ROC Curves ({task_title})', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_pr_curves(ablation_results, task_title, save_path=None):
    """Plot multi-model Precision-Recall curve overlay."""
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab10", len(ablation_results))

    ref_y_true = next(iter(ablation_results.values()))['y_true']
    baseline = np.mean(ref_y_true)

    for i, (model_name, res) in enumerate(ablation_results.items()):
        y_true = res['y_true']
        y_pred = res['y_pred_nn']
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        ap_val = res['nn']['pooled_ap']
        ap_ci_low = res['nn']['ap_ci_lower']
        ap_ci_high = res['nn']['ap_ci_upper']
        lw = 3.5 if 'Full' in model_name else 2.0
        alpha = 1.0 if 'Full' in model_name else 0.85
        ax.plot(
            rec, prec,
            label=f"{model_name} (AP = {ap_val:.3f} [{ap_ci_low:.3f}-{ap_ci_high:.3f}])",
            color=palette[i],
            lw=lw,
            alpha=alpha
        )

    ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=0.6, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax.set_ylabel('Precision (PPV)', fontsize=14)
    ax.set_title(f'Model Component Ablation: PR Curves ({task_title})', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_bars(ablation_results, task_title, save_path=None):
    """Plot bar chart comparing AUC-ROC and Average Precision across ablation models."""
    models = list(ablation_results.keys())
    auc_vals = [ablation_results[m]['nn']['pooled_auc'] for m in models]
    auc_err_low = [auc_vals[i] - ablation_results[m]['nn']['auc_ci_lower'] for i, m in enumerate(models)]
    auc_err_high = [ablation_results[m]['nn']['auc_ci_upper'] - auc_vals[i] for i, m in enumerate(models)]

    ap_vals = [ablation_results[m]['nn']['pooled_ap'] for m in models]
    ap_err_low = [ap_vals[i] - ablation_results[m]['nn']['ap_ci_lower'] for i, m in enumerate(models)]
    ap_err_high = [ablation_results[m]['nn']['ap_ci_upper'] - ap_vals[i] for i, m in enumerate(models)]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(
        x - width/2, auc_vals, width,
        yerr=[auc_err_low, auc_err_high],
        capsize=4,
        label='AUC-ROC',
        color='#378d94',
        edgecolor='black',
        alpha=0.85
    )
    rects2 = ax.bar(
        x + width/2, ap_vals, width,
        yerr=[ap_err_low, ap_err_high],
        capsize=4,
        label='Average Precision',
        color='#9671bd',
        edgecolor='black',
        alpha=0.85
    )

    ax.set_ylabel('Score [95% CI]', fontsize=14)
    ax.set_title(f'Component Ablation Comparison ({task_title})', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
