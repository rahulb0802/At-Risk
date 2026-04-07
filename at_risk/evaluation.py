import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, log_loss

def brier_score_decomposition(y_true, y_prob, n_bins=10):
    """
    Brier Score Decomposition according to Murphy (1973)
    """
    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)

    if len(y_true_np) == 0:
         return {'Reliability': np.nan, 'Resolution': np.nan, 'Uncertainty': np.nan}

    p_bar = y_true_np.mean()
    uncertainty = p_bar * (1 - p_bar)

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    bin_indices = np.digitize(y_prob_np, bins)

    reliability = 0
    resolution = 0
    N = len(y_true_np)

    for i in range(1, n_bins + 1):
        bin_mask = (bin_indices == i)
        if not np.any(bin_mask):
            continue

        y_true_bin = y_true_np[bin_mask]
        y_prob_bin = y_prob_np[bin_mask]
        N_k = len(y_true_bin)

        p_bar_k = y_true_bin.mean() 
        p_k = y_prob_bin.mean() 

        reliability += (N_k / N) * (p_bar_k - p_k)**2
        resolution += (N_k / N) * (p_bar_k - p_bar)**2

    return {'Reliability': reliability, 'Resolution': resolution, 'Uncertainty': uncertainty}

def calculate_performance_metrics(y_true: pd.Series, y_prob: pd.Series) -> dict:
    """Calculates performance metrics"""
    valid_idx = ~np.isnan(y_prob)
    if valid_idx.sum() == 0:
        return None
        
    y_true_eval = y_true[valid_idx]
    y_prob_eval = y_prob[valid_idx]
    
    if len(np.unique(y_true_eval)) < 2:
        return None
        
    pr_auc = average_precision_score(y_true_eval, y_prob_eval)
    roc_auc = roc_auc_score(y_true_eval, y_prob_eval)
    brier = brier_score_loss(y_true_eval, y_prob_eval)
    ll_score = log_loss(y_true_eval, y_prob_eval)
    
    brier_decomp = brier_score_decomposition(y_true_eval, y_prob_eval)
    
    return {
        'PR_AUC': pr_auc,
        'Brier_Score': brier,
        'Resolution': brier_decomp['Resolution'],
        'Reliability': brier_decomp['Reliability'],
        'Uncertainty': brier_decomp['Uncertainty'],
        'ROC_AUC': roc_auc,
        'Log_Loss': ll_score,
        'Num_Forecasts': len(y_prob_eval)
    }
