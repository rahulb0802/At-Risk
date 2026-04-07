import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

def optimize_C_value_first_sample_l1(X_train, y_train, C_values, n_splits=5):
    """Optimize C value using time series cross-validation on the first training sample"""
    if len(X_train) < 50:
        return 1.0

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = float('inf')
    best_C = 1.0

    for C in C_values:
        scores = []
        try:
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = LogisticRegression(penalty='l1', solver='liblinear', C=C, 
                                           max_iter=1000, random_state=42, class_weight='balanced')
                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                brier_score = np.mean((y_val - y_pred_proba) ** 2) # calc brier score
                scores.append(brier_score)
        except:
            continue
        # take mean of brier scores to find the best C
        if scores:
            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_C = C

    return best_C

def optimize_C_value_first_sample_l2(X_train, y_train, C_values, n_splits=5):
    """Optimize C value using time series cross-validation on the first training sample (L2)"""
    if len(X_train) < 50:
        return 1.0

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = float('inf')
    best_C = 1.0

    for C in C_values:
        scores = []
        try:
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # same as previous func but with L2 reg
                model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, 
                                           max_iter=1000, random_state=42, class_weight='balanced')
                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                brier_score = np.mean((y_val - y_pred_proba) ** 2)
                scores.append(brier_score)
        except:
            continue

        if scores:
            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_C = C

    return best_C

def get_empirical_unified_quantile(X_in_sample, y_in_sample, counter_cyclical_vars, horizon, momentum_window=12):
    """Derives a single consensus quantile threshold based on median rank during recessions"""
    y_series = y_in_sample.iloc[:, 0] if isinstance(y_in_sample, pd.DataFrame) else y_in_sample
    X_unified = X_in_sample.copy()
    for var in counter_cyclical_vars:
        if var in X_unified.columns:
            X_unified[var] = X_unified[var] * -1 # convert counter cyclical to cyclical

    recession_periods = y_series[y_series == 1].index
    median_recessionary_quantiles = []

    for var in X_unified.columns:
        rank = X_unified[var].rank(pct=True) # rank quantiles in recessionary periods
        ranks_during_recession = rank.loc[rank.index.intersection(recession_periods)].dropna()

        if not ranks_during_recession.empty:
            median_recessionary_quantiles.append(ranks_during_recession.median()) # get median for this var

    return np.median(median_recessionary_quantiles) if median_recessionary_quantiles else 0.5 # get total median

# same logic but by variable
def get_empirical_variable_quantiles(X_in_sample, y_in_sample, counter_cyclical_vars):
    """Derives individual quantile thresholds for each variable"""
    y_series = y_in_sample.iloc[:, 0] if isinstance(y_in_sample, pd.DataFrame) else y_in_sample
    X_unified = X_in_sample.copy()
    for var in counter_cyclical_vars:
        if var in X_unified.columns:
            X_unified[var] = X_unified[var] * -1

    X_ranks = X_unified.rank(pct=True)
    recession_periods = y_series[y_series == 1].index
    ranks_during_recession = X_ranks.loc[X_ranks.index.intersection(recession_periods)].dropna()

    return ranks_during_recession.median().to_dict()

# same func but by sector
def get_empirical_sector_quantiles(X_in_sample, y_in_sample, variable_groups, counter_cyclical_vars):
    """Derives individual quantile thresholds for each sector"""
    y_series = y_in_sample.iloc[:, 0] if isinstance(y_in_sample, pd.DataFrame) else y_in_sample
    X_unified = X_in_sample.copy()
    for var in counter_cyclical_vars:
        if var in X_unified.columns:
            X_unified[var] = X_unified[var] * -1

    X_ranks = X_unified.rank(pct=True)
    recession_periods = y_series[y_series == 1].index
    ranks_during_recession = X_ranks.loc[X_ranks.index.intersection(recession_periods)].dropna()
    median_variable_ranks = ranks_during_recession.median()

    sector_quantiles = {}
    for sector, var_list in variable_groups.items():
        vars_in_sector = [var for var in var_list if var in median_variable_ranks.index]
        if vars_in_sector:
            sector_quantiles[sector] = median_variable_ranks.loc[vars_in_sector].median()

    return sector_quantiles
