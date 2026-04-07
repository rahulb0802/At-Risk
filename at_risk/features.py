import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

def generate_PCA_Factors(X_transformed_train: pd.DataFrame, n_factors: int = 8) -> tuple:
    """Generates PCA factors and loadings from transformed training data"""
    X_stat = X_transformed_train.copy()
    cols_to_drop_nan = X_stat.columns[X_stat.isna().all()]
    X_stat_valid = X_stat.drop(columns=cols_to_drop_nan)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_stat_valid),
                             index=X_stat_valid.index,
                             columns=X_stat_valid.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed),
                            index=X_imputed.index,
                            columns=X_imputed.columns)

    variances = X_scaled.var()
    constant_cols = variances[variances < 1e-10].index
    X_final_for_pca = X_scaled.drop(columns=constant_cols)

    pca = PCA(n_components=n_factors, random_state=42)
    factors = pca.fit_transform(X_final_for_pca)

    pca_factors_df = pd.DataFrame(factors,
                                  index=X_final_for_pca.index,
                                  columns=[f'PCA_Factor_{i+1}' for i in range(n_factors)])

    loadings_df = pd.DataFrame(pca.components_.T,
                               index=X_final_for_pca.columns,
                               columns=[f'PCA_Factor_{i+1}' for i in range(n_factors)])

    return pca_factors_df, loadings_df

def generate_PCA_Factors_Binary(X_binary_train, n_factors=8):
    """Generates PCA factors from binary indicator states"""
    X_stat = X_binary_train.copy()
    cols_to_drop_nan = X_stat.columns[X_stat.isna().all()]
    X_stat_valid = X_stat.drop(columns=cols_to_drop_nan)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_stat_valid),
                             index=X_stat_valid.index,
                             columns=X_stat_valid.columns)

    variances = X_imputed.var()
    constant_cols = variances[variances < 1e-10].index
    X_final_for_pca = X_imputed.drop(columns=constant_cols)

    pca = PCA(n_components=n_factors, random_state=42)
    factors = pca.fit_transform(X_final_for_pca)

    df_factors = pd.DataFrame(factors,
                              index=X_final_for_pca.index,
                              columns=[f'PCA_Factor_{i+1}' for i in range(n_factors)])

    df_loadings = pd.DataFrame(pca.components_.T,
                               index=X_final_for_pca.columns,
                               columns=[f'PCA_Factor_{i+1}' for i in range(n_factors)])

    return df_factors, df_loadings

def generate_Deter_Indices(X_transformed_train, y_train, horizon, 
                           variable_groups, counter_cyclical_vars,
                           threshold=None, sector_threshold=None, var_threshold=None, 
                           window_size=None):
    """Generates binary deterioration states based on relative quantiles of signals"""
    deterioration_dict = {}
    all_selected_vars = [var for var_list in variable_groups.values() for var in var_list]

    for var in all_selected_vars:
        if var not in X_transformed_train.columns:
            continue
        
        signal = X_transformed_train[var]
        if window_size is not None:
            signal = signal.rolling(window=window_size, min_periods=1).mean()
        
        is_counter = var in counter_cyclical_vars
        
        # Determine quantile based on specified threshold type
        if var_threshold is not None:
            q = var_threshold.get(var)
        elif sector_threshold is not None:
            var_sector = next((s for s, group in variable_groups.items() if var in group), None)
            q = sector_threshold.get(var_sector)
        else:
            q = threshold
        
        if q is None: continue
        
        # Counter-cyclical logic: high value is bad (recession)
        # Cyclical logic: low value is bad (recession)
        target_q = 1 - q if is_counter else q
        det_threshold = signal.quantile(target_q)
        
        state = pd.Series(0.0, index=signal.index, name=var)
        if is_counter:
            state[signal > det_threshold] = 1.0
        else:
            state[signal < det_threshold] = 1.0
        
        deterioration_dict[var] = state

    # concatenate all at once while preserving original categorical order
    if deterioration_dict:
        ordered_vars = [v for v in all_selected_vars if v in deterioration_dict]
        deterioration_states = pd.concat([deterioration_dict[v] for v in ordered_vars], axis=1)
    else:
        deterioration_states = pd.DataFrame(index=X_transformed_train.index)

    return deterioration_states

def add_lags(df, lags_to_add, prefix=''):
    """Adds lags to DF"""
    if not lags_to_add:
        return df

    df_lagged = df.copy()
    for lag in lags_to_add:
        df_shifted = df.shift(lag)
        df_shifted.columns = [f'{prefix}{col}_lag{lag}' for col in df.columns]
        df_lagged = pd.concat([df_lagged, df_shifted], axis=1)

    return df_lagged
