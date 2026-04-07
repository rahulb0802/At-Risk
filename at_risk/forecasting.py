import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm # for progress bar

from .features import generate_PCA_Factors, generate_PCA_Factors_Binary, generate_Deter_Indices, add_lags
from .optimization import (
    optimize_C_value_first_sample_l1, 
    optimize_C_value_first_sample_l2,
    get_empirical_unified_quantile, 
    get_empirical_variable_quantiles, 
    get_empirical_sector_quantiles
)
from .models import ModelRegistry
from .evaluation import calculate_performance_metrics
from .config import RESULTS_PATH, INTERMEDIATE_PATH, OOS_MODELS_PATH, SUB_INDICES_PATH, ALL_POSSIBLE_SETS, TARGET_COLUMN, KEY_MACRO_SUBSET

def run_oos_forecast(
    y_target_full: pd.DataFrame, 
    X_transformed_full: pd.DataFrame, 
    X_untransformed_full: pd.DataFrame, 
    X_yield_full: pd.DataFrame, 
    X_ads_full: pd.DataFrame,
    prediction_horizons: list, 
    models_to_run: dict, 
    c_values: np.ndarray,
    oos_start_date: str, 
    force_rerun_all: bool = False, 
    force_rerun_specific: list = None,
    use_lags: bool = False, 
    lags_to_add: list = None,
    results_path: str = RESULTS_PATH, 
    intermediate_path: str = INTERMEDIATE_PATH,
    variable_groups: dict = None, 
    counter_cyclical_vars: set = None,
    use_subset: bool = False
) -> bool:
    """Main forecasting orchestration for Recursive Out-of-Sample predictions."""
    if lags_to_add is None: lags_to_add = []
    if force_rerun_specific is None: force_rerun_specific = []
    
    for horizon in prediction_horizons:
        print(f"\n{'='*25} Processing Horizon h={horizon} {'='*25}")
        
        # Prepare shifted data
        X_trans_shifted = X_transformed_full.shift(horizon)
        X_untrans_shifted = X_untransformed_full.shift(horizon)

        # Output paths
        suffix = "_subset" if use_subset else "_final_refactored"
        file_path = os.path.join(results_path, 'predictions', f'oos_results_h{horizon}{suffix}.pkl')
        model_path = os.path.join(results_path, 'models', f'h{horizon}')
        os.makedirs(model_path, exist_ok=True)

        # Load existing or initialize new
        res_data = _initialize_results_dict(file_path, force_rerun_all, force_rerun_specific, models_to_run)
        if not res_data['sets_to_run']:
            print(f"No sets to run for h={horizon}.")
            continue

        forecast_dates = _get_forecast_dates(y_target_full, oos_start_date)
        oos_actuals = y_target_full.loc[forecast_dates, TARGET_COLUMN]

        # Loop through time
        opt_params = {'l1': {}, 'l2': {}, 'q': None}
        
        for i, forecast_date in tqdm(enumerate(forecast_dates), total=len(forecast_dates), desc=f"Forecast h={horizon}"):
            train_end_date = forecast_date - pd.DateOffset(months=horizon)
            y_train_full = y_target_full.loc[:train_end_date, TARGET_COLUMN]
            y_actual = oos_actuals.loc[forecast_date]
            
            # check if last iter
            is_last_iter = (i == len(forecast_dates) - 1)

            # data processing
            X_iter_scaled, X_iter_imputed = _preprocess_iteration(
                X_trans_shifted, X_untrans_shifted, forecast_date, use_subset
            )
            
            if i == 0:
                opt_params['q'] = _calibrate_thresholds(
                    X_transformed_full, y_target_full, oos_start_date, horizon, 
                    use_subset, counter_cyclical_vars, variable_groups
                )

            # pred set generation
            data_iter = _generate_iteration_sets(
                X_iter_scaled, X_iter_imputed, X_yield_full.shift(horizon), X_ads_full.shift(horizon),
                forecast_date, y_train_full, horizon, variable_groups, counter_cyclical_vars, 
                opt_params['q'], res_data['sets_to_run']
            )

            # fit and pred
            if i == 0:
                _optimize_logit_c(data_iter, y_train_full, train_end_date, c_values, use_lags, lags_to_add, opt_params)

            _run_fit_predict_cycle(
                data_iter, y_train_full, forecast_date, train_end_date, y_actual,
                models_to_run, use_lags, lags_to_add, opt_params, res_data,
                model_path=model_path, is_last_iter=is_last_iter
            )

        # Save res
        _save_results(file_path, res_data, oos_actuals)

    return True

def _initialize_results_dict(file_path, force_rerun_all, force_rerun_specific, models_to_run):
    oos_probs, oos_errors, oos_importances = {}, {}, {}
    if not force_rerun_all and os.path.exists(file_path):
        # set up results dict
        res = joblib.load(file_path)
        oos_probs, oos_errors = res.get('probabilities', {}), res.get('squared_errors', {})
        oos_importances = res.get('importances', {})

    sets_to_run = ALL_POSSIBLE_SETS if force_rerun_all else [
        s for s in ALL_POSSIBLE_SETS if s not in oos_probs or s in force_rerun_specific
    ]

    # set up dicts for each set
    for s in sets_to_run:
        oos_probs[s] = {m: [] for m in models_to_run}
        oos_errors[s] = {m: [] for m in models_to_run}
        oos_importances[s] = {m: [] for m in models_to_run}
    
    # return dict
    return {'probabilities': oos_probs, 'squared_errors': oos_errors, 'importances': oos_importances, 'sets_to_run': sets_to_run}

def _get_forecast_dates(y_target_full, oos_start_date):
    all_dates = y_target_full.index
    return all_dates[all_dates >= pd.to_datetime(oos_start_date)]

def _preprocess_iteration(X_shifted, X_untrans_shifted, forecast_date, use_subset):
    X_raw_slice = X_shifted.loc[:forecast_date]
    X_untrans_slice = X_untrans_shifted.loc[:forecast_date]
    
    if use_subset:
        # make slices for training
        existing_raw = [v for v in KEY_MACRO_SUBSET if v in X_raw_slice.columns]
        X_raw_slice = X_raw_slice[existing_raw]
        existing_untrans = [v for v in KEY_MACRO_SUBSET if v in X_untrans_slice.columns]
        X_untrans_slice = X_untrans_slice[existing_untrans]

    # Drop columns that are all NaNs in the current window
    X_valid = X_raw_slice.drop(columns=X_raw_slice.columns[X_untrans_slice.isna().all()])
    
    X_imputed = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X_valid), index=X_valid.index, columns=X_valid.columns)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imputed), index=X_imputed.index, columns=X_imputed.columns)
    
    return X_scaled, X_imputed

def _calibrate_thresholds(X_full, y_full, oos_start, horizon, use_subset, counter_vars, groups):
    print("Calibrating thresholds...")
    setup_end = pd.to_datetime(oos_start) - pd.DateOffset(months=horizon)
    X_tuning = X_full.loc[:setup_end]
    
    # handle subset quantiles
    if use_subset:
        existing = [v for v in KEY_MACRO_SUBSET if v in X_tuning.columns]
        X_tuning = X_tuning[existing]

    y_tuning = y_full.loc[:setup_end, TARGET_COLUMN]
    return get_empirical_unified_quantile(X_tuning, y_tuning, counter_vars, horizon) # find quantiles

def _generate_iteration_sets(X_scaled, X_imputed, X_yield_sh, X_ads_sh, forecast_date, y_train, horizon, groups, counter_vars, q, sets_to_run):
    data_iter = {}
    # build dict with all sets
    if 'Full' in sets_to_run: data_iter['Full'] = X_scaled
    if 'Yield' in sets_to_run: data_iter['Yield'] = X_yield_sh.loc[:forecast_date]
    if 'ADS' in sets_to_run: data_iter['ADS'] = X_ads_sh.loc[:forecast_date]
    
    # build at risk predictor sets
    if any('Deter' in s for s in sets_to_run):
        states = generate_Deter_Indices(X_imputed, y_train, horizon, groups, counter_vars, threshold=q)
        if 'Deter_States' in sets_to_run: data_iter['Deter_States'] = states
        if 'Deter_Avg' in sets_to_run: data_iter['Deter_Avg'] = states.mean(axis=1).to_frame('Deter_Avg')
        if 'Deter_PCA' in sets_to_run: data_iter['Deter_PCA'], _ = generate_PCA_Factors_Binary(states, n_factors=8)
    
    # build PCA with 8 factors
    if 'PCA_Factors_8' in sets_to_run: 
        data_iter['PCA_Factors_8'], _ = generate_PCA_Factors(X_scaled, n_factors=8)
        
    # Save sub-indices for debugging if it's the first iteration
    if any('Deter' in s for s in sets_to_run):
        horizon_sub_path = os.path.join(SUB_INDICES_PATH, f'h{horizon}')
        os.makedirs(horizon_sub_path, exist_ok=True)
        fname = os.path.join(horizon_sub_path, f'sub_indices_{forecast_date.date()}.pkl')
        if not os.path.exists(fname):
            states.to_pickle(fname)

    return data_iter

def _optimize_logit_c(data_iter, y_train, train_end, c_values, use_lags, lags_to_add, opt_params):
    for s_name, X_raw in data_iter.items():
        X_tr = X_raw.loc[:train_end]
        if use_lags:
            # add lags
            X_tr = add_lags(X_tr, lags_to_add).dropna()
        
        # find common index
        common_idx = y_train.index.intersection(X_tr.index)
        opt_params['l1'][s_name] = optimize_C_value_first_sample_l1(X_tr.loc[common_idx], y_train.loc[common_idx], c_values)
        opt_params['l2'][s_name] = optimize_C_value_first_sample_l2(X_tr.loc[common_idx], y_train.loc[common_idx], c_values)

def _run_fit_predict_cycle(data_iter, y_train_full, forecast_date, train_end, y_actual, models, use_lags, lags_to_add, opt_params, res_data, model_path=None, is_last_iter=False):
    for s_name in res_data['sets_to_run']:
        X_raw = data_iter.get(s_name) # grab set
        if X_raw is None or X_raw.empty:
            for m in models:
                res_data['probabilities'][s_name][m].append(np.nan)
                res_data['squared_errors'][s_name][m].append(np.nan)
            continue
        
        # build training and forecast sets
        X_tr = X_raw.loc[:train_end]
        X_pred = X_raw.loc[[forecast_date]]

        # add lags
        if use_lags:
            X_tr = add_lags(X_tr, lags_to_add).dropna()
            X_pred = add_lags(X_raw, lags_to_add).loc[[forecast_date]]
        
        # prevent numpy/pandas errors
        X_tr.columns = X_tr.columns.astype(str)
        X_pred.columns = X_pred.columns.astype(str)

        common_idx = y_train_full.index.intersection(X_tr.index)
        X_tr, y_tr = X_tr.loc[common_idx], y_train_full.loc[common_idx]

        for m_name in models.keys():
            try:
                model = ModelRegistry.get_model(m_name, y_train=y_tr, opt_params=opt_params, predictor_set=s_name)
                if model is None:
                    continue
                
                # fit and predict using model
                model.fit(X_tr, y_tr)
                prob = model.predict_proba(X_pred)[:, 1][0]
                
                # save model if last iteration
                if is_last_iter and model_path:
                    m_fname = f"{s_name}_{m_name}_model.pkl"
                    if use_lags: m_fname = f"{s_name}_{m_name}_lagged_model.pkl"
                    joblib.dump(model, os.path.join(model_path, m_fname))

                # fill data
                res_data['probabilities'][s_name][m_name].append(prob)
                res_data['squared_errors'][s_name][m_name].append((y_actual - prob)**2)
                res_data['importances'][s_name][m_name].append(_get_importance(m_name, model, X_tr.columns))
            except Exception as e:
                print(f"Error at {forecast_date} for {s_name}/{m_name}: {e}")
                res_data['probabilities'][s_name][m_name].append(np.nan)
                res_data['squared_errors'][s_name][m_name].append(np.nan)

    # get_model in modelregistry class does the job of setting parameters
    pass

def _get_importance(m_name, model, columns):
    # get importances/coefs
    if 'XGBoost' in m_name:
        return pd.Series(model.get_booster().get_score(importance_type='gain'), index=columns).fillna(0)
    if any(x in m_name for x in ['RandomForest', 'HGBoost']):
        return pd.Series(model.feature_importances_, index=columns)
    if 'Logit' in m_name:
        return pd.Series(np.abs(model.coef_[0]), index=columns)
    return pd.Series(0, index=columns)

def _save_results(file_path, res_data, oos_actuals):
    joblib.dump({
        'probabilities': res_data['probabilities'], 
        'squared_errors': res_data['squared_errors'],
        'actuals': oos_actuals, 
        'importances': res_data['importances']
    }, file_path)
