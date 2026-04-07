import argparse
import sys
import os
import warnings
from at_risk.config import (
    C_VALUES, OOS_START_DATE, 
    VARIABLE_GROUPS, COUNTER_CYCLICAL_VARS
)
from at_risk.data import load_analysis_ready_data, filter_problematic_vars
from at_risk.forecasting import run_oos_forecast
from at_risk.models import ModelRegistry

def run_experiment():
    """Main CLI entry point for running forecasting experiments"""
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description='At Risk: Research-Grade Forecasting CLI')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3], 
                        help='Prediction horizons (e.g. 3 6 12)')
    parser.add_argument('--rerun-all', action='store_true', 
                        help='Force rerun all predictor sets')
    parser.add_argument('--specific-sets', nargs='*', 
                        help='Specific predictor sets to rerun')
    parser.add_argument('--oos-start', type=str, default=OOS_START_DATE, 
                        help='Start date for OOS loop (YYYY-MM-DD)')
    parser.add_argument('--lags', nargs='*', type=int, 
                        help='Lag periods to add (e.g. 3 6 12)')
    parser.add_argument('--use-subset', action='store_true', 
                        help='Only use a subset of 10 key macro variables')
    
    args = parser.parse_args()

    print("--- Initializing At Risk Experiment ---")
    
    # Load and Filter Data
    y_target, X_yield, X_trans, X_untrans, X_ads, tcodes = load_analysis_ready_data()
    X_trans, X_untrans = filter_problematic_vars(
        X_trans, X_untrans, ['ACOGNO', 'TWEXAFEGSMTHx', 'UMCSENTx', 'OILPRICEx']
    )
    
    # run forecast
    models_to_run = ModelRegistry._TEMPLATES 
    
    run_oos_forecast(
        y_target_full=y_target,
        X_transformed_full=X_trans,
        X_untransformed_full=X_untrans,
        X_yield_full=X_yield,
        X_ads_full=X_ads,
        prediction_horizons=args.horizons,
        models_to_run=models_to_run,
        c_values=C_VALUES,
        oos_start_date=args.oos_start,
        force_rerun_all=args.rerun_all,
        force_rerun_specific=args.specific_sets,
        use_lags=True if args.lags else False,
        lags_to_add=args.lags,
        variable_groups=VARIABLE_GROUPS,
        counter_cyclical_vars=COUNTER_CYCLICAL_VARS,
        use_subset=args.use_subset
    )
    
    print("\n--- Experiment Complete ---")

if __name__ == "__main__":
    run_experiment()
