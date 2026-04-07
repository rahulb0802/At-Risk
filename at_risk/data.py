import os
import pandas as pd
from .config import INTERMEDIATE_PATH

def load_analysis_ready_data() -> tuple:
    """Reads all pickle files from the intermediate data dir"""
    print("Loading analysis-ready datasets...")
    y_target = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'y_target.pkl'))
    X_yield = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_yield.pkl'))
    X_transformed = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_transformed_monthly.pkl'))
    X_untransformed = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_untransformed_monthly.pkl'))
    X_ads = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'X_ads.pkl'))
    tcodes = pd.read_pickle(os.path.join(INTERMEDIATE_PATH, 'tcodes.pkl'))

    print(f"Data loading complete. Master shape: {X_transformed.shape}")
    return y_target, X_yield, X_transformed, X_untransformed, X_ads, tcodes

def filter_problematic_vars(X_transformed, X_untransformed, vars_to_remove):
    """Drops problematic vars"""
    existing_vars = [v for v in vars_to_remove if v in X_transformed.columns]
    X_transformed = X_transformed.drop(columns=existing_vars)
    X_untransformed = X_untransformed.drop(columns=existing_vars)
    print(f"Filtered {len(existing_vars)} problematic variables: {existing_vars}")
    return X_transformed, X_untransformed
