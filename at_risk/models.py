import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

class ModelRegistry:    
    _TEMPLATES = {
        'Logit': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
        'Logit_L1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
        'Logit_L2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, objective='binary:logistic'),
    }

    @classmethod
    def get_model(cls, name: str, X_train: pd.DataFrame = None, y_train: pd.Series = None, 
                  opt_params: dict = None, predictor_set: str = None):
        if name not in cls._TEMPLATES:
            return None
            
        model = clone(cls._TEMPLATES[name]) # clone for safety
        params = {}
        
        # Handle class weight stuff
        if y_train is not None:
            if 'XGBoost' in name:
                neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
                if pos > 0:
                    params['scale_pos_weight'] = (neg / pos)
            elif any(x in name for x in ['Logit', 'HGBoost', 'RandomForest']):
                params['class_weight'] = 'balanced'
                
        # get oprimized C params
        if opt_params and predictor_set:
            if 'L1' in name:
                params['C'] = opt_params['l1'].get(predictor_set, 1.0)
            elif 'L2' in name:
                params['C'] = opt_params['l2'].get(predictor_set, 1.0)
                
        if params:
            model.set_params(**params)
            
        return model

def get_model_template(name):
    """Some backward compability after refactor"""
    return ModelRegistry._TEMPLATES.get(name)
