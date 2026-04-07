import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# paths
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

RAW_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "raw")
INTERMEDIATE_PATH = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_PATH = os.path.join(_PROJECT_ROOT, "results")
OOS_PRED_PATH = os.path.join(RESULTS_PATH, "predictions")
VISUALS_PATH = os.path.join(RESULTS_PATH, 'figures')
OOS_MODELS_PATH = os.path.join(RESULTS_PATH, 'models')
SUB_INDICES_PATH = os.path.join(RESULTS_PATH, 'sub_indices_for_tuning')

# oos settings
OOS_START_DATE = '1990-01-01'
PREDICTION_HORIZONS = [3]
USE_LAGS = False

# C opt params and grid
N_GRID = 30
C_VALUES = np.logspace(-3, 1, N_GRID)

# different sets to test
ALL_POSSIBLE_SETS = ['Deter_States', 'Full', 'Deter_PCA', 'Deter_Avg', 'PCA_Factors_8', 'Yield']

# target column
TARGET_COLUMN = 'USRECM'

# subset for parsimonious
KEY_MACRO_SUBSET = ['T10YFFM', 'BAAFFM', 'PAYEMS', 'CLAIMSx', 'UNRATE', 'INDPRO', 'HOUST', 'RETAILx', 'S&P 500', 'M2REAL']

COUNTER_CYCLICAL_VARS = {
    'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV',
    'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'VIXCLSx', 'ISRATIOx'
}

VARIABLE_GROUPS = {
    'Output_Income': ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS', 'CUMFNS'],
    'Labor_Market': ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008', 'CES2000000008', 'CES3000000008'],
    'Housing': ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS', 'PERMITW'],
    'Consumption_Orders_Inventories': ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'AMDMNOx', 'AMDMUOx', 'ANDENOx', 'BUSINVx', 'ISRATIOx'],
    'Money_Credit': ['M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 'REALLN', 'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST'],
    'Interest_Rates_Spreads': ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 'BAAFFM', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx'],
    'Prices': ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'PPICMM', 'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA'],
    'Stock_Market': ['S&P 500', 'S&P div yield', 'S&P PE ratio', 'VIXCLSx']
}
