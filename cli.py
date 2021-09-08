import os
import sys
import pandas as pd

import utils as u
from ml import ml_analysis
from dl import dl_analysis

#####################################################################################################

def app():
    
    data = pd.read_csv(os.path.join(u.DATA_PATH, "data.csv"), engine="python")
    
    # u.breaker()
    # for col in data.columns:
    #     print(col + " - " + repr(data[col].nunique()))
    # u.breaker()
    # print(data.isnull().any())
    # u.breaker()
    # print(data.mean().mean())
    # print(data.std().std())
    # u.breaker()

    features = data.iloc[:, 1:-1].copy().values
    targets  = data.iloc[:, -1].copy().values

    args_1 = "--ml"
    args_2 = "--dl"

    if args_1 in sys.argv:
        ml_analysis(features=features, targets=targets)
    if args_2 in sys.argv:
        dl_analysis(features=features, targets=targets)

#####################################################################################################
