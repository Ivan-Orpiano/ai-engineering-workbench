from __future__ import annotations

import numpy as np
from scipy import stats
import pandas as pd 


import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG = np.random.default_rng(42)

#Data integration
def make_dataset(n: int  = 500) -> tuple[pd.DataFrame, pd.Series, dict]:
    
    x1 = RNG.normal(0,1,n)
    x2 = RNG.normal(0,1, n)
    x3 = RNG.normal(0,1,n)
    x4 = 0.6*x1 + RNG.normal(0,0.8, n)
    
    true_coefs = {"const": 3.0, "x1": 2.5, "x2": -1.8, "x3": 0.0, "x4": 4.0}
    noise = RNG.normal(0,2.0,n)
    
    
    y = (true_coefs["const"]
         + true_coefs["x1"] * x1
         + true_coefs["x2"] * x2
         + true_coefs ["x3"] * x3
         + true_coefs ["x4"] * x4
         + noise  )


    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, pd.Series(y, name="y"), true_coefs

#METRICS HELPERS

def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Penalize R2 for the number of predictors p (excluding intercept)."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def regression_report(name: str, y_true, y_pred, n: int, p: int) -> dict:







