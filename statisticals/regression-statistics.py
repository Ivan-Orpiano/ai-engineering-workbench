from __future__ import annotations

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category = ConvergenceWarning)

import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use("Agg") 
import matplotlib as plt


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso,
    RidgeCV, LassoCV, ElasticNetCV
)


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score


