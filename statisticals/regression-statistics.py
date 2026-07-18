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


RNG = np.random.default_rng(7)
OUT = "/mnt/user-data/outputs"




def true_function (x: np.ndarray) -> np.ndarray:
    return 0.5 * x **3 - 2.0* x**2 + 1.5 * x + 3.0 + 4.0 * np.sin(1.5 * x)

def make_data(n: int =120, noise: float = 6.0):
    x = np.sort(RNG.uniform(-4, 4, n))
    y = true_function(x) + RNG.normal(0, noise, n)
    return x.reshape(-1, 1), y



def poly_pipe(degree: int, model = None):
    model = model if model is not None else LinearRegression()
    return make_pipeline(
        PolynomialFeatures(degree, include_bias=False), StandardScaler(),model,
    )
    
    
    
# POLYNOMIAL DEGREE SWEEEP - VALIDATION CURVE

def degree_sweep(X, y, degrees=range(1,16)):
    kf = KFold(n_splits = 5, shuffle=True, random_state = 7)
    train_rmse, crv_rmse = [],[]
    
    print("=" * 68)
    print("POLYNOMIAL DEGREE SWEEP (bias-variance via train vs CV error)")
    print("=" * 68)
    print(f"{'deg':>3} {'train_RMSE':> 11} {'cv_RMSE' :>9} {'gap':>7}")
    for d in degrees:
        pipe = poly_pipe(d)
        pipe.fit(X, y)
        tr = np.sqrt(mean_squared_error(y, pipe.predict(X)))
        cv= -cross_val_predict(poly_pipe(d), X,y,cv=kf, scoring="neg_root_mean_squared_error").mean()
    best_d = list(degrees)[int(np.argmin(crv_rmse))]
    print(f"\n Best degree by CV: {best_d} (Lowest CV RMSE = {min(crv_rmse):.3f})")
    print(" Low degree -> high bias (underfit); high degree -> high variance")
    print(" (train error keeps dropping while CV error climbs = Overfitting)")
    #plot
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(list(degrees), train_rmse, "o-", label = "Train RMSE", color = "#2563eb")
    ax.plot(list(degrees), crv_rmse, "s-", label = "5-fold CV RMSE", color = "#dc2626" )    
    ax.axvline(best_d, ls="--", color = "#16a34a", alpha = 0.7, label = f"Best degree = {best_d}")
    
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylable("RMSE")
    ax.set_title("Validation Curve: the Overfitting Cliff")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/validation_curve.png", dpi = 130)
    plt.close(fig)
    return best_d









    
    
    
    
    
    
 
 
 
 
 
 
 
 
 