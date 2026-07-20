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


#BOOTSTRAP BIAS OR VARIANCE DECOMPOSITION
def bias_variance(X, y, degrees=(1,3,6,10,15), n_boot = 120):
    x_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    truth = true_function(x_grid.ravel())
    
    print("\n" + "=" * 68)
    print("BOOTSTRAP BIAS-VARIANCE DECOMPOSITION (EXPECTED TEST ERROR)")
    print("=" * 68)
    print(f"{'def':>3}  {'bias^2':>9} {'variance':>9} {'bias^2+var':> 11}  ")
    for d in degrees:
        preds = np.zeros((n_boot, len(x_grid)))
        for b in range(n_boot):
            idx = RNG.integers(0, len(X), len(X))
            pipe = poly_pipe(d).fit(X[idx], y[idx])
            preds[b] = pipe.predict(x_grid)
        mean_pred = preds.mean(axis=0)
        bias2 = np.mean((mean_pred - truth)**2)
        var = np.mean((preds.var(axis=0)))
        print(f"{d:>3}{bias2:>9.3f} {var:>9.3f} {bias2+ var:>11.3f}")
    print("As degree rises: bias^2 falls, variance rises -> U-shaped total error.")



#REGULARIZATION ON A HIGH DEGREE BASIS
def regularization_compare(X_train, X_test, y_train, y_test, degree=12):
    kf = KFold(n_splits = 5, shuffle=True, random_state=7)
    alphas = np.logspace(-3,3,60)
    l1_ratios = [0.1,0.5,0.7,0.9,0.95,1.0]
    
    models = {
        "OLS (NO REGRESSION)": LinearRegression(),
        "RidgeCV (L2)": RidgeCV(alphas = alphas, cv=kf),
        "LassoCV(L1)":  LassoCV(alphas-alphas, l1_ratio=l1_ratios, cv=kf, max_iter=20000, random_state=7),
        "ElasticNetCV": ElasticNetCV(alphas = alphas, l1_ratio=l1_ratios, cv=kf, max_iter=20000, random_state=7),
    }
    
    print("\n" + "=" * 68)
    print(f"REGULARIZATION on degree - {degree} features (tames overfitting)")
    print("=" * 68)
    print(f"{'model' : < 15} {'train_R2':> 9}{'test_R2':>9} {'test_RMSE':>11}"
          f"{'alpha':>10} {'nonzero':>9}")
    for name, est in models.items():
        pipe = poly_pipe(degree, est).fit(X_train, y_train)
        tr_r2 = r2_score(y_train, pipe.predict(X_train))
        te_r2 = r2_score(y_test, pipe.predict(X_test))
        te_rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
        fitted = pipe.named_steps[list(pipe.named_steps)[-1]]
        alpha = getattr(fitted, "alpha_", np.nan)
        nonzero = int(np.sum(np.abs(fitted.coef_) > 1e-6))
        total = len(fitted.coef_)
        print(f"{name :<15}{tr_r2:>9.4f}{te_r2:>9.4f}{te_rmse:>11.3f}" 
              f"{alpha:>10.4f}{nonzero:>6}/{total}")
    print(" OLS overfits (train)R2 >> test_R2). Lasso zeroes redundant powers")
    print(" (sparsity/feature selection); Ridge shrinks all coefs smoothly.")
    
def regularization_paths(X,y, degree = 12):
    poly = make_pipeline(PolynomialFeatures(degree, include_bias = False), StandardScaler())
    Xp = poly.fit_transform(X)
    alphas = np.logspace(-2, 4, 80)
    
    ridge_coefs = np.array([Ridge(alpha = a).fit(Xp, y).coef_ for a in alphas])
    
    fig, axes = plt.subplots(1,2, figsize=(12,5), sharey = True)
    for coefs, ax, title in [(ridge_coefs, axes[0], "Ridge [L2] - Smooth Shrink"),
                             (lasso_coefs, axes[1], "Lassos [L1] - Sparse Selection")]:
        
        for j in range(coefs.shape[1]):
            ax.plot(alphas, coefs[:, j], alpha = 0.8)
        ax.set_xscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (regularization strength)")
        ax.set_title(title)
        ax.axhline(0, color = "k", lw = 0.6)
        ax.grid(alpha=0.3)
        
    axes[0].set_ylabel("coefficient value")
    fig.suptitle(f"Regularization Paths (degree-{degree} polynomial features)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/regularization_paths.png", dpi=130)
    plt.close(fig)
    print("\n  Saved plots: validation_curve.png, regularization_paths.png")
    
    
def main():
    X, y = make_data(n=120, noise=6.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7)
    print(f"Nonlinear data: {len(X)} points on a cubic+sine curve with noise.\n")

    degree_sweep(X, y)
    bias_variance(X, y)
    regularization_compare(X_train, X_test, y_train, y_test, degree=12)
    regularization_paths(X, y, degree=12)
    print("\nDone.")


if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    






    
    
    
    
    
    
 
 
 
 
 
 
 
 
 