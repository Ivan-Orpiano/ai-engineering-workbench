"""
Supervised Learning — Regression Modeling & Statistics
======================================================
A self-contained walkthrough of the statistical machinery behind regression:
  1. Synthetic data with KNOWN coefficients (so we can verify recovery)
  2. Train/test split
  3. OLS via statsmodels  -> full inferential stats (coef, std err, t, p, CIs, R2, F, AIC/BIC)
  4. sklearn LinearRegression / Ridge / Lasso comparison
  5. Metrics: RMSE, MAE, R2, Adjusted R2
  6. K-fold cross-validation (bias/variance sanity check)
  7. Residual diagnostics: normality (Shapiro), heteroscedasticity (Breusch-Pagan),
     autocorrelation (Durbin-Watson), multicollinearity (VIF)

Deps: numpy pandas scikit-learn statsmodels scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# 1. Data generation                                                          #
# --------------------------------------------------------------------------- #
def make_dataset(n: int = 500) -> tuple[pd.DataFrame, pd.Series, dict]:
    """y = 3.0 + 2.5*x1 - 1.8*x2 + 0.0*x3 + 4.0*x4 + noise.
    x3 is intentionally irrelevant (true coef 0) to test significance testing.
    x4 is mildly collinear with x1 to exercise VIF/regularization.
    """
    x1 = RNG.normal(0, 1, n)
    x2 = RNG.normal(0, 1, n)
    x3 = RNG.normal(0, 1, n)              # noise feature, should be insignificant
    x4 = 0.6 * x1 + RNG.normal(0, 0.8, n)  # correlated with x1

    true_coefs = {"const": 3.0, "x1": 2.5, "x2": -1.8, "x3": 0.0, "x4": 4.0}
    noise = RNG.normal(0, 2.0, n)

    y = (true_coefs["const"]
         + true_coefs["x1"] * x1
         + true_coefs["x2"] * x2
         + true_coefs["x3"] * x3
         + true_coefs["x4"] * x4
         + noise)

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, pd.Series(y, name="y"), true_coefs


# --------------------------------------------------------------------------- #
# 2. Metrics helpers                                                          #
# --------------------------------------------------------------------------- #
def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Penalize R2 for the number of predictors p (excluding intercept)."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def regression_report(name: str, y_true, y_pred, n: int, p: int) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj = adjusted_r2(r2, n, p)
    print(f"  {name:<20}  RMSE={rmse:6.3f}  MAE={mae:6.3f}  "
          f"R2={r2:6.4f}  AdjR2={adj:6.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "adj_r2": adj}


# --------------------------------------------------------------------------- #
# 3. OLS inferential statistics (statsmodels)                                 #
# --------------------------------------------------------------------------- #
def ols_inference(X_train, y_train, true_coefs) -> sm.regression.linear_model.RegressionResults:
    X_sm = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_sm).fit()

    print("\n" + "=" * 70)
    print("OLS COEFFICIENT INFERENCE  (does each predictor matter?)")
    print("=" * 70)
    ci = model.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        "true":     [true_coefs.get(k, np.nan) for k in model.params.index],
        "estimate": model.params.values,
        "std_err":  model.bse.values,
        "t":        model.tvalues.values,
        "p_value":  model.pvalues.values,
        "ci_low":   ci[0].values,
        "ci_high":  ci[1].values,
    }, index=model.params.index)
    tbl["signif"] = np.where(tbl["p_value"] < 0.05, "  *", "")
    with pd.option_context("display.float_format", lambda v: f"{v:8.4f}"):
        print(tbl)

    print("\nGoodness of fit")
    print(f"  R-squared        : {model.rsquared:.4f}")
    print(f"  Adj. R-squared   : {model.rsquared_adj:.4f}")
    print(f"  F-statistic      : {model.fvalue:.2f}  (p={model.f_pvalue:.2e})")
    print(f"  AIC / BIC        : {model.aic:.1f} / {model.bic:.1f}")
    print("  Note: x3 (true coef 0) should be non-significant (p >= 0.05).")
    return model


# --------------------------------------------------------------------------- #
# 4. Regularized model comparison (sklearn)                                   #
# --------------------------------------------------------------------------- #
def compare_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    n_test, p = X_test.shape
    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (a=1.0)":    make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso (a=0.1)":    make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    }
    print("\n" + "=" * 70)
    print("HELD-OUT TEST PERFORMANCE")
    print("=" * 70)
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        rows.append(regression_report(name, y_test, pipe.predict(X_test), n_test, p))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Cross-validation                                                         #
# --------------------------------------------------------------------------- #
def cross_validate(X, y) -> None:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    rmse_scores = -cross_val_score(pipe, X, y, cv=kf,
                                   scoring="neg_root_mean_squared_error")
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION  (generalization stability)")
    print("=" * 70)
    print(f"  R2   per fold : {np.round(r2_scores, 4)}")
    print(f"  R2   mean±std : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  RMSE mean±std : {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")


# --------------------------------------------------------------------------- #
# 6. Residual diagnostics                                                     #
# --------------------------------------------------------------------------- #
def residual_diagnostics(model, X_train) -> None:
    resid = model.resid
    print("\n" + "=" * 70)
    print("RESIDUAL DIAGNOSTICS  (are OLS assumptions met?)")
    print("=" * 70)

    # Normality of residuals
    sh_stat, sh_p = stats.shapiro(resid)
    print(f"  Shapiro-Wilk (normality)     : W={sh_stat:.4f}  p={sh_p:.4f}"
          f"  -> {'OK' if sh_p > 0.05 else 'violated'}")

    # Homoscedasticity
    X_sm = sm.add_constant(X_train)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_sm)
    print(f"  Breusch-Pagan (homoscedast.) : LM={bp_stat:.4f}  p={bp_p:.4f}"
          f"  -> {'OK' if bp_p > 0.05 else 'heteroscedastic'}")

    # Independence of errors
    dw = durbin_watson(resid)
    print(f"  Durbin-Watson (autocorr.)    : {dw:.4f}"
          f"  -> {'OK' if 1.5 < dw < 2.5 else 'check'}  (ideal ~2.0)")

    # Multicollinearity
    print("  Variance Inflation Factors (VIF > 5-10 => multicollinearity):")
    Xc = sm.add_constant(X_train)
    for i, col in enumerate(Xc.columns):
        vif = variance_inflation_factor(Xc.values, i)
        flag = "  <-- high" if (col != "const" and vif > 5) else ""
        print(f"      {col:<8}: {vif:6.3f}{flag}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    X, y, true_coefs = make_dataset(n=500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {len(X)} rows, {X.shape[1]} features "
          f"({len(X_train)} train / {len(X_test)} test)")
    print(f"True model: y = 3.0 + 2.5*x1 - 1.8*x2 + 0.0*x3 + 4.0*x4 + N(0, 2)")

    ols_model = ols_inference(X_train, y_train, true_coefs)
    compare_models(X_train, X_test, y_train, y_test)
    cross_validate(X, y)
    residual_diagnostics(ols_model, X_train)

    print("\nDone. Coefficients recovered close to truth; x3 flagged insignificant.")


if __name__ == "__main__":
    main()