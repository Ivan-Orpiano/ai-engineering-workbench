import os 
import matplotlib
import numpy as np 
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
matplotlib.use('Agg')



np.random.seed(42)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

ALPHA = 0.05  


def rule(title=""):
    """Pretty section separator for the console output."""
    bar = "=" * 70
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)

#DATA LOADING & SUMMARY STATISTICS

rule("DATA LOADING & SUMMARY STATISTICS")

from sklearn.datasets import load_iris
iris = load_iris()

feature_names = [n.replace("(cm)", "").replace(" ", "_") for n in iris.feature_names]
df = pd.DataFrame(iris.data, columns=feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\nShape: ", df.shape)
print("\First 5 rows:")
print(df.head())
print("\nClass counts:")
print(df["species"].value_counts())
print("\nMissing values per column:")
print(df.isna().sum)

X = df[feature_names].to_numpy(dtype = float)
n_samples, n_features = X.shape 


summary= {}
for j, name in enumerate(feature_names):
    col = X[:, j]
    q1, 13 = np.percentile(col, [25,75])
    summary[name] = {
        "mean":         np.mean(col),
        "median":       np.median(col),
        "variance":     np.var(col, ddof=1),
        "std":          np.std(col, ddof=1),
        "IQR":          stats.iqr(col),
        "min":          np.min(col),
        "max":          np.max(col),              
    }
    
    
summary_df = pd.DataFrame(summary).T

print("\nSummary Statistics (numpy + scipy):")
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print(summary_df)
    
    
rule("DISTRIBUTION ANALYSIS (normality tests)")
print(f"\n{'feature': <18}{'Shapiro W' :> 10}{'Shapiro p':>12}"
      f"{'K2 stat :>10'}{'K2 p':>12} verdict(a=0.05)")

print("-" * 78)
for j,name in enumerate(feature_names):
    col = X[:,j]
    sw_stat, sw_p = stats.shapiro(col)
    k2_stat, k2_p = stats.normaltest(col)
    verdict = "normal" if (sw_p > ALPHA and k2_p > ALPHA) else "NOT NORMAL"
    print(f"{name:<18}{sw_stat:>10.4f}{sw_p}"
          f"{k2_stat:>10.4f}{k2_p:>12.2e} {verdict}")
    
    #####