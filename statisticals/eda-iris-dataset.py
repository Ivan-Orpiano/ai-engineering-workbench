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
    

def gaussian_kde_curve(data, grid):
    #Manual KDE via scipy.stats.gaussian_kde
    kde = stats.gaussian_kde(data)
    return kde(grid)

#Histograms and KDE Overlay
fig, axes = plt.subplots(2,2, figsize = (11,8))
for j, (name, ax) in enumerate(zip(feature_names, axes.ravel())):
    col = X[:, j]
    ax.hist(col, bins = 20, density = True, color = "#4C72B0",
            alpha = 0.55, edgecolor = "white", label = "histogram")
    grid= np.linspace(col.min(), col.max(), 200)
    ax.plot(grid, gaussian_kde_curve(col, grid), color = "#C44E52", lw = 2, label = "KDE")
    
    mu, sigma = np.mean(col), np.std(col, ddof = 1)
    ax.plot(grid, stats.norm.pdf(grid, mu, sigma), "k--", lw=1.2, label = "normal fit")
    
    ax.set_title(name)
    ax.set_xlabel("value (cm)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    
fig.suptitle("Feature Distributions: Histogram + KDE + Normal Fit", y =1.00)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/01_distributions.png", bbox_inches = "tight")
plt.close(fig)

    
#CORRELATION AND HYPTHESIS TESTING

# PEARSON AND SPEARMAN CORRELATION DIFF.
pearson = np.corrcoef(X, rowvar=False)
spearman, _ = stats.spearman(X)

print ("\nPearson correlation matrix: ")
print(pd.DataFrame(pearson, index = feature_names, columns = feature_names).round(3))
print("\nSpearman correlation matrix: ")
print(pd.DataFrame(spearman, index = feature_names, columns = feature_names).round(3))

mask = ~np.eye(n_features, dtype = bool)
abs_corr = np.where(mask, np.abs(pearson), 0)
i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
print(f"\nStrongest linear pair: {feature_names[i]} < - > {feature_names[j]}" f"(Pearson r = {pearson[i,j]:.3f})")


#ANOVA ACROSS 3 SPECIES FEATURE (SETOSA/VERSICOLOR/VIGINICA)
groups = {sp: X[iris.target == k] for k, sp in enumerate(iris.target_names)}
print(f"\fOne-way ANOVA across species(H0: equal means) a= {ALPHA}")
print("-" * 60)
for j, name in enumerate(feature_names):
    samples = [g[:,j] for g in groups.value()]
    f_stat, p_val = stats.f_oneway(*samples)
    sig = "YES" if p_val < ALPHA else "no"
    print(f"{name:<18}{f_stat:>12.3f}{p_val:>14.3e}{sig}")
    
    
#follow up t-test: versicolor vs virginica
print("\nWelch t-test: Versicolor vs Virginica")
print(f"{'feature' :<18}{'t-stat':>12}{'p-value':>14} significant?")
print("-" * 60)
ver, vir = groups["versicolor"], groups["virginica"]
for j, name in enumerate(feature_names):
    t_stat, p_val = stats.ttest_ind
    sig = "YES" if p_val < ALPHA else "NO"
    print(f"{name:<18}{t_stat:>12.3f}{p_val:>14.3e} {sig}")
    
    
    
#VISUALIZATION | HEATMAP | BOXPLOTS | SCATTER MATRIX



