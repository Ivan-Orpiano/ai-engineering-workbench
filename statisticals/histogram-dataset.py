import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris





#load data 

iris = load_iris(as_frame = True)
df = iris.frame.rename(columns = lambda c: c.replace("(cm)", "").replace(" ", "_"))
features = [c for c in df.columns if c != "target"]


#figure: panel per features

plt.style.use("seaborn-v0_8-whitegrid")
n = len(features)
ncols = 2
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows))
axes = axes.ravel()

summary_rows = []

for ax, col in zip(axes, features):
    data = df[col].dropna().values
    mu, sigma = data.mean(), data.std(ddof = 1)
    
    #histogram density
    ax.hist(data, bins = "auto", density = True, alpha= 0.45, color = "#4C72B0", edgecolor = "white", label = "Histogram")
    
    
    #KDE (empirical dens)
    kde = stats.gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 400)
    ax.plot(xs,kde(xs), color = "#c44e52", lw = 2.2, label = "KDE" )
    
    #normal fit (theoretical pdf from mean and std)
    
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color = "#55a868", lw = 2.2, ls = "--",
            label = f"Normal(μ={mu:.2f}, σ={sigma:.2f})")
    
    ax.axvline(mu, color = "#333333", lw = 1, ls = ":", alpha = 0.7)
    ax.set_title(col, fontsize = 12, fontweight = "bold")
    ax.set_xlabel(col); ax.set_ylabel("Density")
    ax.legend(fontsize = 8, framealpha = 0.9)
    
    #stats for the summary table
    
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    W, p = stats.shapiro(data)
    summary_rows.append({
        "feature" : col, "n": len(data), "mean": mu, "std" : sigma,
        "skew": skew, "excess_kurt": kurt,
        "shapiro_W": W, "shapiro_p": p,
        "normal?" : "yes" if p > 0.05 else "no"
    })
    
    
    
for ax in axes[n:]:
    ax.set_visible(False)
    
fig.suptitle("Histogram + KDE + Normal Fit - Iris Features", 
             fontsize = 15, fontweight = "bold", y=1.00)

#statistical summary
summary = pd.DataFrame(summary_rows).set_index("feature")
pd.set_option("display.float_format", lambda v: f"{v: .4f}")
print("\n" + "=" * 78)
print("STATISTICAL SUMMARY  (Shapiro–Wilk H0: data is normally distributed)")
print("=" * 78)
print(summary.to_string())
print("\nExcess kurtosis: 0 ≈ normal, >0 heavy tails, <0 light tails")
print("Shapiro p > 0.05  →  fail to reject normality (looks normal)")