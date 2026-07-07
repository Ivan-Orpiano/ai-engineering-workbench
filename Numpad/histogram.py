"""
Histograms + KDE + Normal fit for each numeric feature.
Dataset: Iris (scikit-learn). Swap in any DataFrame / CSV to reuse.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris

# ----------------------------------------------------------------------
# 1. Load data  (replace this block with pd.read_csv("yourfile.csv"))
# ----------------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
features = [c for c in df.columns if c != "target"]

# ----------------------------------------------------------------------
# 2. Figure: one panel per feature
# ----------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
n = len(features)
ncols = 2
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows))
axes = axes.ravel()

summary_rows = []

for ax, col in zip(axes, features):
    data = df[col].dropna().values
    mu, sigma = data.mean(), data.std(ddof=1)

    # --- Histogram (density-normalised so KDE / PDF overlay correctly) ---
    ax.hist(data, bins="auto", density=True, alpha=0.45,
            color="#4C72B0", edgecolor="white", label="Histogram")

    # --- KDE (empirical density) ---
    kde = stats.gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 400)
    ax.plot(xs, kde(xs), color="#C44E52", lw=2.2, label="KDE")

    # --- Normal fit (theoretical PDF from sample mean & std) ---
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="#55A868",
            lw=2.2, ls="--", label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")

    ax.axvline(mu, color="#333333", lw=1, ls=":", alpha=0.7)
    ax.set_title(col, fontsize=12, fontweight="bold")
    ax.set_xlabel(col); ax.set_ylabel("Density")
    ax.legend(fontsize=8, framealpha=0.9)

    # --- Stats for the summary table ---
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)              # excess kurtosis (normal = 0)
    W, p = stats.shapiro(data)               # Shapiro–Wilk normality test
    summary_rows.append({
        "feature": col, "n": len(data), "mean": mu, "std": sigma,
        "skew": skew, "excess_kurt": kurt,
        "shapiro_W": W, "shapiro_p": p,
        "normal?": "yes" if p > 0.05 else "no"
    })

# hide any unused subplot
for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle("Histogram + KDE + Normal Fit  —  Iris features",
             fontsize=15, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig("/home/claude/hist_kde_normfit.png", dpi=140, bbox_inches="tight")

# ----------------------------------------------------------------------
# 3. Statistical summary (printed to console)
# ----------------------------------------------------------------------
summary = pd.DataFrame(summary_rows).set_index("feature")
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("\n" + "=" * 78)
print("STATISTICAL SUMMARY  (Shapiro–Wilk H0: data is normally distributed)")
print("=" * 78)
print(summary.to_string())
print("\nExcess kurtosis: 0 ≈ normal, >0 heavy tails, <0 light tails")
print("Shapiro p > 0.05  →  fail to reject normality (looks normal)")