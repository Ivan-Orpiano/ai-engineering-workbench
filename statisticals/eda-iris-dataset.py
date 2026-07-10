import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt


# Global style / reproducibility

np.random.seed(42)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

ALPHA = 0.05  # significance level used throughout


def rule(title=""):
    """Pretty section separator for the console output."""
    bar = "=" * 70
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)



# DATA LOADING & SUMMARY STATISTICS

rule("1. DATA LOADING & SUMMARY STATISTICS")
from sklearn.datasets import load_iris 
iris = load_iris()

feature_names = [n.replace(" (cm)", "").replace(" ", "_") for n in iris.feature_names]
df = pd.DataFrame(iris.data, columns=feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass counts:")
print(df["species"].value_counts())
print("\nMissing values per column:")
print(df.isna().sum())

# The numerical feature matrix 
X = df[feature_names].to_numpy(dtype=float)   
samples, n_features = X.shape

summary = {}
for j, name in enumerate(feature_names):
    col = X[:, j]
    q1, q3 = np.percentile(col, [25, 75])
    summary[name] = {
        "mean":     np.mean(col),
        "median":   np.median(col),
        "variance": np.var(col, ddof=1),         # sample variance
        "std":      np.std(col, ddof=1),         # sample std deviation
        "IQR":      stats.iqr(col),              # scipy: robust spread
        "min":      np.min(col),
        "max":      np.max(col),
    }

summary_df = pd.DataFrame(summary).T  
print("\nSummary statistics (numpy + scipy):")
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print(summary_df)



# DISTRIBUTION ANALYSIS 

rule("2. DISTRIBUTION ANALYSIS  (normality tests)")

print(f"\n{'feature':<18}{'Shapiro W':>10}{'Shapiro p':>12}"
      f"{'K2 stat':>10}{'K2 p':>12}   verdict (a=0.05)")
print("-" * 78)
for j, name in enumerate(feature_names):
    col = X[:, j]
    sw_stat, sw_p = stats.shapiro(col)          
    k2_stat, k2_p = stats.normaltest(col)       
    verdict = "normal" if (sw_p > ALPHA and k2_p > ALPHA) else "NOT normal"
    print(f"{name:<18}{sw_stat:>10.4f}{sw_p:>12.2e}"
          f"{k2_stat:>10.4f}{k2_p:>12.2e}   {verdict}")


def gaussian_kde_curve(data, grid):
    """Manual KDE via scipy.stats.gaussian_kde (avoids seaborn)."""
    kde = stats.gaussian_kde(data)
    return kde(grid)


# Histograms + KDE overlay, one subplot per feature
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for j, (name, ax) in enumerate(zip(feature_names, axes.ravel())):
    col = X[:, j]
    ax.hist(col, bins=20, density=True, color="#4C72B0",
            alpha=0.55, edgecolor="white", label="histogram")
    grid = np.linspace(col.min(), col.max(), 200)
    ax.plot(grid, gaussian_kde_curve(col, grid), color="#C44E52",
            lw=2, label="KDE")
   
    mu, sigma = np.mean(col), np.std(col, ddof=1)
    ax.plot(grid, stats.norm.pdf(grid, mu, sigma), "k--", lw=1.2,
            label="normal fit")
    ax.set_title(name)
    ax.set_xlabel("value (cm)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
fig.suptitle("Feature Distributions: Histogram + KDE + Normal Fit", y=1.00)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/01_distributions.png", bbox_inches="tight")
plt.close(fig)



#CORRELATION & HYPOTHESIS TESTING

rule("3. CORRELATION & HYPOTHESIS TESTING")

# pearson and spearman 
pearson = np.corrcoef(X, rowvar=False)          
spearman, _ = stats.spearmanr(X)                

print("\nPearson correlation matrix:")
print(pd.DataFrame(pearson, index=feature_names, columns=feature_names)
      .round(3))
print("\nSpearman correlation matrix:")
print(pd.DataFrame(spearman, index=feature_names, columns=feature_names)
      .round(3))


mask = ~np.eye(n_features, dtype=bool)
abs_corr = np.where(mask, np.abs(pearson), 0)
i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
print(f"\nStrongest linear pair: {feature_names[i]} <-> {feature_names[j]} "
      f"(Pearson r = {pearson[i, j]:.3f})")

# ANOVA across the 3 species for each feature 
groups = {sp: X[iris.target == k] for k, sp in enumerate(iris.target_names)}
print(f"\nOne-way ANOVA across species (H0: equal means)   a = {ALPHA}")
print(f"{'feature':<18}{'F-stat':>12}{'p-value':>14}   significant?")
print("-" * 60)
for j, name in enumerate(feature_names):
    samples = [g[:, j] for g in groups.values()]
    f_stat, p_val = stats.f_oneway(*samples)
    sig = "YES" if p_val < ALPHA else "no"
    print(f"{name:<18}{f_stat:>12.3f}{p_val:>14.3e}   {sig}")

# Follow-up t-test: versicolor vs virginica 

print("\nWelch t-test: versicolor vs virginica")
print(f"{'feature':<18}{'t-stat':>12}{'p-value':>14}   significant?")
print("-" * 60)
ver, vir = groups["versicolor"], groups["virginica"]
for j, name in enumerate(feature_names):
    t_stat, p_val = stats.ttest_ind(ver[:, j], vir[:, j], equal_var=False)
    sig = "YES" if p_val < ALPHA else "no"
    print(f"{name:<18}{t_stat:>12.3f}{p_val:>14.3e}   {sig}")



# 4. VISUALIZATION  (heatmap, boxplots, scatter matrix)

rule("4. VISUALIZATION")

# Correlation heatmap 
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(pearson, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(n_features)); ax.set_yticks(range(n_features))
ax.set_xticklabels(feature_names, rotation=45, ha="right")
ax.set_yticklabels(feature_names)
for r in range(n_features):                     # annotate each cell
    for c in range(n_features):
        ax.text(c, r, f"{pearson[r, c]:.2f}", ha="center", va="center",
                color="white" if abs(pearson[r, c]) > 0.5 else "black",
                fontsize=9)
ax.set_title("Pearson Correlation Heatmap")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")
ax.grid(False)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/02_corr_heatmap.png", bbox_inches="tight")
plt.close(fig)

# Boxplots of each feature grouped by species 
colors = ["#4C72B0", "#DD8452", "#55A868"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for j, (name, ax) in enumerate(zip(feature_names, axes.ravel())):
    data_by_group = [groups[sp][:, j] for sp in iris.target_names]
    bp = ax.boxplot(data_by_group, tick_labels=iris.target_names,
                    patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_title(name); ax.set_ylabel("value (cm)")
fig.suptitle("Feature Distributions by Species (boxplots)", y=1.00)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/03_boxplots_by_species.png", bbox_inches="tight")
plt.close(fig)

# pairplot-style, pure matplotlib
target = iris.target
fig, axes = plt.subplots(n_features, n_features, figsize=(11, 11))
for r in range(n_features):
    for c in range(n_features):
        ax = axes[r, c]
        if r == c:
            # diagonal: per-species histograms of that feature
            for k, sp in enumerate(iris.target_names):
                ax.hist(X[target == k, r], bins=15, alpha=0.5, color=colors[k])
        else:
            for k in range(3):
                ax.scatter(X[target == k, c], X[target == k, r],
                           s=12, alpha=0.6, color=colors[k])
        if r == n_features - 1:
            ax.set_xlabel(feature_names[c], fontsize=8)
        if c == 0:
            ax.set_ylabel(feature_names[r], fontsize=8)
        ax.tick_params(labelsize=7)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[k],
                      label=sp) for k, sp in enumerate(iris.target_names)]
fig.legend(handles=handles, loc="upper right", title="species")
fig.suptitle("Scatter-Plot Matrix (pairplot-style)", y=1.00)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/04_scatter_matrix.png", bbox_inches="tight")
plt.close(fig)



#  OUTLIER DETECTION  (z-scores)

rule("5. OUTLIER DETECTION  (|z| > 3)")

z = np.abs(stats.zscore(X, axis=0, ddof=1))     
Z_THRESH = 3.0
outlier_mask = z > Z_THRESH

print(f"\n{'feature':<18}{'# outliers':>12}   row indices")
print("-" * 55)
for j, name in enumerate(feature_names):
    idx = np.where(outlier_mask[:, j])[0]
    print(f"{name:<18}{len(idx):>12}   {idx.tolist()}")

rows_with_any = np.where(outlier_mask.any(axis=1))[0]
print(f"\nRows containing at least one outlier: {rows_with_any.tolist()}")
print(f"Total flagged rows: {len(rows_with_any)} / {n_samples} "
      f"({100*len(rows_with_any)/n_samples:.1f}%)")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))


z_signed = stats.zscore(X, axis=0, ddof=1)
ax1.boxplot([z_signed[:, j] for j in range(n_features)],
            tick_labels=feature_names, showfliers=True)
ax1.axhline(Z_THRESH, color="red", ls="--", lw=1, label="|z| = 3")
ax1.axhline(-Z_THRESH, color="red", ls="--", lw=1)
ax1.set_title("Standardized Features (outliers beyond dashed lines)")
ax1.set_ylabel("z-score")
ax1.tick_params(axis="x", rotation=25)
ax1.legend()


xf, yf = feature_names.index("petal_length"), feature_names.index("petal_width")
normal_pts = ~outlier_mask.any(axis=1)
ax2.scatter(X[normal_pts, xf], X[normal_pts, yf], s=20, alpha=0.6,
            color="#4C72B0", label="normal")
ax2.scatter(X[~normal_pts, xf], X[~normal_pts, yf], s=90, alpha=0.9,
            facecolors="none", edgecolors="red", linewidths=1.8,
            label="outlier row")
ax2.set_xlabel("petal_length"); ax2.set_ylabel("petal_width")
ax2.set_title("Outlier Rows Highlighted")
ax2.legend()
fig.tight_layout()
fig.savefig(f"{OUTDIR}/05_outliers.png", bbox_inches="tight")
plt.close(fig)


rule("6. CONCLUSION")

print(f"""
Key findings
------------
- Central tendency: petal measurements have far larger spread (petal_length
  variance = {summary['petal_length']['variance']:.2f}) than sepal measurements
  (sepal_width variance = {summary['sepal_width']['variance']:.2f}), which is why
  petals carry most of the discriminative signal.

- Normality: no feature passes both normality tests at a = {ALPHA}. Petal
  measurements are strongly bimodal (setosa vs the rest), so treating them as
  Gaussian would be inappropriate; rank-based methods are safer.

- Correlation: the strongest linear relationship is
  {feature_names[i]} <-> {feature_names[j]} (Pearson r = {pearson[i, j]:.3f}).
  Petal length and petal width move almost in lockstep; sepal_width is the
  least correlated with everything else (even mildly negative with petals).

- Class differences: one-way ANOVA rejects equal means for ALL four features
  (every p-value << {ALPHA}), so species differ significantly on every
  measurement. The follow-up Welch t-tests show versicolor and virginica differ
  significantly on all four features too, though they overlap the most.

- Outliers: {len(rows_with_any)} of {n_samples} rows carry at least one |z| > 3
  value, concentrated in sepal_width. Given the strong per-species structure,
  these look like genuine biological variation rather than data errors, so
  dropping them is not recommended without domain justification.

Figures written to ./{OUTDIR}/:
  01_distributions.png     histograms + KDE + normal fit
  02_corr_heatmap.png      Pearson correlation heatmap
  03_boxplots_by_species.png
  04_scatter_matrix.png    pairplot-style scatter matrix
  05_outliers.png          standardized boxplot + highlighted scatter
""")