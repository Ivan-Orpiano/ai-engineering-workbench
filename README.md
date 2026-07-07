# iris-statistical-eda

Exploratory statistical analysis of the Iris dataset using **NumPy, SciPy & Matplotlib** — summary stats, normality tests, correlation, ANOVA/t-tests, and outlier detection. No pandas or sklearn used for the statistics.

![demo](figures/demo.gif)

## Features

- **Summary stats** — mean, median, variance, std, IQR per feature
- **Normality** — Shapiro–Wilk + D'Agostino K²
- **Correlation** — Pearson & Spearman matrices
- **Hypothesis tests** — one-way ANOVA across species + Welch t-test
- **Outliers** — z-score detection (|z| > 3)
- **Plots** — distributions, correlation heatmap, boxplots, scatter matrix

## Stack

`numpy` · `scipy` · `matplotlib` — pandas only loads the data, no ML models.


Prints all statistical summaries to the console and writes figures to `./figures/`.

## Output

| File | Content |
|------|---------|
| `01_distributions.png` | Histograms + KDE + normal fit |
| `02_corr_heatmap.png` | Pearson correlation heatmap |
| `03_boxplots_by_species.png` | Feature distributions by class |
| `04_scatter_matrix.png` | Pairplot-style scatter matrix |
| `05_outliers.png` | Standardized boxplot + flagged points |

