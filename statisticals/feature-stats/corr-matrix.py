#corr_matrix.py
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

#load dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

correlation_matrix = df.corr()

#plot heatmap
plt.figure(figsize = (10,8))
sns.heatmap(correlation_matrix, annot = True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


correlated_features = correlation_matrix['target'].sort_values(ascending=False)
print("Features Most Correlated with Target: ")
print(correlated_features)


