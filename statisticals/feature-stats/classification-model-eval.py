from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#load dataset
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#train logistic 
model= LogisticRegression()
model.fit(X_train, y_train)

#predict






