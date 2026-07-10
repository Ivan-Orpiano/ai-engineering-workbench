#PERFORM EDA AND PRE PROCESSING - LINEAR REGRESSION
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[["MedInc, HouseAge", 'AveRooms']]
y = df['MedHouseVal']

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 42)

model = LinearRegression()
model.fit(X_train, y_train)

#predictions

y_pred = model.predict(X_test)

#evaluate the performance 
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression MSE: ", mse)