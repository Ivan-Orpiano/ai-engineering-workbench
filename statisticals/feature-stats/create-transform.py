import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#load bike sharing dataset
df = pd.read_csv("dataset/bike_sharing_daily.csv")


df['dteday'] = pd.to_datetime(df['dteday'])

#creating a new features

df['day_of_week'] = df ['dteday'].dt.day_name
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

X = df[['temp']]
y = df['cnt']

#apply polynomial transformation
poly = PolynomialFeatures(degree =2, include_bias = False)
X_poly = poly.fit_transform(X)


#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_poly_train, X_poly_test = train_test_split(X_poly, test_size=0.2, random_state = 42)

#train and evaluate model with original features
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)

#Train and evaluate model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train   )
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

#comparing the results
print(f"MSE Original: {mse_original:.2f}")
print(f"MSE Polynomial: {mse_poly:.2f}")