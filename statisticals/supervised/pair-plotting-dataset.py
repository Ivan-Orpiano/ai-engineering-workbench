#PERFORM EDA AND PRE PROCESSING
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns


#LOAD DATASET'
data = fetch_california_housing(as_frame=True)
df = data.frame

# #Define features and target
# X = df[['MedInc', 'HouseAge', 'AveRooms']]
# y = df['MedHouseVal']

#Inpect data 
print(df.info())
print(df.describe)

# visualize relationships
sns.pairplot(df, vars = ['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal'])
plt.show()

#check for missing values
print("Missing Values: \n", df.isnull().sum())


