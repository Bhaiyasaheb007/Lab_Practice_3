import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


#preprocessing
df = pd.read_csv("./uber.csv")
print("---------------------head----------------------------")
print(df.head())
print("---------------------describe------------------------")
print(df.describe)
print("-------------------shape-----------------------------")
print(df.shape)
print("---------------------null count--------------------")
print(df.isnull().sum())
print("------------------------------------------------------")
df.drop(labels = "Unnamed: 0", axis = 1, inplace=True)
df.drop(labels = "key", axis=1, inplace=True)

print(df.describe)

df.dropna(inplace=True)

import warnings
warnings.filterwarnings("ignore")

sns.distplot(df["fare_amount"])
plt.show()
sns.distplot(df["pickup_latitude"])
plt.show()
sns.distplot(df["pickup_longitude"])
plt.show()
sns.distplot(df["dropoff_latitude"])
plt.show()
sns.distplot(df["dropoff_longitude"])
plt.show()

#finding outliers
sns.boxplot(df["fare_amount"])
plt.show()
def find_Outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1

    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers

outliers = find_Outliers_IQR(df["fare_amount"])

print("-----------------------------------------")
print("Number of outilers: ", str(len(outliers)))
print("------------------------------------------")
print("Max of outliers: ", str(outliers.max()))
print("------------------------------------------")
print("Min of outliers: ", str(outliers.min()))
print("----------------outliers------------------")
print(outliers)

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()



X = df.drop(["fare_amount", "pickup_datetime"], axis =1 )
Y = df["fare_amount"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state = 0)

#model building
from sklearn.linear_model import LinearRegression
lmodel = LinearRegression()
lmodel.fit(X_train, Y_train)
lmodel_pred = lmodel.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
rmodel = RandomForestRegressor(n_estimators = 5, random_state = 6)
rmodel.fit(X_train, Y_train)
rmodel_pred = rmodel.predict(X_test)


#RSME metrics
from sklearn.metrics import mean_squared_error
linear_RSME = np.sqrt(mean_squared_error(lmodel_pred, Y_test))
print("----------------------------------------------------")
print("RSME value for linear Regression model: ", linear_RSME)
random_RSME = np.sqrt(mean_squared_error(rmodel_pred, Y_test))
print("----------------------------------------------------")
print("RSME value for Random Forest Regression Model: ",random_RSME )

