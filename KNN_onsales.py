import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('./sales_data_sample.csv', encoding='unicode_escape')
print("---------------head------------------")
print(df.head())
print("-------------INFO-------------------------")
print(df.info)


to_drop = ["ADDRESSLINE1", "ADDRESSLINE2", "STATE", "POSTALCODE", "PHONE"]
df = df.drop(to_drop, axis=1)

print("---------------------------------head--------------------------")
print(df.head)


print("--------------------null count-------------------------------")
print(df.isnull().sum())

df.dropna(inplace=True)
print("---------------after removing nulls -------------------")
print(df.head())

df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"])

import datetime as dt 

snapshot_date = df["ORDERDATE"].max() + dt.timedelta(days = 1)

df_RFM = df.groupby(["CUSTOMERNAME"]).agg({
    'ORDERDATE': lambda X : (snapshot_date - X.max()).days,
    'ORDERNUMBER': 'count',
    'SALES': 'sum'
})

df_RFM.rename(columns = {
    'ORDERDATE':'Recency',
    'ORDERNUMBER':'Frequency',
    'SALES':'MonetaryValue'
}, inplace=True)

print("--------------------------After renaming columns-------------------------")
print(df_RFM.head())


#dividing data into 4 quartiles

df_RFM['M'] = pd.qcut(df_RFM['MonetaryValue'], q=4, labels = range(1, 5))
df_RFM['R'] = pd.qcut(df_RFM['Recency'], q=4, labels = range(4, 0, -1))
df_RFM['F'] = pd.qcut(df_RFM['Frequency'], q=4 ,labels = range(1, 5))

print('-------------after dividing into 4 quantiles-------------------')
print(df_RFM.head())


df_RFM['RFM_score'] = df_RFM[['R', 'F', 'M']].sum(axis=1)

print("---------------------RFM score---------------------------")
print(df_RFM.head())

data = df_RFM[['Recency', 'Frequency', 'MonetaryValue']]
print("-------------------------------head-------------------------")
print(data.head())

#log transformation
data_log = np.log(data)
print("------------------------------after log transformation removing skweed data--------------------")
print(data_log.head())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_log)
data_normalized = pd.DataFrame(data_normalized, index = data_log.index, columns = data_log.columns)

print("-------------normalized data----------------------")
print(data_normalized.head())

from sklearn.cluster import KMeans
Mapping = {}

for k in range(1, 21):
    Kmeans = KMeans(n_clusters = k, random_state = 1)
    Kmeans.fit(data_normalized)
    Mapping[k] = Kmeans.inertia_


import warnings
warnings.filterwarnings("ignore")
plt.title("Elbow plot")
sns.pointplot(x = list(Mapping.keys()), y = list(Mapping.values()))
plt.show()

#selecting k-values and buolding model
Kmeans = KMeans(n_clusters=5, random_state=1)
Kmeans.fit(data_normalized)
cluster_labels = Kmeans.labels_

data_rfm = data.assign(Cluster = cluster_labels)
print("-------------------After clustring---------------------")
print(data_rfm.head())