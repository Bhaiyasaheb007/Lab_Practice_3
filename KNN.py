import numpy as np
import pandas as pd

data = pd.read_csv('./diabetes.csv')
print(data.head())
print(data.isnull().sum())

#relpacing zeros with mean value
for column in data.columns[1:-3]:
	data[column].replace(0, np.NaN, inplace=True)
	data[column].fillna(round(data[column].mean(skipna=True)), inplace=True)

print(data.head(10))

x = data.iloc[:, :8]
y = data.iloc[:, 8:]
print(x)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_fit = knn.fit(X_train, Y_train.values.ravel())
knn_pred = knn_fit.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
print("Confusion Matrix")
print(confusion_matrix(Y_test, knn_pred))
print("Accuracy Score:", accuracy_score(Y_test, knn_pred))
print("Recall Score:", recall_score(Y_test, knn_pred))
print("F1 Score:", f1_score(Y_test, knn_pred))
print("precision Score:", precision_score(Y_test, knn_pred))

