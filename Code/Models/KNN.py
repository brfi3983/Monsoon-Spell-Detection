from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

climate_data = pd.read_csv(
	"C:\\Users\\user\\Documents\\Personal\\Research\\Lead DataSets\\Lead_1.csv")

climate_data.columns = ['test1', 'test2', 'test3', 'test4', 'test5', 'class']
print(climate_data['class'].value_counts())

X = climate_data.drop(['class'], axis=1)
Y = climate_data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=15)

#Train the model using the training sets
knn.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
