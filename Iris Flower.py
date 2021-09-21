#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data
data_set = pd.read_csv('IRIS.csv')
cols = data_set.shape[1] #Number of Columns
X = data_set.iloc[:, 0:cols-1] #Independent variables
y = data_set.iloc[:, cols-1:cols] #dependent variable

print('*'*50)
print('Data:')
print(data_set.head(10))
print('*'*50)
print('*'*50)
print('Data.describe:')
print(data_set.describe())
print('*'*50)

#Encoding the data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Splitting the data to Training & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

#ML model LogisticRegression in training data
from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression(random_state= 0)
classifier.fit(X_train, y_train)

#Prediction 
y_pred = classifier.predict(X_test)

#Evaluating the model
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
cm = confusion_matrix(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#Printing Results
print('*'*50)
print('Confusion Matrix:')
print(cm)
print('*'*50)
print('*'*50)
print("R-Squared Value= ", R2)
print('*'*50)
print('*'*50)
print('Mean-Squared-Error= ', mse)
print('*'*50)

#Dimensionality Reduction Using Backward Elimination
import statsmodels.api as sm
X.insert(0,'Ones', 1)
X_opt = np.array(X.iloc[:, [1,3,4]])
regressor = sm.OLS(y, X_opt).fit()

#OLS Regression Results
print(regressor.summary())

#Vislualization
setosa = data_set[data_set['species'].isin(['Iris-setosa'])]
versicolor = data_set[data_set['species'].isin(['Iris-versicolor'])]
virginica = data_set[data_set['species'].isin(['Iris-virginica'])]

fig, ax= plt.subplots()
ax.scatter(setosa['sepal_length'], setosa['petal_width'],
           c= 'b', marker= 'o', label= 'setosa'
           )
ax.scatter(versicolor['sepal_length'], versicolor['petal_width'],
           c= 'r', marker= 'x', label= 'versicolor'
           )
ax.scatter(virginica['sepal_length'], virginica['petal_width'],
           c= 'g', marker= '+', label= 'virginica'
           )
ax.set_xlabel('sepal_length')
ax.set_ylabel('petal_width')
ax.legend()

