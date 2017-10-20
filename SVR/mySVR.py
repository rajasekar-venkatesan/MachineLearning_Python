#PolyReg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

##Splitting to train and test
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Feature Scaling - No inherent feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape((-1,1)))

#Fitting PolyReg
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Prediction with LinReg and PolyReg
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(6.5)))

#Visualizing LinReg and PolyReg
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'black')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'black')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()