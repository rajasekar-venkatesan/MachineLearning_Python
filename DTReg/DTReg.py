#DTReg
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

##Feature Scaling - No inherent feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y.reshape((-1,1)))

#Fitting DTReg
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Prediction with DTReg
y_pred = regressor.predict(6.5)

#Visualizing DTReg
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'black')
plt.title('DTReg')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'black')
plt.title('DTReg')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

