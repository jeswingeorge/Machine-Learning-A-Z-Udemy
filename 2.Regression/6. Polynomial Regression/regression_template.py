import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#  to get matrix of X
X = dataset.iloc[:,1:2].values
# to get y as vector
y = dataset.iloc[:,2]

# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# No need for feature scaling for now as Linear Regression functions will take care of it.
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScalerardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)


# Fitting the regression model to the dataset
## Create your regressor here


# Predicting the new results with Polynomial regression
y_pred = regressor.predict()

# Visualizing the regression results
plt.scatter(X, y, color = 'r');
plt.plot(X, regressor.predict(X), color = 'b');
plt.title("Polynomial Regression results with degree ");
plt.xlabel('xlabel');
plt.ylabel('ylabel');


# Visualizing the Regression results for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'r');
plt.plot(X_grid, regressor.predict(X_grid), color = 'b');
plt.title("");
plt.xlabel('xlabel');
plt.ylabel('ylabel');