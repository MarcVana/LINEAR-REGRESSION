"""
Created on Thu Sep 24 16:34:06 2020

LINEAR REGRESSION MODEL PROJECT

@author: Marc
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv('Ecommerce Customers.csv')
plot = sns.pairplot(customers)

# Data preprocessing
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y = customers['Yearly Amount Spent']

# Split the data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

# Create and fit the linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)

# Results shown in a scatter plot
plt.figure(2)
predictions = model.predict(X_test)
plt.scatter(Y_test, predictions)
plt.xlabel('True values')
plt.ylabel('Predicted values')

# Model evaluation
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(Y_test, predictions))
print('MSE: ', metrics.mean_squared_error(Y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
print('EVS: ', metrics.explained_variance_score(Y_test, predictions))

# Coefficients
cdf = pd.DataFrame(model.coef_, X.columns, columns = ['Coeff'])
print('------------------------------------')
print(cdf)