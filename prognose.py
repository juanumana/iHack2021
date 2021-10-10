# Adapting the Code from Jaques Grobler, License: BSD 3 clause in order to predict Flooding situations according to the input data from IoT Sensor of aQuartier
# IM AUFBAU
# Model should be deployed in order to be used for practical porpuses in aQuartier

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

s_X, s_y = datasets.loads(return_X_y=True)

# Use only one feature
s_X = s_X[:, np.newaxis, 2]

# Split the data into training/testing sets
s_X_train = s_X[:-20]
s_X_test = s_X[-20:]

# Split the targets into training/testing sets
s_y_train = s_y[:-20]
s_y_test = s_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(s_X_train, s_y_train)

# Make predictions using the testing set
s_y_pred = regr.predict(s_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(s_y_test, s_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(s_y_test, s_y_pred))

# Plot outputs
plt.scatter(s_X_test, s_y_test,  color='black')
plt.plot(s_X_test, s_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
