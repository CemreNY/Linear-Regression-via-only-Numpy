import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

'''

Boston Housing dataset is originally a part of UCI Machine Learning Repository
There are 506 observations and 13 features. The objective is to predict
prices of the house using the given features. Uncomment the following to see
a description of the features.

print(boston_dataset.DESCR)

'''

# Create a new column PRC and add it to the DataFrame. It was under the 'target' key of the dataset.
boston['PRC'] = boston_dataset.target

# features: all rows, all columns but the last one
X = boston.iloc[:,:-1]

# insert a vector of ones as the zeroth feature
X.insert(loc = 0, column = 'X_0', value = np.ones(X.shape[0]))

# target: all rows, only the last column, can refer to it by name
y = boston['PRC']

# Split the data into two sets: training (80%) and test (20%)
# We will use the test set to assess the model's performance on unseen data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

'''
We could simply use the following to fit a linear regression model.
But we want to practice and code gradient descent ourselves.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

'''
def mean_squared_error(theta, X, y):
	"""
	This is the cost function.
	theta, X, and y are NumPy arrays
	theta is the vector of coefficients 
	X is the matrix of observations
		Each row corresponds to a training example
		Each column corresponds to a feature
	y is the vector of targets in the training set
	"""

	y_hat = np.dot(X, theta)
	return np.sum((y - y_hat) ** 2) / (2 * X.shape[0])

def gradient_descent(theta, X, y, precision = 1e-6, learning_rate = 0.000006, print_interval = 10000, max_iters = 100000):
	"""
	This implements the batch gradient descent.
	theta, X, and y are NumPy arrays
	theta is the vector of coefficients 
	X is the matrix of observations
		Each row corresponds to a training example
		Each column corresponds to a feature
	y is the vector of targets in the training set

	Returns a better theta
	"""
	print("\nStarting gradient descent...")
	iter = 0
	while True:
		y_hat = np.dot(X, theta)

		# notice the use of matrix operations to calculate the gradient as a whole, instead of a loop which calculates just one partial derivative in each iteration
		step = learning_rate * np.dot(X.transpose(), (y_hat - y)) / X.shape[0]
		theta = theta - step
		if iter % print_interval == 0:
			print("Iteration: %7d\tMSE: %f" % (iter, mean_squared_error(theta, X, y)))
		if np.linalg.norm(step) < precision:
			break
		if iter >= max_iters:
			break
		iter += 1
	print("\nGradient descent terminated after %d iterations..." % (iter))
	print("...with Theta: {}".format(theta))
	print("\n...and MSE: \t%f" % (mean_squared_error(theta, X, y)))
	return theta


# Let's use normal equations to calculate optimal solution.
theta = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
np.set_printoptions(suppress = True, precision = 6)
print("Optimal Theta: {}".format(theta))
print("\nOptimal MSE for Training Set: \t%f" % (mean_squared_error(theta, X_train, y_train)))
print("Optimal MSE for Test Set: \t%f" % (mean_squared_error(theta, X_test, y_test)))
print("A big difference means we have overfitting problem (high variance)")

# Now, let's use gradient descent
theta = np.zeros(X_train.shape[1]) # initialize theta
theta_hat = gradient_descent(theta, X_train, y_train)
print("Test set MSE: \t%f" % (mean_squared_error(theta_hat, X_test, y_test)))
print("A big difference means we have overfitting problem (high variance)")

#Now, let's use feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# First, delete the dummy first column. We don't want to have that re-scaled. We are gonna add later.
X_train = X_train.drop("X_0", axis = 1)
X_test = X_test.drop("X_0", axis = 1)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the resulting NumPy.ndarray back into Pandas.DataFrame
X_train_scaled = pd.DataFrame(data = X_train_scaled, index = X_train.index, columns = X_train.columns)
X_test_scaled = pd.DataFrame(data = X_test_scaled, index = X_test.index, columns = X_test.columns)

# Insert back the dummy 0th column
X_train_scaled.insert(loc = 0, column = 'X_0', value = np.ones(X_train_scaled.shape[0]))
X_test_scaled.insert(loc = 0, column = 'X_0', value = np.ones(X_test_scaled.shape[0]))


# Let's use normal equations to calculate optimal solution.
theta = np.dot(np.dot(np.linalg.inv(np.dot(X_train_scaled.transpose(), X_train_scaled)), X_train_scaled.transpose()), y_train)
np.set_printoptions(suppress = True, precision = 6)
print("\nScaled Version Optimal Theta: {}".format(theta))
print("\nScaled Version Optimal MSE for Training Set: \t%f" % (mean_squared_error(theta, X_train_scaled, y_train)))
print("Scaled Version Optimal MSE for Test Set: \t%f" % (mean_squared_error(theta, X_test_scaled, y_test)))
print("A big difference means we have overfitting problem (high variance)")

# Now, let's use gradient descent
theta = np.zeros(X_train_scaled.shape[1]) # initialize theta

# Please observe how we can afford a much larger learning rate and the resulting much higher convergence rate
theta_hat = gradient_descent(theta, X_train_scaled, y_train, learning_rate = 0.3)
print("Test set MSE: \t%f" % (mean_squared_error(theta_hat, X_test_scaled, y_test)))
print("A big difference means we have overfitting problem (high variance)")
