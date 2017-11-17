"""Fit the curve."""

import numpy as np
import pandas as pd
import compute_cost
import matplotlib.pyplot as plt


def gradient_descent(X, Y, theta, alpha, iterations, normal_factor):
    """Apply batch gradient_descent.

    Parameters
    ----------
    X : np.matrix
        Value of features.
    Y : np.matrix
        Target Value.
    theta : np.matrix
        Feature matrix.
    alpha : float
        Learning rate.
    iterations : int
        Number of epochs.

    Returns
    -------
    np.matrix
        Optimized weights.

    """
    m = Y.shape[0]
    print(m)
    J_accumulate = np.zeros((iterations, 1), dtype=np.float64)
    for i in range(iterations):
        # print(theta)
        theta = theta - alpha * (1. / m) * np.dot(X.T, (np.dot(X, theta) - Y))
        J_accumulate[i] = compute_cost.compute_cost(X, Y, theta)
#     return theta, J_accumulate
    theta[0] = theta[0] * normal_factor
    return theta, J_accumulate


iterations = 10000
alpha = 0.5
theta = np.zeros((2, 1), dtype=np.float64)
house_data = pd.read_csv("house-prices.csv", header=None)
house_data_array = house_data.as_matrix()
X = house_data_array[:, 1]
Px = X
normal_factor = np.max(Px)

# normalizing X
X = X / normal_factor
# X = X / 100000.
X = np.asmatrix(X)
X = X.T
# Adding a bias to X
X = np.hstack((np.ones((X.shape[0], 1)), X))
# X = X / np.max(Px)

Y = house_data_array[:, 0]
Py = Y
# normalizing Y
Y = Y / normal_factor
Y = np.asmatrix(Y)
Y = Y.T
theta, J_history = gradient_descent(
    X, Y, theta, alpha, iterations, normal_factor)
print("optimized theta [A B]:\n>>", theta)
print("\nfinal_cost:J_history[-1]:\n>>", J_history[-1])
# print(compute_cost.compute_cost(X, Y, theta))
print("\nPlot of linear fit:\n")
Px_mat = np.asmatrix(Px)
Px_mat = Px_mat.T
Px_mat = np.hstack((np.ones((Px_mat.shape[0], 1)), Px_mat))
fig = plt.figure()
plt.plot(Px, Py,  'bo', Px, np.dot(Px_mat, theta),
         'r')
fig.suptitle('Line Plot')
fig.savefig('line.png')
plt.close(fig)
