"""Fit the curve."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optimizer

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
theta, J_history = optimizer.gradient_descent(
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
