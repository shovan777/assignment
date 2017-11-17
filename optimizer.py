"""Optimize the function to reduce cost."""
import numpy as np
import compute_cost


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
        theta = theta - alpha * (1. / m) * np.dot(X.T, (np.dot(X, theta)-Y))
        J_accumulate[i] = compute_cost.compute_cost(X, Y, theta)
#     return theta, J_accumulate
    theta[0] = theta[0] * normal_factor
    return theta, J_accumulate
