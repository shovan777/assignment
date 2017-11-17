"""Compute the cost."""
import numpy as np


def compute_cost(X, Y, theta):
    """Compute the LMS cost.

    Parameters
    ----------
    X : np.matrix
        Value of features.
    Y : np.matrix
        Target Value.
    theta : np.matrix
        Feature matrix.

    Returns
    -------
    np.float64
        C.

    """
    m = X.shape[0]
    J = 0.
    J = (1. / (2 * m)) * np.sum(np.square(np.dot(X, theta) - Y))
    return J
