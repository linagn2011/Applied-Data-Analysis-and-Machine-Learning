import numpy as np

def mean_squared_error(y, targets):
    n = y.shape[0]
    mse_loss = np.sum((y - targets) ** 2) / (2 * n)
    return mse_loss

def mean_squared_error_gradient(y, targets):
    n = y.shape[0]
    mse_gradient = (y - targets) / n
    return mse_gradient

def binary_cross_entropy(y, targets):
    not_0 = 10**-20
    n = len(targets)
    loss = -np.sum((targets * np.log(y + not_0) + (1 - targets) * np.log(1 - y + not_0))) / n
    return loss

def binary_cross_entropy_gradient(targets, y):
    not_0 = 10**(-20)
    gradient = -(targets / y - (1 - targets) / (1 - y))
    return gradient

#