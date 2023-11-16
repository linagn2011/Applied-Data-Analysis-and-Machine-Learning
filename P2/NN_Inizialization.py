import numpy as np
from input_data import *
from NN_own import NN
from sklearn.model_selection import train_test_split
from cost_func import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def r2_score(y, y_pred):
    mean_y = np.mean(y)
    ss_total = np.sum((y - mean_y)**2)
    ss_residual = np.sum((y - y_pred)**2)
    
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Create code
n = 40         # n x n number of data points
sigma2 = 0.01   # irreducible error
sigma = np.sqrt(sigma2)

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
noise = np.random.normal(0, sigma, int(n*n))

# create mesh and unravel
x, y = np.meshgrid(x, y)
x = np.ravel(x)
y = np.ravel(y)

# observed data
z = FrankeFunction(x, y) + noise

x.ravel()
y.ravel()

X = np.c_[x,y]

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

neurons = [3,3,3,1]

model_none= NN(X_train,z_train,neurons, act_func_h = 'sigmoid')
model_xavier=NN(X_train,z_train,neurons, act_func_h = 'sigmoid',initialization='xavier')
model_he=NN(X_train,z_train,neurons, act_func_h = 'sigmoid',initialization='he')

model_none.train()
model_xavier.train()
model_he.train()

mse_train_none =model_none.errors
mse_train_xavier =model_xavier.errors
mse_train_he =model_he.errors

plt.plot(np.arange(0,1000),mse_train_none)
plt.plot(np.arange(0,1000),mse_train_xavier)
plt.plot(np.arange(0,1000),mse_train_he)





