import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from input_data import *
import matplotlib.pyplot as plt
import seaborn as sns



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



X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,random_state=12)

#Scaling function for the design matrix. obtains mean= 0 and variance=1
def scale(X_train, X_test, y_train,y_test):
    X_train_scaled = (X_train - X_train.mean()) / (np.std(X_train))
    X_test_scaled = (X_test - X_train.mean()) / (np.std(X_train))
 
    y_train_scaled = (y_train - y_train.mean()) /(np.std(y_train)) 
    y_test_scaled = (y_test - y_train.mean()) / (np.std(y_train)) 
    return X_train_scaled, X_test_scaled, y_train_scaled,y_test_scaled



X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train,z_test)

lam = 0
eta = 0.1
n_iterations = 1000

model_sig = MLPRegressor(hidden_layer_sizes=(3,3),
                      activation='logistic',
                      solver='sgd',
                      batch_size=20,
                      learning_rate='constant',
                      learning_rate_init=eta,
                      alpha=lam,
                      max_iter=n_iterations,
                      momentum=0,
                      random_state=12
                      )
model_relu = MLPRegressor(hidden_layer_sizes=(3,3),
                      activation='relu',
                      solver='sgd',
                      batch_size=20,
                      learning_rate='constant',
                      learning_rate_init=eta,
                      alpha=lam,
                      max_iter=n_iterations,
                      momentum = 0,
                      random_state=12)

model_sig.fit(X_train,z_train)
model_relu.fit(X_train,z_train)

z_pred_sig = model_sig.predict(X_train)
z_tilde_sig= model_sig.predict(X_test)

z_pred_relu = model_relu.predict(X_train)
z_tilde_relu= model_relu.predict(X_test)

mse_train_sig = mse(z_pred_sig,z_train)
mse_train_relu = mse(z_pred_relu,z_train)

mse_test_sig =mse(z_tilde_sig,z_test)
mse_test_relu = mse(z_tilde_relu,z_test)

print(mse_train_relu,mse_train_sig)
print(mse_test_relu,mse_test_sig)





