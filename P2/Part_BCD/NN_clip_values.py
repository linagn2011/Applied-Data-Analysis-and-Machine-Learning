import numpy as np
from input_data import *

from NN_own import NN
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=12)

#Scaling function for the design matrix. obtains mean= 0 and variance=1
def scale(X_train, X_test, y_train,y_test):
    X_train_scaled = (X_train - X_train.mean()) / (np.std(X_train))
    X_test_scaled = (X_test - X_train.mean()) / (np.std(X_train))
 
    y_train_scaled = (y_train - y_train.mean()) /(np.std(y_train)) 
    y_test_scaled = (y_test - y_train.mean()) / (np.std(y_train)) 
    return X_train_scaled, X_test_scaled, y_train_scaled,y_test_scaled



X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train,z_test)


neurons = [3,3,1]

model_sig= NN(X_train,z_train,neurons, act_func_h = 'sigmoid')
model_relu=NN(X_train,z_train,neurons, act_func_h = 'relu')
model_leakyrelu =NN(X_train,z_train,neurons, act_func_h = 'leakyrelu')

lam = 0
eta = 0.1
n_iterations = 1000

clip_values = np.logspace(-6, 2, 8)

mse_sig_list =[]
mse_relu_list =[]
mse_leakyrelu_list =[]

for clip_value in clip_values:
    
    model_sig.train(batch_size = 20, lmb = lam, learning_rate = eta, epochs = n_iterations, clip_value=clip_value)
    model_relu.train(batch_size = 20, lmb = lam, learning_rate = eta, epochs = n_iterations, clip_value=clip_value)
    model_leakyrelu.train(batch_size = 20, lmb = lam, learning_rate = eta, epochs = n_iterations, clip_value=clip_value)
    
    z_pred_sig= model_sig.predict(X_test)
    z_pred_relu=model_relu.predict(X_test)
    z_pred_leakyrelu=model_leakyrelu.predict(X_test)
    
    mse_sig = mse(z_test,z_pred_sig)
    mse_relu = mse(z_test,z_pred_relu)
    mse_leakyrelu = mse(z_test,z_pred_leakyrelu)
    
    mse_sig_list.append(mse_sig)
    mse_relu_list.append(mse_relu)
    mse_leakyrelu_list.append(mse_leakyrelu)
    
    error_sig = model_sig.errors
    error_relu = model_relu.errors
    error_leakyrelu = model_leakyrelu.errors
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(error_sig)), error_sig, label='Sigmoid', color='blue')
    ax.plot(np.arange(len(error_relu)), error_relu, label='ReLU', color='green')
    ax.plot(np.arange(len(error_leakyrelu)), error_leakyrelu, label='Leaky ReLU', color='red')
    
    ax.set_title(f'Comparison of Error Curves, gradient_clip = {clip_value}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.legend()
    
    plt.show()

df_mse = pd.DataFrame({'Key': clip_values, 'MSE_sig': mse_sig_list, 'MSE_relu': mse_relu_list, 'MSE_leakyrelu': mse_leakyrelu_list})

