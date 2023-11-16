import numpy as np
def sigmoid(x):
        return 1.0/ (1.0 + np.exp(-x))
        
        
def relu(x):
        x[x <= 0] = 0
        return x
        

def leakyrelu(x, alpha=0.2):
    x[x <= 0] = alpha * x[x <= 0]
    return x


def elu(x, alpha =0.2):
    neg = x < 0.0
    x[neg] = alpha * (np.exp(x[neg]) - 1.0)
    return x


def softmax(x):
    a = np.exp(x)
    p = a / np.sum((a), axis=1, keepdims=True)
    return p

        #Derivatives
def sigmoid_d(x):
    return x * (1 - x)
    
    
def relu_d(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

    
    
def leakyrelu_d(x, alpha=0.2):
    x[x > 0] = 1
    x[x <= 0] = alpha
    return x

    
    
def elu_d(x, alpha =0.2):
    x[x > 0] = 1
    x[x <= 0] = x[x <= 0] + alpha
    return x
    
    
def softmax_d(x):
    s = softmax(x)
    return s * (1 - s)

def identity(x):
    s=x
    return s

def identity_d(x):
    return 1
    
    