'''Codes for Non Linear Models

Classes:
-------------------------------------------
Gradient Descent
Gradient Descent with momentum
Stochastic Gradient Descent
Stochastic Gradient Descent with momentum'''


# Gradient Descent Linear Regression Class
class GradientDescentLinearRegression:
    '''Docstring'''
    
    # Initialize the object with a learning rate and n number of iterations specified as input
    def __init__(self, learning_rate=0.01, n_iterations=1000): 
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    # fit method, takes in design matrix X and target value y as input
    def fit(self, X, y): #
        n_samples, n_features = X.shape # Extracting datapoints and columns from design matrix X, neccessary to determine "theta"
        self.theta = np.random.randn(n_features) # Coefficients or weights of the model
        
        # Gradient Descent algorithm
        for _ in range(self.n_iterations):
            # Gradients, used to update the model parameters in the direction thath minimizes losses.
            gradients = (2 / n_samples) * X.T @ (X @ self.theta - y)  

            # self.learning_rate is the step-size
            self.theta -= self.learning_rate * gradients # Moving in the direction of steepest descent.
            
    # Method to make a prediction (on new test data)
    def predict(self, X):
        return X @ self.theta

    # Method for calculating Mean Squared Error
    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    # Method for calculating R2 score
    def r2(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    


# Gradient Descent Linear Regression Class WITH MOMENTUM
class GD_momentum:
    '''Explain the parameters'''
    
    # Initialize the object with a learning rate and n number of iterations specified as input
    def __init__(self, learning_rate=0.01, n_iterations=1000, momentum = 0): 
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None # Initializing the weights/parameters
        self.momentum = momentum

    # fit method, takes in design matrix X and target value y as input
    def fit(self, X, y): #
        n_samples, n_features = X.shape # Extracting datapoints and columns from design matrix X, neccessary to determine "theta"
        self.theta = np.random.randn(n_features) # Coefficients or weights of the model
        
        # Adding momentum
        change = 0.0
        delta_momentum = 0.3

        # Gradient Descent algorithm
        for _ in range(self.n_iterations):
            gradients = (2 / n_samples) * X.T @ (X @ self.theta - y)  # Gradients, used to update the model parameters in the direction that minimizes losses.
            new_change = self.learning_rate*gradients + delta_momentum*change # calculating update
            self.theta -= new_change # Moving in the direction of steepest descent. self.learning_rate is the step-size
            change = new_change # saving the change
            
    # Method to make a prediction (on new test data)
    def predict(self, X):
        return X @ self.theta

    # Method for calculating Mean Squared Error
    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    # Method for calculating R2 score
    def r2(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    

# Now using Stochastic Gradient Descent, adding mini-batches and epochs, WITHOUT momentum

class SGD_plain:
    '''Explain the parameters'''
    
    # Initialize the object with a learning rate and n number of iterations specified as input
    def __init__(self, learning_rate=0.01, n_iterations=1000, n_epochs = 10, batch_size = 5): 
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.theta = None # Initializing the weights/parameters

    # fit method, takes in design matrix X and target value y as input
    def fit(self, X, y): #
        n_samples, n_features = X.shape # Extracting datapoints and columns from design matrix X, neccessary to determine "theta"
        self.theta = np.random.randn(n_features) # Coefficients or weights of the model
        
        # Adding mini-batches and epochs
        m = int(self.n_iterations / self.batch_size) #number of minibatches
        j = 0
        for epoch in range(1,self.n_epochs+1):
            for i in range(m):
                k = np.random.randint(m) #Pick the k-th minibatch at random
                #Compute the gradient using the data in minibatch Bk
                xi = X[k:k+self.batch_size]
                yi = y[k:k+self.batch_size]
                gradients = (2 / self.batch_size) * xi.T @ (xi @ self.theta - yi)  
                self.theta -= self.learning_rate * gradients # Moving in the direction of steepest descent.
                #Compute new suggestion for 
                j += 1
            
    # Method to make a prediction (on new test data)
    def predict(self, X):
        return X @ self.theta

    # Method for calculating Mean Squared Error
    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    # Method for calculating R2 score
    def r2(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
import numpy as np 

# Now using Stochastic Gradient Descent, adding mini-batches and epochs, WITH momentum

class SGD_momentum:
    '''Explain the parameters'''
            # Initialize the object with a learning rate and n number of iterations specified as input
    def __init__(self, learning_rate=0.01, n_iterations=1000, n_epochs = 10, batch_size = 5, momentum = 0): 
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.theta = None # Initializing the weights/parameters

    # fit method, takes in design matrix X and target value y as input
    def fit(self, X, y): #
        n_samples, n_features = X.shape # Extracting datapoints and columns from design matrix X, neccessary to determine "theta"
        self.theta = np.random.randn(n_features) # Coefficients or weights of the model
        
        # Adding mini-batches and epochs
        m = int(self.n_iterations / self.batch_size) #number of minibatches
        j = 0
        # Adding momentum
        change = 0.0
        delta_momentum = 0.3
        for epoch in range(1,self.n_epochs+1):
            for i in range(m):
                k = np.random.randint(m) #Pick the k-th minibatch at random
                #Compute the gradient using the data in minibatch Bk
                xi = X[k:k+self.batch_size]
                yi = y[k:k+self.batch_size]
                gradients = (2 / self.batch_size) * xi.T @ (xi @ self.theta - yi) 
                new_change = self.learning_rate*gradients + delta_momentum*change # calculating update
                self.theta -= new_change # Moving in the direction of steepest descent. self.learning_rate is the step-size
                change = new_change # saving the change
                #Compute new suggestion for 
                j += 1
            
    # Method to make a prediction (on new test data)
    def predict(self, X):
        return X @ self.theta

    # Method for calculating Mean Squared Error
    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    # Method for calculating R2 score
    def r2(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
