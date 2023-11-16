from activation_func import *
from cost_func import *

import activation_func
import cost_func

import numpy as np
class NN:
    def __init__(self,
                 X,
                 target,
                 n_neurons_list,
                 act_func_h = 'sigmoid',
                 act_func_o = identity,
                 initialization = None,
                 classification = False):
        """
        Initialize a Neural Network.

        Parameters:
        - X (matrix): Input data.
        - target (vector): Target values for the input data.
        - n_neurons_list (list): List containing the number of neurons in each hidden and output layer.
        - act_func_h (str): Activation function for hidden layers. Accepted values: 'sigmoid', 'relu', etc.
        - act_func_o (str): Activation function for the output layer. Accepted values: 'sigmoid', 'softmax', etc.
        - initialization (str): Weight initialization technique. Accepted values: 'xavier', 'he', or None for random initialization.
        - classification (bool): Type of output for the network ('classification' or 'regression'). Default is False for "regression".

        """
        self.X = X
        self.target = target
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_neurons_list = n_neurons_list
        self.layers = len(n_neurons_list)
        self.initialization = initialization
        self.weights, self.biases = self.initialize_weights_and_biases()
        self.layer_outputs =[]
        self.probabilities = list()
        self.act_func_o = act_func_o
        self.d_act_func_o = activation_func.identity_d
        self.cost = mean_squared_error
        self.d_cost = mean_squared_error_gradient
        self.classification = classification
        
        
        
        
        
        if(act_func_h == "sigmoid"):
            self.act_func_h = activation_func.sigmoid
            self.d_act_func_h = activation_func.sigmoid_d
        elif(act_func_h == "relu"):
            self.act_func_h = activation_func.relu
            self.d_act_func_h= activation_func.relu_d
        elif(act_func_h == "leakyrelu"):
            self.act_func_h = activation_func.leakyrelu
            self.d_act_func_h = activation_func.leakyrelu_d
        elif(act_func_h == "elu"):
            self.act_function = activation_func.elu
            self.d_act_func_h = activation_func.d_elu
        else:
            raise ValueError("please choose ,'sigmoid','relu','leakyrelu' or 'elu' \
                  as your activation function for ther hidden layers. Try again")
        
 
        if self.classification:
            if (self.act_func_o == "sigmoid"):
                self.act_func_o = activation_func.sigmoid
                self.d_act_func_o = activation_func.sigmoid_d
            elif (self.act_func_o == "softmax"):
                self.act_func_o = activation_func.softmax
                self.d_act_func_o= activation_func.softmax_d
            else:
                print('no activation function has been chosen, default activation function: identity has been set. ')
            self.cost = binary_cross_entropy
            self.d_cost= binary_cross_entropy_gradient
            
        
        
        
    def initialize_weights_and_biases(self):
        """
        Initialize weights and biases for each layer in a neural network.

        This method initializes the weights and biases of each layer in a neural network based on the specified
        weight initialization technique. The supported techniques are 'xavier' and 'he'. If no specific technique
        is provided, the default initialization is random.
    
        Returns:
        - weights (list of 2D arrays): List containing weight matrices for each layer.
        - biases (list of 2D arrays): List containing bias vectors for each layer.
    
        Raises:
        - ValueError: If an invalid weight initialization technique is provided. Choose 'xavier' or 'he'.
        """
        np.random.seed(0)
        n_layers = len(self.n_neurons_list)
        n_neurons = [self.n_features] + self.n_neurons_list

        if self.initialization is not None:
            if self.initialization == 'xavier':
                weights = [np.random.normal(0,np.sqrt(1.0/n_neurons[i]),(n_neurons[i], n_neurons[i-1])) for i in range(1, n_layers + 1)]
                biases = [np.random.randn(n_neurons[i], 1) for i in range(1, n_layers + 1)]
                
        
            elif(self.initialization == "he"):
                weights = [np.random.normal(0,np.sqrt(2.0/n_neurons[i]),(n_neurons[i], n_neurons[i-1])) for i in range(1, n_layers + 1)]
                biases = [np.random.randn(n_neurons[i], 1) for i in range(1, n_layers + 1)]
                
            else:
                raise ValueError("Invalid weight initialization. Choose 'xavier' or 'he'.")
        else:
            weights = [np.random.randn(n_neurons[i], n_neurons[i-1]) for i in range(1, n_layers + 1)]
            biases = [np.random.randn(n_neurons[i], 1) for i in range(1, n_layers + 1)]
        
        return weights, biases
    
    def batch_generator(self,X, y, batch_size, shuffle=True):
        """
        Generate batches from input data X and target data y.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n, m), where n is the number of samples and m is the number of features.
        - y (numpy.ndarray): Target data vector of shape (n,).
        - batch_size (int): Size of each batch.
        - shuffle (bool): Whether to shuffle the input data before generating batches. Default is True.

        Yields:
        - X_batch (numpy.ndarray): Batch of input data of shape (batch_size, m).
        - y_batch (numpy.ndarray): Batch of target data of shape (batch_size,).
        """
        np.random.seed(0)
        n = X.shape[0]
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        num_batches = n // batch_size
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices, :]
            y_batch = y[batch_indices]
          
            yield X_batch, y_batch

        # If there's a remainder, yield the last batch
        if n % batch_size != 0:
            
            start_idx = num_batches * batch_size
            batch_indices = indices[start_idx:]

            X_batch = X[batch_indices, :]
            y_batch = y[batch_indices]

            yield X_batch, y_batch
    
    def clip_gradients(self,gradients, clip_value):
        """
        Clip gradients to prevent the exploding gradient problem.
    
        Parameters:
        - gradients: List or array of gradients.
        - clip_value: Threshold for clipping.
    
        Returns:
        - clipped_gradients: List of clipped gradients.
        """
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
        scale = clip_value / grad_norm if grad_norm > clip_value else 1.0
        clipped_gradients = [g * scale for g in gradients]

        return clipped_gradients
        
    
    
    def forward_pass(self, X, training = False):
        """
        Performs a forward pass through the NN.

        Parameters:
        - X (numpy.ndarray): Input data of shape (n_samples, n_features).
        - Training (boolean): defines if the forward pass is used for training. Defalut is False.

        Returns:
        - output (numpy.ndarray): Output of the network.
        """
        self.output_v = []
        self.output_a = []
        
        layer_output = X
        
        layer_outputs =[layer_output]
        
        # forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            # Perform matrix multiplication
            
            weighted_input = layer_output @ self.weights[i].T 
            layer_input = weighted_input  + self.biases[i].T
            layer_output = self.act_func_h(layer_input)
            layer_outputs.append(layer_output)
        #forward pass through output layer
        
        output_layer_value = layer_output @ self.weights[-1].T + self.biases[-1].T
        output = self.act_func_o(output_layer_value)
        
        self.output_v = output_layer_value
        self.output_a = output
        
        layer_outputs.append(output)
        
        if training:
        # Store the layer outputs in memory after each forward pass, if the network is training
            self.layer_outputs.append(layer_outputs)
        
        return output
    
    def backward_pass(self, X, target, lmb, clip_value):
        """
        Perform the backward pass (backpropagation) of the neural network to update weights and biases.
    
        Parameters:
        - X (matrix): Input data.
        - target (vector): Target values for the input data.
        - lmb (float): Regularization parameter (lambda) for controlling overfitting.
        - clip_value (float or None): Optional parameter to clip the gradient values during backpropagation.
                                     If None, no gradient clipping is applied.
        """
        
        self.lmb = lmb
        
        #First layer(outputlayer)
        # Retrieve stored layer outputs from the forward pass
        layer_outputs = self.layer_outputs.pop()
      
        delta = self.d_cost(layer_outputs[-1], target)
      
        # Gradient Clipping
        if clip_value is not None:

            gradients = [layer_outputs[-1].T @ delta, np.sum(delta, axis=0, keepdims=True)]
            gradients = self.clip_gradients(gradients, clip_value=clip_value)
      
            dweights = gradients[0]
            dbiases = gradients[1]
        else:
            dweights = layer_outputs[-1].T @ delta
            dbiases = np.sum(delta, axis=0, keepdims=True)

        if self.lmb > 0.0:
            dweights = dweights + self.lmb * self.weights[-1].T

        # Update output layer parameters GD
        self.weights[-1] -= self.learning_rate * dweights.T
        self.biases[-1] -= self.learning_rate * dbiases.reshape(-1, 1)
        
        #Hidden layers
        for i in range(len(self.weights) - 2, 0, -1):
            delta = delta @ self.weights[i + 1] * self.d_act_func_h(layer_outputs[i])

            # Calculate gradients for hidden layers
            if clip_value is not None:
                
                gradients = [layer_outputs[i].T @ delta, np.sum(delta, axis=0, keepdims=True)]
                gradients = self.clip_gradients(gradients, clip_value=clip_value)
          
                dweights = gradients[0]
                dbiases = gradients[1]
                
            else:
                
                dweights = layer_outputs[i].T @ delta
                dbiases = np.sum(delta, axis=0, keepdims=True)

            if self.lmb > 0.0:
                dweights += self.lmb * self.weights[i].T
            # Update hidden layer parameters GD
            self.weights[i] -= self.learning_rate * dweights.T
            self.biases[i] -= self.learning_rate * dbiases.reshape(-1, 1)
    
        
        #Last layer (Input layer)
        
        delta = delta @ self.weights[1]
      
        if clip_value is not None:
            
            gradients = [layer_outputs[0].T @ delta, np.sum(delta, axis=0, keepdims=True)]
            gradients = self.clip_gradients(gradients, clip_value=clip_value)
      
            dweights = gradients[0]
            dbiases = gradients[1]
        else:
            dweights = layer_outputs[0].T @ delta
            dbiases = np.sum(delta, axis=0, keepdims=True)

        if self.lmb > 0.0:
            dweights += self.lmb * self.weights[0].T
        
        #Update input layer parameters GD
        self.weights[0] -= self.learning_rate * dweights.T
        self.biases[0] -= self.learning_rate * dbiases.reshape(-1, 1)
    
    
    
    def train(self, epochs=1000, learning_rate=0.01, batch_size = 20, lmb = 0, clip_value = None):
        """
       Trains the neural network using backpropagation.

       Parameters:
       - epochs (int, optional): Number of training epochs.
       - learning_rate (float, optional): Learning rate for weight updates.
       - batch_size (int, optional): Size of each training batch.
       - lmd (float, optional): Regularization parameter L1
       - clip_value (float or None): Optional parameter to clip the gradient values during backpropagation.
                                    If None, no gradient clipping is applied.

       Returns:
       None
       """
        self.clip_value = clip_value
        self.lmb = lmb
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.errors = []
        total_error = 0
        
        for epoch in range(epochs):
            #X_batch, target_batch = self.batch_generator(self.X, self.target)
            batches = self.batch_generator(self.X, self.target, self.batch_size)
        
            
            for X_batch, target_batch in batches:
                output =self.forward_pass(X_batch, training=True)
                self.backward_pass(X_batch, self.learning_rate,lmb = self.lmb, clip_value=self.clip_value)
                
                # Calculate error using the cost function
                batch_error = self.cost(output, target_batch.reshape(-1))
                total_error += np.sum(batch_error)
            
            
                
            # Calculate average error for the epoch
            avg_error = total_error / self.n_samples
            self.errors.append(avg_error)
            total_error = 0
    
            # Print or log the error for monitoring
            print(f"Epoch {epoch + 1}/{epochs}, Average Error: {avg_error}")
            
    def predict(self, X):
        """
       Make predictions using the trained neural network.

       Parameters:
       - X (numpy.ndarray): Input data for making predictions.

       Returns:
       numpy.ndarray: Predicted outputs.
       """
        output = self.forward_pass(X)
        if self.classification:
            if self.act_func_o == activation_func.softmax:
                print('the networks confidence in each value output has been returned,\
                  in order to get the output values: "self.v"')
            else:
                for i in range(len(output)):
                    if (output[i] < 0.5):
                        output[i] = int(0)
                    elif (output[i]>= 0.5):
                        output[i] = int(1)
                return output.ravel()
        
        return output
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    

            
        
            