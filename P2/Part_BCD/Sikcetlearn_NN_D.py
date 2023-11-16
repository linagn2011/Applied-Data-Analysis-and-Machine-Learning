from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
BC = load_breast_cancer()
X = BC.data
z = BC.target


X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2,random_state=12)

scaler_X = StandardScaler()
X_train_normalized = scaler_X.fit(X_train)
X_test_normalized = scaler_X.fit(X_test)

def accuracy_score_own(target, y):
    return np.sum(target == y) / len(target)

neurons_list1 = [15, 15, 15, 15, 15, 15, 1]
neurons_list2 = [5, 5, 5, 5, 5, 5, 1]
neurons_list3 = [30, 30, 30, 30, 30, 30, 1]
neurons_list_of_lists = [neurons_list1, neurons_list2, neurons_list3]
act_func_list = ['logistic', 'relu']

batch_size = 20
n_epochs = 1000
eta = 0.10
layers_sim =[]
result_dict = {}  # Initialize an empty dictionary to store DataFrames
for act in act_func_list:
    result_df = pd.DataFrame()  # Initialize an empty DataFrame for the current activation function

    for neurons_list in neurons_list_of_lists:
        acc_list = []
        acc_sciket_list = []

        for i in range(1, len(neurons_list) + 1):
            neurons = neurons_list[:i]

            # Append the last element of neurons_list if not already present
            if neurons[-1] != neurons_list[-1]:
                neurons.append(neurons_list[-1])
            layers_sim.append(neurons)
            model = MLPClassifier(hidden_layer_sizes=neurons,
                                  max_iter=n_epochs, 
                                  random_state=12, 
                                  solver='sgd',
                                  activation=str(act), 
                                  batch_size= batch_size,
                                  learning_rate_init=eta,
                                  momentum=0
                                  )
            model.fit(X_train, z_train)
            
            z_pred_prob = model.predict(X_train)
            z_tilde = model.predict(X_test).reshape(-1, 1)

            acc_test = accuracy_score_own(z_test, z_tilde)
            acc_sciket = accuracy_score(z_test, z_tilde)

            acc_list.append(acc_test)
            acc_sciket_list.append(acc_sciket)

        # Create a DataFrame for the current loop iteration
        df = pd.DataFrame({
            'Neurons': [neurons_list[:i] for i in range(1, len(neurons_list) + 1)],
            'Accuracy_Test': acc_list,
            'Accuracy_Scikit': acc_sciket_list
        })

        # Concatenate the current DataFrame to the result_df
        result_df = pd.concat([result_df, df], ignore_index=True)

    # Add the DataFrame to the dictionary with the activation function as the key
    result_dict[act] = result_df

# Access DataFrames by activation function
print(result_dict['logistic'])
print(result_dict['relu'])


# Calculate the accuracy of the model
accuracy = accuracy_score(z_test, z_tilde)
classification_report_str = classification_report(z_test, z_tilde)


