from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from NN_own import NN
import numpy as np
import matplotlib.pyplot as plt
BC = load_breast_cancer()
X = BC.data
z = BC.target


X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)



scaler_X = StandardScaler()
X_train_normalized = scaler_X.fit(X_train)
X_test_normalized = scaler_X.fit(X_test)


# # Normalize target data (z)
# scaler_z = StandardScaler()
# z_train_reshaped = z_train.reshape(-1, 1)
# z_test_reshaped = z_test.reshape(-1, 1)
# z_train_normalized = scaler_z.fit_transform(z_train_reshaped)
# z_test_normalized = scaler_z.transform(z_test_reshaped)


# X_train, X_test, z_train, z_test = X_train_normalized, X_test_normalized, z_train_normalized, z_test_normalized


neurons = [3, 3, 1]
batch_size = 20
n_epochs = 1000
eta = 0.10

model =NN(X_train,z_train, neurons,act_func_h='sigmoid', act_func_o = "sigmoid", classification = True)



model.train(epochs = n_epochs,learning_rate = eta)

z_pred_prob = model.predict(X_train)

outputs = model.output_v

z_tilde = model.predict(X_test).reshape(-1,1)


def accuracy_score_own(target, pred):
    return np.sum(target == pred) / len(target)

acc_train = accuracy_score_own(z_train,outputs)
acc_test = accuracy_score_own(z_test, z_tilde)

print(acc_train)
print(acc_test)


acc_sickit = accuracy_score(z_test,z_tilde)
print(acc_sickit)

np.hidden()




# print(len(z_test))
# print(len(z_pred))
# print(z_pred)
# print(z_test)