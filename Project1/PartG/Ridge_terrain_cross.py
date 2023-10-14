
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import KFold

from imageio import imread

from Mycode.linearmodels import RidgeRegression
from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

# Load the terrain
terrain = imread("n35_e138_1arc_v3.tif")

slice_idx = 10
terrain = terrain[::slice_idx, ::slice_idx]

nx = np.shape(terrain)[0]
ny = np.shape(terrain)[1]

# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x, y = np.meshgrid(x, y)
x = np.ravel(x)
y = np.ravel(y)

z = terrain.ravel()
n = len(z)
print(n)
degrees = [5, 8, 12, 15]

# setup for cross-validation
N = 20
lmbdas = np.logspace(-12, 5, N)

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

MSE = []
fig = plt.figure(figsize=(8, 6))

for i, deg in enumerate(degrees):

    # Generate design matrix
    X = designMatrix(x, y, deg, with_intercept=False)
    
    # Perform the cross-validation to estimate MSE
    scores_KFold_train = np.zeros((N, k))
    scores_KFold_test = np.zeros((N, k))

    for j, lmbda in enumerate(lmbdas):
    
        k = 0
        for train_inds, test_inds in kfold.split(x):
            X_train = X[train_inds]
            z_train = z[train_inds]

            X_test = X[test_inds]
            z_test = z[test_inds]

            # Scale the data, i.e, subtract the mean and divide by std (based on the training set)
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept=False)
            z_train_scaled, z_test_scaled = normalize_target(z_train, z_test)

            ridge_reg = RidgeRegression(lmbda=lmbda, fit_intercept=True)
            ridge_reg.fit(X_train_scaled, z_train_scaled)
            scores_KFold_train[j, k] = ridge_reg.mse(X_train_scaled, z_train_scaled)
            scores_KFold_test[j, k] = ridge_reg.mse(X_test_scaled, z_test_scaled)
        k += 1
    
    estimated_mse_KFold_train = np.mean(scores_KFold_train, axis=1)
    estimated_mse_KFold_test = np.mean(scores_KFold_test, axis=1)
    MSE.append(estimated_mse_KFold_test)
    ax = fig.add_subplot(2, 2, i+1)
    ax.grid(alpha=0.3)
    ax.plot(np.log10(lmbdas), estimated_mse_KFold_train, label='Train MSE')
    ax.plot(np.log10(lmbdas), estimated_mse_KFold_test, label='Test MSE')
    ax.set_xlabel(r"$\log_{10} \lambda$")
    ax.set_ylabel("MSE")
    ax.set_title(f"Model Complexity {deg}")
    ax.legend()

print(MSE)
plt.tight_layout()
plt.show()
fig.savefig(fig_path("Cross_validation_Ridge_terrain_n_1600.jpg"), dpi=300, transparent=True)
