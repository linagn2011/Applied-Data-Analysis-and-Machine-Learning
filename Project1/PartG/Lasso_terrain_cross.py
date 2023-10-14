import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

from imageio import imread

from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

# Load the terrain
terrain = imread("n35_e138_1arc_v3.tif")

slice_idx = 91
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
degrees = [5, 8, 12, 15]

# setup for cross-validation
N = 20
lmbdas = np.logspace(-12, 2, N)

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

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

            lasso_reg = Lasso(alpha=lmbda, fit_intercept=True)
            lasso_reg.fit(X_train_scaled, z_train_scaled)
            
            z_fit = lasso_reg.predict(X_train_scaled)

            # Make predictions using the test set
            z_pred = lasso_reg.predict(X_test_scaled)
            
            scores_KFold_train[j, k] = mse(z_train_scaled, z_fit)
            scores_KFold_test[j, k] = mse(z_test_scaled, z_pred)
        k += 1
    
    estimated_mse_KFold_train = np.mean(scores_KFold_train, axis=1)
    estimated_mse_KFold_test = np.mean(scores_KFold_test, axis=1)
    
    ax = fig.add_subplot(2, 2, i+1)
    ax.grid(alpha=0.3)
    ax.plot(np.log10(lmbdas), estimated_mse_KFold_train, label='Train MSE')
    ax.plot(np.log10(lmbdas), estimated_mse_KFold_test, label='Test MSE')
    ax.set_xlabel(r"$\log_{10} \lambda$")
    ax.set_ylabel("MSE")
    ax.set_title(f"Model Complexity {deg}")
    ax.legend()

plt.tight_layout()
plt.show()
fig.savefig(fig_path("Cross_validation_Lasso_terrain_n_1600.jpg"), dpi=300, transparent=True)