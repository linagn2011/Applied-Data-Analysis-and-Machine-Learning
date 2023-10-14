import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

np.random.seed(41)

n = 40          # n x n number of data points
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
z = frankeFunction(x, y) + noise

degrees = [2,3,4,5]

#setup for ridge
N = 100
lambdas = np.logspace(-10, 5, N)

Lasso_r2_pred = np.zeros(N)
Lasso_r2_train = np.zeros(N)

fig = plt.figure(figsize=(8, 6))

for i, deg in enumerate(degrees):
    
    # Generate design matrix
    X = designMatrix(x, y, deg, with_intercept=False)

    # Split data into train and test sets
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # scale the data
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept=False)
    z_train_scaled, z_test_scaled = normalize_target(z_train, z_test)
    
    for j, lmbda in enumerate(lambdas):
        lasso_reg = Lasso(alpha=lmbda, fit_intercept=True)
        lasso_reg.fit(X_train_scaled, z_train_scaled)
        
        z_pred= lasso_reg.predict(X_test_scaled) #target*Beta_ridge
        z_tilde=lasso_reg.predict(X_test_scaled) #target*Beta_ridge
        
        Lasso_r2_pred[j] = r2(z_test_scaled, z_pred)
        Lasso_r2_train[j] = r2(z_test_scaled, z_tilde)

    ax = fig.add_subplot(2, 2, i+1)
    ax.grid(alpha=0.3)
    ax.plot(np.log10(lambdas),Lasso_r2_pred, label='r2_test')
    ax.plot(np.log10(lambdas),Lasso_r2_train, label='r2_train')
    ax.set_xlabel(r"$\log_{10} \lambda$")
    ax.set_ylabel("Error")
    ax.set_title(f"Model Complexity {deg}")
    ax.legend()
    
print(Lasso_r2_pred, Lasso_r2_train)
plt.tight_layout()
plt.show()
fig.savefig(fig_path("lasso_lamdba_r2.jpg"), dpi=300, transparent=True)
