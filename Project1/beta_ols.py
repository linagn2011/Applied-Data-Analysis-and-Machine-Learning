import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Mycode.linearmodels import LinearRegression, RidgeRegression
from Mycode.ml_tools import *
from Mycode.project_tools import fig_path
# Create data
np.random.seed(42)

n = 40         # number of data points
sigma2 = 0.1   # irreducible error
deg = 5

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
noise = np.random.normal(0, sigma2, int(n*n))

# Create mesh and unravel
x, y = np.meshgrid(x, y)
x = np.ravel(x)
y = np.ravel(y)

# Observed data
z = frankeFunction(x, y) + noise

# Generate design matrix
X = designMatrix(x, y, deg, with_intercept=False)

# Split data into train and test sets
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

# Scale the data, i.e, subtract the mean and divide by the std
# scaler.fit computes the mean and std to be used for later scaling
# It is important that scaling is done based on the training set
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create linear regression object
ols_linreg = LinearRegression(fit_intercept=True)

# Train the model using the (scaled) training sets
ols_linreg.fit(X_train_scaled, z_train)

# Get coefficients
beta = ols_linreg.coef_

# CI
z = 1.96  # 95% CI
beta_var = np.diag(np.linalg.pinv(X_train_scaled.T @ X_train_scaled))
beta_std = np.sqrt(beta_var)

CI = [[beta[i] - z * beta_std[i], beta[i] + z * beta_std[i]] for i in range(len(beta))]

# Make a nice plot
fig = plt.figure(figsize=(8, 6))

cmap = plt.get_cmap("Reds")
norm = matplotlib.colors.Normalize(vmin=-10, vmax=len(CI))

for i in range(len(CI)):
    plt.plot(CI[i], (i, i), color=cmap(norm(i)))
    plt.plot(CI[i], (i, i), "o", color=cmap(norm(i)))

    
plt.show()