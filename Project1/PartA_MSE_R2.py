from sklearn.model_selection import train_test_split

from Mycode.linearmodels import LinearRegression
from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

# Create data
np.random.seed(40)

n = 40         # n x n number of data points
sigma2 = 0.01   # irreducible error
sigma = np.sqrt(sigma2)

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
noise = np.random.normal(0, sigma, int(n*n))
                         
# Create mesh and unravel
x, y = np.meshgrid(x, y)
x = np.ravel(x)
y = np.ravel(y)

# Observed data
z = frankeFunction(x, y) + noise

deg = 5          # degree


# Generate design matrix
X = designMatrix(x, y, deg, with_intercept=False)
print(f"Design matrix shape: {X.shape}")

# Split data into train and test sets
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

# Scale the data, i.e, subtract the mean and divide by std (based on the training set)
X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept=False)
z_train_scaled, z_test_scaled = normalize_target(z_train, z_test)

# Create linear regression object
ols_linreg = LinearRegression(fit_intercept=True)

# Train the model using the (scaled) training sets
ols_linreg.fit(X_train_scaled, z_train_scaled)

# Intercept and coefficients
beta0 = ols_linreg.intercept_
beta = ols_linreg.coef_
print(f"Intercept: {beta0}")
print("Coefficients: \n", beta)
print(f"Coefficients shape: {beta.shape}")

# MSE
mse_train = ols_linreg.mse(X_train_scaled, z_train_scaled)
mse_test = ols_linreg.mse(X_test_scaled, z_test_scaled)
print(f"Train MSE: {mse_train}")
print(f"Test MSE: {mse_test}")

# R2
r2_train = ols_linreg.r2(X_train_scaled, z_train_scaled)
r2_test = ols_linreg.r2(X_test_scaled, z_test_scaled)
print(f"Train R2: {r2_train}")
print(f"Test R2: {r2_test}")