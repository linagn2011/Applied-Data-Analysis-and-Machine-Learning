import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from Mycode.linearmodels import LinearRegression
from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

# Create data
np.random.seed(40)

n = 20      # n x n number of data points
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

maxdeg = 15
degrees = np.arange(0, maxdeg, dtype=int)

n_bootstraps = 400


bias2 = np.zeros(maxdeg)
variance = np.zeros(maxdeg)
MSE = np.zeros(maxdeg)
cycles = 20



for i, deg in enumerate(degrees):

    # Generate design matrix
    X = designMatrix(x, y, deg, with_intercept=False)
    
    bias2_boot = np.zeros(cycles)
    variance_boot = np.zeros(cycles)
    
    for j in range(cycles):
        seed_value = np.random.randint(4000)
        np.random.seed(seed_value)
        
        # Split data into train and test sets
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        # Scale the data, i.e, subtract the mean and divide by std (based on the training set)
        X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept=False)
        z_train_scaled, z_test_scaled = normalize_target(z_train, z_test)
        
        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        
        for trial_idx in range(n_bootstraps):
            X_, z_ = resample(X_train_scaled, z_train_scaled)
            ols_linreg = LinearRegression(fit_intercept=True)
            ols_linreg.fit(X_, z_)
            z_pred[:, trial_idx] = ols_linreg.predict(X_test_scaled)
            
        bias2_boot[j] = np.mean((z_test_scaled - np.mean(z_pred, axis=1))**2) # bias^2
        variance_boot[j] = np.mean(np.var(z_pred, axis=1))

    bias2[i] = np.mean(bias2_boot)
    variance[i] = np.mean(variance_boot)

MSE = bias2+variance

fig = plt.figure(figsize=(8, 6))
plt.title(r'Bias-Variance trade off for OLS with resampling and %g Datapoints' % (n))
plt.xlabel("Model Complexity")
plt.ylabel("Error")
plt.grid(True)
plt.plot(degrees, bias2, label="Bias")
plt.plot(degrees, variance, label="Variance")
plt.plot(degrees, MSE, label="MSE")
plt.legend()
# plt.yscale('log')
# plt.show()
plt.tight_layout()

fig.savefig(fig_path("Bias_Variance_trade_off_OLS_with_resampling_20.jpg"), dpi=300, transparent=True)
