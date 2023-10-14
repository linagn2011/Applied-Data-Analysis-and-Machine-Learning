import pandas as pd

from sklearn.model_selection import train_test_split

from Mycode.linearmodels import LinearRegression, RidgeRegression
from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

# Create data
np.random.seed(40)

n = 40       # n x n number of data points
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

maxdeg = 5
degrees = np.arange(1, maxdeg+1, dtype=int)

mse = pd.DataFrame(columns=["train", "test"], index=degrees-1)
r2 = pd.DataFrame(columns=["train", "test"], index=degrees-1)

# Initialize an empty dictionary to store beta values
beta_dict = {}


for i, deg in enumerate(degrees):

    # Generate design matrix
    X = designMatrix(x, y, deg, with_intercept=False)

    # Split data into train and test sets
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Scale the data, i.e, subtract the mean and divide by std (based on the training set)
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept=False)
    z_train_scaled, z_test_scaled = normalize_target(z_train, z_test)

    # Create linear regression object
    ols_linreg = LinearRegression(fit_intercept=True)

    # Train the model using the (scaled) training sets
    ols_linreg.fit(X_train_scaled, z_train_scaled)

    # # Get coefficients
    beta_dict[f'beta_{i+1}'] = ols_linreg.coef_
    beta_g= list(beta_dict.values())
    
    z_fit = ols_linreg.predict(X_train_scaled)

    # Make predictions using the test set
    z_pred = ols_linreg.predict(X_test_scaled)

    # Statistical metrics
    mse["train"][i] = ols_linreg.mse(X_train_scaled, z_train_scaled)
    mse["test"][i] = ols_linreg.mse(X_test_scaled, z_test_scaled)    
    r2["train"][i] = ols_linreg.r2(X_train_scaled, z_train_scaled)
    r2["test"][i] = ols_linreg.r2(X_test_scaled, z_test_scaled)
    
   
fig = plt.figure(figsize=(8, 6))
for i, beta in enumerate(beta_g):
    plt.plot(np.arange(len(beta)), beta, label=f'Degree {degrees[i]}')

# Customize the plot
plt.xlabel('Coefficient Index')
plt.ylabel('Beta Value')
plt.title('Coefficients with Model Complexity')
plt.legend()
plt.grid(True)
fig.savefig(fig_path("Beta_model_complex.jpg"), dpi=300, transparent=True)

fig, (ax1, ax2) = plt.subplots(2,figsize=(9,8), sharex=True)
#fig.suptitle(r'MSE and R^2 with model complexity')
ax1.plot(degrees, mse["train"], "o--", label="Training data_MSE")
ax1.plot(degrees, mse["test"], "o--", label="Test data_MSE")
ax1.title.set_text(r'MSE with model complexity')
ax1.set_ylabel('MSE')
ax1.legend()

ax2.plot(degrees,r2["train"], "o--", label="Training data_R^2")
ax2.plot(degrees,r2["test"], "o--", label="Test data_R^2")
ax2.title.set_text(r'R^2 with model complexity')
ax2.set_xlabel("Model Complexity")
ax2.set_ylabel('R2')
ax2.legend()
fig.savefig(fig_path("MSE_R^2_forced_overfit.jpg"), dpi=300, transparent=True)