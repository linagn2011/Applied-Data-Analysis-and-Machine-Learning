
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import KFold

from Mycode.linearmodels import LinearRegression
from Mycode.ml_tools import *
from Mycode.project_tools import fig_path

np.random.seed(40)

n = 40        # number of data points
sigma2 = 0.01   # irreducible error
sigma = np.sqrt(sigma2)


maxdeg = 15
degrees = np.arange(maxdeg)

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
noise = np.random.normal(0, sigma, int(n*n))

# Create mesh and unravel
x, y = np.meshgrid(x, y)
x = np.ravel(x)
y = np.ravel(y)

# Observed data
z = frankeFunction(x, y) + noise

scores_KFold_train = np.zeros(maxdeg)
scores_KFold_test = np.zeros(maxdeg)
      

MSE_test = np.zeros(maxdeg)
MSE_train = np.zeros(maxdeg)

k = 5
kfold = KFold(n_splits = k)

for degree in range(maxdeg):
    MSE_test[degree], MSE_train[degree] = Kfold_ols(k, x, y, z, degree, n)
    

fig = plt.figure(figsize=(8, 6))
plt.grid(alpha=0.5)
plt.plot(degrees, MSE_train, label='Train MSE')
plt.plot(degrees,MSE_test, label='Test MSE')
plt.legend()
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.title(r'K-fold Cross Validation on OLS, k =%g, n = %g' % (k, n))
plt.yticks([10**n for n in range(-4,2)])
plt.yscale('log')
plt.show()

fig.savefig(fig_path("Cross_validation_OLS_n_40_k_5.jpg"), dpi=300, transparent=True)