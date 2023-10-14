from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

from imageio import imread

from Mycode.linearmodels import LinearRegression
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
n = len(z)
# setup for Cross-validation
maxdeg = 20
degrees = np.arange(maxdeg)


MSE_test = np.zeros(len(degrees))
MSE_train = np.zeros(len(degrees))
k = 5
for degree in degrees:
    MSE_test[degree], MSE_train[degree] = Kfold_ols(k, x, y, z, degree, n) #Kfold with small f is our own
    

fig = plt.figure(figsize=(8, 6))
plt.grid(alpha=0.5)
plt.plot(degrees, MSE_train, label='Train MSE')
plt.plot(degrees,MSE_test, linestyle='--', label='Test MSE')
plt.legend()
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.title(r'K-fold Cross Validation, k = 5, slice index = %g n = %g' % (slice_idx, n))
plt.yticks([10**n for n in range(-4,2)])
plt.yscale('log')
plt.show()
fig.savefig(fig_path("Cross_validation_OLS_terrain.jpg"), dpi=300, transparent=True)
print(MSE_train)
print(MSE_test)
print(degrees)