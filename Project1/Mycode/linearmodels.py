import numpy as np


class StatisticalMetrics:

    def rss(self, data, target):
        """
        RSS - Residual Sum of Squares
        """
        return np.sum((target - self.predict(data))**2)

    def sst(self, target):
        """
        SST - Sum of Squares Total
        """
        return np.sum((target - np.mean(target))**2)

    def r2(self, data, target):
        """
        Calculate the R^2-score, coefficient of determination (R^2-score)
        """
        return 1 - self.rss(data, target) / self.sst(target)

    def mse(self, data, target):
        """
        MSE - Mean Squared Error
        """
        return np.mean((target - self.predict(data))**2)
    
    def bias(self, data, target):
        """
        Bias - Simplification of the model
        """
        return np.mean((target - np.mean(self.prdict(data)))**2)
    
class LinearRegression(StatisticalMetrics):
    """
    Ordinary Least Squares (OLS) Regression
    """

    # the intercept, adds 1 with True.
    def __init__(self, fit_intercept=True, normalize=False):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept
        self._normalize = normalize

    def normalize_data(self):
        """
        Normalize data with the exception of the intercept column
        """
        if self._fit_intercept:
            self.data_mean = np.mean(self.data, axis=0)
            self.data_std = np.std(self.data, axis=0)

            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return data_norm
        else:
            self.data_mean = np.mean(self.data[:, 1:], axis=0)
            self.data_std = np.std(self.data[:, 1:], axis=0)
            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return np.c_[np.ones(X.shape[0]), data_norm]

    def fit(self, X, y):
        """
        Fit the model
        ----------
        Input: design matrix (data), target data
        """
        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:  # find shape of array
            # reshape takes all data with [-1] and makes it 2D
            _X = self.data.reshape(-1, 1)
        else:
            _X = self.data

        # if normalize data
        if self._normalize:
            _X = self.normalize_data()

        # add bias if fit_intercept
        if self._fit_intercept:
            _X = np.c_[np.ones(X.shape[0]), _X]

        self._inv_xTx = np.linalg.pinv(_X.T @ _X)  # pseudo-inverse
        beta = self._inv_xTx @ _X.T @ self.target

        # set attributes
        if self._fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = np.mean(self.target)
            self.coef_ = beta

        return self.coef_

    def predict(self, X):
        """
        Model prediction
        """
        # reshapes if X is wrong
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_

    def coef_CI(self, critical_value=1.96):
        """
        Estimate a confidence interval of the coefficients

        The critical value for a 90% confidence interval is 1.645
        The critical value for a 95% confidence interval is 1.96
        The critical value for a 98% confidence interval is 2.326
        The critical value for a 99% confidence interval is 2.576

        Returns lower and upper bound as sets in a list.
        """
        beta_std = np.sqrt(np.diag(self._inv_xTx))
        beta = self.coef_
        data_mse_sqrt = np.sqrt(self.mse(self.data, self.target))
        CI = [[beta[i] - critical_value * beta_std[i] * data_mse_sqrt, beta[i] +
               critical_value * beta_std[i] * data_mse_sqrt]for i in range(len(beta))]
        return CI


class RidgeRegression(StatisticalMetrics):
    """
    Linear Model Using Ridge Regression.
    """

    def __init__(self, lmbda=1.0, fit_intercept=True, normalize=False):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept
        self._normalize = normalize

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        if isintstance(value, (int, float)):
            self._lmbda = value
        else:
            raise ValueError("Penalty must be int or float")

    def normalize_data(self):
        """
        Normalize data with the exception of the intercept column
        """
        if self._fit_intercept:
            self.data_mean = np.mean(self.data, axis=0)
            self.data_std = np.std(self.data, axis=0)

            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return data_norm
        else:
            self.data_mean = np.mean(self.data[:, 1:], axis=0)
            self.data_std = np.std(self.data[:, 1:], axis=0)
            data_norm = (
                self.data[:, 1:] - self.data_mean[np.newaxis, :]) / self.data_std[np.newaxis, :]
            return np.c_[np.ones(X.shape[0]), data_norm]

    def fit(self, X, y):

        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:  # find shape of array
            # reshape takes all data with [-1] and makes it 2D
            _X = self.data.reshape(-1, 1)
        else:
            _X = self.data

        # if normalize data
        if self._normalize:
            _X = self.normalize_data()

        # add bias if fit_intercept
        if self._fit_intercept:
            _X = np.c_[np.ones(X.shape[0]), _X]

        # calculate coefficients
        xTx = _X.T @ _X
        lmb_eye = self._lmbda * np.identity(xTx.shape[0])
        _inv_xTx = np.linalg.pinv(xTx + lmb_eye)  # pseudo-inverse
        coef = _inv_xTx @ _X.T @ self.target

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = np.mean(self.target)
            self.coef_ = coef

        return self.coef_

    def predict(self, X):
        """
        Model prediction
        """
        # reshapes if X is wrong
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_