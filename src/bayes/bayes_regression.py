# Implementing bayesian regression. This has the form y = Xtheta + epsilon
import numpy as np
from src.core.base import Regressor

class BayesianRegression(Regressor):
    def __init__(self, prior_mean=None, prior_cov=None, noise_var=1.0):
        self.m0 = prior_mean
        self.S0 = prior_cov
        self.sigma2 = noise_var

        self.mN = None
        self.SN = None
        self.fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        if self.m0 is None:
            self.m0 = np.zeros(n_features)
        if self.S0 is None:
            self.S0 = np.eye(n_features) * 1e6

        S0_inv = np.linalg.inv(self.S0)
        precision = S0_inv + (X.T @ X) / self.sigma2
        self.SN = np.linalg.inv(precision)

        b = S0_inv @ self.m0 + (X.T @ y) / self.sigma2
        self.mN = self.SN @ b

        self.fitted = True
        return self

    def predict(self, X, return_std=False):
        if not self.fitted:
            raise RuntimeError("Model must be fitted")
        
        X_new = np.asanyarray(X)
        mean = X_new @ self.mN
        if not return_std:
            return mean
        
        var = np.sum(X_new @ self.SN * X_new, axis=1) + self.sigma2
        std = np.sqrt(var)
        return mean, std
    

