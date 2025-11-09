# Implementing linear regression from scratch
import numpy as np
from src.core.base import Regressor


class LinearRegression(Regressor):
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.weights = w
        return self

    def predict(self, X):
        return X @ self.weights

    def fit_ridge(self, X, y, alpha=1.0):
        """
        Fit the model using Ridge Regression (L2 regularization) built from scratch.
        
        Ridge regression solves the regularized least squares problem:
            w* = argmin ||Xw - y||² + alpha * ||w||²
        
        The analytical solution is:
            w* = (X^T X + alpha * I)^(-1) X^T y
        
        This implementation uses the pseudoinverse for numerical stability.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        alpha : float, default=1.0
            Regularization strength (lambda). Larger values mean more regularization.
            Must be non-negative.
        
        Returns:
        --------
        self : LinearRegression
            Returns self for method chaining.
        """
        n_samples, n_features = X.shape
        
        # Ridge regression: add alpha * I to X^T X for L2 regularization
        # Using pseudoinverse for numerical stability
        w = np.linalg.pinv(X.T @ X + alpha * np.eye(n_features)) @ X.T @ y
        self.weights = w
        return self

    def set_regularization(self, X, y, reg):
        n_samples, n_features = X.shape

        # we solve analytically for weight vector
        w = np.linalg.pinv(X.T @ X + reg * np.eye(n_features)) @ X.T @ y
        self.weights = w
        return self
    
    def opt_lambda(self, X_train, y_train, X_valid, y_valid, lambda_range=np.linspace(0, 100, 100)):
        """
        Find optimal lambda by fitting on training set and evaluating on validation set.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_valid : array-like
            Validation features
        y_valid : array-like
            Validation targets
        lambda_range : array-like
            Range of lambda values to search
            
        Returns:
        --------
        best_lambda : float
            Lambda value that minimizes validation MSE
        """
        best_lambda = None
        best_mse = float('inf')
        for l in lambda_range:
            # Fit on training set
            w = np.linalg.pinv(X_train.T @ X_train + l * np.eye(X_train.shape[1])) @ X_train.T @ y_train
            # Evaluate on validation set
            mse = np.mean((y_valid - X_valid @ w) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_lambda = l
        return best_lambda