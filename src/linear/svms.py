# implementing linear support vector machines.

import numpy as np
from src.core.base import Classifier
from src.core.utils import sigmoid
from src.core.metrics import accuracy, precision, recall

class LinearSVM(Classifier):
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.n_iterations = 1000
        self.learning_rate = 0.01

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        for _ in range(self.n_iterations):
            scores = X @ self.w
            viol = y * scores < 1

            grad_w = self.w - self.C * (X[viol].T @ y[viol]) + self.w
            self.w -= self.learning_rate * grad_w
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w)
    
    def evaluate(self, X, y):
        return accuracy(y, self.predict(X)), precision(y, self.predict(X)), recall(y, self.predict(X))
