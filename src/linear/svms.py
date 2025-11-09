# implementing linear support vector machines.

import numpy as np
from src.core.base import Classifier

class LinearSVM(Classifier):
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)