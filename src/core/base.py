from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

Array = np.ndarray

class Regressor(ABC):
    @abstractmethod
    def fit(self, X: Array, y: Array) -> "Regressor": ...
    @abstractmethod
    def predict(self, X: Array) -> Array: ...
    def score(self, X: Array, y: Array) -> float:
        from .metrics import r2_score
        return float(r2_score(y, self.predict(X)))

class Classifier(ABC):
    @abstractmethod
    def fit(self, X: Array, y: Array) -> "Classifier": ...
    @abstractmethod
    def predict(self, X: Array) -> Array: ...
    def score(self, X: Array, y: Array) -> float:
        from .metrics import accuracy
        return float(accuracy(y, self.predict(X)))
