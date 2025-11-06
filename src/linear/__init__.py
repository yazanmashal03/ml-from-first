from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .ridge_regression import RidgeRegression

__all__ = ["LinearRegression", "LogisticRegression", "RidgeRegression"]

REGISTRY = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "ridge_regression": RidgeRegression,
}