import numpy as np
from src.linear.linear_regression import LinearRegression
from src.core.metrics import mse

def test_linear_regression_exact_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    true_w = np.array([2.0, -3.0])
    y = X @ true_w + 5.0
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    assert mse(y, preds) < 1e-10
