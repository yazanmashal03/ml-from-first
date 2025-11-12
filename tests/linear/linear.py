import numpy as np
from src.linear.linear_regression import LinearRegression
from src.core.utils import train_valid_test_split
from src.linear.logistic_regression import LogisticRegression

def test_linear_regression():
    # shape
    m = 300
    n = 10

    # feature matrix, Xij ~ Unif([0, 1]), X.shape = (300, 20)
    X = np.random.uniform(size = (m, n))

    # generate random noise eps, eps.shape = (300,)
    eps = np.random.normal(size = m)

    # uniformly random weights weight, weight.shape = (20,)
    weight = np.random.uniform(size = n)

    # y = X weight + eps, y.shape = (300,)
    y = X @ weight + eps

    # split the data into train, valid, and test
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)

    # fit the model
    model = LinearRegression().fit(X_train, y_train)

    # we now regularize the model
    lr = model.opt_lambda(X_train, y_train, X_valid, y_valid) 
    print(f"Optimal lambda: {lr}")
    model.fit_ridge(X_train, y_train, lr)

    # predict the model
    y_pred = model.predict(X_test)

    # calculate the MSE
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Test loss: {mse}")

    return 0

def test_logistic_regression():
    n, d = 300, 10
    X = np.random.normal(size = (n, d))
    w = np.random.normal(size = d)
    y = np.sign(X @ w)

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)
    model = LogisticRegression()
    model.find_opt_lamb(X_train, X_valid, y_train, y_valid)
    print(f"Optimal lambda: {model.lamb}")
    model.fit(X_train, y_train)

    # we now predict
    y_pred = model.predict(X_test)
    accuracy, precision, recall = model.evaluate(X_test, y_test)
    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test accuracy: {accuracy}")
    return 0

def main():
    np.random.seed(42)
    # test_linear_regression()
    test_logistic_regression()

if __name__ == "__main__":
    main()
