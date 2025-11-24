import numpy as np
import matplotlib.pyplot as plt
from src.linear.linear_regression import LinearRegression
from src.core.utils import train_valid_test_split
from src.linear.logistic_regression import LogisticRegression
from src.linear.svms import LinearSVM
from src.bayes.bayes_regression import BayesianRegression

def test_linear_regression(X, y):
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

def test_logistic_regression(X, y):

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)
    model = LogisticRegression()
    model.find_opt_lamb(X_train, X_valid, y_train, y_valid)
    print(f"Optimal lambda: {model.lamb}")
    model.fit(X_train, y_train)

    # we now predict
    accuracy, precision, recall = model.evaluate(X_test, y_test)
    print(f"Logistic test precision: {precision}")
    print(f"Logistic test recall: {recall}")
    print(f"Logistic test accuracy: {accuracy}")
    return 0

def test_linear_svm(X, y):
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)
    model = LinearSVM()
    model.fit(X_train, y_train)

    # we now predict the model
    y_pred = model.predict(X_test)

    # calculate the accuracy
    accuracy, precision, recall = model.evaluate(X_test, y_test)
    print(f"SVM test precision: {precision}")
    print(f"SVM test recall: {recall}")
    print(f"SVM test accuracy: {accuracy}")
    return 0

def test_linear_bayes(X, y):
    # split the data into train, valid, and test
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)

    # fit the model
    prior_mean = np.zeros(1)
    prior_cov = np.eye(1) * 1
    model = BayesianRegression(prior_mean, prior_cov, 1)
    model.fit(X_train, y_train)

    # predict the model
    y_mean, y_std = model.predict(X_test, return_std=True)

    sort_idx = np.argsort(X_test[:, 0])
    X_sorted = X_test[sort_idx, 0]
    y_mean_sorted = y_mean[sort_idx]
    y_std_sorted = y_std[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(X_sorted, y_mean_sorted, 'b-', label='Mean prediction')
    plt.fill_between(X_sorted, 
                    y_mean_sorted - 2*y_std_sorted, 
                    y_mean_sorted + 2*y_std_sorted, 
                    alpha=0.3, color='blue', label='±2σ uncertainty')
    plt.scatter(X_test[:, 0], y_test, color='red', alpha=0.5, s=20, label='True values')
    plt.xlabel('X_test (first feature)')
    plt.ylabel('y')
    plt.legend()
    plt.title('Bayesian Regression: Predictions with Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.show()

    # calculate the MSE
    mse = np.mean((y_test - y_mean) ** 2)
    print(f"Bayesian test loss: {mse}")

    return 0

def main():
    np.random.seed(42)
    n, d = 300, 10

    # skewed distribution, but perfect for hinge or logistic regression
    # X1 = np.random.normal(loc=1, scale=1, size = (250, d))
    # X2 = np.random.normal(loc=100, scale=0.1, size =(50, d))
    # X = np.concatenate([X1, X2])
    # np.random.shuffle(X)
    # w = np.random.uniform(low=-0.05, high=0.95, size = d)

    X = np.random.normal(size = (n, d))
    w = np.random.normal(size = d)
    y = np.sign(X @ w)

    # regression case
    eps = np.random.normal(size = n)
    y = X @ w + eps

    test_linear_regression(X, y)
    test_logistic_regression(X, y)
    test_linear_svm(X, y)
    test_linear_bayes(X, y)

if __name__ == "__main__":
    main()
