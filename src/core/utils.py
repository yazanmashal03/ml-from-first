# we make a function that takes a matrix X and based on the given 
# percentage, returns a train, valid, and test split.

def train_valid_test_split(X, y, train_size=0.6, valid_size=0.3, test_size=0.1):
    # check if the sum of the sizes is 1
    if abs(train_size + valid_size + test_size - 1.0) > 1e-10:
        raise ValueError("The sum of the sizes must be 1")

    n, m = X.shape
    train_size = int(train_size * n)
    valid_size = int(valid_size * n)
    test_size = int(test_size * n)

    return X[:train_size], X[train_size:train_size+valid_size], X[train_size+valid_size:], y[:train_size], y[train_size:train_size+valid_size], y[train_size+valid_size:]
