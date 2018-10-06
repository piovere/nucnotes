import numpy as np
import numpy.linalg as la

class LinearRegression():
    def __init__(self):
        self.coefficients_ = None

    def atleast_2d(self, x):
        if len(x.shape) < 2:
            x = np.atleast_2d(x).reshape((-1, 1))
        return x

    def add_intercept_column(self, x):
        x = self.atleast_2d(x)

        o = np.ones_like(x[:,0])

        return np.hstack([x, o])

    def fit(self, x, y, add_intercept=True):
        self.cond_ = la.cond(x)

        x = self.add_intercept_column(x)

        self.coefficients_ = la.inv(x.T @ x) @ x.T @ y

    def predict(self, x, add_intercept=True):
        if add_intercept:
            x = self.add_intercept_column(x)

        # x dimensions must match
        assert x.shape[1] == self.coefficients_.shape[0]

        return x @ self.coefficients_
