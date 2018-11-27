import numpy as np
from numpy.random import random


class NN():
    """A neural network with one hidden layer
    """
    def __init__(self, ninputs, nhidden, noutput=1):
        self.w = random((ninputs+1, nhidden))
        self.v = random((nhidden+1, noutput))

    def predict(self, xt):
        xt = np.atleast_2d(xt)
        ox = np.atleast_2d(np.ones_like(xt[:,0])).T
        xt = np.hstack([xt, ox])

        self.z = self.sigmoid(xt @ self.w)
        oz = np.atleast_2d(np.ones_like(self.z[:,0])).T
        self.z = np.hstack([self.z, oz])
        self.yp = self.z @ self.v
        
        self.yp = np.exp(self.yp) / np.sum(np.exp(self.yp))

        return self.yp

    def fit(self, xt, yt, lr=0.001):
        xt = np.atleast_2d(xt)
        # One-pad xt for bias
        ox = np.atleast_2d(np.ones_like(xt[:,0])).T
        xtp = np.hstack([xt, ox])

        yp = self.predict(xt)
        yt = np.atleast_2d(yt)

        e = yt - yp

        # Calc output layer weight delta
        delta_v = lr * self.z.T @ e
        assert delta_v.shape == self.v.shape
        try:
            assert not(np.isnan(delta_v[0,0]))
        except AssertionError:
            raise AssertionError('delta_v is NaN')

        # Calc hidden layer weight delta
        delta_w = lr * xtp.T @ ((e @ self.v.T) * self.derivative(self.z))
        delta_w = delta_w[:,:-1]  # Shave off the bias from the output layer
        try:
            assert delta_w.shape == self.w.shape
        except AssertionError:
            raise AssertionError(f"delta_w shape: {delta_w.shape}, w shape: {self.w.shape}")
        try:
            assert not(np.isnan(delta_w[0,0]))
        except AssertionError:
            raise AssertionError('delta_w is NaN')

        # Update layer weights
        self.v += delta_v
        self.w += delta_w

        try:
            assert np.sum(yp) == 1.0
        except AssertionError:
            print(yp)
            print(np.sum(yp))

        return np.max(yt - yp)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def rmse(self, xv, yv):
        yp = self.predict(xv)
        yv = np.atleast_2d(yv)
        if yv.shape[0] == 1:
            yv = yv.T
        e = (yv - yp)[0,0]
        es = e ** 2
        return np.sum(es)**0.5

    def tp(self, xv, yv):
        yp = self.predict(xv)[:,0]

    def prob(self, x):
        scores = self.predict(x)
        return np.exp(scores) / np.sum(scores)
    
    def misclassification(self, xv, yv):
        yp = np.round(self.predict(xv))

        print(yp)

        totlen = yv.shape[0]
        corrects = np.argwhere(np.all(yp == yv))

        print(corrects)
        print(totlen)


        return 1 - corrects.shape[0] / totlen
