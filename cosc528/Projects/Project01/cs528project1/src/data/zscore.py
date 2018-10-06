import numpy as np


class Zscorer():
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, x):
        self.means_ = np.mean(x.values, axis=0)
        self.stds_ = np.std(x.values, axis=0)

    def transform(self, x):

