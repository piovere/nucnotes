import numpy as np
import numpy.linalg as la


class Scaler():
    def __init__(self):
        self.means_ = None
        self.stds_ = None
    
    def fit(self, x):
        self.means_ = np.mean(x, axis=0)
        self.stds_ = np.std(x, axis=0)
        return self
    
    def transform(self, x):
        xs = (x - self.means_) / self.stds_
        return xs


class PCA():
    def __init__(self, num_pcs):
        self.scaler_ = None
        self.num_pcs = num_pcs
        self.v_ = None
        self.s_ = None
    
    def fit(self, x):
        self.scaler_ = Scaler().fit(x)
        xs = self.scaler_.transform(x)
        s, vt = la.svd(xs, full_matrices=False)[1:]
        self.v_ = vt.T
        self.s_ = s
    
    def transform(self, x):
        raise NotImplementedError
