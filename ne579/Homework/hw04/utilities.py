import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.io as sio
from sklearn.base import BaseEstimator, TransformerMixin

def load_matlab_data(filename):
    dd = sio.loadmat(filename)
    
    x = dd.get('x')
    y = dd.get('y')
    
    return x, y

def zscore(x, m=None, s=None):
    """Scale `x` to mean-centered, unit variance
    
    If an already-scaled matrix is provided along with vector
    of means and a vector of standard deviations, apply the same
    transformation without recalculating.
    
    Parameters
    ----------
    x : numpy.ndarray
        2d numpy array of shape `(n, m)` consisting of `n`
        samples of `m` dimensions each
    m : numpy.ndarray, optional
        The mean values to use to transform x
    s : numpy.ndarray, optional
        The standard deviation values to use to transform x

    Returns
    -------
    tuple of numpy.ndarray
        0: `n` rows (samples) of `m` dimensions (columns)
        1: 1-d vector of mean values
        2: 1-d vector of standard deviation values
    
    Raises
    ------
    Exception
        If only `m` or only `s` is provided
    """
    if m is None and s is None:
        m = np.mean(x, axis=0)
        s = np.std(x, axis=0)
        return (x - m) / s, m, s
    elif m is not None and s is not None:
        return (x - m) / s, m, s
    else:
        raise Exception("Something went wrong in zscore")

def train_test_val_split(x, y, train_f=0.7, test_f=0.15, val_f=0.15,
                         seed=None):
    """DOCSTRING!!!
    
    Hopefully this is deprecated soon in favor of cross validation
    
    Parameters
    x, y : numpy.ndarray
        Input and output data, respectively
    train_f, test_f, val_f : float
        Fractions of the data to put in each respective set. This function
        will normalize them for you
    
    Returns
    -------
    tuple of numpy.ndarray
        x and y for the training, testing, and validation set respectively
    """
    # Get length of data
    rows, cols = x.shape

    # Generate randomized array of row numbers
    rand_ind = np.random.permutation(rows)

    # Find the index row with the max and min in each column
    min_indices = [np.argmin(x[:,i]) for i in range(cols)]
    max_indices = [np.argmax(x[:,i]) for i in range(cols)]
    # Filter non-unique values
    minmax_indices = min_indices + max_indices
    minmax_indices = list(set(minmax_indices))

    # Convert the train, test, val fractions into numbers
    # Take floor of each of those numbers
    train_len = np.floor(train_f * rows).astype(int)
    test_len = np.floor(test_f * rows).astype(int)
    val_len = np.floor(val_f * rows).astype(int)

    # Split array of row numbers into train_ix, test_ix, val_ix
    val_ixs = list(rand_ind[:val_len])
    test_ixs = list(rand_ind[val_len:val_len + test_len])
    train_ixs = list(rand_ind[val_len + test_len:])

    # Add the max and min indices to training set
    train_ixs.extend(minmax_indices)

    # Slice up x and y
    x_train = x[train_ixs]
    y_train = y[train_ixs]

    x_test = x[test_ixs]
    y_test = y[test_ixs]

    x_val = x[val_ixs]
    y_val = y[val_ixs]

    return x_train, y_train, x_test, y_test, x_val, y_val

class linear_regression():
    def __init__(self):
        self._params = None
    
    def fit(self, x, y):
        x = np.hstack([x, np.ones_like(x[:,0]).reshape((-1, 1))])
        self._params = la.inv(x.T @ x) @ x.T @ y
    
    def predict(self, x):
        if len(x.shape) < 2:
            print(x.shape)
            x = np.atleast_2d(x)
            print(x.shape)
        x = np.hstack([x, np.ones_like(x[:,0]).reshape((-1, 1))])
        return x @ self._params
