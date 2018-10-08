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

    
def rmse(y_true, y_false):
    return np.sqrt(np.sum((y_true - y_false) ** 2) / y_true.flatten().shape[0])
