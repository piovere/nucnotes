import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio

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
        m = np.mean(x, axis=1)
        s = np.std(x, axis=1)
    elif m is not None and s is not None:
        return (x - m) / s, m, s
    else:
        raise Exception("Something went wrong in zscore")

def train_test_val_split(x, y, train_f=0.7, test_f=0.15, val_f=0.15):
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
    # Generate randomized array of row numbers
    # Find the index row with the max and min in each column
    # Convert the train, test, val fractions into numbers
    # Take floor of each of those numbers
    # Split array of row numbers into train_ix, test_ix, val_ix
    # Add leftover rows to the training set
    # Add the max and min indices to training set
    # Slice up x and y
    return x_train, y_train, x_test, y_test, x_val, y_val
    
def correllation(a, b):
    """Calculate the correlation between two vectors
    """
    np.var(a) * np.var(b) / np.std(a) / np.std(b)
    raise Exception("I forget how to calculat correlation")
