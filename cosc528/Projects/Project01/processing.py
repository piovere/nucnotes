import numpy as np

def zscore(x):
    """Scale a 2d matrix `x` such that each column is mean-centered with unit
    variance
    
    Parameters
    ----------
    x : numpy.ndarray
        The input data to be scaled. This will be in the form of a 2d array
    
    Returns
    -------
    tuple of numpy.ndarray
        0 - Scaled data with 0-mean and 1 standard deviation. Shape is the 
            same as the original data
        1 - 1d array of the column means of `x`
        2 - 1d array of the column standard deviations of `x`
    
    To Do
    -----
    - [ ] Add checking to validate that every column has variance, raise
          ValueError otherwise
    - [ ] Add doctests to this function
    """
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)

    xs = (x - means) / stds

    return xs, means, stds

def add_intercept_column(x):
    """Adds a column of ones to a 2d matrix of data

    Parameters
    ----------
    x : numpy.ndarray
        2d array of n rows with m columns each (shape `(n,m)`)
    
    Returns
    -------
    numpy.ndarray
        2d array of data with a column of ones added on the right
    
    Examples
    --------
    >>> x = np.arange(9).reshape((3, 3))
    >>> add_intercept_column(x)
    array([[0, 1, 2, 1],
       [3, 4, 5, 1],
       [6, 7, 8, 1]])
    
    >>> np.arange(9).reshape((3, 3)).astype(np.float64)
    >>> add_intercept_column(x)
    array([[0., 1., 2., 1.],
       [3., 4., 5., 1.],
       [6., 7., 8., 1.]])
    """
    one_col = np.ones_like(x[:,0]).reshape((-1, 1))

    return np.hstack((x, one_col))

def cross_val_split(x, train_frac, val_frac, test_frac, y=None):
    """Split data into three groups: training, validation, and testing.

    Can optionally split the output into the same arrangement.

    Parameters
    ----------
    x : numpy.ndarray
        Input data
    train_frac, val_frac, test_frac : float
        Fraction of the data to place in each of the three new sets.

        These values will be normalized such that they sum to one. Checks are
        performed to ensure that all rows of the data set are used--any extra
        rows are added to the training set.
    
    Returns
    -------
    tuple of numpy.ndarray
        0 - training set
        1 - validation set
        2 - test set

    To Do
    -----
    - [ ] Make sure that max and min values for each variable are in the
          training data set
    - [ ] Add doctests to this function
    """
    # Find number of rows in dataset
    rows = x.shape[0]

    # Create iterable containing all the row numbers

    # Calculate the number of rows in each returned set
    
