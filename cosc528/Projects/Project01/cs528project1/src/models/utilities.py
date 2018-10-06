import numpy as np
import math

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

    >>> x = np.arange(9).reshape((3, 3)).astype(np.float64)
    >>> add_intercept_column(x)
    array([[0., 1., 2., 1.],
           [3., 4., 5., 1.],
           [6., 7., 8., 1.]])
    """
    one_col = np.ones_like(x[:,0]).reshape((-1, 1))

    return np.hstack((x, one_col))

def rmse(truth, prediction):
    """Gives the root-mean-squared erorr between prediction and truth

    Parameters
    ----------
    truth : numpy.ndarray
    prediction : numpy.ndarray

    Returns
    -------
    float

    Examples
    --------
    >>> a = np.array([3, 4, 5])
    >>> b = np.array([3, 4, 5])
    >>> rmse(a, b)
    0.0

    >>> c = np.array([4, 3, 6])
    >>> rmse(a, c)
    1.0

    >>> d = np.array([3, 4])
    >>> rmse(a, d)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 2, in rmse
    AssertionError
    """
    assert truth.shape == prediction.shape
    rmse = np.linalg.norm(truth - prediction) / truth.shape[0] ** 0.5
    return rmse

def cross_val_split(x, train_frac, val_frac, test_frac, y=None):
    """Split data into three groups: training, validation, and testing.

    Can optionally split the output into the same arrangement.

    To Do:
    - [ ] Make sure that max and min values for each variable are in the
          training data set

    Parameters
    ----------
    x : numpy.ndarray
        Input data
    y : numpy.ndarray, optional, default is None
        Targets for the regression
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

        If a y column was provided, it will be returned as the last column of
        each dataset.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(3)
    >>> td = np.arange(9).reshape((-1, 3))
    >>> y = np.arange(3).reshape((-1, 1))
    >>> cross_val_split(td, 1, 1, 1, y)
    (array([[3, 4, 5, 1]]), array([[0, 1, 2, 0]]), array([[6, 7, 8, 2]]))

    >>> np.random.seed(3)
    >>> td = np.arange(9).reshape((-1, 3))
    >>> y = np.arange(3).reshape((-1, 1))
    >>> cross_val_split(td, 1, 1, 1)
    (array([[3, 4, 5]]), array([[0, 1, 2]]), array([[6, 7, 8]]))

    >>> np.random.seed(3)
    >>> td = np.arange(12).reshape((-1, 3))
    >>> y = np.arange(3).reshape((-1, 1))
    >>> cross_val_split(td, 1, 1, 1)
    (array([[ 9, 10, 11],
           [ 3,  4,  5]]), array([[0, 1, 2]]), array([[6, 7, 8]]))
    """
    # Find number of rows in dataset
    numrows = x.shape[0]

    # Normalize the number of rows
    total_frac = train_frac + val_frac + test_frac
    train_frac /= total_frac
    val_frac /= total_frac
    test_frac /= total_frac

    # Create iterable containing all the row numbers

    # Calculate the number of rows in each returned set
    train_num = math.floor(train_frac * numrows)
    test_num = math.floor(test_frac * numrows)
    val_num = math.floor(val_frac * numrows)

    # Verify we have used all rows
    used_num = train_num + test_num + val_num
    while numrows > used_num:
        train_num += 1
        used_num = train_num + test_num + val_num
    # I THINK that we have used all the rows, but just to check:
    try:
        assert numrows == train_num + test_num + val_num
    except AssertionError as e:
        print(f'{numrows - (train_num + test_num + val_num)} unassigned')
        raise(e)

    # Join `y` to `x`
    if y is not None:
        data = np.hstack([x, y.reshape(-1, 1)])
    else:
        data = x

    # Shuffle the array
    np.random.shuffle(data)

    # Pick out the train, test, validation
    train = data[:train_num]
    test = data[train_num:train_num + test_num]
    val = data[train_num + test_num:]

    return train, test, val

class Scaler():
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, x, y=None):
        if y is not None:
            data = np.hstack([x, y.reshape((-1, 1))])
        else:
            data = x

        self.means_ = np.mean(data, axis=0)
        self.stds_ = np.std(data, axis=0)

    def transform(self, x, y=None):
        if y is not None:
            data = np.hstack([x, y.reshape((-1, 1))])
        else:
            data = x

        data -= self.means_
        data /= self.stds_

        if y is not None:
            return data[:,:-1], data[:, -1]
        else:
            return data

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        self.transform(x, y)

    def inverse_transform(self, xs, ys=None):
        if ys is not None:
            ds = np.hstack([xs, ys.reshape((-1, 1))])
        else:
            ds = xs

        ds = ds * self.stds_ + self.means_

        if ys is not None:
            return ds[:, :-1], ds[:, -1]
        else:
            return ds
