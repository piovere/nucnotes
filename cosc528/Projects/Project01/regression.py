import numpy as np

def regress(x, y):
    """Return the coefficients of a linear regression

    `x` should not be one-padded

    Parameters
    ----------
    x : numpy.ndarray
        Input data with dimensions (num_samples, num_features)
    y : numpy.ndarray
        Output to be regressed against with dimensions (num_samples,)
    
    Returns
    -------
    numpy.ndarray
        Coefficients for each input and intercept
    
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(3)
    >>> x = np.random.rand(21).reshape((-1, 3))
    >>> y = np.sum(x, axis=1) + 1
    >>> regress(x, y)
    array([1., 1., 1., 1.])

    >>> import numpy as np
    >>> x = np.random.rand(12).reshape((-1, 3))
    >>> y = np.sum(x, axis=1) + 1
    >>> regress(x, y)
    array([1., 1., 1., 1.])
    """
    # One-pad `x`
    ones = np.ones_like(x[:,1]).reshape(-1, 1)
    x = np.hstack([x, ones])

    # Calculate the regression
    # tmp = np.linalg.inv(x.T @ x)

    # print(np.eye(4))
    # print(tmp * (x.T @ x))
    # assert np.allclose(np.dot(tmp, np.dot(x.T, x)), np.eye(4))

    # tmp = np.dot(tmp, x.T)
    
    coefficients = np.linalg.inv(x.T @ x) @ x.T @ y

    return coefficients.flatten()

if __name__ == "__main__":
    import doctest
    doctest.testmod()