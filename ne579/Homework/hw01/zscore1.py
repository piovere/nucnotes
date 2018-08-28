import numpy as np

def zscore1(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)

    xs = (x - means[np.newaxis, :]) / stds[np.newaxis, :]

    return xs, means, stds
