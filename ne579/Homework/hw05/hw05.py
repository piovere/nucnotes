import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from utilities import *


# Set seed to match earlier homeworks
np.random.seed(3)

# Load data from matlab file
x, y = load_matlab_data("data/hwkdataNEW.mat")

# Split into training, testing, validation
xtr, ytr, xts, yts, xv, yv = train_test_val_split(x, y)

# The PLSRegression object scales the data by default
plsr = PLSRegression()
plsr.fit(xtr, ytr)

e = rmse(yts, plsr.predict(xts))
print(f"Error with default parameters: {e}")

# Vary number of components
for i in range(1, x.shape[1]):
    p = PLSRegression(n_components=i)
    p.fit(xtr, ytr)
    e = rmse(yts, p.predict(xts))
    print(f"Error for {i} components: {e}")

models = [PLSRegression(n_components=i).fit(xtr, ytr) for i in range(1, x.shape[1]+1)]
# map(lambda m: m.fit(xtr, ytr), models)

es = [rmse(yts, m.predict(xts)) for m in models]

f = plt.figure()
plt.plot(range(1, x.shape[1]+1), es)
plt.xlabel("Number of components")
plt.ylabel("Test set RMSE")
plt.show()
