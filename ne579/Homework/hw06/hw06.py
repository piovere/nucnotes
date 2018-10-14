from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_matlab_data, train_test_val_split, rmse
from scipy.stats import uniform, norm


# Set random seed
np.random.seed(3)

# Brief description of ridge regression

# Create ill-conditioned data set
t = np.arange(1, 101).reshape((-1, 1))
x1 = np.sin(t) + 0.01 * uniform().rvs(t.shape[0])
x2 = 10 * np.sin(t) + 0.01 * uniform().rvs(t.shape[0])
x = np.hstack([x1, x2])
y = 0.1 * x[:,0] + 0.9 * x[:,1] + uniform().rvs(t.shape[0])
y = y.reshape((-1, 1))

# Split the training and testing sets for this
xtr, ytr, xts, yts, xv, yv = train_test_val_split(x, y)

# Compare linear regression and ridge regression on these two
lr = Pipeline([
    ('scale', StandardScaler()),
    ('linear', LinearRegression())
])
rr_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('ridge', Ridge())
])

# First fit the linear regression
lr.fit(xtr, ytr)

# Find the best alpha value by cross validation

# Try a number of CV candidates to shrink the search range
# Trying values from 0.1 to 1000
alpha_candidates = np.logspace(-1, 3)
params = {
    'ridge__alpha': alpha_candidates
}
rr = GridSearchCV(rr_pipe, param_grid=params,
                  scoring=make_scorer(rmse, greater_is_better=False))
rr.fit(xtr, ytr)

# Then use a randomized search to find the best
# alpha_candidates are chosen from a normal distribution centered at the
# best result from the grid search above
alpha_candidates = norm(loc=rr.best_params_['ridge__alpha'], scale=10)
params = {
    'ridge__alpha': alpha_candidates
}
rr = RandomizedSearchCV(rr_pipe, param_distributions=params, 
                        scoring=make_scorer(rmse, greater_is_better=False), 
                        n_iter=10000)
rr.fit(xtr, ytr)

print(f"Best alpha value: {rr.best_params_['ridge__alpha']}")

print(f"Linear regression training error: {rmse(ytr, lr.predict(xtr))}")
print(f"Ridge regression training error: {rmse(ytr, rr.predict(xtr))}")
print(f"Linear regression testing error: {rmse(yts, lr.predict(xts))}")
print(f"Ridge regression testing error: {rmse(yts, rr.predict(xts))}")

# Load the data from the body fat set
x, y = load_matlab_data('data/hwkdataNew.mat')
xtr, ytr, xts, yts, xv, yv = train_test_val_split(x, y, seed=3)
