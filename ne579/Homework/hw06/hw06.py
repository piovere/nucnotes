from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
import numpy as np
import numpy.linalg as la
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

# Will use a separate scaler for convenience
# Even though estimators have one built in
scaler = StandardScaler()
scaler.fit(xtr)

# Examine how ill-conditioned the training set is
print(f"Training matrix condition number: {la.cond(xtr.T @ xtr):0.2e}")
xs = scaler.transform(xtr)
u, s, v = la.svd(xs)
print(f"Scaled training matrix condition number: " \
      f"{np.max(s) / np.min(s):0.2e}")

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
# Trying values from 0.01 to 1000
alpha_candidates = np.logspace(-3, 3)
params = {
    'ridge__alpha': alpha_candidates
}
rr = GridSearchCV(rr_pipe, param_grid=params,
                  scoring=make_scorer(rmse, greater_is_better=False), 
                  cv=70)
rr.fit(xtr, ytr)

best_alpha = rr.best_params_['ridge__alpha']
print(f"Best alpha from grid search: {best_alpha:0.6f}")

# Plot the solution norm vs the residuals for the "l-curve" method
l_curve_alphas = np.linspace(0.01, 1000)
solution_norms = []
residual_norms = []
for a in l_curve_alphas:
    reg = Ridge(alpha=a, fit_intercept=False)
    reg.fit(xs, ytr)
    solution_norms.append(la.norm(reg.coef_))
    residual_norms.append(rmse(ytr, reg.predict(xs)))

plt.plot(residual_norms, solution_norms, '.')
plt.xlabel(r"$|y-\beta x|$")
plt.ylabel(r"$|\beta|$")

#plt.xscale('log')
#plt.yscale('log')
#plt.show()

plt.clf()

# Then use a randomized search to find the best
# alpha_candidates are chosen from a normal distribution centered at the
# best result from the grid search above
alpha_candidates = norm(loc=best_alpha, scale=best_alpha)

# scale=np.log(best_alpha)/np.log(10))

params = {
    'ridge__alpha': alpha_candidates
}
rr = RandomizedSearchCV(rr_pipe, param_distributions=params, 
                        scoring=make_scorer(rmse, greater_is_better=False), 
                        n_iter=10, cv=70)
rr.fit(xtr, ytr)

best_alpha = rr.best_params_['ridge__alpha']
print(f"Best alpha value: {best_alpha:0.2f}")

# Calculate modified condition number
ridge_cond = la.cond(xtr.T @ xtr + best_alpha ** 2 * np.eye(xtr.shape[1]))
scaled_bond = la.cond(xs.T @ xs + best_alpha ** 2 * np.eye(xs.shape[1]))
print(f"Modified condition number: {ridge_cond:0.2f}")
print(f"Scaled modified condition number: {scaled_bond:0.2f}")

print(f"Linear regression training error: {rmse(ytr, lr.predict(xtr)):0.2f}")
print(f"Ridge regression training error: {rmse(ytr, rr.predict(xtr)):0.2f}")
print(f"Linear regression testing error: {rmse(yts, lr.predict(xts)):0.2f}")
print(f"Ridge regression testing error: {rmse(yts, rr.predict(xts)):0.2f}")
print(f"Linear regression validation error: {rmse(yv, lr.predict(xv)):0.2f}")
print(f"Ridge regression validation error: {rmse(yv, rr.predict(xv)):0.2f}")

# Load the data from the body fat set
x, y = load_matlab_data('data/hwkdataNew.mat')
xtr, ytr, xts, yts, xv, yv = train_test_val_split(x, y, seed=3)
