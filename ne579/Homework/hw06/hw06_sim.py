import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from utilities import train_test_val_split, load_matlab_data, rmse
from sklearn.metrics import make_scorer


t = np.arange(1, 101)
x1 = np.sin(t) + 0.01 * np.random.rand(t.shape[0])
x2 = 10 * np.sin(t) + 0.01 * np.random.rand(t.shape[0])
x = np.hstack([x1.reshape((-1, 1)), x2.reshape((-1, 1))])
y = 0.1 * x[:,0] + 0.9 * x[:,1] + np.random.rand(t.shape[0])

xtr, ytr, xts, yts, xv, yv = train_test_val_split(x, y, seed=42)
ytr = ytr.reshape((-1, 1))
yts = yts.reshape((-1, 1))
yv = yv.reshape((-1, 1))

xscaler = StandardScaler().fit(xtr)
yscaler = StandardScaler().fit(ytr)

xs = xscaler.transform(xtr)
ys = yscaler.transform(ytr)

lin = LinearRegression().fit(xs, ys)

# Find best alpha value with Leave One Out Cross validation
alpha_params = np.logspace(-2, 2)
params = {'alpha': alpha_params}
ridge_loo = GridSearchCV(Ridge(fit_intercept=False), param_grid=params, 
                     scoring=make_scorer(rmse, greater_is_better=False), 
                     cv=LeaveOneOut())
ridge_loo.fit(xs, ys)
a_loo = ridge_loo.best_estimator_.alpha
print(f"Best alpha for leave one out was {a_loo}")

# Find best alpha value with 10-fold cross validation
alpha_params = np.logspace(-2, 2)
params = {'alpha': alpha_params}
ridge_cv10 = GridSearchCV(Ridge(fit_intercept=False), param_grid=params, 
                     scoring=make_scorer(rmse, greater_is_better=False), 
                     cv=10)
ridge_cv10.fit(xs, ys)
a_cv10 = ridge_cv10.best_estimator_.alpha
print(f"Best alpha for 10-fold was {a_cv10}")

# Find best alpha value with 3-fold cross validation
alpha_params = np.logspace(-2, 2)
params = {'alpha': alpha_params}
ridge_cv3 = GridSearchCV(Ridge(fit_intercept=False), param_grid=params, 
                     scoring=make_scorer(rmse, greater_is_better=False), 
                     cv=3)
ridge_cv3.fit(xs, ys)
a_cv3 = ridge_cv3.best_estimator_.alpha
print(f"Best alpha for 3-fold was {a_cv3}")

def best_alpha(folds, x, y):
    alpha_params = np.logspace(-2, 2)
    params = {'alpha': alpha_params}
    r = GridSearchCV(Ridge(fit_intercept=False), 
                     param_grid=params, 
                     scoring=make_scorer(rmse, greater_is_better=False), 
                     cv=folds)
    r.fit(x, y)
    a = r.best_estimator_.alpha
    return a

folds = np.arange(2, xs.shape[0])
alphas = [best_alpha(f, xs, ys) for f in folds]
f = plt.figure()
plt.plot(folds, alphas)
plt.xlabel("Number of folds")
plt.ylabel(r"Optimal $\alpha$")
plt.savefig("images/alpha_vs_folds.png", dpi=300)
plt.clf()

# Now try L-curve
alpha_params = np.logspace(-3, 2)
# Generate a model for each alpha value
models = [Ridge(alpha=a, fit_intercept=False).fit(xs, ys) for a in alpha_params]
norm_b = [la.norm(m.coef_) for m in models]
ytr_scaled_pred = [m.predict(xscaler.transform(xtr)) for m in models]
rmse_vals = [rmse(ytr, yscaler.inverse_transform(y)) for y in ytr_scaled_pred]

# Examine norm of coefficients vs RMSE
plt.plot(rmse_vals, norm_b)
plt.ylabel("|B|")
plt.xlabel("RMSE")
plt.savefig("images/norm_b_vs_rmse.png", dpi=300)
plt.clf()

# Figure out where to zoom in
opt_b = np.array(rmse_vals) ** 2 + np.array(norm_b) ** 2
opt_b_loc = np.where(opt_b == np.min(opt_b))[0][0]
start_ind = max([0, opt_b_loc-10])
end_ind = min([opt_b.shape[0], opt_b_loc+10])
plt.plot(rmse_vals[start_ind:end_ind], norm_b[start_ind:end_ind], 'x-')
plt.plot(rmse_vals[opt_b_loc], norm_b[opt_b_loc], 'r*')
plt.ylabel("|B|")
plt.xlabel("RMSE")
plt.savefig("images/norm_b_vs_rmse_zoomed.png", dpi=300)
plt.clf()

# Look at effect of increasing alpha on RMSE
# Pretty evident that bias increases as alpha goes up
plt.plot(alpha_params, rmse_vals)
plt.xscale('log')
plt.xlabel(r"$\alpha$")
plt.ylabel("RMSE")
plt.savefig("images/rmse_vs_alpha.png", dpi=300)
plt.clf()

# Look at effect of increasing alpha on coefficient norm
plt.plot(alpha_params, norm_b)
plt.xscale('log')
plt.xlabel(r"$\alpha$")
plt.ylabel("|B|")
plt.savefig("images/norm_b_vs_alpha.png", dpi=300)
plt.clf()

# Examine the stability of the result
def cond_ridge(x, alpha):
    s = la.svd(x.T @ x)[1]
    return (np.max(s) + alpha**2) / (np.min(s) + alpha**2)

# Effect of alpha on condition number
plt.plot(alpha_params, [cond_ridge(xs, a) for a in alpha_params])
plt.xscale('log')
plt.xlabel(r"$\alpha$")
plt.ylabel("Condition number")
plt.savefig("images/condition_number_vs_alpha.png", dpi=300)
plt.clf()

print(f"Best alpha from L-curve method: {alpha_params[opt_b_loc]}")

# Original training matrix
c_unscaled = np.linalg.cond(xtr.T @ xtr)
print(f"Condition number of unscaled matrix: {c_unscaled}")

# Scaled training matrix
c_scaled = np.linalg.cond(xs.T @ xs)
print(f"Condition number of scaled matrix: {c_scaled}")

# Ridge regression condition number
c_ridge = cond_ridge(xs, alpha_params[opt_b_loc])
print(f"Condition number for ridge regression: {c_ridge}")

# Compare model performance
lin_predictions = yscaler.inverse_transform(
    lin.predict(xscaler.transform(xts))
)
linear_performance = rmse(yts, lin_predictions)
print(f"Linear regression RMSE: {linear_performance}")

ridge_lc = Ridge(alpha=alpha_params[opt_b_loc], 
                 fit_intercept=False).fit(xs, ys)
ridge_lc_predictions = yscaler.inverse_transform(
    ridge_lc.predict(xscaler.transform(xts))
)
ridge_performance = rmse(yts, ridge_lc_predictions)
print(f"Ridge regression RMSE: {ridge_performance}")
