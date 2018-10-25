import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from utilities import train_test_val_split, load_matlab_data, rmse
from itertools import product
import pandas as pd


x, y = load_matlab_data("data/hwkdataNEW.mat")
xtr, ytr, xts, yts, xv, yv = train_test_val_split(x, y, seed=3)

class LocallyWeightedRegression(BaseEstimator, RegressorMixin):
    def __init__(self, bandwidth=0.25, order=1):
        self.bandwidth = bandwidth
        self.order = order
        
        # Save the scalers
        self.xscaler_ = None
        self.yscaler_ = None
        
        # Save training data
        self.xs_ = None
        self.ys_ = None
        
    def fit(self, x, y):
        self.xscaler_ = StandardScaler().fit(x)
        self.yscaler_ = StandardScaler().fit(y)
        
        self.xs_ = self.xscaler_.transform(x)
        self.ys_ = self.yscaler_.transform(y)
        
        return self
    
    def predict(self, x):
        xs = self.xscaler_.transform(x)
        results = []
        
        for x in xs:
            # Calculate the distances
            ds = cdist(self.xs_, np.atleast_2d(x))
            
            # Calculate the weights
            ws = norm.pdf(ds, scale=self.bandwidth)
            
            # Normalize the weights
            ws /= sum(ws)
            
            # Convert to 1d array
            ws = ws[:,0]
            
            # Add in the polynomial terms
            p = PolynomialFeatures(self.order)
            xt = p.fit_transform(self.xs_)
            
            # Perform the regression to get the coefficients
            l = LinearRegression()
            l.fit(xt, self.ys_, sample_weight=ws)            
            
            # Calculated the predicted value
            res = l.predict(p.transform(x.reshape((1, -1))))
            
            # Add that to the results array
            results.append(res)
        
        res_np = np.array(res).reshape(-1, 1)
        
        return self.yscaler_.inverse_transform(res_np)
    
    def score(self, x, y):
        yp = self.predict(x)
        return -rmse(y, yp)

r_dict = {
    'Bandwidth': [],
    'Order': [],
    'RMSE (% bodyfat)': []
}
bandwidths = [0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
orders = [0, 1, 2]

params = product(orders, bandwidths)
print("Bandwidth\tOrder\t\tRMSE")
print("----------------------------------------")
for o, b in params:
    lwr = LocallyWeightedRegression(b, o)
    lwr.fit(xtr, ytr)
    print(f"{b}\t\t{o}\t\t{-lwr.score(xts, yts):0.2f}")
    r_dict['Bandwidth'].append(b)
    r_dict['Order'].append(o)
    r_dict['RMSE (% bodyfat)'].append(f"{-lwr.score(xts, yts):0.2f}")

rdf = pd.DataFrame(r_dict)

with open("table.tab", "w") as f:
    f.write(rdf.to_latex(index=False))

# Validation error
# Best model is first order with bandwidth 0.5
lwr = LocallyWeightedRegression(bandwidth=0.5, order=1)
lwr.fit(xtr, ytr)
print(f"Best model val score: {lwr.score(xv, yv)}")
print(f"{lwr.get_params}")

