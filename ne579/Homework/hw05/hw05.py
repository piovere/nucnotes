import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from utilities import load_matlab_data, train_test_val_split, rmse
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


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

pls_models = [PLSRegression(n_components=i).fit(xtr, ytr) for i in range(1, x.shape[1]+1)]
# map(lambda m: m.fit(xtr, ytr), pls_models)

ets = [rmse(yts, m.predict(xts)) for m in pls_models]
etr = [rmse(ytr, m.predict(xtr)) for m in pls_models]

f = plt.figure()
plt.plot(range(1, x.shape[1]+1), ets, label="Test error")
plt.plot(range(1, x.shape[1]+1), etr, label="Training error")
plt.xlabel("Number of components")
plt.ylabel("PLS RMSE")
plt.legend(loc="upper right")
plt.savefig("images/rmse_vs_number_components.png", dpi=1000)
plt.show()

def PCR(n_features):
    pcr = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=n_features)),
        ('linear', LinearRegression())
    ])
    return pcr

pcr_models = [PCR(i).fit(xtr, ytr) for i in range(1, x.shape[1]+1)]
ets = [rmse(yts, m.predict(xts)) for m in pcr_models]
etr = [rmse(ytr, m.predict(xtr)) for m in pcr_models]

f = plt.figure()
plt.plot(range(1, x.shape[1]+1), ets, label="Test error")
plt.plot(range(1, x.shape[1]+1), etr, label="Training error")
plt.xlabel("Number of PCA loadings")
plt.ylabel("PCR RMSE")
plt.legend(loc="upper right")
plt.savefig("images/rmse_vs_pca_loadings.png", dpi=1000)
plt.show()

f, axes = plt.subplots(nrows=5, ncols=2, sharey='row')

tmp = pls_models[4]
tmp2 = pcr_models[4]

for i in range(5):
    l = tmp.x_loadings_.T[i]
    ax = axes[i, 0]
    ax.bar(x=range(l.shape[0]), height=l)

for i in range(5):
    l = tmp2.named_steps.pca.components_[i]
    ax = axes[i, 1]
    ax.bar(x=range(len(l)), height=l)

plt.savefig("images/pls_vs_pcr_loadings.png", dpi=1000)
plt.show()
