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

colnames = [
    'Age',
    'Weight',
    'Height',
    'Adiposity index',
    'Neck circumference',
    'Chest circumference',
    'Abdomen circumference',
    'Hip circumference',
    'Thigh circumference',
    'Knee circumference',
    'Ankle circumference',
    'Extended bicep circumference',
    'Forearm circumference',
    'Wrist circumference'
]

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
plt.xlabel("Number of latent variables")
plt.ylabel("PLS RMSE")
plt.legend(loc="upper right")
plt.savefig("images/rmse_vs_number_components.png", dpi=300)
plt.gcf().clear()

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
plt.savefig("images/rmse_vs_pca_loadings.png", dpi=300)
plt.gcf().clear()

f, axes = plt.subplots(nrows=5, ncols=2, sharey='row', figsize=(20, 20))

tmp = pls_models[4]
tmp2 = pcr_models[4]

for i in range(5):
    l = tmp.x_weights_.T[i]
    ax = axes[i, 0]
    ax.bar(x=range(l.shape[0]), height=l)
    ax.set_ylabel("Contribution")
    ax.text(0.99, 0.99, f'Latent variable {i+1}',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    if i == 4:
        ax.set_xlabel("PLS weights", fontsize=18)

for i in range(5):
    l = tmp2.named_steps.pca.components_[i]
    ax = axes[i, 1]
    ax.bar(x=range(len(l)), height=l)
    ax.text(0.99, 0.99, f'Principal component {i+1}',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    if i == 4:
        ax.set_xlabel("PLR loading", fontsize=18)

plt.tight_layout()
plt.savefig("images/pls_vs_pcr_loadings.png", dpi=300)
plt.gcf().clear()

for i in range(4):
    p = pls_models[-1]
    p = p.x_weights_.T[i]

f = plt.figure(figsize=(15, 9))

c = np.corrcoef(x, y, rowvar=False)[-1]
p = pls_models[-1].x_weights_.T[0]

ax1 = f.add_subplot(121)
ax1.bar(x=colnames, height=c[:-1])
plt.xticks(rotation=90)
ax1.text(0.99, 0.99, 'Correlation',
         verticalalignment='top', 
         horizontalalignment='right', 
         transform=ax1.transAxes, 
         color='green', fontsize=12)

ax2 = f.add_subplot(122, sharey=ax1)
ax2.bar(x=colnames, height=p)
plt.xticks(rotation=90)
ax2.text(0.99, 0.99, 'PLS weights',
         verticalalignment='top', 
         horizontalalignment='right', 
         transform=ax2.transAxes, 
         color='green', fontsize=12)

plt.tight_layout()
plt.savefig("images/pls_weights_and_correlation.png", dpi=300)
plt.gcf().clear()

p = pls_models[-1]
w1 = p.x_weights_.T[0]
w2 = p.x_weights_.T[1]
w3 = p.x_weights_.T[2]
w4 = p.x_weights_.T[3]

f = plt.figure()

ax1 = f.add_subplot(221)
ax1.text(0.99, 0.99, "LV 1", 
         verticalalignment='top', 
         horizontalalignment='right', 
         transform=ax1.transAxes, 
         color='green')
ax1.bar(x=range(14), height=w1)

ax2 = f.add_subplot(222)
ax2.text(0.99, 0.99, "LV 2", 
         verticalalignment='top', 
         horizontalalignment='right', 
         transform=ax2.transAxes, 
         color='green')
ax2.bar(x=range(14), height=w2)

ax3 = f.add_subplot(223)
ax3.text(0.99, 0.99, "LV 3", 
         verticalalignment='top', 
         horizontalalignment='right', 
         transform=ax3.transAxes, 
         color='green')
#ax3.bar(x=colnames, height=w3)
ax3.bar(x=range(14), height=w3)
#plt.xticks(rotation=90)

ax4 = f.add_subplot(224)
ax4.text(0.99, 0.99, "LV 4", 
         verticalalignment='top', 
         horizontalalignment='right', 
         transform=ax4.transAxes, 
         color='green')
#ax4.bar(x=colnames, height=w4)
ax4.bar(x=range(14), height=w4)
#plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("images/first_four_latent_variables.png", dpi=300)
plt.gcf().clear()

p = pls_models[0]
lr = LinearRegression()
lr.fit(p.x_scores_, p.y_scores_)
plt.plot(p.x_scores_, p.y_scores_, '.')
plt.plot(p.x_scores_, lr.predict(p.x_scores_))
plt.xlabel("X scores (T)")
plt.ylabel("Y scores (U)")
plt.savefig("images/pls_first_scores_regression.png", dpi=300)
plt.gcf().clear()

perf = rmse(yv, pls_models[5].predict(xv))
print(f"Final validation performance: {perf}")
