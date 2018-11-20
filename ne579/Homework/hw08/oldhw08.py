import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utilities import load_matlab_data


def t2(data, pca, scaler):
    """Calculate the T^2 score for a data sample
    """
    l = np.diag(pca.explained_variance_)
    t = pca.transform(data)
    return np.array([tt @ l @ tt.T for tt in t])

def q(data, pca, scaler):
    """Calculate Q-score for a data sample
    """
    # Get dimensionality of the data
    i = data.shape[1]
    
    pp = pca.components_.T @ pca.components_
    
    return np.array([x @ (np.eye(i) - pp) @ x.T for x in data])

data, test1, test2 = load_matlab_data('data/hwk8data.mat')

scale = StandardScaler().fit(data)

pca = PCA().fit(scale.transform(data))

t2scores1 = t2(test1, pca, scale)
t2scores2 = t2(test2, pca, scale)
qscores1 = q(test1, pca, scale)
qscores2 = q(test2, pca, scale)
