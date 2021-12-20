import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


def kNN(X, n_neighbors, n_jobs):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
    dists, inds = neigh.kneighbors(X)
    return dists, inds


def Levina_Bickel(X, dists, k):
    m = np.log(dists[:, k:k+1] / dists[:, 1:k])
    m = (k-2) / np.sum(m, axis=1)
    dim = np.mean(m)
    return dim


def start_matlab_engine():
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(os.path.split(__file__)[0], 'matlab_codes'))
    return eng


def MiND_ML(X, dists, k):
    import matlab.engine
    eng = start_matlab_engine()
    X_mat = matlab.double(X.T.tolist())
    dists_mat = matlab.double(dists[:, :k+2].tolist())
    dim = eng.MiND_ML(X_mat, 'dists', dists_mat, 'normalized', False, 'optimize', True)
    return dim


def MiND_KL(X, dists, k, maxDim=30):
    import matlab.engine
    eng = start_matlab_engine()
    X_mat = matlab.double(X.T.tolist())
    dists_mat = matlab.double(dists[:, :k+2].tolist())
    dim = eng.MiND_KL(X_mat, 'k', matlab.double([k]), 'maxDim', matlab.double([maxDim]),
                      'dists', dists_mat, 'normalized', False, nargout=1)
    return dim


def DANCo(X, dists, inds, k, maxDim=30):
    import matlab.engine
    eng = start_matlab_engine()
    X_mat = matlab.double(X.T.tolist())
    dists_mat = matlab.double(dists[:, :k+2].tolist())
    inds_mat = matlab.int32((inds[:, :k+2]+1).tolist())  # fit Matlab indices
    dim = eng.DANCo(X_mat, 'k', matlab.double([k]), 'maxDim', matlab.double([maxDim]), 'fractal', True, 
                    'dists', dists_mat, 'inds', inds_mat, 'normalized', False, nargout=1)
    return dim


def Hein_CD(X):
    import matlab.engine
    eng = start_matlab_engine()
    X_mat = matlab.double(X.T.tolist())
    dim = eng.GetDim(X_mat, nargout=1)
    dim = np.array(dim)[0]
    return dim[0], dim[1]