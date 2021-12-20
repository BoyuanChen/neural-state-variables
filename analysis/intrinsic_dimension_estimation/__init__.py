import numpy as np
from .methods import *

class ID_Estimator:
    def __init__(self, method='Levina_Bickel'):
        self.all_methods = ['Levina_Bickel', 'MiND_ML', 'MiND_KL', 'Hein', 'CD']
        self.set_method(method)
    
    def set_method(self, method='Levina_Bickel'):
        if method not in self.all_methods:
            assert False, 'Unknown method!'
        else:
            self.method = method
        
    def fit(self, X, k_list=20, n_jobs=4):
        if self.method in ['Hein', 'CD']:
            dim_Hein, dim_CD = Hein_CD(X)
            return dim_Hein if self.method=='Hein' else dim_CD
        else:
            if np.isscalar(k_list):
                k_list = np.array([k_list])
            else:
                k_list = np.array(k_list)
            kmax = np.max(k_list) + 2
            dists, inds = kNN(X, kmax, n_jobs)
            dims = []
            for k in k_list:
                if self.method == 'Levina_Bickel':
                    dims.append(Levina_Bickel(X, dists, k))
                elif self.method == 'MiND_ML':
                    dims.append(MiND_ML(X, dists, k))
                elif self.method == 'MiND_KL':
                    dims.append(MiND_KL(X, dists, k))
                else:
                    pass
            if len(dims) == 1:
                return dims[0]
            else:
                return np.array(dims)
    
    def fit_all_methods(self, X, k_list=[20], n_jobs=4):
        k_list = np.array(k_list)
        kmax = np.max(k_list) + 2
        dists, inds = kNN(X, kmax, n_jobs)
        dim_all_methods = {method:[] for method in self.all_methods}
        dim_all_methods['Hein'], dim_all_methods['CD'] = Hein_CD(X)
        for k in k_list:
            dim_all_methods['Levina_Bickel'].append(Levina_Bickel(X, dists, k))
            dim_all_methods['MiND_ML'].append(MiND_ML(X, dists, k))
            dim_all_methods['MiND_KL'].append(MiND_KL(X, dists, k))
        for method in self.all_methods:
            dim_all_methods[method] = np.array(dim_all_methods[method])
        return dim_all_methods