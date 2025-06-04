# +
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from itertools import chain, combinations

class Statistics:
    """
    Provides auxiliary statistical methods 
    """
    
    def get_proximities(bin_edges, p, q):
        """
        Applies the empirical model and returns the list of proximites for one climatic variable.
        """

        F = np.cumsum(q)
        u = p * (F - q /2 - 1 / 2)
        
        n = len(p)
        g = np.zeros(n)
 
        P, Q = np.meshgrid(p, q)
        g[0] = np.sum(np.tril((P * Q)))
        g[0] -= (p[0] * (1 - q[0])) / 2
        g[0] -= np.sum(np.dot(p, q)) / 2
        
        for k in range(n-1):
            g[k+1] = g[k] + u[k] + u[k+1]
            
        return 1 - g
    
    def get_auc(y, y_predict):
        
        fpr, tpr, thresholds = roc_curve(y, y_predict)
        
        return auc(fpr, tpr)
    
    def rmse(y, y_predict):
        return np.sqrt(((np.ravel(y) - np.ravel(y_predict)) ** 2).mean())

    def get_subsets(iterable, min_size=1):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        res = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
        res_copy = res.copy()
        for subset in res_copy:
            if len(subset) < min_size:
                res.remove(subset)
        return res
