# +
import numpy as np
import pandas as pd

class Statistics:
    """
    Provides auxiliary statistical methods 
    """
    
    def get_proximities(distrib1, distrib2):
        p, bin_edges = distrib1
        q, bin_edges = distrib2

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
        return g
