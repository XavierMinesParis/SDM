# +
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

class Statistics:
    """
    Provides auxiliary statistical methods 
    """
    
    def get_proximities(bin_edges, p, q):

        F = np.cumsum(q)
        u = p * (F - q /2 - 1 / 2)
        #plt.plot(u, label='u')
        #plt.plot(p, label='p')
        #plt.plot(q, label='q')
        #plt.legend()
        #plt.show()
        n = len(p)
        g = np.zeros(n)
 
        P, Q = np.meshgrid(p, q)
        g[0] = np.sum(np.tril((P * Q)))
        g[0] -= (p[0] * (1 - q[0])) / 2
        g[0] -= np.sum(np.dot(p, q)) / 2
        
        for k in range(n-1):
            g[k+1] = g[k] + u[k] + u[k+1]
            
        return 1 - g
