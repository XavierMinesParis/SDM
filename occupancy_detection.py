# +
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults
from scipy.special import expit

class OccupancyDetection(GenericLikelihoodModel):
    
    def __init__(self, x, y, K=2, **kwds):
        
        endog, exog = y, x
        super(OccupancyDetection, self).__init__(endog, exog, **kwds)
        self.K = K
        self.res = None
        self.m = self.exog.shape[1]

    def nloglikeobs(self, params):
        
        m = self.exog.shape[1] # Number of climatic variables
        
        alpha0 = params[0]
        alpha = params[1: m + 1]
        beta0 = params[m + 1]
        beta = params[m+2: ]
        
        psi = expit(beta0 + np.dot(self.exog, beta)) # Occupancy
        p = expit(alpha0 + np.dot(self.exog, alpha)) # Detection
        
        # For n (ie. self.endog) different from 0
        ll_non_null = np.log(psi) + self.endog * np.log(p) + (self.K - self.endog) * np.log(1 - p)

        # For n = 0
        ll_null = np.log(psi * (1 - p) ** self.K + (1 - psi))
        
        ll  = (self.endog != 0) * ll_non_null + (self.endog == 0) * ll_null
        return -ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        
        m = self.exog.shape[1] # Number of climatic variables
        if start_params == None:
            start_params = np.random.normal(size=2 * m + 2) # Reasonable initialization
            
        self.res = super(OccupancyDetection, self).fit(start_params=start_params, maxiter=maxiter, 
                                                       maxfun=maxfun, disp=False, **kwds)
    
    def predict(self, x):
        params = self.res.params
        
        m = x.shape[1] # Number of climatic variables
        alpha0 = params[0]
        alpha = params[1: m + 1]
        beta0 = params[m + 1]
        beta = params[m + 2: ]
        
        psi = expit(beta0 + np.dot(x, beta)) # Occupancy
        p = expit(alpha0 + np.dot(x, alpha)) # Detection
        
        return psi, p
    
    def get_aic(self):
        return GenericLikelihoodModelResults(self, self.res).aic
