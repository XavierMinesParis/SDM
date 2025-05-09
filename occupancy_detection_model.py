# +
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import expit

class Occupancy_Detection_Model(GenericLikelihoodModel):
    
    def __init__(self, x, y, K=2, **kwds):
        
        endog, exog = y, np.array(x).transpose()
        super(Occupancy_Detection_Model, self).__init__(endog, exog, **kwds)
        self.K = K
        self.res = None

    def nloglikeobs(self, params):
        
        m = self.exog.shape[1] # Number of climatic variables
        alpha0 = params[0]
        alpha = params[1: m + 1]
        beta0 = params[m + 1]
        beta = params[m+2: ]
        
        occupancy = expit(alpha0 + np.dot(self.exog, alpha))
        detection = expit(beta0 + np.dot(self.exog, beta))
        
        # For n (ie. self.endog) different from 0
        ll_non_null = np.log(occupancy) + self.endog * np.log(detection) + (self.K - self.endog) * np.log(1 - detection)

        # For n = 0
        ll_null = np.log(occupancy * (1 - detection) ** self.K + (1 - occupancy))
        
        ll  = (self.endog != 0) * ll_non_null + (self.endog == 0) * ll_null
        return -ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        
        m = self.exog.shape[1] # Number of climatic variables
        if start_params == None:
            start_params = np.random.normal(size=2 * m + 2) # Reasonable initialization
            
        self.res = super(Occupancy_Detection_Model, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
    
    def predict(self, x):
        params = self.res.params
        x = np.array(x).transpose()
        
        m = x.shape[1] # Number of climatic variables
        alpha0 = params[0]
        alpha = params[1: m + 1]
        beta0 = params[m + 1]
        beta = params[m + 2: ]
        
        occupancy = expit(alpha0 + np.dot(x, alpha))
        detection = expit(beta0 + np.dot(x, beta))
        return occupancy, detection


# -

"""
class Occupancy_Detection_Model(GenericLikelihoodModel):
    
    def __init__(self, x, y, K=2, **kwds):
        
        endog, exog = y, np.array(x).transpose()
        super(Occupancy_Detection_Model, self).__init__(endog, exog, **kwds)
        self.K = K
        self.res = None

    def nloglikeobs(self, params):
        
        m = self.exog.shape[1] # Number of climatic variables
        alpha0 = params[0]
        alpha = params[1: m + 1]
        beta0 = params[m + 1]
        beta = params[m+2: ]
        
        occupancy_regression = alpha0 + np.dot(self.exog, alpha) 
        detection_regression = beta0 + np.dot(self.exog, beta)
        
        # For n (ie. self.endog) different from 0
        occupancy_part = occupancy_regression - np.log(1 + np.exp(occupancy_regression))
        detection_part = self.endog * detection_regression - self.K * np.log(1 + np.exp(detection_regression))
        ll_non_null = occupancy_part + detection_part # Log-likelihood
        
        l_null = expit(occupancy_regression) * (1 - expit(detection_regression)) ** self.K + (1 - expit(occupancy_regression))
        ll_null = np.log(l_null)
        
        ll  = (self.endog != 0) * ll_non_null + (self.endog == 0) * ll_null
        return -ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        
        m = self.exog.shape[1] # Number of climatic variables
        if start_params == None:
            start_params = np.random.normal(size=2 * m + 2) # Reasonable initialization
            
        self.res = super(Occupancy_Detection_Model, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
    
    def predict(self, x):
        params = self.res.params
        x = np.array(x).transpose()
        
        m = x.shape[1] # Number of climatic variables
        alpha0 = params[0]
        alpha = params[1: m + 1]
        beta0 = params[m + 1]
        beta = params[m + 2: ]
        
        occupancy = expit(alpha0 + np.dot(x, alpha))
        detection = expit(beta0 + np.dot(x, beta))
        return occupancy, detection
"""

"""
x = ([2, 3, 2, 3, 2, 1, 0, -1], [7, 9, 10, 12, 11, 11, 11, 11])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
model = Occupancy_Detection_Model(x, y, K=2)
model.fit()

model.predict(([2, 3, -1], [7, 9, 11]))
"""
