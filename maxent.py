# +
import numpy as np
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults

class Maxent(GenericLikelihoodModel):
    
    def __init__(self, presence, background, base_alpha=1):
        self.presence = presence
        self.background = background
        self.m = presence.shape[1] # Number of climatic variables
        self.n_presence = len(presence)
        self.alpha = base_alpha * np.std(self.presence, axis=0) / np.sqrt(self.n_presence)
        self.res = None

        endog, exog = np.ones(self.n_presence), presence
        super(Maxent, self).__init__(endog=endog, exog=exog)
    
    def loglike(self, params):
        linear_term = np.sum(np.dot(self.exog, params)) # Sum over the presence points
        
        z = np.exp(np.dot(self.background, params)) # Partition function
        log_partition = self.n_presence * np.log(np.sum(z) + 1e-12)

        l1_penalty = np.sum(self.alpha * np.abs(params))

        return linear_term - log_partition - l1_penalty

    def fit(self, start_params=None, maxiter=500, maxfun=5000, **kwds):
        
        if start_params is None:
            start_params = np.random.normal(size=self.m)
            
        self.res = super(Maxent, self).fit(start_params=start_params, maxiter=maxiter,
                                           maxfun=maxfun, disp=False, **kwds)
        self.params_ = self.res.params

    def predict(self, x):
        return np.exp(np.dot(x, self.params_))

    def get_aic(self):
        return GenericLikelihoodModelResults(self, self.res).aic
