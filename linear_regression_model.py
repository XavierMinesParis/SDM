# +
import numpy as np
from statsmodels.api import GLM
import statsmodels.api as sm

class Linear_Regression_Model(GLM):
    
    def __init__(self, **kwds):
        self.model = None
        self.res = None

    def fit(self, x, y):
        endog, exog = y, np.array(x).transpose()
        exog = sm.add_constant(exog)
        model = sm.GLM(endog, exog, family=sm.families.Binomial())
        self.res = model.fit()
        self.model = model
    
    def predict(self, x):
        x_test = np.array(x).transpose()
        x_test = sm.add_constant(x_test)
        return self.res.predict(x_test)
    
    def get_aic(self):
        return self.res.aic
