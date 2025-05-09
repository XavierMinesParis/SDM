# +
import numpy as np
import statsmodels.api as sm

class Linear_Regression_Model:
    
    def __init__(self, model=None):
        self.model = model
    
    def fit(self, x, y, verbose=False):
        x_train = np.column_stack(x)
        model = sm.GLM(y, sm.add_constant(x_train), family=sm.families.Binomial())
        self.model = model.fit()
        if verbose:
            print(model.summary())
    
    def predict_proba(self, x):
        x_test = np.column_stack(x)
        return self.model.predict(sm.add_constant(x_test))
    
    def predict(self, x):
        proba_prediction = Linear_Regression_Model.predict_proba(self, x)
        return (proba_prediction >= 0.5).astype(int)
