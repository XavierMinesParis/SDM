# +
import numpy as np
import statsmodels.api as sm

class LinearRegression:
    
    def __init__(self):
        self.model = None
        self.res = None
        self.m = None  # Save number of features (including constant)
    
    def fit(self, x, y):
        x = sm.add_constant(x, has_constant='add') # Add intercept column
        self.m = x.shape[1]
        self.model = sm.GLM(y, x, family=sm.families.Binomial())
        self.res = self.model.fit()
    
    def predict(self, x):
        x = sm.add_constant(x, has_constant='add') # Add intercept

        if x.shape[1] != self.m:
            raise ValueError(f"Prediction input has {x.shape[1]} features, expected {self.m} (including intercept).")

        return self.res.predict(x)
    
    def get_aic(self):
        return self.res.aic
