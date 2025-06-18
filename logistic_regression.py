# +
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from statistics import *
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

class LogisticRegression:
    
    def __init__(self):
        self.model = None
        self.res = None
        self.m = None  # Save number of features (including constant)
        self.auc, self.rmse, self.spearman = None, None, None
    
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
    
    def get_auc(self, x_test, y_test):
        y_pred = LogisticRegression.predict(self, x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        self.auc = metrics.auc(fpr, tpr)
        return self.auc
    
    def get_rmse(self, x_test, y_test):
        """
        y_test belongs to [0, 1]
        """
        y_pred = LogisticRegression.predict(self, x_test)
        self.rmse = Statistics.rmse(y_test, y_pred)
        return self.rmse
    
    def get_spearman(self, x_test, y_test):
        """
        y_test belongs to [0, 1]
        """
        y_pred = LogisticRegression.predict(self, x_test)
        self.spearman = spearmanr(y_test, y_pred)[0]
        return self.spearman
    
    def __repr__(self):
        
        text = "| Logistic Regression"
        text += "\n| Number of variables: " + str(self.m)
        text += "\n| RMSE: " + str(self.rmse)[: 4]
        text += "\n| AUC: " + str(self.auc)[: 4]
        text += "\n| Spearman's Rank Correlation Index: " + str(self.spearman)[: 4]

        return text
