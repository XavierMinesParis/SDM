# +
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, mean_squared_error, log_loss
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

class LogisticRegression2:
    
    def __init__(self):
        """
        A logistic regression with L1 regularization.
        """
        
        self.model = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='saga', random_state=0)
        self.res = None
        self.m = None  # Number of climate variables
        self.auc, self.rmse, self.spearman = None, None, None
    
    def fit(self, x, y):
        self.m = x.shape[1]
        self.res = self.model.fit(x, y)
        self.aic = 2 * self.m - 2 * (-1 * np.log(log_loss(y, LogisticRegression2.predict(self, x))))
    
    def predict(self, x):
        return self.res.predict_proba(x)[: , 1] # Two classes, probability of 1 are visible on the second column
    
    def get_aic(self):
        return self.aic
    
    def get_auc(self, x_test, y_test):
        y_pred = LogisticRegression2.predict(self, x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        self.auc = auc(fpr, tpr)
        return self.auc
    
    def get_rmse(self, x_test, y_test):
        """
        Computes the Root Mean Squared Error value, with y_test belonging to [0, 1].
        """
        y_pred = LogisticRegression2.predict(self, x_test)
        self.rmse = mean_squared_error(y_test, y_pred , squared=False)
        return self.rmse
    
    def get_spearman(self, x_test, y_test):
        """
        Computes the Spearman's rank correlation coefficient, with y_test belonging to [0, 1]
        """
        y_pred = LogisticRegression2.predict(self, x_test)
        self.spearman = spearmanr(y_test, y_pred)[0]
        return self.spearman
    
    def __repr__(self):
        
        text = "| Logistic Regression with L1 regularization"
        text += "\n| Number of variables: " + str(self.m)
        text += "\n| RMSE: " + str(self.rmse)[: 4]
        text += "\n| AUC: " + str(self.auc)[: 4]
        text += "\n| Spearman's Rank Correlation Index: " + str(self.spearman)[: 4]

        return text
