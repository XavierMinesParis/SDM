# +
import numpy as np
from statistics import *
from stations import *
from sklearn import metrics
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

class EmpiricalModel:
    
    def __init__(self, bin_edges=None, proximities=None, concentrations=None, optimum_range=None,
                optimum_value=None, indicator_power=None):
        self.bin_edges = bin_edges
        self.proximities = proximities
        self.concentrations = concentrations
        self.optimum_range = optimum_range
        self.optimum_value = optimum_value
        self.indicator_power = indicator_power
        self.x, self.y = None, None
        self.auc, self.rmse, self.spearman = None, None, None
    
    def fit(self, x, y=None, scenario='current', bins=100, verbose=False):
        """
        By default, a fit for presence-absence data provided with x and y.
        In this case, the ubiquist proximities are computed.
        If y is set to None, x is supposed to contain presence-only data.
        In this case, the ubiquist proximities are imported from the Stations class
        """
        self.x, self.y = x, y
        m = x.shape[1]
        self.bin_edges = []
        self.proximities = []
        self.concentrations = []
        self.optimum_range = []
        self.optimum_value = []
        self.indicator_power = []
            
        for i in range(m):
            
            if isinstance(x, pd.DataFrame):
                variable = x[x.columns[i]]
            else:
                variable = x[: , i]
                
            if y is None:
                column = x.columns[i]
                p, bin_edges = Stations.distributions[scenario][column] # PDF of the stations
                ubiquist_proximities = Stations.ubiquist_proximities[scenario][column]
                
                q, bin_edges = np.histogram(variable, bins=bin_edges, density=True) # PDF of the detections
            else:
                p, bin_edges = np.histogram(variable, bins=bins, density=True) # PDF of the stations
                p /= np.sum(p)
                ubiquist_proximities = Statistics.get_proximities(bin_edges, p, p)
            
                q, bin_edges = np.histogram(variable[y == 1], bins=bins, density=True) # PDF of the detections
            
            q /= np.sum(q)
            proximities = Statistics.get_proximities(bin_edges, p, q)
            
            concentrations = 1 - (1 - proximities) / (1 - ubiquist_proximities + 10**(-6))
            optimum_range = np.argmax(concentrations)
            optimum_value = bin_edges[optimum_range]
            indicator_power = np.max(concentrations)
            
            self.bin_edges.append(bin_edges)
            self.proximities.append(proximities)
            self.concentrations.append(concentrations)
            self.optimum_range.append(optimum_range)
            self.optimum_value.append(optimum_value)
            self.indicator_power.append(indicator_power)

    def predict(self, x):
        m = x.shape[1]
        probas = 0
        for i in range(m):
            
            if isinstance(x, pd.DataFrame):
                variable = x[x.columns[i]]
            else:
                variable = x[: , i]
                
            indices = np.digitize(variable, self.bin_edges[i])
            indices = list(np.maximum(np.minimum(len(self.bin_edges[i]) - 2, indices), 0))
            probas += self.concentrations[i][indices] / m
            
        return probas
        
    def get_aic(self):
        y_pred = self.predict(self.x)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
        return Statistics.rmse(self.y, y_pred)
    
    def get_auc(self, x_test, y_test):
        y_pred = EmpiricalModel.predict(self, x_test)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) # Min max normalization
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        self.auc = metrics.auc(fpr, tpr)
        return self.auc
    
    def get_rmse(self, x_test, y_test):
        """
        y_test belongs to [0, 1]
        """
        y_pred = EmpiricalModel.predict(self, x_test)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) # Min max normalization
        self.rmse = Statistics.rmse(y_test, y_pred)
        return self.rmse
    
    def get_spearman(self, x_test, y_test):
        """
        y_test belongs to [0, 1]
        """
        y_pred = EmpiricalModel.predict(self, x_test)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) # Min max normalization
        self.spearman = spearmanr(y_test, y_pred)[0]
        return self.spearman
    
    def __repr__(self):
        
        text = "| Empirical Model"
        text += "\n| Number of variables: " + str(self.m)
        text += "\n| RMSE: " + str(self.rmse)[: 4]
        text += "\n| AUC: " + str(self.auc)[: 4]
        text += "\n| Spearman's Rank Correlation Index: " + str(self.spearman)[: 4]

        return text
