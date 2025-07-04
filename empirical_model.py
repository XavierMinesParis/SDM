# +
from constants import *
from extractor import *
from sklearn.metrics import roc_curve, auc, mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

class EmpiricalModel:
    
    def __init__(self, bin_edges=None, proximities=None, concentrations=None, optimum_range=None,
                optimum_value=None, indicator_power=None):
        """
        This model 

        Attributes :
        bin_edges (list): Its ith element is a list of bins for the histogram of the ith climate variable.
        proximities (list): Its element of index [i][j] is the value of proximity at the jth bin of the ith climate variable.
        concentrations (list): Same as proximities.
        optimum_range (list): Its ith element is the index of the bin in the ith histogram with highest concentration.
        optimum_value (list): Its ith element is the the bin value in the ith histogram with highest concentration.
        indicator_power (list): Its ith element is the maximum concentration value of the ith histogram.
        x (pd.Dataframe or np.ndarray): Climate data.
        y (pd.Series or np.ndarray): Presence-absence data, labeled 0 and 1.
        auc (float): AUC discrimination value.
        rmse (float): Root Mean Squared Error.
        spearman (float): Spearman's rank correlation coefficient.
        """
        
        self.bin_edges = bin_edges
        self.proximities = proximities
        self.concentrations = concentrations
        self.optimum_range = optimum_range
        self.optimum_value = optimum_value
        self.indicator_power = indicator_power
        self.x, self.y = None, None
        self.auc, self.rmse, self.spearman = None, None, None
    
    def fit(self, x, y=None, stations=None, bins=100, verbose=False):

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
                
            if stations is None: # Simulations
                p = p, bin_edges = np.histogram(variable, bins=bins, density=True)
                p /= np.sum(p)
                ubiquist_proximities = EmpiricalModel.get_proximities(bin_edges, p, p)
            else: # Real case studies
                column = CLIMATE_VARIABLES[i]
                p, bins = stations.distributions[column]
                ubiquist_proximities = stations.ubiquist_proximities[column]
                
            q, bin_edges = np.histogram(variable[y == 1], bins=bins, density=True) # PDF of the detections
            q /= np.sum(q)
            proximities = EmpiricalModel.get_proximities(bin_edges, p, q)
            
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
            
    @staticmethod
    def get_proximities(bin_edges, p, q):
        """
        Applies the empirical model and returns the list of proximites for one climate variable.
        """

        F = np.cumsum(q)
        u = p * (F - q /2 - 1 / 2)
        
        n = len(p)
        g = np.zeros(n)
 
        P, Q = np.meshgrid(p, q)
        g[0] = np.sum(np.tril((P * Q)))
        g[0] -= (p[0] * (1 - q[0])) / 2
        g[0] -= np.sum(np.dot(p, q)) / 2
        
        for k in range(n-1):
            g[k+1] = g[k] + u[k] + u[k+1]
            
        return 1 - g

    def predict(self, x):
        m = x.shape[1]
        probas = 0
        for i in range(m):
            
            if isinstance(x, pd.DataFrame):
                variable = x[x.columns[i]]
            else:
                variable = x[: , i]
                
            indices = np.digitize(variable, self.bin_edges[i])
            # Handling values out of the histogram.
            indices = list(np.maximum(np.minimum(len(self.bin_edges[i]) - 2, indices), 0))
            probas += self.concentrations[i][indices] / m
            
        return probas
        
    def get_aic(self):
        """
        Computes a quantity that is not the AIC (Akaike criterion) value.
        But this syntax is used to perform model selection, as in other MLE models.
        """
        
        y_pred = self.predict(self.x)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
        return mean_squared_error(self.y, y_pred , squared=False)
    
    def get_auc(self, x_test, y_test):
        """
        Computes the AUC discrimination value, with y_test belonging to {0, 1}.
        """
        
        y_pred = EmpiricalModel.predict(self, x_test)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) # Min max normalization
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        self.auc = auc(fpr, tpr)
        return self.auc
    
    def get_rmse(self, x_test, y_test):
        """
        Computes the Root Mean Squared Error value, with y_test belonging to [0, 1].
        """
        
        y_pred = EmpiricalModel.predict(self, x_test)
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred)) # Min max normalization
        self.rmse = mean_squared_error(y_test, y_pred , squared=False)
        return self.rmse
    
    def get_spearman(self, x_test, y_test):
        """
        Computes the Spearman's rank correlation coefficient, with y_test belonging to [0, 1]
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
