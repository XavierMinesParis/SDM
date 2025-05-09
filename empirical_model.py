# +
import numpy as np
from statistics import *

class Empirical_Model:
    
    def __init__(self, bin_edges=None, proximities=None, concentrations=None, optimum_range=None,
                optimum_value=None, indicator_power=None):
        self.bin_edges = bin_edges
        self.proximities = proximities
        self.concentrations = concentrations
        self.optimum_range = optimum_range
        self.optimum_value = optimum_value
        self.indicator_power = indicator_power
    
    def fit(self, x, y, bins=100, verbose=False):
        n = len(x)
        self.bin_edges = []
        self.proximities = []
        self.concentrations = []
        self.optimum_range = []
        self.optimum_value = []
        self.indicator_power = []
        
        for i, variable in enumerate(x):
        
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
    
    def predict_proba(self, x):
        n = len(x)
        probas = 0
        for i, variable in enumerate(x):
            indices = np.digitize(variable, self.bin_edges[i])
            indices = list(np.maximum(np.minimum(len(self.bin_edges[i]) - 2, indices), 0))
            probas += self.concentrations[i][indices] / n
        return probas
