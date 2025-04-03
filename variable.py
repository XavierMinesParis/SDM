# +
from stations import *

class Variable:
    """
    Climatic variable distribution of a species and its resulting features such as proximities, concentrations and optimums.
    """
    
    def __init__(self, column=None, label=None, bin_edges=None, counts=None, proximities=None, optimum_range=None,
                optimum_value=None, indicator_power=None):
        """
        Climatic variables object.

        Attributes:
        column (str): Normalized name present in the input database
        label (str): Name for plotting purpose
        bin_edges (list): Borders of the histogram
        counts (list): Density probability function
        proximities (list): List of proximity values
        concentrations (list): List of concentrations values
        optimum_range (int): Range of the maximum in concentrations list
        optimum_value (float): Value of the optimum
        indicator_power (float): Maximum concentration
        """
        
        self.column = column
        self.label = label
        
        self.bin_edges = bin_edges
        self.counts = counts
        self.proximities = proximities
        self.optimum_range = optimum_range
        self.optimum_value = optimum_value
        self.indicator_power = indicator_power
        
    def set_results(self, values):
        
        counts_stations, bin_edges = Stations.distributions[self.column]
        counts, bin_edges = np.histogram(values, bins=bin_edges, density=True)
        counts /= np.sum(counts)
        self.bin_edges = bin_edges
        self.counts = counts
        
        self.proximities = Statistics.get_proximities(bin_edges, counts_stations, counts)
        g = 1 - self.proximities
        g_u = 1 - Stations.ubiquist_proximities[self.column]
        self.concentrations = 1 - g / (g_u + 10**(-6))
        
        self.optimum_range = np.argmax(self.concentrations)
        self.optimum_value = bin_edges[self.optimum_range]
        self.indicator_power = np.max(self.concentrations)
        
