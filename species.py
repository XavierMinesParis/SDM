# -*- coding: utf-8 -*-
# +
from stations import *

class Species:
    """
    Description to be added
    """
    
    names = pd.read_csv('PASA_2013_names_all.csv', sep=",", on_bad_lines='skip')
    pasa_var_clim = pd.read_csv('pasa_var_clim_distinct_examples.csv', sep=",", on_bad_lines='skip')
    
    def __init__(self, id_pasa=None, plt_id_cfvvf=None, latin_name=None, data=None,
                 columns=None, distributions=None, proximities=None, concentrations=None,
                 optimums=None):
        """
        Species object.

        Attributes:
        id_pasa (int): ID with abundance (pasa = "plante à seuil d'abondance" in French)
        plt_id_cfvvf (int): ID (cfvvf = "code forestier..." in French)
        latin_name (str): Name of the species in the dataframe
        data (pd.Dataframe): Contains all the samples with the species (36 climatic variables, geographical info)
        columns (list): Names of the 36 climatic variables
        distributions (dict): Keys are columns' names and values are tuples containing counts and bin edges values
        proximities (dict): Keys are columns' names and values are lists with proximites for every bin edge (cf. above)
        concentrations (dict): Keys are columns' names and values are lists with concentrations for every bin edge (cf. above)
        optimums (dict): Level of the layer for each node
        """
        
        if id_pasa is None and latin_name is None:
            id_pasa = 1
        elif id_pasa is None:
            id_pasa = int(Species.names.loc[Species.names['latin_name'].values == latin_name]['id_pasa'].values[0])
        self.id_pasa = id_pasa
        if plt_id_cfvvf is None:
            plt_id_cfvvf = Species.names.loc[Species.names['id_pasa'].values == id_pasa]['plt_id_cfvvf'].values[0]
        self.plt_id_cfvvf = plt_id_cfvvf
        if latin_name is None:
            latin_name = Species.names.loc[Species.names['id_pasa'].values == id_pasa]['latin_name'].values[0]
        self.latin_name = latin_name[: 20]
        
        if data is None:
            data = Species.pasa_var_clim.loc[Species.pasa_var_clim['id_pasa'] == id_pasa]
            data = data.drop(columns=['id_clim', 'id_pasa', 'lon', 'lat', 'altitude_fr_1km', 'classesph', 'mat1'])
            
        self.data = data
        self.columns = data.columns
        
        self.set_distributions()
        self.set_proximities()
        self.set_concentrations()
        self.set_optimums()
        
    def __repr__(self):
        text = "| Name: " + self.latin_name
        text += "\n| ID (pasa): " + str(self.id_pasa)
        text += "\n| ID (cfvvf): " + str(self.plt_id_cfvvf)
        text += "\n| Number of samples: " + str(len(self.data))
        #text += "\n| Variables: " + str(self.data.columns)
        return text
        
    
    def set_distributions(self):
        self.distributions = dict()
        for column in self.columns:
            counts_stations, bin_edges = Stations.distributions[column]
            variable = self.data[column]
            counts, bin_edges = np.histogram(variable.values, bins=bin_edges, density=True)
            self.distributions[column] = counts / np.sum(counts), bin_edges
        
    def set_proximities(self):
        self.proximities = dict()
        for column in self.columns:
            self.proximities[column] = 1 - Statistics.get_proximities(Stations.distributions[column],
                                                                     self.distributions[column])
    
    def set_concentrations(self):
        self.concentrations = dict()
        for column in self.columns:
            g = 1 - self.proximities[column]
            g_u = 1 - Stations.ubiquist_proximities[column]
            self.concentrations[column] = 1 - g / (g_u + 10**(-6))
    
    def set_optimums(self):
        self.optimums = dict()
        for column in self.columns:
            bin_edges = self.distributions[column][1]
            concentration = self.concentrations[column]
            optimum = np.argmax(concentration)
            indicator_power = np.max(concentration)
            self.optimums[column] = (optimum, indicator_power)
