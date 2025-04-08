# -*- coding: utf-8 -*-
# +
from variable import *

class Species:
    """
    Description to be added
    """
    
    names = pd.read_csv('PASA_2013_names_all.csv', sep=",", on_bad_lines='skip')
    pasa_var_clim = pd.read_csv('pasa_var_clim_distinct_examples.csv', sep=",", on_bad_lines='skip')
    
    def __init__(self, id_pasa=None, plt_id_cfvvf=None, latin_name=None, locations=None, variables=None):
        """
        Species object.

        Attributes:
        id_pasa (int): ID with abundance (pasa = "plante à seuil d'abondance" in French)
        plt_id_cfvvf (int): ID (cfvvf = "code forestier..." in French)
        latin_name (str): Name of the species in the dataframe
        locations (pd.Dataframe): Longitude, latitude and altitude of samples
        variables (dict): Keys are columns and values are Variable objects that store the result of statistical analysis.
        """
        
        self.id_pasa = id_pasa
        
        if plt_id_cfvvf is None:
            plt_id_cfvvf = Species.names.loc[Species.names['id_pasa'].values == id_pasa]['plt_id_cfvvf'].values[0]
        self.plt_id_cfvvf = plt_id_cfvvf
        
        if latin_name is None:
            latin_name = Species.names.loc[Species.names['id_pasa'].values == id_pasa]['latin_name'].values[0]
        self.latin_name = latin_name[: 20]
        
        if locations is None:
            samples = Species.pasa_var_clim.loc[Species.pasa_var_clim['id_pasa'] == id_pasa]
            locations = samples[['lon', 'lat', 'altitude_fr_1km']]
            samples = samples.drop(columns=['id_clim', 'id_pasa', 'lon', 'lat', 'altitude_fr_1km', 'classesph', 'mat1'])
            variables = dict()
            for column in samples.columns:
                variables[column] = Variable(column=column, label=Stations.dict_variables[column])
                variables[column].set_results(samples[column].values)
        
        self.locations = locations
        self.variables = variables
        
    def get_concentrations(self, column):
        return self.variables[column].bin_edges, self.variables[column].concentrations
    
    def get_optimum_value(self, column):
        return self.variables[column].optimum_value
        
    def __repr__(self):
        text = "| Name: " + self.latin_name
        text += "\n| ID (pasa): " + str(self.id_pasa)
        text += "\n| ID (cfvvf): " + str(self.plt_id_cfvvf)
        text += "\n| Number of samples: " + str(len(self.locations))
        return text
