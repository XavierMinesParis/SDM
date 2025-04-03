# -*- coding: utf-8 -*-
# +
from variable import *

class Species:
    """
    Description to be added
    """
    
    names = pd.read_csv('PASA_2013_names_all.csv', sep=",", on_bad_lines='skip')
    pasa_var_clim = pd.read_csv('pasa_var_clim_distinct_examples.csv', sep=",", on_bad_lines='skip')
    
    def __init__(self, id_pasa=None, plt_id_cfvvf=None, latin_name=None, samples=None, variables=None):
        """
        Species object.

        Attributes:
        id_pasa (int): ID with abundance (pasa = "plante à seuil d'abondance" in French)
        plt_id_cfvvf (int): ID (cfvvf = "code forestier..." in French)
        latin_name (str): Name of the species in the dataframe
        samples (pd.Dataframe): Contains 36 climatic variables and geographical info
        variables (dict): Keys are columns and values are Variable objects that store the result of statistical analysis.
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
        
        if samples is None:
            samples = Species.pasa_var_clim.loc[Species.pasa_var_clim['id_pasa'] == id_pasa]
            samples = samples.drop(columns=['id_clim', 'id_pasa', 'lon', 'lat', 'altitude_fr_1km', 'classesph', 'mat1'])
            
        self.samples = samples
        
        if variables is None:
            variables = dict()
            for column in self.samples.columns:
                variables[column] = Variable(column=column, label=Stations.dict_variables[column])
                variables[column].set_results(self.samples[column].values)
        self.variables = variables
        
    def __repr__(self):
        text = "| Name: " + self.latin_name
        text += "\n| ID (pasa): " + str(self.id_pasa)
        text += "\n| ID (cfvvf): " + str(self.plt_id_cfvvf)
        text += "\n| Number of samples: " + str(len(self.samples))
        return text
