# -*- coding: utf-8 -*-
# +

from visualization import *

class Species:
    
    def __init__(self, file_name, id_name=None, id_=None):
        """
        Species object.

        Attributes:
        file_name (str):
        id_name (str):
        id_ (int): 
        locations
        x
        latin_name (str)
        """
        self.file_name = file_name
        self.id_name
        self.id_ = id_
        
        df = pd.read_csv(file_name, sep=",")
        if id_ is not None:
            df = df[df[id_name] == id_]
        
        self.locations = df[["lon", "lat"]]
        self.x = df[list(dict_variables.values())]
        
        names = pd.read_csv("names.csv", sep=",")
        self.latin_name = str(df[df.loc[df[id_name] == id_]]['latin_name'])
        
        
    def __repr__(self):
        text = "| Name: " + self.latin_name
        text += "\n| ID (" + str(self.id_) + "): " + str(self.id_pasa)
        text += "\n| Number of samples: " + str(len(self.locations))
        return text
