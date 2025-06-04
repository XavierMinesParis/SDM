# -*- coding: utf-8 -*-
# +
from extractor import *
import pandas as pd
import numpy as np
from linear_regression import *
from empirical_model import *

class Species:
    
    def __init__(self, file_name, id_name=None, id_=None, id_stations_name=None, n_presence=None, scenario='current'):
        """
        Species object.

        Attributes:
        file_name (str): Name of the csv file with species locations
        scenario (str): Name of the csv file with climatic data
        id_name (str): Name of the column providing stations ids in the species table
        id_ (int): 
        locations
        presence
        latin_name (str)
        id_stations (pd.Series)
        """
        
        
        df = pd.read_csv('Data/' + file_name, sep=",")
        if id_ is not None:
            df = df[df[id_name] == id_]
        
        
        if id_stations_name is not None:
            id_stations = df[id_stations_name]
            
        names = pd.read_csv("Data/names.csv", sep=",", on_bad_lines='skip')
        species_names = names.loc[names[id_name] == id_]['latin_name']
        latin_name = str(species_names.values[0])[: 20]
            
        
        self.file_name = file_name
        self.id_name = id_name
        self.id_ = id_
        self.locations = df[["lon", "lat"]]
        self.n_presence = len(self.locations)
        self.ids_stations = id_stations
        self.latin_name = latin_name

        self.lr_model = None
        self.em_model = None
        self.scenario = scenario
        
        if scenario == 'ssp245':
            self.dict_variables = ssp245_variables
            self.climate_name = "Data/stations_climate_ssp245.csv"
        elif scenario == 'ssp585':
            self.dict_variables = ssp245_variables
            self.climate_name = "Data/stations_climate_ssp585.csv"
        else:
            self.dict_variables = current_variables
            self.climate_name = "Data/stations_climate_current.csv"
            
        self.presence = df[list(self.dict_variables.values())]
        
    def train_models(self):
            
        background = pd.read_csv(self.climate_name, sep=",")
        x_background = background[self.dict_variables.values()].values
        
        presence = self.presence
        ids_stations = self.ids_stations.values
        x_presence = presence.values
        y_presence = len(x_presence) * [1]

        absence = background[~background['id'].isin(ids_stations)]
        x_absence = absence[self.dict_variables.values()].values
        y_absence = len(x_absence) * [0]

        x_train = np.concatenate((x_presence, x_absence))
        y_train = np.concatenate((y_presence, y_absence))

        lr_model = LinearRegression()
        lr_model.fit(x_train, y_train)
        self.lr_model = lr_model

        em_model = EmpiricalModel()
        em_model.fit(presence, scenario=scenario)
        self.em_model = em_model
        
    def test_models(self, file_name):
        
        test = pd.read_csv(file_name, sep=",")
        x_test = test[dict_variables.values()].values
        
        return lr_model.predict(x_test), em_model.predict(x_test)
        
        
    def __repr__(self):
        
        text = "| Name: " + self.latin_name
        text += "\n| ID (" + self.id_name + "): " + str(self.id_)
        text += "\n| Number of samples: " + str(self.n_presence)
        text += "\n| Source of species locations: " + self.file_name
        text += "\n| Scenario: " + self.scenario

        return text
