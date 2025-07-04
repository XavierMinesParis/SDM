# -*- coding: utf-8 -*-
# +
from stations import *
from logistic_regression import *
from logistic_regression2 import *
from empirical_model import *
from extractor import *

class Species:
    """
    A Species object designed for ecological applications.
    It is defined with a Stations object, and thus with climate data.
    To date, this object can only encompass presence-absence SDMs.
    """
    
    def __init__(self, file_name, stations, id_name=None, id_=None,
                 id_stations_name=None, n_presence=None):
        """
        Extracts presence-absence data from the species locations and the stations file.

        Attributes:
        file_name (str): Name of the csv file with species locations.
        stations (Stations): Used to keep only valid points at climate stations, and used for training.
        id_name (str): Name of the column providing species ids in the species table.
        id_ (int): Species id.
        locations (pd.Dataframe): Locations of samples.
        n_presence (int): Number of points where the species was censed.
        id_stations (pd.Series): Column of the dataframe with stations ids at presence locations.
        latin_name (str): Scientific name.
        x_train (np.ndarray): Climate data.
        y_train (np.ndarray): Presence-absence data, with labels 0 and 1.
        models (dict): Keys are names chosen by the user and values might be different kinds of models:
            - LogisticRegression
            - LogisticRegresson2
            - EmpiricalModel
        """
        
        df = pd.read_csv('Data/' + file_name, sep=",")
        if id_ is not None:
            df = df[df[id_name] == id_]
        
        if id_stations_name is not None:
            df = df.drop_duplicates(subset=id_stations_name, keep='first') # Dropping multiple records at one location
            df = df[df[id_stations_name].isin(stations.ids)] # Keeping only presence points at valid climate stations
            id_stations = df[id_stations_name]
            
        names = pd.read_csv("Data/names.csv", sep=",", on_bad_lines='skip')
        species_names = names.loc[names[id_name] == id_]['latin_name']
        latin_name = str(species_names.values[0])[: 20]
            
        self.file_name = file_name
        self.stations = stations
        self.id_name = id_name
        self.id_ = id_
        self.locations = df[["lon", "lat"]]
        self.n_presence = len(self.locations)
        self.id_stations = id_stations
        self.latin_name = latin_name
        
        # Building the training dataset.
        id_stations = stations.ids.values
        
        background = stations.climate_data
        
        presence = background[np.isin(id_stations, self.id_stations)]
        x_presence = presence.values
        y_presence = len(x_presence) * [1]
        
        absence = background[~np.isin(id_stations, self.id_stations)]
        x_absence = absence.values
        y_absence = len(x_absence) * [0]
        
        x_train = np.concatenate((x_presence, x_absence))
        y_train = np.concatenate((y_presence, y_absence))
        self.x_train = x_train
        self.y_train = y_train
        self.models = dict()
        
        distributions = dict()
        for column in stations.climate_variables:
            variable = presence[column].values
            counts, bin_edges = np.histogram(variable, bins=np.unique(variable), density=True)
            counts /= np.sum(counts)
            distributions[column] = (counts, bin_edges)
        self.distributions = distributions
        
    def add_model(self, name, model):
        """
        Adds a trained model to the models attribute.
        """
        
        model.fit(self.x_train, self.y_train)
        self.models[name] = model
        
    def test_models(self, locations_file_name=None, climate_folder=None, save=False):
        """
        Returns a dictionary where keys are names of models and values are predictions.
        """
        
        extractor = Extractor(locations_file_name, climate_folder)
        climate_data = extractor.extract(verbose=False)[self.stations.climate_variables]
        x_test = climate_data.values
        
        res = dict()
        
        for name, model in self.models.items():
            res[name] = model.predict(x_test)
            
        if save: # Building a pandas dataframe and saving it as csv.
            results = locations.copy()
            for name, model in self.models.items():
                results[name] = res[name]
            results.to_csv('Data/' + file_name[:-4] + '_results', index=False)
            
        return res
        
    def __repr__(self):
        
        text = "| Name: " + self.latin_name
        text += "\n| ID (" + self.id_name + "): " + str(self.id_)
        text += "\n| Number of samples: " + str(self.n_presence)
        text += "\n| Source of stations locations: " + self.stations.file_name
        text += "\n| Source of species locations: " + self.file_name
        text += "\n| Number of stations: " + str(len(self.x_train))
        for name, model in self.models.items():
            text += "\n|" + name + " training AUC: " + str(model.get_auc(self.x_train, self.y_train))[: 4]

        return text
