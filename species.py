# -*- coding: utf-8 -*-
# +
from logistic_regression import *
from empirical_model import *
from extractor import *

class Species:
    
    def __init__(self, file_name, stations, id_name=None, id_=None,
                 id_stations_name=None, n_presence=None, train_status=False):
        """
        Species object.

        Attributes:
        file_name (str): Name of the csv file with species locations
        stations (Stations): Used to keep only valid points at climate stations, and used for training.
        id_name (str): Name of the column providing species ids in the species table
        id_ (int): Name of the species id
        id_stations_name: Column of the dataframe with stations ids
        locations (pd.Dataframe): Locations of samples.
        n_presence
        
        id_stations (pd.Series)
        latin_name (str)
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

        self.train_status = train_status
        self.x_train = None
        self.y_train = None
        self.lr_model = None
        self.em_model = None
        
    def train_models(self, stations=None):
        
        self.train_status = True
        
        if stations is None:
            stations = self.stations
            
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

        lr_model = LogisticRegression()
        lr_model.fit(x_train, y_train)
        self.lr_model = lr_model
        

        em_model = EmpiricalModel()
        em_model.fit(x_train, y_train, stations=stations)
        self.em_model = em_model
        
    def test_models(self, locations_file_name=None, climate_folder=None):
        
        extractor = Extractor(locations_file_name, climate_folder)
        climate_data = extractor.extract(verbose=False)[CLIMATE_VARIABLES]
        x_test = climate_data.values
        
        return self.lr_model.predict(x_test), self.em_model.predict(x_test)
        
    def __repr__(self):
        
        text = "| Name: " + self.latin_name
        text += "\n| ID (" + self.id_name + "): " + str(self.id_)
        text += "\n| Number of samples: " + str(self.n_presence)
        text += "\n| Source of species locations: " + self.file_name
        
        if self.train_status:
            text += "\n| Number of stations: " + str(len(self.x_train))
            text += "\n|LR training AUC: " + str(self.lr_model.get_auc(self.x_train, self.y_train))[: 4]
            text += "\n|EM training AUC: " + str(self.em_model.get_auc(self.x_train, self.y_train))[: 4]

        return text
