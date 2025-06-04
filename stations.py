# +
from statistics import *
from extractor import *
from constants import *

class Stations:
    
    df = {'current': pd.read_csv("Data/stations_climate_current.csv", sep=",", on_bad_lines='skip'),
          'ssp245': pd.read_csv("Data/stations_climate_ssp245.csv", sep=",", on_bad_lines='skip'),
          'ssp585': pd.read_csv("Data/stations_climate_ssp245.csv", sep=",", on_bad_lines='skip')}
    
    locations = dict()
    distributions = dict()
    ubiquist_proximities = dict()
    medians = dict()
    
    for scenario in df.keys():
        
        locations[scenario] = df[scenario][['lon', 'lat']]
        distributions[scenario] = dict()
        ubiquist_proximities[scenario] = dict()
        medians[scenario] = dict()
        
        if scenario == 'ssp245':
            dict_variables = ssp245_variables
        elif scenario == 'ssp585':
            dict_variables = ssp245_variables
        else:
            dict_variables = current_variables
    
        for column in dict_variables.values():
            variable = df[scenario][column].values
            counts, bin_edges = np.histogram(variable, bins=np.unique(variable), density=True)
            counts /= np.sum(counts)
            distributions[scenario][column] = (counts, bin_edges)
            ubiquist_proximities[scenario][column] = Statistics.get_proximities(bin_edges, counts, counts)
            medians[scenario][column] = np.median(variable)
