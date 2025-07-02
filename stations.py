# +
from constants import *
from empirical_model import *

class Stations:
    
    def __init__(self, file_name, id_name, ids=None, locations=None, 
                 data=None, distributions=None, ubiquist_proximities=None):
        """
        A Stations object designed for ecological applications.
        
        Attributes:
        file_name (str): Source file of stations.
        id_name (str): Column for stations ids.
        ids (pd.Series): Ids of the stations.
        locations (pd.Dataframe): Longitude and latitude of the stations.
        data (pd.Dataframe): Climate data.
        distributions (dict): Keys are names of climate variables and values are tuple of size 2.
        The two values are the counts ands bin edges of the climate histogram of the corresponding climate variable.
        ubiquist_proximities (dict): Keys are names of climate variables and values are lists of proximity values.
        """
        
        self.file_name = file_name
        self.id_name = id_name 
        
        df = pd.read_csv('Data/' + file_name, sep=",", on_bad_lines='skip')
        self.locations = df[['lon', 'lat']]
        self.ids = df[id_name]
        self.climate_data = df[CLIMATE_VARIABLES]
        
        distributions = dict()
        ubiquist_proximities = dict()
    
        for column in CLIMATE_VARIABLES:
            variable = df[column].values
            counts, bin_edges = np.histogram(variable, bins=np.unique(variable), density=True)
            counts /= np.sum(counts)
            distributions[column] = (counts, bin_edges)
            ubiquist_proximities[column] = EmpiricalModel.get_proximities(bin_edges, counts, counts)
            
        self.distributions = distributions
        self.ubiquist_proximities = ubiquist_proximities
