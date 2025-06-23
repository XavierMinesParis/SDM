# +
from constants import *
from empirical_model import *

class Stations:
    
    def __init__(self, file_name, id_name, locations=None, ids=None,  
                 data=None, distributions=None, ubiquist_proximities=None):
        
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
