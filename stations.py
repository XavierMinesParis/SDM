# +
from statistics import *
from extractor import *

class Stations:
    
    df = pd.read_csv("stations_climate.csv", sep=",", on_bad_lines='skip')
    locations = df[['lon', 'lat']]
    distributions = dict()
    ubiquist_proximities = dict()
    medians = dict()
    
    for column in dict_variables.values():
        variable = df[column].values
        counts, bin_edges = np.histogram(variable.values, bins=np.unique(variable), density=True)
        counts /= np.sum(counts)
        distributions[column] = (counts, bin_edges)
        ubiquist_proximities[column] = Statistics.get_proximities(bin_edges, counts, counts)
        medians[column] = np.median(variable.values)
