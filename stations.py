# +
from statistics import *

class Stations:
    
    climat_fr = pd.read_csv('climat_fr_61_90_bis.csv', sep=",")
    dict_variables = {'rr01_61_90': 'Rainfall January', 'rr02_61_90': 'Rainfall February',
                  'rr03_61_90': 'Rainfall March', 'rr04_61_90': 'Rainfall April',
                  'rr05_61_90': 'Rainfall May', 'rr06_61_90': 'Rainfall June',
                  'rr07_61_90': 'Rainfall July', 'rr08_61_90': 'Rainfall August',
                  'rr09_61_90': 'Rainfall September', 'rr10_61_90': 'Rainfall October',
                  'rr11_61_90': 'Rainfall November', 'rr12_61_90': 'Rainfall December',
                  'tx01_61_90': 'Tmax January', 'tx02_61_90': 'Tmax February', 'tx03_61_90': 'Tmax March',
                  'tx04_61_90': 'Tmax April', 'tx05_61_90': 'Tmax May', 'tx06_61_90': 'Tmax June',
                  'tx07_61_90': 'Tmax July', 'tx08_61_90': 'Tmax August', 'tx09_61_90': 'Tmax September',
                  'tx10_61_90': 'Tmax October', 'tx11_61_90': 'Tmax November', 'tx12_61_90': 'Tmax December',
                  'tn01_61_90': 'Tmin January', 'tn02_61_90': 'Tmin February', 'tn03_61_90': 'Tmin March',
                  'tn04_61_90': 'Tmin April', 'tn05_61_90': 'Tmin May', 'tn06_61_90': 'Tmin June',
                  'tn07_61_90': 'Tmin July', 'tn08_61_90': 'Tmin August', 'tn09_61_90': 'Tmin September',
                  'tn10_61_90': 'Tmin October', 'tn11_61_90': 'Tmin November', 'tn12_61_90': 'Tmin December'}

    data = climat_fr[dict_variables.keys()]
    distributions = dict()
    ubiquist_proximities = dict()
    medians = dict()
    
    for column in data.columns:
        variable = data[column]
        counts, bin_edges = np.histogram(variable.values, bins=np.unique(variable), density=True)
        counts /= np.sum(counts)
        distributions[column] = (counts, bin_edges)
        ubiquist_proximities[column] = Statistics.get_proximities(bin_edges, counts, counts)
        medians[column] = np.median(variable.values)
