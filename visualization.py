# +
import matplotlib. pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 5)
from stations import *
from species import *

class Visualization:
    """
    Displays information about species and its relationships with climatic variables: the distributions,
    the concentrations, the climatic optimums, etc.
    """
    
    def plot_concentration(species, column):
        plt.plot(Stations.ubiquist_proximities[column], label="Poximity of the ubiquist species")
        plt.plot(species.proximities[column], label="Proximity of " + species.latin_name)
        plt.plot(species.concentrations[column], label='Concentration')
        plt.legend()
        plt.title('Concentration for ' + Stations.dict_variables[column])
        plt.show()
    
    def plot_hist(species, column):
        
        if not isinstance(species, list):
            species = [species]
        
        counts, bin_edges = Stations.distributions[column]
        plt.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges), align='edge', alpha=0.5, color='grey')
        for s in species:
            data = s.data
            if column in data.columns:
                counts, bin_edges = s.distributions[column]
                plt.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges),
                        align='edge', alpha=0.5, label=s.latin_name)
                plt.axvline(bin_edges[s.optimums[column][0]])
        plt.legend()
        plt.title("Distribution by " + Stations.dict_variables[column])
        plt.show()
        
    def plot_summary(species):
        
        if not isinstance(species, list):
            species = [species]
            
        labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle(str([s.latin_name[: 20] for s in species]))
        plt.setp((ax1, ax2, ax3), xticks=np.arange(12), xticklabels=labels)
        
        tmax_columns = ['tx' + str(i).zfill(2) + '_61_90' for i in range (1, 13)]
        tmin_columns = ['tn' + str(i).zfill(2) + '_61_90' for i in range (1, 13)]
        rain_columns = ['rr' + str(i).zfill(2) + '_61_90' for i in range (1, 13)]
        
        for s in species:
            tmax_powers = np.array([s.optimums[column][0] for column in tmax_columns])
            tmax_values = []
            for tmax_power, column in zip(tmax_powers, tmax_columns):
                bin_edges = s.distributions[column][1]
                tmax_values.append(bin_edges[tmax_power])
            tmax_optimums = [s.optimums[column][1] for column in tmax_columns]
            plt.xticks(np.arange(12), labels)
            ax1.bar(np.arange(12), tmax_values, alpha=0.4, label=s.latin_name)
            ax1.set_title("Max Temperature")
            
            tmin_powers = np.array([s.optimums[column][0] for column in tmin_columns])
            tmin_values = []
            for tmin_power, column in zip(tmin_powers, tmin_columns):
                bin_edges = s.distributions[column][1]
                tmin_values.append(bin_edges[tmin_power])
            tmin_optimums = [s.optimums[column][1] for column in tmin_columns]
            plt.xticks(np.arange(12), labels)
            ax2.bar(np.arange(12), tmin_values, alpha=0.4)
            ax2.set_title("Min Temperature")
            
            rain_powers = np.array([s.optimums[column][0] for column in rain_columns])
            rain_values = []
            for rain_power, column in zip(rain_powers, rain_columns):
                bin_edges = s.distributions[column][1]
                rain_values.append(bin_edges[rain_power])
            rain_optimums = [s.optimums[column][1] for column in rain_columns]
            plt.xticks(np.arange(12), labels)
            ax3.bar(np.arange(12), rain_values, alpha=0.4)
            ax3.set_title("Rainfall")
        
        fig.tight_layout()
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='upper left', ncol=1)
        plt.show()
