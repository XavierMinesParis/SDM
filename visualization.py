# -*- coding: utf-8 -*-
# +
import matplotlib. pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 5)
from stations import *
from sklearn.metrics import roc_curve, auc

class Visualization:
    """
    Displays information about species and its relationships with climatic variables: the distributions,
    the concentrations, the climatic optimums, etc.
    """
    
    def plot_area(species):
        grid = pd.read_csv('Grid/final.csv', sep=",")
        plt.rcParams['figure.figsize'] = (10, 7)
        plt.scatter(grid['lon'], grid['lat'], c='beige', s=1)
        for s in species:
            plt.scatter(s.locations['lon'], s.locations['lat'], s=1, label=s.latin_name)
        plt.ylim(41, 52)
        plt.xlim(-5, 10)
        plt.legend()
        plt.show()
        
    def plot_prediction_maps(species):
        grid = pd.read_csv('Grid/grid_climate.csv', sep=",")
        stations = pd.read_csv('stations_climate.csv', sep=",")
        x_test = grid[dict_variables.values()].values
        n_species = len(species)

        plt.rcParams['figure.figsize'] = (15, 5 * n_species)
        fig, axs = plt.subplots(n_species, 3, squeeze=False)  # ensures axs is always 2D

        for i, s in enumerate(species):
            
            # Plot Presence and Absence points
            axs[i, 0].scatter(stations['lon'], stations['lat'], c='gray', cmap='viridis', s=1, alpha=0.07, label="Absence points")
            axs[i, 0].scatter(s.locations['lon'], s.locations['lat'], c="lime", s=1, alpha=0.5, label="Presence points")
            axs[i, 0].set_title(s.latin_name + ' - Sampling Points')
            axs[i, 0].set_xlim(-5, 10)
            axs[i, 0].set_ylim(41, 52)
            
            # Predictions
            lr_pred = s.lr_model.predict(x_test)
            em_pred = s.em_model.predict(x_test)

            # Plot Logistic Regression
            axs[i, 1].scatter(grid['lon'], grid['lat'], c=np.log(1 + lr_pred), cmap='viridis', s=8)
            axs[i, 1].set_title(s.latin_name + ' — Logistic Regression (log scale)')
            axs[i, 1].set_xlim(-5, 10)
            axs[i, 1].set_ylim(41, 52)

            # Plot Empirical Model
            axs[i, 2].scatter(grid['lon'], grid['lat'], c=em_pred, cmap='viridis', s=8)
            axs[i, 2].set_title(s.latin_name + ' — Empirical Model')
            axs[i, 2].set_xlim(-5, 10)
            axs[i, 2].set_ylim(41, 52)

        plt.tight_layout()
        plt.show()

    
    def plot_concentration(species, column):
        plt.plot(Stations.ubiquist_proximities[column], label="Poximity of the ubiquist species")
        plt.plot(species.variables[column].proximities, label="Proximity of " + species.latin_name)
        plt.plot(species.variables[column].concentrations, label='Concentration')
        plt.legend()
        plt.title('Concentration for ' + Stations.dict_variables[column])
        plt.show()
    
    def plot_hist(species, column):
        
        if not isinstance(species, list):
            species = [species]
        
        counts, bin_edges = Stations.distributions[column]
        plt.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges),
                align='edge', alpha=0.5, color='grey', label='Ubiquist species')
        for s in species:
            variable = s.variables[column]
            plt.bar(x=bin_edges[:-1], height=variable.counts, width=np.diff(bin_edges),
                    align='edge', alpha=0.5, label=s.latin_name)
            plt.axvline(bin_edges[variable.optimum_range])
        plt.legend()
        plt.title("Distribution by " + Stations.dict_variables[column])
        plt.show()
        
    def plot_summary(species):
        
        if not isinstance(species, list):
            species = [species]
            
        labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle("Optimums of " + str([s.latin_name[: 20] for s in species]))
        plt.setp((ax1, ax2, ax3), xticks=np.arange(12), xticklabels=labels)
        
        tmax_columns = ['tx' + str(i).zfill(2) + '_61_90' for i in range (1, 13)]
        tmin_columns = ['tn' + str(i).zfill(2) + '_61_90' for i in range (1, 13)]
        rain_columns = ['rr' + str(i).zfill(2) + '_61_90' for i in range (1, 13)]
        
        
        for s in species:
            tmax_values = [s.variables[column].optimum_value for column in tmax_columns]
            plt.xticks(np.arange(12), labels)
            ax1.bar(np.arange(12), tmax_values, alpha=0.4, label=s.latin_name)
            
            tmin_values = [s.variables[column].optimum_value for column in tmin_columns]
            plt.xticks(np.arange(12), labels)
            ax2.bar(np.arange(12), tmin_values, alpha=0.4)
            
            rain_values = [s.variables[column].optimum_value for column in rain_columns]
            plt.xticks(np.arange(12), labels)
            ax3.bar(np.arange(12), rain_values, alpha=0.4)
        
        ubi_tmax = [Stations.medians[column] for column in tmax_columns]
        ax1.bar(np.arange(12), ubi_tmax, edgecolor='black', facecolor=(0, 0, 0, 0), label="Ubiquist species")
        ubi_tmin = [Stations.medians[column] for column in tmin_columns]
        ax2.bar(np.arange(12), ubi_tmin, edgecolor='black', facecolor=(0, 0, 0, 0))
        ubi_rain = [Stations.medians[column] for column in rain_columns]
        ax3.bar(np.arange(12), ubi_rain, edgecolor='black', facecolor=(0, 0, 0, 0))
           
        ax1.set_title("Temperature Day")
        ax2.set_title("Temperature Night")
        ax3.set_title("Rainfall")
        
        fig.tight_layout()
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='upper left', ncol=1)
        plt.show()
        
    def plot_roc(y, y_predict, title='ROC curve'):
        fpr, tpr, thresholds = roc_curve(y, y_predict)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        
    
