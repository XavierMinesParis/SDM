# -*- coding: utf-8 -*-
# +
import matplotlib. pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 5)
from stations import *
from constants import *
from sklearn.metrics import roc_curve, auc

class Visualization:
    """
    Displays information about species and its relationships with climatic variables: the distributions,
    the concentrations, the climatic optimums, etc.
    """
    
    def plot_relationships(simulation):
        s = simulation
        plt.rcParams['figure.figsize'] = (15, 8)
        fig, axs = plt.subplots(4, 2)
        for i in range(4):
            for j in range(2):
                if (i, j) in s.scenarios:
                    axs[i, j].scatter(s.cA_train, s.psi_train[j, i], s=1, label="phi (occupancy)", alpha=0.1)
                    axs[i, j].scatter(s.cA_train, s.p_train[j, i], s=1, label="p (conditional detection)", alpha=0.1)
                    axs[i, j].scatter(s.cA_train, s.d_train[j, i], s=1, label="d (detection for one trial)", alpha=0.1)
                    axs[i, j].scatter(s.cA_train, s.dsk_train[j, i], s=1, label="dk (detection for the K trials)", alpha=0.1)
                    axs[i, j].set_title(Simulation.LABELS[j, i])

        for ax in axs.flat:
            ax.set(xlabel='Covariable cA', ylabel='Probability')
            ax.label_outer()

        fig.tight_layout()
        plt.legend()
        plt.show()
        
    def plot_best_models(simulation, i, j):
        s = simulation
        lr_models, od_models, em_models, maxent_models = s.get_best_models()
        od_model = od_models[i][j]
        lr_model = lr_models[i][j]
        em_model = em_models[i][j]
        maxent_model = maxent_models[i][j]

        plt.rcParams['figure.figsize'] = (15, 6)
        fig, axs = plt.subplots(2, 5)

        psi_grid = s.psi_test[i][j].reshape(s.cA_grid.shape)
        axs[0, 0].matshow(psi_grid, cmap='jet_r', alpha=0.6)
        axs[0, 0].scatter(50 * (s.cA_train / np.sqrt(3) + 1), 50 * (-s.cB_train / np.sqrt(3) + 1),
                          cmap='jet_r', c=s.Z_train[i, j], alpha=0.3, s=12)
        axs[0, 0].set_title('Probability of occupancy')

        dsk_grid = s.dsk_test[i][j].reshape(s.cA_grid.shape)
        axs[1, 0].matshow(dsk_grid , cmap='jet_r', alpha=0.6)
        axs[1, 0].scatter(50 * (s.cA_train / np.sqrt(3) + 1), 50 * (-s.cB_train / np.sqrt(3) + 1),
                          cmap='jet_r', c=s.Y_train[i, j], alpha=0.3, s=12)
        axs[1, 0].set_title('Probability of detection')

        od_occupancy, od_detection = od_model.predict(s.X_test[s.od_ids[i][j]])
        od_occupancy, od_detection = od_occupancy.reshape(s.cA_grid.shape), od_detection.reshape(s.cA_grid.shape)
        axs[0, 1].matshow(od_occupancy, cmap='jet_r')
        title = 'OD (occupancy): RMSE=' + str(od_model.rmse)[: 4]
        title += '\nAUC=' + str(od_model.auc)[: 4] + ' Spearman:' + str(od_model.spearman)[: 4]
        axs[0, 1].set_title(title)
        axs[1, 1].matshow(od_detection, cmap='jet_r')
        axs[1, 1].set_title('OD (detection)')

        lr_prediction = lr_model.predict(s.X_test[s.lr_ids[i][j]])
        lr_prediction = lr_prediction.reshape(s.cA_grid.shape)
        axs[0, 2].matshow(lr_prediction, cmap='jet_r')
        title = 'LR: RMSE=' + str(lr_model.rmse)[: 4]
        title += '\nAUC=' + str(lr_model.auc)[: 4] + ' Spearman:' + str(lr_model.spearman)[: 4]
        axs[0, 2].set_title(title)
        
        maxent_prediction = maxent_model.predict(s.X_test[s.maxent_ids[i][j]])
        maxent_prediction = maxent_prediction.reshape(s.cA_grid.shape)
        axs[0, 3].matshow(maxent_prediction, cmap='jet_r')
        title = 'Maxent: RMSE=' + str(maxent_model.rmse)[: 4]
        title += '\nAUC=' + str(maxent_model.auc)[: 4] + ' Spearman:' + str(maxent_model.spearman)[: 4]
        axs[0, 3].set_title(title)

        em_prediction = em_model.predict(s.X_test[s.em_ids[i][j]])
        em_prediction = em_prediction.reshape(s.cA_grid.shape)
        axs[0, 4].matshow(em_prediction, cmap='jet_r')
        title = "EM: RMSE=" + str(em_model.rmse)[: 4] + '\nAUC=' + str(em_model.auc)[: 4]
        title += ' Spearman:' + str(em_model.spearman)[: 4]
        axs[0, 4].set_title(title)
        
        axs[1, 2].set_axis_off()
        axs[1, 3].set_axis_off()
        axs[1, 4].set_axis_off()

        plt.show()
    
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
        
        stations = Stations("stations_climate_current.csv", id_name='id')
        
        current_grid = pd.read_csv('Data/grid_reduced_current.csv')
        x_current = current_grid[CLIMATE_VARIABLES]
        lr_current = species.lr_model.predict(x_current)
        em_current = species.em_model.predict(x_current)
        ssp245_grid = pd.read_csv('Data/grid_reduced_ssp245.csv')
        x_ssp245 = ssp245_grid[CLIMATE_VARIABLES]
        lr_ssp245 = species.lr_model.predict(x_ssp245)
        em_ssp245 = species.em_model.predict(x_ssp245)
        ssp585_grid = pd.read_csv('Data/grid_reduced_ssp585.csv')
        x_ssp585 = ssp585_grid[CLIMATE_VARIABLES]
        lr_ssp585 = species.lr_model.predict(x_ssp585)
        em_ssp585 = species.em_model.predict(x_ssp585)
        
        x_train, y_train = species.x_train, species.y_train

        plt.rcParams['figure.figsize'] = (16, 8)
        fig, axs = plt.subplots(2, 4, squeeze=False)

        # Plot Presence and Absence points
        axs[0, 0].scatter(stations.locations['lon'], stations.locations['lat'], c='gray',
                          cmap='viridis', s=1, alpha=0.07, label="Absence points")
        axs[0, 0].scatter(species.locations['lon'], species.locations['lat'], c="lime",
                          s=1, alpha=0.5, label="Presence points")
        axs[0, 0].set_title(species.latin_name + ' - Sampling Points')
        axs[0, 0].set_xlim(-5, 10)
        axs[0, 0].set_ylim(41, 52)
        
        axs[1, 0].scatter(stations.locations['lon'], stations.locations['lat'], c='gray',
                          cmap='viridis', s=1, alpha=0.07, label="Absence points")
        axs[1, 0].scatter(species.locations['lon'], species.locations['lat'], c="lime",
                          s=1, alpha=0.5, label="Presence points")
        axs[1, 0].set_title(species.latin_name + ' - Sampling Points')
        axs[1, 0].set_xlim(-5, 10)
        axs[1, 0].set_ylim(41, 52)

        # Plot Logistic Regression
        axs[0, 1].scatter(current_grid['lon'], current_grid['lat'], c=lr_current, cmap='viridis', s=8)
        axs[0, 1].set_title('LR Current \n' + "LR training AUC: " + str(species.lr_model.get_auc(x_train, y_train))[: 4])
        axs[0, 1].set_xlim(-5, 10)
        axs[0, 1].set_ylim(41, 52)
        
        axs[0, 2].scatter(ssp245_grid['lon'], ssp245_grid['lat'], c=lr_ssp245, cmap='viridis', s=8)
        axs[0, 2].set_title('LR SSP245')
        axs[0, 2].set_xlim(-5, 10)
        axs[0, 2].set_ylim(41, 52)
        
        axs[0, 3].scatter(ssp585_grid['lon'], ssp585_grid['lat'], c=lr_ssp585, cmap='viridis', s=8)
        axs[0, 3].set_title('LR SSP585')
        axs[0, 3].set_xlim(-5, 10)
        axs[0, 3].set_ylim(41, 52)

        # Plot Empirical Model
        axs[1, 1].scatter(current_grid['lon'], current_grid['lat'], c=em_current, cmap='viridis', s=8)
        axs[1, 1].set_title('EM Current \n' + "EM training AUC: " + str(species.em_model.get_auc(x_train, y_train))[: 4])
        axs[1, 1].set_xlim(-5, 10)
        axs[0, 1].set_ylim(41, 52)
        
        axs[1, 2].scatter(ssp245_grid['lon'], ssp245_grid['lat'], c=em_ssp245, cmap='viridis', s=8)
        axs[1, 2].set_title('EM SSP245')
        axs[1, 2].set_xlim(-5, 10)
        axs[1, 2].set_ylim(41, 52)
        
        axs[1, 3].scatter(ssp585_grid['lon'], ssp585_grid['lat'], c=em_ssp585, cmap='viridis', s=8)
        axs[1, 3].set_title('EM SSP585')
        axs[1, 3].set_xlim(-5, 10)
        axs[1, 3].set_ylim(41, 52)

        plt.tight_layout()
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
    
