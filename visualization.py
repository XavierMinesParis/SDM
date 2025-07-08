# -*- coding: utf-8 -*-
# +
import matplotlib. pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 5)
from stations import *
from constants import *
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import TwoSlopeNorm

class Visualization:
    """
    Displays information about species and its relationships with climate variables: the distributions,
    the concentrations, the climatic optimums, etc.
    """
    
    def plot_relationships(simulation, save=False):
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
        
        if save:
            plt.savefig('Figures/relationships.png', bbox_inches='tight')
        plt.show()
        
    def plot_best_models(simulation, i, j, save=False):
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
        
        if save:
            plt.savefig('Figures/simulations_best_models.png', bbox_inches='tight')
        plt.show()
        
    def plot_range_simulations(n_trains, res, i, j, save=False):
        
        rmse, auc, spearman = res
        
        lr_rmse, od_rmse, em_rmse, maxent_rmse = rmse
        lr_auc, od_auc, em_auc, maxent_auc = auc
        lr_spearman, od_spearman, em_spearman, maxent_spearman = spearman
        
        lr_rmse, od_rmse, em_rmse, maxent_rmse = lr_rmse[i, j], od_rmse[i, j], em_rmse[i, j], maxent_rmse[i, j]
        lr_auc, od_auc, em_auc, maxent_auc = lr_auc[i, j], od_auc[i, j], em_auc[i, j], maxent_auc[i, j]
        lr_spearman, od_spearman = lr_spearman[i, j], od_spearman[i, j]
        em_spearman, maxent_spearman = em_spearman[i, j], maxent_spearman[i, j]
        
        fig, axs = plt.subplots(1, 3)

        plt.xlim(0, 2100)

        axs[0].plot(n_trains, np.median(lr_rmse, axis=1), label='Logistic Regression')
        axs[0].plot(n_trains, np.median(od_rmse, axis=1), label='Occupancy Detection Model')
        axs[0].plot(n_trains, np.median(em_rmse, axis=1), label='Empirical Model')
        axs[0].plot(n_trains, np.median(maxent_rmse, axis=1), label='Maxent')
        box = axs[0].boxplot(lr_rmse.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('blue')
        for patch in box['boxes']:
            patch.set_facecolor('blue')
            patch.set_alpha(0.2)
        box = axs[0].boxplot(od_rmse.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('orange')
        for patch in box['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.2)
        box = axs[0].boxplot(em_rmse.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('green')
        for patch in box['boxes']:
            patch.set_facecolor('green')
            patch.set_alpha(0.2)
        box = axs[0].boxplot(maxent_rmse.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('red')
        for patch in box['boxes']:
            patch.set_facecolor('red')
            patch.set_alpha(0.2)
        axs[0].set_title('RMSE')
        axs[0].legend()

        axs[1].plot(n_trains, np.median(lr_auc, axis=1), label='Logistic Regression')
        axs[1].plot(n_trains, np.median(od_auc, axis=1), label='Occupancy Detection Model')
        axs[1].plot(n_trains, np.median(em_auc, axis=1), label='Empirical Model')
        axs[1].plot(n_trains, np.median(maxent_auc, axis=1), label='Maxent')
        box = axs[1].boxplot(lr_auc.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('blue')
        for patch in box['boxes']:
            patch.set_facecolor('blue')
            patch.set_alpha(0.2)
        box = axs[1].boxplot(od_auc.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('orange')
        for patch in box['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.2)
        box = axs[1].boxplot(em_auc.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('green')
        for patch in box['boxes']:
            patch.set_facecolor('green')
            patch.set_alpha(0.2)
        box = axs[1].boxplot(maxent_auc.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('red')
        for patch in box['boxes']:
            patch.set_facecolor('red')
            patch.set_alpha(0.2)
        axs[1].set_title('AUC')

        axs[2].plot(n_trains, np.median(lr_spearman, axis=1), label='Logistic Regression')
        axs[2].plot(n_trains, np.median(od_spearman, axis=1), label='Occupancy Detection Model')
        axs[2].plot(n_trains, np.median(em_spearman, axis=1), label='Empirical Model')
        axs[2].plot(n_trains, np.median(maxent_spearman, axis=1), label='Maxent')
        box = axs[2].boxplot(lr_spearman.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('blue')
        for patch in box['boxes']:
            patch.set_facecolor('blue')
            patch.set_alpha(0.2)
        box = axs[2].boxplot(od_spearman.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('orange')
        for patch in box['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.2)
        box = axs[2].boxplot(em_spearman.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('green')
        for patch in box['boxes']:
            patch.set_facecolor('green')
            patch.set_alpha(0.2)
        box = axs[2].boxplot(maxent_spearman.transpose(), positions=n_trains,
                             patch_artist=True, widths=80, showfliers=False)
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for item in box[element]:
                item.set_color('red')
        for patch in box['boxes']:
            patch.set_facecolor('red')
            patch.set_alpha(0.2)
        axs[2].set_title('Spearman')

        if save:
            plt.savefig('Figures/simulations_rmse_auc_spearman.png', bbox_inches='tight')
        plt.show()
        
    def plot_prediction_maps(species, save=False):

        stations = species.stations
        climate_variables = stations.climate_variables
        x_train, y_train = species.x_train, species.y_train
        
        current_grid = pd.read_csv('Data/grid_reduced_current.csv').copy()
        x_current = current_grid[climate_variables]
        ssp245_grid = pd.read_csv('Data/grid_reduced_ssp245.csv').copy()
        x_ssp245 = ssp245_grid[climate_variables]
        ssp585_grid = pd.read_csv('Data/grid_reduced_ssp585.csv').copy()
        x_ssp585 = ssp585_grid[climate_variables]
        
        models = species.models
        n = len(models)
        
        current_columns, ssp245_columns, ssp585_columns = list(), list(), list()
        for name, model in models.items():
            current_grid[name + "_current"] = model.predict(x_current)
            ssp245_grid[name + "_ssp245"] = model.predict(x_ssp245)
            ssp585_grid[name + "_ssp585"] = model.predict(x_ssp585)
            current_columns.append(name + "_current")
            ssp245_columns.append(name + "_ssp245")
            ssp585_columns.append(name + "_ssp585")
            
        common_keys = pd.merge(current_grid[['lon', 'lat']], ssp245_grid[['lon', 'lat']],
                               on=['lon', 'lat'], how='inner')
        df1 = pd.merge(current_grid, common_keys, on=['lon', 'lat'], how='inner')[['lon', 'lat'] + current_columns]
        df2 = pd.merge(ssp245_grid, common_keys, on=['lon', 'lat'], how='inner')[['lon', 'lat'] + ssp245_columns]
        df3 = pd.merge(ssp585_grid, common_keys, on=['lon', 'lat'], how='inner')[['lon', 'lat'] + ssp585_columns]
        df4 = pd.merge(df1, df2, on=['lon', 'lat'], how='inner')
        df = pd.merge(df3, df4, on=['lon', 'lat'], how='inner')

        plt.rcParams['figure.figsize'] = (16, 10 * n)
        fig, axs = plt.subplots(2 * n, 3, squeeze=False)
        norm = TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=0.3)
        
        for i, item in enumerate(models.items()):
            
            name, model = item

            # Plot Presence and Absence points
            axs[2 * i + 1, 0].scatter(stations.locations['lon'], stations.locations['lat'], c='gray',
                              cmap='viridis', s=1, alpha=0.07, label="Absence points")
            axs[2 * i + 1, 0].scatter(species.locations['lon'], species.locations['lat'], c="lime",
                              s=1, alpha=0.5, label="Presence points")
            axs[2 * i + 1, 0].set_title(species.latin_name + ' - Sampling Points')
            axs[2 * i + 1, 0].set_xlim(-5, 10)
            axs[2 * i + 1, 0].set_ylim(41, 52)

            # Plot max and min values on all predictions
            vmax = max(max(np.max(df[name + '_current'].values), np.max(df[name + '_ssp245'].values)),
                       np.max(df[name + '_ssp585'].values))
            vmin = min(min(np.min(df[name + '_current'].values), np.min(df[name + '_ssp245'].values)),
                       np.min(df[name + '_ssp585'].values))
            
            scatter = axs[2 * i, 0].scatter(df['lon'], df['lat'], c=df[name + '_current'], cmap='viridis', s=8,
                                           vmin=vmin, vmax=vmax)
            axs[2 * i, 0].set_title(name + ' Current \n' + "Training AUC: " + str(model.get_auc(x_train, y_train))[: 4])
            axs[2 * i, 0].set_xlim(-5, 10)
            axs[2 * i, 0].set_ylim(41, 52)
            fig.colorbar(scatter, ax=axs[2 * i, 0], pad=0)
            
            scatter = axs[2 * i, 1].scatter(df['lon'], df['lat'], c=df[name + '_ssp245'], cmap='viridis', s=8,
                                           vmin=vmin, vmax=vmax)
            axs[2 * i, 1].set_title(name + ' SSP245')
            axs[2 * i, 1].set_xlim(-5, 10)
            axs[2 * i, 1].set_ylim(41, 52)
            fig.colorbar(scatter, ax=axs[2 * i, 1], pad=0)

            scatter = axs[2 * i, 2].scatter(df['lon'], df['lat'], c=df[name + '_ssp585'], cmap='viridis', s=8,
                                           vmin=vmin, vmax=vmax)
            axs[2 * i, 2].set_title(name + ' SSP585')
            axs[2 * i, 2].set_xlim(-5, 10)
            axs[2 * i, 2].set_ylim(41, 52)
            fig.colorbar(scatter, ax=axs[2 * i, 2], pad=0)

            scatter = axs[2 * i + 1, 1].scatter(df['lon'], df['lat'], c=df[name + '_ssp245'] - df[name + '_current'],
                                        cmap='seismic_r', s=8, norm=norm)
            axs[2 * i + 1, 1].set_title(name + ' SSP245 Trend')
            axs[2 * i + 1, 1].set_xlim(-5, 10)
            axs[2 * i + 1, 1].set_ylim(41, 52)
            fig.colorbar(scatter, ax=axs[2 * i + 1, 1], pad=0)

            scatter = axs[2 * i + 1, 2].scatter(df['lon'], df['lat'], c=df[name + '_ssp585'] - df[name + '_current'],
                                        cmap='seismic_r', s=8, norm=norm)
            axs[2 * i + 1, 2].set_title(name + ' SSP585 Trend')
            axs[2 * i + 1, 2].set_xlim(-5, 10)
            axs[2 * i + 1, 2].set_ylim(41, 52)
            fig.colorbar(scatter, ax=axs[2 * i + 1, 2], pad=0)

        plt.tight_layout()
        if save:
            fig.savefig('Figures/' + str(species.id_) + '.png')
        else:
            plt.show()

    def plot_hist(species, column, save=False):
        
        plt.rcParams['figure.figsize'] = (15, 5)
        
        if not isinstance(species, list):
            species = [species]
        
        ubi_counts, bin_edges = species[0].stations.distributions[column]
        plt.bar(x=bin_edges[:-1], height=ubi_counts, width=np.diff(bin_edges),
                align='edge', alpha=0.5, color='grey', label='Ubiquist species - Background data')
        
        for s in species:
            counts, bin_edges = s.distributions[column]
            plt.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges),
                    align='edge', alpha=0.5, label=s.latin_name)
            
        plt.legend()
        plt.title("Distribution by " + column)
        
        if save:
            plt.savefig('/Figures/hist' + column + '_' + '.png', bbox_inches='tight')
        plt.show()
        
    def plot_em_summary(species):
        """
        Outdated.
        """
        
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
    
# -


