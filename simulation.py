# +
from empirical_model import *
from logistic_regression import *
from occupancy_detection import *
from maxent import *
from scipy.special import expit
from itertools import chain, combinations

class Simulation:
    """
    This class aims to perform simulations for the LogisticRegression, OccupancyDetection, EmpiricalModel and Maxent models.
     The environmental space is bidimensional. Because cross-variables are considered, the final space is 5-dimensional.
    """
    
    LABELS = np.array([["Steep, constant", "Steep, positive", "Steep, negative", "Steep, independent"], 
                   ["Gentle, constant", "Gentle, positive", "Gentle, negative", "Gentle, independent"]])
    
    SCENARIOS = [(i, j) for i in range(2) for j in range(4)]
    
    def __init__(self, n_train=400, K=2, min_size=1, scenarios=SCENARIOS):
        """
        Computes probabilities of occupancy, detection.
        Creates the train and test datasets.
        
        Attributes:
        n_train (int): Number of presence-absent points.
        K (int): Number of attempts at one sampling location.
        scenarios (set): Contains tuples that identify the climate scenarios considered in the simulation.
        min_size (int): Between 1 and 5. It is the minimum size of subsets in the environmental space to achieve model selection.
        cA_train (np.ndarray): Draws of the first climate variable.
        cB_train (np.ndarray): Draws of the second climate variable.
        X_train (list): Contains climate data for different subsets of variables.
        psi_train (np.ndarray): Probabilities of occupancy.
        p_train (np.ndarray): Probabilities of detection given species presence.
        d_train (np.ndarray): Probabilities  of detection.
        psk_train (np.ndarray): Probabilites of detection given species presence with K trials.
        dsk_train (np.ndarray): Probabilities of detection with K trials.
        Z_train (np.ndarray): Binary draw for occupancy, shape (2, 4, n_train).
        Y_k_train (np.ndarray): # Binomial draw for detection.
        Y_train (np.ndarray): # Binary detection, shape (2, 4, n_train).
        cA_test (np.ndarray): Draws of the first climate variable.
        cB_test (np.ndarray): Draws of the second climate variable.
        X_test (list): Contains climate data for different subsets of variables.
        psi_test (np.ndarray): Probabilities of occupancy.
        p_test (np.ndarray): Probabilities of detection given species presence.
        d_test (np.ndarray): Probabilities  of detection.
        psk_test (np.ndarray): Probabilites of detection given species presence with K trials.
        dsk_test (np.ndarray): Probabilities of detection with K trials.
        Z_test (np.ndarray): Binary draw for occupancy, shape (2, 4, n_test).
        Y_k_test (np.ndarray): # Binomial draw for detection.
        Y_test (np.ndarray): # Binary detection, shape (2, 4, n_test).
        """

        # Train data
        self.n_train = n_train
        self.K = K
        self.scenarios = scenarios
        self.cA_train = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_train)
        self.cB_train = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_train)
        train_subsets = Simulation.get_subsets([self.cA_train, self.cB_train, self.cA_train ** 2,
                                               self.cB_train ** 2, self.cA_train * self.cB_train], min_size=min_size)
        self.X_train = [np.array(subset).transpose() for subset in train_subsets]
        # Shape of X_train: (n_scenarios, n_train, m)
        
        # Probabilities of occupancy
        psi = np.array([expit(1 + 3 * self.cA_train), expit(self.cA_train)])  # Steep, gentle
        self.psi_train = np.tile(psi[:, np.newaxis, :], (1, 4, 1))

        # Conditional probabilities of detection
        p = np.array([n_train * [0.5], expit(self.cA_train),
                       expit(-self.cA_train), expit(self.cB_train)])  # Constant, positive, negative, independent
        self.p_train = np.tile(p[np.newaxis, :, :], (2, 1, 1))

        # Probabilities of detection, with 1 trial
        self.d_train = self.p_train * self.psi_train

        # Probabilities of detection with K trials
        self.psk_train = 1 - (1 - self.p_train) ** K
        self.dsk_train = self.psk_train * self.psi_train

        self.Z_train = np.random.binomial(1, self.psi_train) # Binary draw for occupancy, shape (2, 4, n_train)
        self.Y_K_train = np.random.binomial(K, self.Z_train * self.d_train) # Binomial draw for detection
        self.Y_train = 1 * (self.Y_K_train != 0) # Binary detection, shape (2, 4, n_train)
        
        # Test data
        cA_test, cB_test = np.linspace(-np.sqrt(3), np.sqrt(3), 100), np.linspace(-np.sqrt(3), np.sqrt(3), 100)
        cB_test = np.flip(cB_test)
        self.cA_grid, self.cB_grid = np.meshgrid(cA_test, cB_test)
        self.cA_test, self.cB_test = np.ravel(self.cA_grid), np.ravel(self.cB_grid)
        self.n_test = len(self.cA_test)
        test_subsets = Simulation.get_subsets([self.cA_test, self.cB_test, self.cA_test ** 2,
                                              self.cB_test ** 2, self.cA_test * self.cB_test], min_size=min_size)
        self.X_test = [np.array(subset).transpose() for subset in test_subsets]
        psi_test = np.array([expit(1 + 3 * self.cA_test), expit(self.cA_test)])  # Steep, gentle
        self.psi_test = np.tile(psi_test[:, np.newaxis, :], (1, 4, 1))
        p_test = np.array([self.n_test * [0.5], expit(self.cA_test),
                       expit(-self.cA_test), expit(self.cB_test)])  # Constant, positive, negative, independent
        self.p_test = np.tile(p_test[np.newaxis, :, :], (2, 1, 1))
        self.d_test = self.p_test * self.psi_test
        self.psk_test = 1 - (1 - self.p_test) ** K
        self.dsk_test = self.psk_test * self.psi_test
        
        self.Z_test = np.random.binomial(1, self.psi_test) # Binary draw for occupancy
        self.Y_K_test = np.random.binomial(K, self.Z_test * self.d_test) # Binomial draw for detection
        self.Y_test = 1 * (self.Y_K_train != 0) # Binary detection, shape (2, 4, n_train)
        
        self.em_models = None
        self.lr_models = None
        self.od_models = None
        self.maxent_models = None
    
    @staticmethod
    def get_subsets(iterable, min_size=1):
        """
        Returns the set of subsets of the iterable object.
        All subsets with cardinality lower to min_size are discarded.
        """
        
        s = list(iterable)
        res = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
        res_copy = res.copy()
        for subset in res_copy:
            if len(subset) < min_size:
                res.remove(subset)
        return res
        
    def create_models(self, verbose=True):
        """
        Creates and trains the four models: empirical model, logistic regression, occupancy-detection, Maxent.
        """
        
        lr_models = [4*[None], 4*[None]]
        em_models = [4*[None], 4*[None]]
        od_models = [4*[None], 4*[None]]
        maxent_models = [4*[None], 4*[None]]
        lr_ids = [4*[None], 4*[None]]
        em_ids = [4*[None], 4*[None]]
        od_ids = [4*[None], 4*[None]]
        maxent_ids = [4*[None], 4*[None]]

        if verbose:
            print("New simulation")
        for i in range(2):
            for j in range(4):
                if (i, j) in self.scenarios:
                    if verbose:
                        print("Scenario", Simulation.LABELS[i][j])
                    occupancy_label = self.Z_train[i, j]
                    y = self.Y_train[i, j]
                    y_K = self.Y_K_train[i, j]

                    lr_family = []
                    em_family = []
                    od_family = []
                    maxent_family = []

                    for x in self.X_train:

                        # Logistic regression
                        lr_model = LogisticRegression()
                        lr_model.fit(x, y)
                        lr_family.append(lr_model)

                        # Empirical model
                        em_model = EmpiricalModel()
                        em_model.fit(x, y)
                        em_family.append(em_model)

                        # Occupancy-detection model
                        od_model = OccupancyDetection(x, y_K, K=self.K)
                        od_model.fit()
                        od_family.append(od_model)

                        # Maximum Entropy model
                        presence, background = x[y == 1], x
                        maxent_model = Maxent(presence, background)
                        maxent_model.fit()
                        maxent_family.append(maxent_model)

                    # Model selection
                    lr_ids[i][j] = Simulation.get_best_model(lr_family)[0]
                    lr_models[i][j] = lr_family

                    em_ids[i][j] = Simulation.get_best_model(em_family)[0]
                    em_models[i][j] = em_family

                    od_ids[i][j] = Simulation.get_best_model(od_family)[0]
                    od_models[i][j] = od_family

                    maxent_ids[i][j] = Simulation.get_best_model(maxent_family)[0]
                    maxent_models[i][j] = maxent_family
                    
        self.lr_models = lr_models
        self.em_models = em_models
        self.od_models = od_models
        self.maxent_models = maxent_models
        self.lr_ids = lr_ids
        self.em_ids = em_ids
        self.od_ids = od_ids
        self.maxent_ids = maxent_ids
        
    @staticmethod
    def get_best_model(family):
        """
        Returns the model with the lowest Akaike information criterion (AIC)
        """

        aics = [model.get_aic() for model in family]
        id_ = np.argmin(aics)

        return id_, family[id_]
    
    def get_best_models(self):
        """
        Performs model selection in the respective families of the four models under study.
        """
        
        lr = [4*[None], 4*[None]]
        od = [4*[None], 4*[None]]
        em = [4*[None], 4*[None]]
        maxent = [4*[None], 4*[None]]
        for i in range(2):
            for j in range(4):
                if (i, j) in self.scenarios:
                    od_models = self.od_models[i][j]
                    lr_models = self.lr_models[i][j]
                    em_models = self.em_models[i][j]
                    maxent_models = self.maxent_models[i][j]

                    # Extracting the best models
                    od_model = Simulation.get_best_model(od_models)[1]
                    lr_model = Simulation.get_best_model(lr_models)[1]
                    em_model = Simulation.get_best_model(em_models)[1]
                    maxent_model = Simulation.get_best_model(maxent_models)[1]

                    # Extracting data
                    y_proba = self.psi_test[i][j] # Between 0 and 1
                    y_test = self.Z_test[i][j]

                    x_test = self.X_test[self.od_ids[i][j]]
                    auc = od_model.get_auc(x_test, y_test)
                    rmse = od_model.get_rmse(x_test, y_proba)
                    spearman = od_model.get_spearman(x_test, y_proba)
                    od[i][j] = od_model

                    x_test = self.X_test[self.lr_ids[i][j]]
                    auc = lr_model.get_auc(x_test, y_test)
                    rmse = lr_model.get_rmse(x_test, y_proba)
                    spearman = lr_model.get_spearman(x_test, y_proba)
                    lr[i][j] = lr_model

                    x_test = self.X_test[self.em_ids[i][j]]
                    auc = em_model.get_auc(x_test, y_test)
                    rmse = em_model.get_rmse(x_test, y_proba)
                    spearman = em_model.get_spearman(x_test, y_proba)
                    em[i][j] = em_model

                    x_test = self.X_test[self.maxent_ids[i][j]]
                    auc = maxent_model.get_auc(x_test, y_test)
                    rmse = maxent_model.get_rmse(x_test, y_proba)
                    spearman = maxent_model.get_spearman(x_test, y_proba)
                    maxent[i][j] = maxent_model
                
        return np.array(lr), np.array(od), np.array(em), np.array(maxent)
    
    @staticmethod
    def range_results(n_trains=[100, 400, 1000, 2000], n_simul=2, scenarios=SCENARIOS, verbose=True):
        """
        Creates and runs n_simul simulations for every n_train value provided in n_trains.
        Returns RMSE, AUC and Spearman's rank correlation coefficients for all simulations.
        """
        
        lr_rmse = np.zeros((2, 4, len(n_trains), n_simul))
        od_rmse = np.zeros((2, 4, len(n_trains), n_simul))
        em_rmse = np.zeros((2, 4, len(n_trains), n_simul))
        maxent_rmse = np.zeros((2, 4, len(n_trains), n_simul))

        lr_auc = np.zeros((2, 4, len(n_trains), n_simul))
        od_auc = np.zeros((2, 4, len(n_trains), n_simul))
        em_auc = np.zeros((2, 4, len(n_trains), n_simul))
        maxent_auc = np.zeros((2, 4, len(n_trains), n_simul))

        lr_spearman = np.zeros((2, 4, len(n_trains), n_simul))
        od_spearman = np.zeros((2, 4, len(n_trains), n_simul))
        em_spearman = np.zeros((2, 4, len(n_trains), n_simul))
        maxent_spearman = np.zeros((2, 4, len(n_trains), n_simul))

        for i, n_train in enumerate(n_trains):
            for j in range(n_simul):
                
                if verbose:
                    print("New batch", j, i)
                    
                simul = Simulation(n_train=n_train, min_size=1, scenarios=scenarios)
                simul.create_models(verbose=False)

                lr_models, od_models, em_models, maxent_models = simul.get_best_models()
                for k in range(2):
                    for l in range(4):
                        if (k, l) in simul.scenarios:
                            od_model = od_models[k][l]
                            lr_model = lr_models[k][l]
                            em_model = em_models[k][l]
                            maxent_model = maxent_models[k][l]

                            lr_rmse[k, l, i, j] = lr_model.rmse
                            od_rmse[k, l, i, j] = od_model.rmse
                            em_rmse[k, l, i, j] = em_model.rmse
                            maxent_rmse[k, l, i, j] = maxent_model.rmse

                            lr_auc[k, l, i, j] = lr_model.auc
                            od_auc[k, l, i, j] = od_model.auc
                            em_auc[k, l, i, j] = em_model.auc
                            maxent_auc[k, l, i, j] = maxent_model.auc

                            lr_spearman[k, l, i, j] = lr_model.spearman
                            od_spearman[k, l, i, j] = od_model.spearman
                            em_spearman[k, l, i, j] = em_model.spearman
                            maxent_spearman[k, l, i, j] = maxent_model.spearman

        rmse = [lr_rmse, od_rmse, em_rmse, maxent_rmse]
        auc = [lr_auc, od_auc, em_auc, maxent_auc]
        spearman = [lr_spearman, od_spearman, em_spearman, maxent_spearman]

        return rmse, auc, spearman
