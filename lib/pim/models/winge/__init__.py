from datetime import datetime
from loguru import logger
import numpy as np
from pim.models.winge.dynamicnanobrain.BeeSimulator import analyse

from ...setup import Experiment, ExperimentResults
from pim.models.winge.dynamicnanobrain.beesim import beeplotter
from pim.models.winge.dynamicnanobrain.beesim import trialflight as trials


class WingeResults(ExperimentResults):
    def __init__(self, name: str, parameters: dict) -> None:
        super().__init__(name, parameters)

    def report(self):
        pass

    def serialize(self):
        return "winge results"

class WingeExperiment(Experiment):
    def __init__(self, parameters: dict) -> None:
        super().__init__()
        self.parameters = parameters

    def run(self, name: str) -> ExperimentResults:
        logger.info(f"running winge model (experiment {name})")

        #%% Here we specify the arguments of interest and generate the bulk of data
        # needed for the analysis. In addition to saving the minimal distance to the 
        # nest during the inbound flight, we also keep track of the size of the search
        # pattern, perhaps this will prove interesting. 

        N = 5 # number of trials for each parameter to test
        N_dists = 3 # number of logarithmic distance steps
        #distances = np.round(10 ** np.linspace(1, 4, N_dists)).astype('int')
        distances = np.round(10 ** np.linspace(1, 3, N_dists)).astype('int')

        # List the parameter values of interest
        memupdate_vals = [0.001, 0.005, 0.0025, 0.0050, 0.01]
        memupdate_vals = [0.0005,0.001]#, 0.005, 0.0025, 0.0050, 0.01]

        # Specify the dict of parameters
        #param_dicts = [{'n':N_dists, 'T_outbound': distances, 'T_inbound': distances}]
        param_dicts = [{'n':N_dists, 'memupdate': [mem]*N_dists, 'T_outbound': distances, 'T_inbound': distances} for mem in memupdate_vals]
        #param_dicts.append({'n':N_dists, 'T_outbound': distances, 'T_inbound': distances, 'random_homing':[True]*N_dists})

        min_dists_l = []
        min_dist_stds_l = []
        search_dists_l=[]
        search_dist_stds=[]
            
        for param_dict in param_dicts:
            min_dists, min_dist_stds , search_dist, search_dist_std = analyse(N, param_dict)
            min_dists_l.append(min_dists)
            min_dist_stds_l.append(min_dist_stds)
            search_dists_l.append(search_dist)
            search_dist_stds.append(search_dist_std)
            
        #%% Produce a plot of the success of the specific parameters over distance. 
        # A label for the parameter can be sent just after the parameter values
        fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, memupdate_vals, 'Memory update')
        fig.show()

        #%% Produce a plot showing the search pattern width. Here we adjust the 
        # ylabel using an optional variable
        fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, memupdate_vals, 'Memory update', ylabel='Search pattern width')
        fig.show()

        #%% Single flight can be generated like this
        T=100
        OUT, INB = trials.generate_dataset(T,T,1)

        # Output is after statistical analysis (mean and std)
        min_dist, min_dist_std, search_dist, search_dist_std = trials.analyze_inbound(INB,T,T)


        return WingeResults(name, self.parameters)

        
