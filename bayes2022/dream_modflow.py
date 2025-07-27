# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Last edition on on Sun Nov 21 19:27:54 2021
@author: Pablo Merchán-Rivera

DREAM setup for MODFLOW
"""


# %% Import libraries

import f_runModflow_da as rm
import numpy as np
import spotpy
import dream_init as di


# %% Setup model class (parameters, simulations, evaluation and likelihood)

class spot_setup(object):


# %% Define uncertain parameters
    
    def __init__(self, _used_algorithm):
        self._used_algorithm = _used_algorithm
        self.params = di.param_distros


    def parameters(self):
        return spotpy.parameter.generate(self.params)


# %% Run simulations

    def simulation(self, x):
        
        # model discretization
        numSP = 300
        numRows = 260
        numCols = 260
        
        # deterministic parameters
        HK_SC = 10**(-4.5)
        SS_SA, SS_SB, SS_SC = 10**(-4.5), 10**(-4.5), 10**(-4.5)
        SY_SC = 0.275
        
        # stochastic parameters
        HK_SA, HK_SB = 10**x[0], 10**x[1]
        SY_SA, SY_SB = x[2], x[3]
        RCH = x[4]
        CON_CHA1, CON_CHA2 = 10**x[5], 10**x[6]
        CON_RIV1, CON_RIV2, CON_RIV3 = 10**x[7], 10**x[8], 10**x[9]
        STA = x[10]
        
        param_values_model =  [HK_SA, HK_SB, HK_SC,
                               SS_SA, SS_SB, SS_SC,
                               SY_SA, SY_SB, SY_SC,
                               RCH,
                               CON_CHA1, CON_CHA2,
                               CON_RIV1, CON_RIV2, CON_RIV3,
                               STA]
              
        # run model simulations
        sim = rm.runModflow(param_values_model, numSP, numRows, numCols)
        
        # get simulation outcomes
        return sim


# %% Import observed values

    def evaluation(self):
        obs = np.loadtxt('./obs/obs_append_300sp.txt')
        return obs


# %% Define likelihood function

    def objectivefunction(self, simulation, evaluation, params=None):
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation, simulation)
        return like
