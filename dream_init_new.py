# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

DREAM initial values for new model with updated parameters
"""

# %% Import libraries

import spotpy
import numpy as np

# %% Setup random state for the whole process
   
# set the random state
my_seed = 246
np.random.seed(my_seed)

# parameter distributions for the new model
# Based on GWM function signature: hk1, hk2, hk3, hk4, hk5, sy1, sy2, sy3, sy4, sy5, 
#                                  D_Isar, Kriv_Isar, Kriv_Muhlbach, Kriv_Giessen, 
#                                  Kriv_Griesbach, Kriv_Schwabinger_Bach, Kriv_Wiesackerbach, 
#                                  D_rch1, D_rch2

names = ['hk1', 'hk2', 'hk3', 'hk4', 'hk5', 
         'sy1', 'sy2', 'sy3', 'sy4', 'sy5',
         'D_Isar', 'Kriv_Isar', 'Kriv_Muhlbach', 'Kriv_Giessen', 
         'Kriv_Griesbach', 'Kriv_Schwabinger_Bach', 'Kriv_Wiesackerbach',
         'D_rch1', 'D_rch2']

# Define parameter distributions based on physical constraints and typical groundwater modeling ranges
param_distros = [
    # Hydraulic conductivity parameters (m/d) - log-uniform distribution
    spotpy.parameter.Uniform(names[0], low=np.log10(1e-3), high=np.log10(1e2)),  # hk1
    spotpy.parameter.Uniform(names[1], low=np.log10(1e-3), high=np.log10(1e2)),  # hk2
    spotpy.parameter.Uniform(names[2], low=np.log10(1e-3), high=np.log10(1e2)),  # hk3
    spotpy.parameter.Uniform(names[3], low=np.log10(1e-3), high=np.log10(1e2)),  # hk4
    spotpy.parameter.Uniform(names[4], low=np.log10(1e-3), high=np.log10(1e2)),  # hk5
    
    # Specific yield parameters (-) - linear distribution
    spotpy.parameter.Uniform(names[5], low=0.05, high=0.40),   # sy1
    spotpy.parameter.Uniform(names[6], low=0.05, high=0.40),   # sy2
    spotpy.parameter.Uniform(names[7], low=0.05, high=0.40),   # sy3
    spotpy.parameter.Uniform(names[8], low=0.05, high=0.40),   # sy4
    spotpy.parameter.Uniform(names[9], low=0.05, high=0.40),   # sy5
    
    # Stage adjustment for Isar river (m)
    spotpy.parameter.Uniform(names[10], low=-2.0, high=2.0),   # D_Isar
    
    # River bed conductance parameters (m2/d) - log-uniform distribution
    spotpy.parameter.Uniform(names[11], low=np.log10(1e-6), high=np.log10(1e-1)),  # Kriv_Isar
    spotpy.parameter.Uniform(names[12], low=np.log10(1e-6), high=np.log10(1e-1)),  # Kriv_Muhlbach
    spotpy.parameter.Uniform(names[13], low=np.log10(1e-6), high=np.log10(1e-1)),  # Kriv_Giessen
    spotpy.parameter.Uniform(names[14], low=np.log10(1e-6), high=np.log10(1e-1)),  # Kriv_Griesbach
    spotpy.parameter.Uniform(names[15], low=np.log10(1e-6), high=np.log10(1e-1)),  # Kriv_Schwabinger_Bach
    spotpy.parameter.Uniform(names[16], low=np.log10(1e-6), high=np.log10(1e-1)),  # Kriv_Wiesackerbach
    
    # Recharge scaling factors (-)
    spotpy.parameter.Uniform(names[17], low=0.1, high=3.0),    # D_rch1 (background recharge)
    spotpy.parameter.Uniform(names[18], low=0.1, high=3.0),    # D_rch2 (urban recharge)
]

# number of repetitions and chains
rep = 1000  # Reduced from 10000 for initial testing
numSamples = rep
convEvals = 100  # Reduced from 300
numParams = len(param_distros)
nChains = 3  # Reduced from 6 for faster testing

# Generate initial samples for parameter space exploration
samples = np.zeros((numSamples, numParams))
for i in range(numSamples):
    gen_samples = spotpy.parameter.generate(param_distros)
    for j in range(numParams):
        samples[i,j] = gen_samples[j][0]        

# Create unique identifier for this run
flag = 'dream-r%d-c%d-s%d' % (rep, convEvals, my_seed)
