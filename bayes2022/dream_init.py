# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Last edition on on Sun Nov 21 19:27:54 2021
@author: Pablo Merchán-Rivera

DREAM initial values 
"""


# %% Import libraries

import spotpy
import numpy as np


# %% Setup random state for the whole process
   
# set the random state
my_seed = 246
np.random.seed(my_seed)

# parameter distributions

names = ['HK_SA', 'HK_SB', 'SY_SA', 'SY_SB', 'RCH', 'CON_CHA1', 'CON_CHA2', 'CON_RIV1', 'CON_RIV2', 'CON_RIV3', 'STA']

param_distros = [spotpy.parameter.Uniform(names[0], low=np.log10(1e-4), high=np.log10(1e-1)),
                 spotpy.parameter.Uniform(names[1], low=np.log10(1e-4), high=np.log10(1e-1)),
                 spotpy.parameter.Uniform(names[2], low=0.10, high=0.40),
                 spotpy.parameter.Uniform(names[3], low=0.10, high=0.40),
                 spotpy.parameter.Uniform(names[4], low=0.00, high=2.00),
                 spotpy.parameter.Uniform(names[5], low=np.log10(1e-5), high=np.log10(9e-1)),
                 spotpy.parameter.Uniform(names[6], low=np.log10(1e-5), high=np.log10(9e-1)),
                 spotpy.parameter.Uniform(names[7], low=np.log10(1e-7), high=np.log10(9e-4)),
                 spotpy.parameter.Uniform(names[8], low=np.log10(1e-7), high=np.log10(9e-4)),
                 spotpy.parameter.Uniform(names[9], low=np.log10(1e-7), high=np.log10(9e-4)),
                 spotpy.parameter.Uniform(names[10], low=-0.145, high=0.145)
                 ]

# number of repetitions and chains
rep = 10000
numSamples = rep
convEvals = 300
numParams = len(param_distros)
nChains = 6

samples = np.zeros((numSamples, numParams))
for i in range(numSamples):
    gen_samples = spotpy.parameter.generate(param_distros)
    for j in range(numParams):
        samples[i,j] = gen_samples[j][0]        


flag = 'dream-r%d-c%d-s%d' % (rep, convEvals, my_seed)

    
