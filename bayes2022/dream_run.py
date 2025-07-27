# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Last edition on on Sun Nov 21 19:27:54 2021
@author: Pablo Merchán-Rivera

DREAM algorithm settings and executable
"""


# %% Import libraries

import numpy as np
import spotpy
import dream_init as di
from dream_modflow import spot_setup
from datetime import datetime

# reinitiate log file
f = open('./logs/log_dream.txt', 'r+')
f.truncate(0)
f.close()


# %% Run DREAM algorithm


if __name__ == "__main__":

    # datetime object containing current date and time
    time_start = datetime.now()
    time_start_string = time_start.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
    
    # set the random state
    random_state = di.my_seed
    np.random.seed(random_state)
       
    # general settings
    parallel ='seq'   
    spot_setup = spot_setup(_used_algorithm='dream')
    
    # select number of maximum repetitions
    rep = di.rep
    
    # acceptance test option
    ato = 6
    
    # number of crossover values
    nCr = 4
    
    # number of chains and convergence limit (Gelman-Rubin). 
    # Gelman–Rubin convergence diagnostic provides a numerical convergence summary based on multiple chains
    nChains = di.nChains
    convergence_limit = 1.3  # maximum Gelman–Rubin diagnostic across all model parameters

    
    # define DREAM algorithm parameters (further details in Vrugt, 2016)
    runs_after_convergence = di.convEvals
    epsilon = 0.001  # tolerance threshold (Turner & Sederberg, 2012)
    
    # initiate DREAM algorithm
    sampler = spotpy.algorithms.dream(spot_setup, 
                                      dbname='dream_FLOOD', 
                                      dbformat='csv',
                                      db_precision=np.float32, 
                                      save_sim=True,
                                      random_state=random_state)
    
    r_hat = sampler.sample(rep, 
                           nChains=nChains, 
                           convergence_limit=convergence_limit,
                           runs_after_convergence=runs_after_convergence,
                           eps=epsilon,
                           acceptance_test_option=ato,
                           nCr=nCr)
    
    # load the likelihood, and the parameter values of all simulations
    results = sampler.getdata()   
    
    
    # %% Register the end of the algorithm
    
    time_end = datetime.now()
    time_end_string = time_end.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
    
    f = open('./logs/log_dream.txt', 'a+')
    
    if rep > len(results):
        convergence_eval = 'yes'
    else:
        convergence_eval = 'no'
    
    summary = ''' 
    
->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

The DREAM algorithm finished.

- Beginning of the simulation:..... %s
- End of the simulation:........... %s 
- Random seed:..................... %d
- Convergence reached:............. %s
- Acceptance test type:............ %d
- Maximum repetitions:............. %d
- Runs after convergence:.......... %d 
- Gelman-Rubin convergence limit:.. %d
- Epsilon tolerance threshold:..... %f
- Number of crossover values:...... %d
- Number of chains:................ %d

->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    ''' % (time_start_string, time_end_string, random_state, convergence_eval, 
    ato, rep, runs_after_convergence, convergence_limit, epsilon, nCr, nChains)
    
    f.write((summary))
    f.close()
