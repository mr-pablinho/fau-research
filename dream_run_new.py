# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

DREAM algorithm settings and executable for new model
"""

# %% Import libraries

import numpy as np
import spotpy
import dream_init_new as di
from dream_modflow_new import spot_setup
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# reinitiate log file
f = open('./logs/log_dream_new.txt', 'w')
f.close()

# %% Run DREAM algorithm

if __name__ == "__main__":

    # datetime object containing current date and time
    time_start = datetime.now()
    time_start_string = time_start.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
    
    print(f"Starting DREAM algorithm at {time_start_string}")
    print(f"Parameters to calibrate: {len(di.param_distros)}")
    print(f"Parameter names: {di.names}")
    
    # set the random state
    random_state = di.my_seed
    np.random.seed(random_state)
       
    # general settings
    parallel = 'seq'   
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
    
    print(f"DREAM settings:")
    print(f"  - Max repetitions: {rep}")
    print(f"  - Chains: {nChains}")
    print(f"  - Convergence limit: {convergence_limit}")
    print(f"  - Runs after convergence: {runs_after_convergence}")
    print(f"  - Random seed: {random_state}")
    
    # initiate DREAM algorithm
    sampler = spotpy.algorithms.dream(spot_setup, 
                                      dbname='dream_GWM_new', 
                                      dbformat='csv',
                                      db_precision=np.float32, 
                                      save_sim=True,
                                      random_state=random_state)
    
    print("Starting DREAM sampling...")
    print(f"Database will be saved as: dream_GWM_new.csv")
    
    try:
        r_hat = sampler.sample(rep, 
                               nChains=nChains, 
                               convergence_limit=convergence_limit,
                               runs_after_convergence=runs_after_convergence,
                               eps=epsilon,
                               acceptance_test_option=ato,
                               nCr=nCr)
        print(f"DREAM sampling completed successfully")
        print(f"Gelman-Rubin diagnostic: {r_hat}")
    except Exception as e:
        print(f"Error during DREAM sampling: {e}")
        print("Continuing with result processing...")
    
    # Check if the CSV file was created
    import os
    csv_filename = 'dream_GWM_new.csv'
    if os.path.exists(csv_filename):
        print(f"Results file '{csv_filename}' created successfully")
        file_size = os.path.getsize(csv_filename)
        print(f"File size: {file_size} bytes")
    else:
        print(f"Warning: Results file '{csv_filename}' not found")
        print("Checking for alternative file formats...")
        for ext in ['.db', '.hdf5', '.pkl']:
            alt_file = f'dream_GWM_new{ext}'
            if os.path.exists(alt_file):
                print(f"Found alternative database file: {alt_file}")
    
    # load the likelihood, and the parameter values of all simulations
    try:
        results = sampler.getdata()   
        print(f"DREAM completed. Total samples: {len(results)}")
    except AttributeError:
        # For newer SPOTPY versions, results might be saved automatically
        print("DREAM completed. Results saved to database file.")
        results = None
    
    # %% Register the end of the algorithm
    
    time_end = datetime.now()
    time_end_string = time_end.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
    
    f = open('./logs/log_dream_new.txt', 'a+')
    
    if results is not None and rep > len(results):
        convergence_eval = 'yes'
    else:
        convergence_eval = 'no'
    
    summary = ''' 
    
->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

The DREAM algorithm finished for the new GWM model.

- Beginning of the simulation:..... %s
- End of the simulation:........... %s 
- Random seed:..................... %d
- Convergence reached:............. %s
- Acceptance test type:............ %d
- Maximum repetitions:............. %d
- Runs after convergence:.......... %d 
- Gelman-Rubin convergence limit:.. %.1f
- Epsilon tolerance threshold:..... %f
- Number of crossover values:...... %d
- Number of chains:................ %d
- Number of parameters:............ %d
- Observation points:.............. %d
- Stress periods:.................. %d

Parameter names:
%s

->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    ''' % (time_start_string, time_end_string, random_state, convergence_eval, 
    ato, rep, runs_after_convergence, convergence_limit, epsilon, nCr, nChains,
    len(di.param_distros), 13, 139, ', '.join(di.names))
    
    f.write(summary)
    f.close()
    
    print(f"Log saved to './logs/log_dream_new.txt'")
    print(f"Results saved to 'dream_GWM_new.csv'")
    print(summary)
