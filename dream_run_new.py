# -*- coding: utf-8 -*-
"""
DREAM algorithm settings and executable for new GWM model
Adapted from bayes2022 DREAM experiment
@author: Pablo Merchán-Rivera (adapted)

DREAM algorithm settings and executable for new GWM groundwater model
"""

import numpy as np
import spotpy
import dream_init_new as di
from dream_modflow_new import spot_setup
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs('./logs', exist_ok=True)

# Reinitiate log file
f = open('./logs/log_dream_gwm.txt', 'w')
f.write("DREAM algorithm log for new GWM model\n")
f.write("=" * 50 + "\n")
f.close()

if __name__ == "__main__":

    # datetime object containing current date and time
    time_start = datetime.now()
    time_start_string = time_start.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
    
    print(f"Starting DREAM algorithm for GWM model at: {time_start_string}")
    
    # set the random state
    random_state = di.my_seed
    np.random.seed(random_state)
       
    # general settings
    parallel = 'seq'   # Use sequential processing for debugging
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
    
    print(f"DREAM parameters:")
    print(f"  - Number of chains: {nChains}")
    print(f"  - Maximum repetitions: {rep}")
    print(f"  - Runs after convergence: {runs_after_convergence}")
    print(f"  - Random seed: {random_state}")
    
    # Create timestamped database name
    timestamp = time_start.strftime('%Y%m%d_%H%M%S')
    dbname = f'dream_GWM_{timestamp}'
    
    # initiate DREAM algorithm
    sampler = spotpy.algorithms.dream(spot_setup, 
                                      dbname=dbname, 
                                      dbformat='csv',
                                      db_precision=np.float32, 
                                      save_sim=True,
                                      random_state=random_state)
    
    print("Starting DREAM sampling...")
    
    # Run DREAM algorithm
    r_hat = sampler.sample(rep, 
                           nChains=nChains, 
                           convergence_limit=convergence_limit,
                           runs_after_convergence=runs_after_convergence,
                           eps=epsilon,
                           acceptance_test_option=ato,
                           nCr=nCr)
    
    # load the likelihood, and the parameter values of all simulations
    try:
        results = sampler.getdata()
    except AttributeError:
        # Alternative way to get results if getdata() fails
        import pandas as pd
        try:
            results = pd.read_csv(f'{dbname}.csv')
            print(f"Results loaded from CSV file: {len(results)} simulations")
        except FileNotFoundError:
            print(f"Warning: Could not load results from sampler or CSV file ({dbname}.csv)")
            results = None   
    
    if results is not None:
        print(f"DREAM algorithm completed. Total simulations: {len(results)}")
    else:
        print("DREAM algorithm completed, but could not retrieve results")
    
    # Register the end of the algorithm
    time_end = datetime.now()
    time_end_string = time_end.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
    duration = time_end - time_start
    
    f = open('./logs/log_dream_gwm.txt', 'a+')
    
    if results is not None and rep > len(results):
        convergence_eval = 'yes'
    else:
        convergence_eval = 'no'
    
    summary = f''' 

->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

The DREAM algorithm for GWM model finished.

- Beginning of the simulation:..... {time_start_string}
- End of the simulation:........... {time_end_string} 
- Duration:........................ {duration}
- Random seed:..................... {random_state}
- Convergence reached:............. {convergence_eval}
- Acceptance test type:............ {ato}
- Maximum repetitions:............. {rep}
- Actual simulations run:.......... {len(results) if results is not None else 'Unknown'}
- Runs after convergence:.......... {runs_after_convergence} 
- Gelman-Rubin convergence limit:.. {convergence_limit}
- Epsilon tolerance threshold:..... {epsilon}
- Number of crossover values:...... {nCr}
- Number of chains:................ {nChains}
- Database name:................... {dbname}
- Final R-hat value:............... {r_hat if r_hat is not None else 'Not converged'}

Parameter names: {di.names}

->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    '''
    
    f.write(summary)
    f.close()
    
    print("Results saved to dream_GWM.csv")
    print("Log saved to ./logs/log_dream_gwm.txt")
    print(f"Final convergence diagnostic (R-hat): {r_hat}")