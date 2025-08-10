# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

DREAM algorithm settings and executable for new model
"""

import numpy as np
import spotpy
import dream_init_new as di
from dream_modflow_new import spot_setup
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)


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
    parallel = "seq"
    spot_setup = spot_setup(_used_algorithm="dream")

    # select number of maximum repetitions
    rep = di.rep

    # acceptance test option
    ato = 6

    # number of crossover values
    nCr = 4

    # number of chains and convergence limit (Gelman-Rubin).
    # Gelman–Rubin convergence diagnostic provides a numerical convergence summary based on multiple chains
    nChains = di.nChains
    convergence_limit = (
        1.0  # maximum Gelman–Rubin diagnostic across all model parameters
    )

    # define DREAM algorithm parameters (further details in Vrugt, 2016)
    runs_after_convergence = di.convEvals
    epsilon = 1e-5  # tolerance threshold (Turner & Sederberg, 2012)

    print(f"DREAM settings:")
    print(f"  - Max repetitions: {rep}")
    print(f"  - Chains: {nChains}")
    print(f"  - Convergence limit: {convergence_limit}")
    print(f"  - Runs after convergence: {runs_after_convergence}")
    print(f"  - Random seed: {random_state}")

    # Create timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"dream_GWM_{timestamp}"
    csv_filename = f"{base_filename}.csv"
    log_filename = f"./logs/log_dream_{timestamp}.txt"

    # initiate DREAM algorithm
    sampler = spotpy.algorithms.dream(
        spot_setup,
        dbname=base_filename,
        dbformat="csv",
        db_precision=np.float32,
        save_sim=True,
        random_state=random_state,
    )

    print("Starting DREAM sampling...")
    print(f"Database will be saved as: {csv_filename}")
    print(f"Log will be saved as: {log_filename}")

    try:
        r_hat = sampler.sample(  # Gelman-Rubin diagnostic
            rep,
            nChains=nChains,
            convergence_limit=convergence_limit,
            runs_after_convergence=runs_after_convergence,
            eps=epsilon,
            acceptance_test_option=ato,
            nCr=nCr,
        )
        print(f"DREAM sampling completed successfully")
        print(f"Gelman-Rubin diagnostic: {r_hat}")
    except Exception as e:
        print(f"Error during DREAM sampling: {e}")
        print("Continuing with result processing...")

    if os.path.exists(csv_filename):
        print(f"Results file '{csv_filename}' created successfully")
        file_size = os.path.getsize(csv_filename)
        print(f"File size: {file_size} bytes")
    else:
        print(f"Warning: Results file '{csv_filename}' not found")
        print("Checking for alternative file formats...")
        for ext in [".db", ".hdf5", ".pkl"]:
            alt_file = f"{base_filename}{ext}"
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

    # Initialize log file with timestamp
    f = open(log_filename, "w")

    if results is not None and rep > len(results):
        convergence_eval = "yes"
    else:
        convergence_eval = "no"

    summary = f""" 
    
->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

The DREAM algorithm finished for the new GWM model.

- Beginning of the simulation:..... {time_start_string}
- End of the simulation:........... {time_end_string}
- Random seed:..................... {random_state}
- Convergence reached:............. {convergence_eval}
- Acceptance test type:............ {ato}
- Maximum repetitions:............. {rep}
- Runs after convergence:.......... {runs_after_convergence}
- Gelman-Rubin convergence limit:.. {convergence_limit:.1f}
- Epsilon tolerance threshold:..... {epsilon}
- Number of crossover values:...... {nCr}
- Number of chains:................ {nChains}
- Number of parameters:............ {len(di.param_distros)}

Parameter names:
{", ".join(di.names)}

->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    """

    f.write(summary)
    f.close()

    print(f"Log saved to '{log_filename}'")
    print(f"Results saved to '{csv_filename}'")
    print(summary)
