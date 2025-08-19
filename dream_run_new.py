# -*- coding: utf-8 -*-
"""
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

    time_start = datetime.now()
    time_start_string = time_start.strftime("%d/%m/%Y %H:%M:%S")

    print(f"Starting DREAM algorithm at {time_start_string}")
    print(f"Parameters to calibrate: {len(di.param_distros)}")
    print(f"Parameter names: {di.names}")

    np.random.seed(di.my_seed)

    # General settings
    spot_setup = spot_setup(_used_algorithm="dream")

    print(f"DREAM settings:")
    print(f"  - Max repetitions: {di.rep}")
    print(f"  - Chains: {di.nChains}")
    print(f"  - Convergence limit: {di.convergence_limit}")
    print(f"  - Runs after convergence: {di.convEvals}")
    print(f"  - Random seed: {di.my_seed}")

    # Use the timestamp from dream_init_new.py for consistent naming
    timestamp = di.timestamp
    base_filename = f"dream_GWM_{timestamp}"
    csv_filename = f"{base_filename}.csv"
    log_filename = f"./logs/log_dream_{timestamp}.txt"

    # Initiate DREAM algorithm
    sampler = spotpy.algorithms.dream(
        spot_setup,
        dbname=base_filename,
        dbformat="csv",
        db_precision=np.float32,
        save_sim=True,
        random_state=di.my_seed,
    )

    print("Starting DREAM sampling...")
    print(f"Database will be saved as: {csv_filename}")
    print(f"Log will be saved as: {log_filename}")

    try:
        r_hat = sampler.sample(  # Gelman-Rubin diagnostic
            di.rep,
            nChains=di.nChains,
            convergence_limit=di.convergence_limit,
            runs_after_convergence=di.convEvals,
            eps=di.epsilon,
            acceptance_test_option=di.ato,
            nCr=di.nCr,
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

    # Load the likelihood, and the parameter values of all simulations
    try:
        results = sampler.getdata()
        print(f"DREAM completed. Total samples: {len(results)}")
    except AttributeError:
        # For newer SPOTPY versions, results might be saved automatically
        print("DREAM completed. Results saved to database file.")
        results = None

    time_end = datetime.now()
    time_end_string = time_end.strftime("%d/%m/%Y %H:%M:%S")

    f = open(log_filename, "w")
    if results is not None and di.rep > len(results):
        convergence_reached = "yes"
    else:
        convergence_reached = "no"

    summary = f""" 
    
->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

The DREAM algorithm finished for the new GWM model.

- Beginning of the simulation:..... {time_start_string}
- End of the simulation:........... {time_end_string}
- Random seed:..................... {di.my_seed}
- Convergence reached:............. {convergence_reached}
- Acceptance test type:............ {di.ato}
- Maximum repetitions:............. {di.rep}
- Runs after convergence:.......... {di.convEvals}
- Gelman-Rubin convergence limit:.. {di.convergence_limit:.1f}
- Epsilon tolerance threshold:..... {di.epsilon}
- Number of crossover values:...... {di.nCr}
- Number of chains:................ {di.nChains}
- Number of parameters:............ {len(di.param_distros)}
- Results file:.................... {csv_filename}
- Log file:........................ {log_filename}

Parameter names:
{", ".join(di.names)}

->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    """

    f.write(summary)
    f.close()

    print(f"Log saved to '{log_filename}'")
    print(f"Results saved to '{csv_filename}'")
    print(summary)
