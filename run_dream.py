# run_dream.py

import os
import warnings
import numpy as np
import spotpy
from spotpy_setup import GWM_Spotpy_Setup # Import the class we just created
from config import OptimizationConfig
import multiprocessing as mp

# Required for Windows multiprocessing - MUST be at the very top
if __name__ == '__main__':
    mp.freeze_support()
    
    # For reproducibility, you can set a random seed
    np.random.seed(42)

    # 1. Initialize the spotpy setup class for your model
    spotpy_setup = GWM_Spotpy_Setup()

    # 2. Initialize the DREAM sampler 
    # For initial testing, start with sequential processing to avoid multiprocessing issues
    # Change parallel to 'mpc' once everything works
    processing = 'mpc'  # Use 'mpc' for multiprocessing later
    sampler = spotpy.algorithms.dream(
        spotpy_setup,
        dbname=OptimizationConfig.DB_NAME,
        dbformat=OptimizationConfig.DB_FORMAT,
        parallel=processing  # Use 'seq' for initial testing, 'mpc' for multiprocessing
    )

    # 3. Start the sampling process with parallel execution
    # Get settings from configuration
    n_cores = mp.cpu_count()
    n_chains = OptimizationConfig.DREAM_CHAINS
    repetitions = OptimizationConfig.REPETITIONS

    print(f"Available CPU cores: {n_cores}")
    print(f"Starting DREAM with {n_chains} chains and {repetitions} repetitions")
    print(f"Running in {processing} mode for initial testing")
    print(f"Total model runs: {n_chains * repetitions}")

    # For DREAM algorithm requirements:
    # - Need at least 2*n_parameters + 1 chains for proper sampling
    n_params = len(spotpy_setup.params)
    min_chains = 2 * n_params + 1
    if n_chains < min_chains:
        print(f"❌ ERROR: DREAM needs at least {min_chains} chains for {n_params} parameters")
        print(f"Current setting: {n_chains} chains")
        print("Updating chains to minimum requirement...")
        n_chains = min_chains
        print(f"Using {n_chains} chains instead")

    print(f"Final configuration: {n_chains} chains × {repetitions} repetitions = {n_chains * repetitions} total runs")

    try:
        sampler.sample(repetitions, nChains=n_chains)
        
        # 4. Analyze the results
        results = sampler.getdata()
        print("DREAM run finished.")
        print("Best parameter set found:")

        # Use the spotpy.analyser to get the best parameter set from the results
        best_params = spotpy.analyser.get_best_parameterset(results)
        print(best_params)

        # You can also get the corresponding best objective function value
        print(f"\nBest objective value: {results['like1'].max()}")

        print(f"Number of model runs completed: {len(results)}")

        # You can also use spotpy's built-in plotting tools (commented out for now)
        # spotpy.analyser.plot_parameter_trace(results)
        # spotpy.analyser.plot_posterior_parameter_histogram(results)
        
    except Exception as e:
        print(f"❌ Error during DREAM execution: {e}")
        print("This might be due to:")
        print("1. Insufficient number of chains for DREAM algorithm")
        print("2. Model execution failures")
        print("3. Multiprocessing issues")
        
        # Try to get any available results
        try:
            if hasattr(sampler, 'datawriter') and sampler.datawriter is not None:
                results = sampler.getdata()
                print(f"Partial results available: {len(results)} runs completed")
            else:
                print("No results available - sampling may not have started properly")
        except:
            print("Could not retrieve any results")