# run_dream.py

import numpy as np
import spotpy
from spotpy_setup import GWM_Spotpy_Setup # Import the class we just created

# For reproducibility, you can set a random seed
np.random.seed(42)

# 1. Initialize the spotpy setup class for your model
spotpy_setup = GWM_Spotpy_Setup()

# 2. Initialize the DREAM sampler
# dbname: The name of the file to save results to.
# dbformat: 'csv' is easy to read. 'ram' is faster for long runs.
sampler = spotpy.algorithms.dream(
    spotpy_setup,
    dbname='DREAM_GWM_run',
    dbformat='csv'
)

# 3. Start the sampling process
# nchains: Number of parallel chains. Should be > 1. A good rule is to use
#          the number of available CPU cores.
# repetitions: The total number of model runs (iterations). This will be
#              divided among the chains.
# For a real calibration, you would use a much higher number (e.g., 50,000+).
n_chains = 4  # Minimum for testing - use more (8-20) for real calibration
repetitions = 100  # Very small for quick functionality test

sampler.sample(repetitions, nChains=n_chains)

# 4. Analyze the results
results = sampler.getdata()
print("DREAM run finished.")
print("Best parameter set found:")

# Use the spotpy.analyser to get the best parameter set from the results
best_params = spotpy.analyser.get_best_parameterset(results)
print(best_params)

# You can also get the corresponding best objective function value and simulation
best_likelihood = spotpy.analyser.get_maxlikehood(results)
best_simulation = spotpy.analyser.get_best_simulation(results)

print(f"\nBest Likelihood: {best_likelihood}")

# You can also use spotpy's built-in plotting tools
# Plot the posterior distributions of the parameters
spotpy.analyser.plot_parameter_trace(results)
spotpy.analyser.plot_posterior_parameter_histogram(results, bins=30)