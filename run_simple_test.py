# run_simple_test.py - For quick functionality testing

import numpy as np
import spotpy
from spotpy_setup import GWM_Spotpy_Setup

# For reproducibility, you can set a random seed
np.random.seed(42)

# 1. Initialize the spotpy setup class for your model
spotpy_setup = GWM_Spotpy_Setup()

# 2. Use Monte Carlo sampling for quick testing (no chain requirements)
sampler = spotpy.algorithms.mc(
    spotpy_setup,
    dbname='MC_GWM_test',
    dbformat='csv'
)

# 3. Start with very few runs just to test functionality
repetitions = 10  # Just 10 runs to test if everything works

print(f"Starting Monte Carlo test with {repetitions} repetitions")
sampler.sample(repetitions)

# 4. Analyze the results
results = sampler.getdata()
print("Test run finished.")
print(f"Number of model runs completed: {len(results)}")

if len(results) > 0:
    print("✓ Model is running successfully!")
    print("Best parameter set found:")
    best_params = spotpy.analyser.get_best_parameterset(results)
    print(best_params)
    print(f"Best objective value: {results['like1'].max()}")
else:
    print("✗ No results generated - check for errors")
