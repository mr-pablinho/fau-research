# debug_test.py - Debug version to see what's happening

import numpy as np
import spotpy
from spotpy_setup import GWM_Spotpy_Setup

# Test the setup directly
setup = GWM_Spotpy_Setup()

print("Testing setup initialization...")
print(f"Number of parameters: {len(setup.parameters())}")
print(f"Observation data shape: {setup.obs_data.shape}")

# Test a single simulation
print("\nTesting single simulation...")
test_vector = [5000, 5000, 5000, 5000, 5000, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 500, 500, 500, 500, 500, 500, 1.0, 0.5]  # Middle values
print(f"Test parameter vector: {test_vector[:5]}...")  # Show first 5 values

try:
    simulation_result = setup.simulation(test_vector)
    print(f"Simulation result shape: {simulation_result.shape}")
    print(f"Simulation result sample: {simulation_result[0, :3]}...")  # First few values
    print(f"Simulation has NaN values: {np.isnan(simulation_result).any()}")
    
    evaluation_result = setup.evaluation()
    print(f"Evaluation result shape: {evaluation_result.shape}")
    print(f"Evaluation has NaN values: {np.isnan(evaluation_result).any()}")
    print(f"Evaluation result sample: {evaluation_result[0, :3]}...")  # First few values
    
    objective = setup.objectivefunction(simulation_result, evaluation_result)
    print(f"Objective function result: {objective}")
    
except Exception as e:
    import traceback
    print(f"❌ Error during simulation test:")
    traceback.print_exc()
