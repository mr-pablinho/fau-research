# mock_spotpy_setup.py
"""
Mock version of the SPOTPY setup using the mock groundwater model.
This allows rapid testing of the DREAM optimization setup.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import spotpy
from config import OptimizationConfig

# Import the mock model functions
from mock_model import mock_GWM, mock_get_heads_with_params

class Mock_GWM_Spotpy_Setup:
    """
    Mock version of GWM_Spotpy_Setup that uses a fast mathematical model
    instead of the expensive FloPy groundwater model.
    """
    # Class variable for max_runs only
    max_runs = None
    # File to store repetition count
    counter_file = os.path.join(tempfile.gettempdir(), 'mock_gwm_spotpy_counter.txt')

    def __init__(self, obs_path=None, max_runs=None):
        # Use configuration for observation path
        if obs_path is None:
            obs_path = OptimizationConfig.OBS_PATH
            
        # Get parameter set from configuration
        self.params = OptimizationConfig.get_parameter_set()
        self.default_params = OptimizationConfig.get_default_parameters()
        
        # Load observation data (or create synthetic observations)
        try:
            self.obs_data = pd.read_csv(obs_path).values
            print(f"Loaded {len(self.obs_data)} observation time steps")
        except Exception as e:
            print(f"Warning: Could not load observation data from {obs_path}: {e}")
            print("Creating synthetic observation data for testing...")
            self.obs_data = self._create_synthetic_observations()
        
        # Print configuration info
        OptimizationConfig.print_config()
        print("🚀 Using MOCK MODEL for fast testing!")
        
        # Set max_runs only if provided
        if max_runs is not None:
            Mock_GWM_Spotpy_Setup.max_runs = max_runs

        # Reset the counter file to 0 at the start of a new run
        try:
            with open(Mock_GWM_Spotpy_Setup.counter_file, 'w') as f:
                f.write('0')
        except Exception as e:
            print(f"Warning: Could not reset repetition counter file: {e}")

    def _create_synthetic_observations(self):
        """Create synthetic observation data with known optimal response."""
        try:
            obs_data = pd.read_csv(OptimizationConfig.OBS_PATH)
            n_time_steps, n_obs_points = obs_data.shape
            print(f"Creating synthetic observations with real data dimensions: {n_time_steps}×{n_obs_points}")
        except:
            n_time_steps = 139  # From actual obs file
            n_obs_points = 13   # From actual obs file
            print(f"Creating synthetic observations with fallback dimensions: {n_time_steps}×{n_obs_points}")
        
        # Generate "true" observations using optimal parameters
        optimal_params = {
            'hk1': 5000.0,
            'hk2': 4000.0,
            'hk3': 3000.0,
            'hk4': 2500.0,
            'hk5': 2000.0,
        }
        
        # Create synthetic heads using the mock model with optimal parameters
        synthetic_heads = mock_get_heads_with_params(
            model_ws=None, 
            obs_csv_path=OptimizationConfig.OBS_PATH,  # Use OBS_PATH instead of OBS_CSV_PATH
            **optimal_params
        )
        
        # Add a bit of measurement noise to make it realistic
        np.random.seed(123)  # Different seed from the model
        noise = np.random.normal(0, 0.2, synthetic_heads.shape)
        
        return synthetic_heads + noise

    def parameters(self):
        """Returns the parameter set definition to SPOTPY."""
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        """
        Runs a single simulation of the mock model.
        `vector` is a parameter set proposed by the DREAM algorithm.
        """
        # File-based repetition counter (robust to multiprocessing)
        import threading
        lock = threading.Lock()
        with lock:
            try:
                with open(Mock_GWM_Spotpy_Setup.counter_file, 'r+') as f:
                    count = int(f.read().strip()) + 1
                    f.seek(0)
                    f.write(str(count))
                    f.truncate()
            except FileNotFoundError:
                count = 1
                with open(Mock_GWM_Spotpy_Setup.counter_file, 'w') as f:
                    f.write(str(count))
            except Exception:
                count = -1  # fallback if error
        
        # Print repetition info
        if Mock_GWM_Spotpy_Setup.max_runs is not None:
            print(f"🔄 Running MOCK simulation {count} out of {Mock_GWM_Spotpy_Setup.max_runs}")
        else:
            print(f"🔄 Running MOCK simulation {count}")
        
        temp_dir = tempfile.mkdtemp(prefix="mock_gwm_run_")
        
        # Create parameter dictionary from optimization vector
        p = {param.name: val for param, val in zip(self.params, vector)}
        
        # Merge with default parameters (for parameters not being optimized)
        full_params = self.default_params.copy()
        full_params.update(p)

        try:
            # Run the MOCK model with the complete parameter set
            model, out_dir = mock_GWM(
                hk1=full_params['hk1'], hk2=full_params['hk2'], 
                hk3=full_params['hk3'], hk4=full_params['hk4'], hk5=full_params['hk5'],
                sy1=full_params['sy1'], sy2=full_params['sy2'], 
                sy3=full_params['sy3'], sy4=full_params['sy4'], sy5=full_params['sy5'],
                D_Isar=full_params['D_Isar'],
                Kriv_Isar=full_params['Kriv_Isar'], Kriv_Muhlbach=full_params['Kriv_Muhlbach'],
                Kriv_Giessen=full_params['Kriv_Giessen'], Kriv_Griesbach=full_params['Kriv_Griesbach'],
                Kriv_Schwabinger_Bach=full_params['Kriv_Schwabinger_Bach'],
                Kriv_Wiesackerbach=full_params['Kriv_Wiesackerbach'],
                D_rch1=full_params['D_rch1'], D_rch2=full_params['D_rch2'],
                custom_out_dir=temp_dir
            )

            # Get synthetic heads from the mock model
            sim_heads = mock_get_heads_with_params(
                model_ws=out_dir,
                obs_csv_path=OptimizationConfig.OBS_PATH,  # Use the obs_values.csv file
                **p  # Pass the optimized parameters to the mock response function
            )
            
            # Match simulation output shape to evaluation data shape
            # Use the last 100 time steps to match what the real model does
            if sim_heads.shape[0] > 100:
                sim_output = sim_heads[-100:, :]
            else:
                sim_output = sim_heads
                
            # CRITICAL: SPOTPY expects flattened 1D arrays, not 2D arrays
            # Flatten the simulation output to match SPOTPY's expectations
            sim_flattened = sim_output.flatten()
            print(f"    🔧 Simulation output shape: {sim_output.shape} -> flattened to {sim_flattened.shape}")
            
            return sim_flattened

        except Exception as e:
            import traceback
            print(f"⚠️ Mock model run failed with parameters {vector}:")
            traceback.print_exc() 
            # Return flattened NaN array with correct shape
            expected_shape = self.obs_data[-100:, :].shape
            nan_array = np.full(expected_shape, np.nan)
            return nan_array.flatten()
        
    def evaluation(self):
        """Returns the observation data to SPOTPY."""
        # CRITICAL: Return flattened observation data to match simulation output
        obs_2d = self.obs_data[-100:, :]
        obs_flattened = obs_2d.flatten()
        print(f"    🔧 Evaluation output shape: {obs_2d.shape} -> flattened to {obs_flattened.shape}")
        return obs_flattened

    def objectivefunction(self, simulation, evaluation, params=None):
        """Calculates the log-likelihood for DREAM."""
        if np.isnan(simulation).any():
            return -np.inf
        
        # Create mask for valid (non-NaN) observation data
        valid_mask = ~np.isnan(evaluation)
        
        # Only compare where we have valid observations
        if not valid_mask.any():
            return -np.inf  # No valid observations
        
        sim_valid = simulation[valid_mask]
        obs_valid = evaluation[valid_mask]
        
        # Simple RMSE-based objective function
        rmse = np.sqrt(np.mean((sim_valid - obs_valid) ** 2))
        
        # Convert RMSE to a likelihood (lower RMSE = higher likelihood)
        likelihood = -rmse
        
        # Print objective function value for monitoring
        print(f"    📊 RMSE: {rmse:.4f}, Log-likelihood: {likelihood:.4f}")
        
        return likelihood
