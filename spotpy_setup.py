# spotpy_setup.py

import os
import tempfile
import numpy as np
import pandas as pd
import spotpy
from config import OptimizationConfig

# Import your groundwater model function from your script
# Make sure your script file is named 'your_model_script.py'
from GWM_model_run import GWM, get_heads_from_obs_csv

class GWM_Spotpy_Setup:
    """
    A class that connects the Garching FloPy model (GWM) with the SPOTPY framework,
    using configurable parameter sets for testing and full optimization.
    """
    # Class variable for max_runs only
    max_runs = None
    # File to store repetition count
    counter_file = os.path.join(tempfile.gettempdir(), 'gwm_spotpy_counter.txt')

    def __init__(self, obs_path=None, max_runs=None):
        # Use configuration for observation path
        if obs_path is None:
            obs_path = OptimizationConfig.OBS_PATH
            
        # Get parameter set from configuration
        self.params = OptimizationConfig.get_parameter_set()
        self.default_params = OptimizationConfig.get_default_parameters()
        
        # Load observation data
        self.obs_data = pd.read_csv(obs_path).values
        
        # Print configuration info
        OptimizationConfig.print_config()
        # Set max_runs only if provided
        if max_runs is not None:
            GWM_Spotpy_Setup.max_runs = max_runs

        # Reset the counter file to 0 at the start of a new run
        try:
            with open(GWM_Spotpy_Setup.counter_file, 'w') as f:
                f.write('0')
        except Exception as e:
            print(f"Warning: Could not reset repetition counter file: {e}")

    def parameters(self):
        """Returns the parameter set definition to SPOTPY."""
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        """
        Runs a single simulation of the FloPy model.
        `vector` is a parameter set proposed by the DREAM algorithm.
        """
        # File-based repetition counter (robust to multiprocessing)
        import threading
        lock = threading.Lock()
        with lock:
            try:
                with open(GWM_Spotpy_Setup.counter_file, 'r+') as f:
                    count = int(f.read().strip()) + 1
                    f.seek(0)
                    f.write(str(count))
                    f.truncate()
            except FileNotFoundError:
                count = 1
                with open(GWM_Spotpy_Setup.counter_file, 'w') as f:
                    f.write(str(count))
            except Exception:
                count = -1  # fallback if error
        # Print repetition info
        if GWM_Spotpy_Setup.max_runs is not None:
            print(f"Running simulation {count} out of {GWM_Spotpy_Setup.max_runs}")
        else:
            print(f"Running simulation {count}")
        temp_dir = tempfile.mkdtemp(prefix="gwm_run_")
        
        # Create parameter dictionary from optimization vector
        p = {param.name: val for param, val in zip(self.params, vector)}
        
        # Merge with default parameters (for parameters not being optimized)
        full_params = self.default_params.copy()
        full_params.update(p)

        try:
            # Run the model with the complete parameter set
            model, out_dir = GWM(
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

            sim_heads = get_heads_from_obs_csv(
                model_ws=out_dir,
                obs_csv_path=OptimizationConfig.OBS_CSV_PATH
            )
            
            # Match simulation output shape to evaluation data shape
            if sim_heads.shape[0] > 100:
                 return sim_heads[-100:, :]
            else:
                 return sim_heads

        except Exception as e:
            import traceback
            print(f"⚠️ Model run failed with parameters {vector}:")
            # This next line is the crucial debugging step
            traceback.print_exc() 
            return np.full(self.obs_data[-100:, :].shape, np.nan)
        
    def evaluation(self):
        """Returns the observation data to SPOTPY."""
        # Return the same slice of observations as in the simulation method.
        return self.obs_data[-100:, :]

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
        
        # Simple RMSE-based objective function that should work reliably
        rmse = np.sqrt(np.mean((sim_valid - obs_valid) ** 2))
        
        # Convert RMSE to a likelihood (lower RMSE = higher likelihood)
        # Use negative RMSE as log-likelihood (higher values are better for SPOTPY)
        likelihood = -rmse
        
        return likelihood