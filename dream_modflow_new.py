# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

DREAM setup for new MODFLOW model
"""

# %% Import libraries

import GWM_model_run as gwm
import numpy as np
import pandas as pd
import spotpy
import dream_init_new as di
import os
import tempfile
import shutil

# %% Setup model class (parameters, simulations, evaluation and likelihood)

class spot_setup(object):

    # %% Define uncertain parameters
    
    def __init__(self, _used_algorithm):
        self._used_algorithm = _used_algorithm
        self.params = di.param_distros

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    # %% Run simulations

    def simulation(self, x):
        
        # Get parameter definitions and calibration settings from dream_init_new
        param_names = di.CALIBRATE_PARAMS
        deterministic_vals = di.deterministic_values
        log_transform_params = di.LOG_TRANSFORM_PARAMS
        
        # Start with deterministic values for all parameters
        param_values = deterministic_vals.copy()
        
        # Update with calibrated parameter values
        for i, param_name in enumerate(param_names):
            if param_name in log_transform_params:
                # Transform log-space parameters back to linear space
                param_values[param_name] = 10**x[i]
            else:
                # Use linear space parameters directly
                param_values[param_name] = x[i]
        
        # Extract all parameter values in the order expected by GWM function
        hk1 = param_values['hk1']
        hk2 = param_values['hk2'] 
        hk3 = param_values['hk3']
        hk4 = param_values['hk4']
        hk5 = param_values['hk5']
        
        sy1 = param_values['sy1']
        sy2 = param_values['sy2']
        sy3 = param_values['sy3']
        sy4 = param_values['sy4']
        sy5 = param_values['sy5']
        
        D_Isar = param_values['D_Isar']
        
        Kriv_Isar = param_values['Kriv_Isar']
        Kriv_Muhlbach = param_values['Kriv_Muhlbach']
        Kriv_Giessen = param_values['Kriv_Giessen']
        Kriv_Griesbach = param_values['Kriv_Griesbach']
        Kriv_Schwabinger_Bach = param_values['Kriv_Schwabinger_Bach']
        Kriv_Wiesackerbach = param_values['Kriv_Wiesackerbach']
        
        D_rch1 = param_values['D_rch1']
        D_rch2 = param_values['D_rch2']
        
        # Print calibrated parameter values for debugging
        print(f"Calibrating: {', '.join([f'{p}={param_values[p]:.2e}' if p in log_transform_params else f'{p}={param_values[p]:.3f}' for p in param_names])}")
        
        try:
            # Create a unique temporary directory for this simulation
            temp_dir = tempfile.mkdtemp(prefix='dream_sim_')
            
            # Run the GWM model with parameter values
            model, out_dir = gwm.GWM(
                hk1=hk1, hk2=hk2, hk3=hk3, hk4=hk4, hk5=hk5,
                sy1=sy1, sy2=sy2, sy3=sy3, sy4=sy4, sy5=sy5,
                D_Isar=D_Isar, 
                Kriv_Isar=Kriv_Isar, 
                Kriv_Muhlbach=Kriv_Muhlbach,
                Kriv_Giessen=Kriv_Giessen,
                Kriv_Griesbach=Kriv_Griesbach,
                Kriv_Schwabinger_Bach=Kriv_Schwabinger_Bach,
                Kriv_Wiesackerbach=Kriv_Wiesackerbach,
                D_rch1=D_rch1,
                D_rch2=D_rch2,
                custom_out_dir=temp_dir
            )
            
            # Extract simulated head values for all observation points
            sim_heads = gwm.get_heads_from_obs_csv(
                model_ws=temp_dir, 
                obs_csv_path='Output1_Input2/obs.csv'
            )
            
            # Flatten the array: convert from [n_stress_periods x n_obs_points] to 1D array
            # This matches the format expected by SPOTPY
            sim_heads_flat = sim_heads.flatten()
            
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Continue even if cleanup fails
            
            return sim_heads_flat
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            # Return array of NaNs if simulation fails
            # Size should match expected output: 139 stress periods × 13 observation points
            return np.full(139 * 13, np.nan)

    # %% Import observed values

    def evaluation(self):
        """
        Load observed head values from obs_values.csv
        
        Returns flattened array matching the simulation output format:
        [stress_period_0_obs_0, stress_period_0_obs_1, ..., stress_period_0_obs_12,
         stress_period_1_obs_0, stress_period_1_obs_1, ..., stress_period_1_obs_12,
         ...]
        """
        try:
            # Load observation values, skip header row
            obs_df = pd.read_csv('Output1_Input2/obs_values.csv', header=0)
            
            # Convert to numpy array and ensure numeric data only
            obs_array = obs_df.iloc[:, :13].values.astype(np.float64)  # Take only first 13 columns
            
            # Flatten the array to match simulation output format
            obs_flat = obs_array.flatten()
            
            print(f"Debug: Loaded observations with shape {obs_array.shape}, flattened to {obs_flat.shape}")
            
            return obs_flat
            
        except Exception as e:
            print(f"Error loading observations: {e}")
            # Return array of zeros if loading fails
            return np.zeros(139 * 13)

    # %% Define likelihood function

    def objectivefunction(self, simulation, evaluation, params=None):
        """
        Calculate likelihood using Gaussian likelihood with measurement error
        """
        # Ensure both arrays are numpy arrays with float64 dtype
        simulation = np.array(simulation, dtype=np.float64)
        evaluation = np.array(evaluation, dtype=np.float64)
        
        # Remove NaN values from both simulation and evaluation
        valid_mask = ~(np.isnan(simulation) | np.isnan(evaluation))
        
        if np.sum(valid_mask) == 0:
            # If no valid data points, return very low likelihood
            return -np.inf
        
        sim_valid = simulation[valid_mask]
        eval_valid = evaluation[valid_mask]
        
        # Calculate Gaussian likelihood
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(eval_valid, sim_valid)
        
        return like
