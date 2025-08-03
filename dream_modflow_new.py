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
        
        # Transform log-space parameters back to linear space
        hk1 = 10**x[0]
        hk2 = 10**x[1] 
        hk3 = 10**x[2]
        hk4 = 10**x[3]
        hk5 = 10**x[4]
        
        # Specific yield parameters (already in linear space)
        sy1 = x[5]
        sy2 = x[6]
        sy3 = x[7]
        sy4 = x[8]
        sy5 = x[9]
        
        # Stage adjustment for Isar river
        D_Isar = x[10]
        
        # Transform log-space river conductance parameters back to linear space
        Kriv_Isar = 10**x[11]
        Kriv_Muhlbach = 10**x[12]
        Kriv_Giessen = 10**x[13]
        Kriv_Griesbach = 10**x[14]
        Kriv_Schwabinger_Bach = 10**x[15]
        Kriv_Wiesackerbach = 10**x[16]
        
        # Recharge scaling factors
        D_rch1 = x[17]
        D_rch2 = x[18]
        
        try:
            # Create a unique temporary directory for this simulation
            temp_dir = tempfile.mkdtemp(prefix='dream_sim_')
            
            # Run the GWM model with transformed parameters
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
