# -*- coding: utf-8 -*-
"""
DREAM setup for new GWM MODFLOW model
Adapted from bayes2022 DREAM experiment
@author: Pablo Merchán-Rivera (adapted)

DREAM setup for new GWM MODFLOW model
"""

import numpy as np
import spotpy
import dream_init_new as di
from GWM_model_run import GWM, get_heads_from_obs_csv
import os
import tempfile

class spot_setup(object):
    """Setup model class for DREAM algorithm with new GWM model"""

    def __init__(self, _used_algorithm):
        self._used_algorithm = _used_algorithm
        self.params = di.param_distros

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, x):
        """
        Run GWM model simulation with parameter vector x
        """
        try:
            # Create parameter dictionary starting with default values for fixed parameters
            param_values = di.default_values.copy()
            
            # Update with uncertain parameter values from DREAM
            for i, param_name in enumerate(di.names):
                param_info = di.ALL_PARAMETERS[param_name]
                if param_info['transform'] == 'log':
                    param_values[param_name] = 10**x[i]  # Transform from log scale
                else:
                    param_values[param_name] = x[i]  # Linear scale
            
            # Extract all parameters for GWM function
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
            
            # Create unique output directory for this simulation
            temp_dir = tempfile.mkdtemp(prefix='gwm_dream_')
            
            # Run GWM model
            model, out_dir = GWM(
                hk1=hk1, hk2=hk2, hk3=hk3, hk4=hk4, hk5=hk5,
                sy1=sy1, sy2=sy2, sy3=sy3, sy4=sy4, sy5=sy5,
                D_Isar=D_Isar,
                Kriv_Isar=Kriv_Isar, Kriv_Muhlbach=Kriv_Muhlbach, 
                Kriv_Giessen=Kriv_Giessen, Kriv_Griesbach=Kriv_Griesbach,
                Kriv_Schwabinger_Bach=Kriv_Schwabinger_Bach, 
                Kriv_Wiesackerbach=Kriv_Wiesackerbach,
                D_rch1=D_rch1, D_rch2=D_rch2,
                custom_out_dir=temp_dir
            )
            
            # Extract simulated heads at observation points
            sim_heads = get_heads_from_obs_csv(out_dir, 'Output1_Input2/obs.csv')
            
            # Flatten the simulation results (all time steps and observation points)
            sim_heads_flat = sim_heads.flatten()
            
            # Clean up temporary directory (optional - comment out for debugging)
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors
                
            return sim_heads_flat
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            # Return a large penalty value in case of model failure
            # This should match the size of flattened observations
            return np.full(1794, 999.0)  # 139 stress periods * 13 obs points - adjust if needed

    def evaluation(self):
        """
        Import observed values from new model format
        """
        try:
            obs_df = np.loadtxt('Output1_Input2/obs_values.csv', delimiter=',', skiprows=1)
            # Flatten the observed values (all time steps and observation points)
            return obs_df.flatten()
        except Exception as e:
            print(f"Failed to load observations: {e}")
            # Return dummy data if file not found
            return np.full(1794, 0.0)  # Adjust size as needed

    def objectivefunction(self, simulation, evaluation, params=None):
        """
        Define likelihood function for DREAM algorithm
        """
        try:
            # Remove any NaN or invalid values
            valid_mask = ~(np.isnan(simulation) | np.isnan(evaluation) | 
                          np.isinf(simulation) | np.isinf(evaluation))
            
            if np.sum(valid_mask) == 0:
                return -1e10  # Very low likelihood for invalid simulations
            
            sim_valid = simulation[valid_mask]
            obs_valid = evaluation[valid_mask]
            
            # Use Gaussian likelihood with measurement error
            like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(obs_valid, sim_valid)
            return like
            
        except Exception as e:
            print(f"Likelihood calculation failed: {e}")
            return -1e10  # Very low likelihood for failed calculations