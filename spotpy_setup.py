# spotpy_setup.py

import os
import tempfile
import numpy as np
import pandas as pd
import spotpy

# Import your groundwater model function from your script
# Make sure your script file is named 'your_model_script.py'
from GWM_model_run import GWM, get_heads_from_obs_csv

class GWM_Spotpy_Setup:
    """
    A class that connects the Garching FloPy model (GWM) with the SPOTPY framework,
    using the full set of 19 uncertain parameters.
    """
    def __init__(self, obs_path='Output1_Input2/obs_values.csv'):
        # Define the full set of 19 uncertain parameters
        self.params = [
            # Hydraulic Conductivities (m/d)
            spotpy.parameter.Uniform(name='hk1', low=100, high=10000),
            spotpy.parameter.Uniform(name='hk2', low=100, high=10000),
            spotpy.parameter.Uniform(name='hk3', low=100, high=10000),
            spotpy.parameter.Uniform(name='hk4', low=100, high=10000),
            spotpy.parameter.Uniform(name='hk5', low=100, high=10000),
            # Specific Yield (-)
            spotpy.parameter.Uniform(name='sy1', low=0.05, high=0.35),
            spotpy.parameter.Uniform(name='sy2', low=0.05, high=0.35),
            spotpy.parameter.Uniform(name='sy3', low=0.05, high=0.35),
            spotpy.parameter.Uniform(name='sy4', low=0.05, high=0.35),
            spotpy.parameter.Uniform(name='sy5', low=0.05, high=0.35),
            # River Stage and Conductance
            spotpy.parameter.Uniform(name='D_Isar', low=-0.5, high=0.5),
            spotpy.parameter.Uniform(name='Kriv_Isar', low=10, high=1000),
            spotpy.parameter.Uniform(name='Kriv_Muhlbach', low=10, high=1000),
            spotpy.parameter.Uniform(name='Kriv_Giessen', low=10, high=1000),
            spotpy.parameter.Uniform(name='Kriv_Griesbach', low=10, high=1000),
            spotpy.parameter.Uniform(name='Kriv_Schwabinger_Bach', low=10, high=1000),
            spotpy.parameter.Uniform(name='Kriv_Wiesackerbach', low=10, high=1000),
            # Recharge Multipliers (-)
            spotpy.parameter.Uniform(name='D_rch1', low=0, high=3),
            spotpy.parameter.Uniform(name='D_rch2', low=0, high=1)
        ]

        # Load observation data
        self.obs_data = pd.read_csv(obs_path).values

    def parameters(self):
        """Returns the parameter set definition to SPOTPY."""
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        """
        Runs a single simulation of the FloPy model.
        `vector` is a parameter set proposed by the DREAM algorithm.
        """
        temp_dir = tempfile.mkdtemp(prefix="gwm_run_")
        p = {param.name: val for param, val in zip(self.params, vector)}

        try:
            # Run the model with the full parameter set from the vector.
            # No dummy values are needed now.
            model, out_dir = GWM(
                hk1=p['hk1'], hk2=p['hk2'], hk3=p['hk3'], hk4=p['hk4'], hk5=p['hk5'],
                sy1=p['sy1'], sy2=p['sy2'], sy3=p['sy3'], sy4=p['sy4'], sy5=p['sy5'],
                D_Isar=p['D_Isar'],
                Kriv_Isar=p['Kriv_Isar'], Kriv_Muhlbach=p['Kriv_Muhlbach'],
                Kriv_Giessen=p['Kriv_Giessen'], Kriv_Griesbach=p['Kriv_Griesbach'],
                Kriv_Schwabinger_Bach=p['Kriv_Schwabinger_Bach'],
                Kriv_Wiesackerbach=p['Kriv_Wiesackerbach'],
                D_rch1=p['D_rch1'], D_rch2=p['D_rch2'],
                custom_out_dir=temp_dir
            )

            sim_heads = get_heads_from_obs_csv(
                model_ws=out_dir,
                obs_csv_path='Output1_Input2/obs.csv'
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
        
        likelihood = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation, simulation)
        return likelihood