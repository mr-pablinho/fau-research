# -*- coding: utf-8 -*-
"""
DREAM initial values for new GWM groundwater model
Adapted from bayes2022 DREAM experiment
@author: Pablo Merchán-Rivera (adapted)

DREAM initial values for new GWM model
"""

import spotpy
import numpy as np

# Setup random state for the whole process
my_seed = 246
np.random.seed(my_seed)

# =============================================================================
# PARAMETER SELECTION - MODIFY THIS SECTION TO SELECT UNCERTAIN PARAMETERS
# =============================================================================

# Define all possible parameters with their default values and distributions
ALL_PARAMETERS = {
    # Hydraulic conductivity (m/d) - log-uniform distribution
    'hk1': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('hk1', low=np.log10(1e-2), high=np.log10(1e2)), 'transform': 'log'},
    'hk2': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('hk2', low=np.log10(1e-2), high=np.log10(1e2)), 'transform': 'log'},
    'hk3': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('hk3', low=np.log10(1e-2), high=np.log10(1e2)), 'transform': 'log'},
    'hk4': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('hk4', low=np.log10(1e-2), high=np.log10(1e2)), 'transform': 'log'},
    'hk5': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('hk5', low=np.log10(1e-2), high=np.log10(1e2)), 'transform': 'log'},
    
    # Specific yield (dimensionless)
    'sy1': {'default': 0.2, 'distribution': spotpy.parameter.Uniform('sy1', low=0.05, high=0.35), 'transform': 'linear'},
    'sy2': {'default': 0.2, 'distribution': spotpy.parameter.Uniform('sy2', low=0.05, high=0.35), 'transform': 'linear'},
    'sy3': {'default': 0.2, 'distribution': spotpy.parameter.Uniform('sy3', low=0.05, high=0.35), 'transform': 'linear'},
    'sy4': {'default': 0.2, 'distribution': spotpy.parameter.Uniform('sy4', low=0.05, high=0.35), 'transform': 'linear'},
    'sy5': {'default': 0.2, 'distribution': spotpy.parameter.Uniform('sy5', low=0.05, high=0.35), 'transform': 'linear'},
    
    # River stage adjustment (m)
    'D_Isar': {'default': 0.0, 'distribution': spotpy.parameter.Uniform('D_Isar', low=-2.0, high=2.0), 'transform': 'linear'},
    
    # River bed conductance (m2/d) - log-uniform distribution
    'Kriv_Isar': {'default': 10.0, 'distribution': spotpy.parameter.Uniform('Kriv_Isar', low=np.log10(1e-5), high=np.log10(1e3)), 'transform': 'log'},
    'Kriv_Muhlbach': {'default': 10.0, 'distribution': spotpy.parameter.Uniform('Kriv_Muhlbach', low=np.log10(1e-5), high=np.log10(1e3)), 'transform': 'log'},
    'Kriv_Giessen': {'default': 10.0, 'distribution': spotpy.parameter.Uniform('Kriv_Giessen', low=np.log10(1e-5), high=np.log10(1e3)), 'transform': 'log'},
    'Kriv_Griesbach': {'default': 10.0, 'distribution': spotpy.parameter.Uniform('Kriv_Griesbach', low=np.log10(1e-5), high=np.log10(1e3)), 'transform': 'log'},
    'Kriv_Schwabinger_Bach': {'default': 10.0, 'distribution': spotpy.parameter.Uniform('Kriv_Schwabinger_Bach', low=np.log10(1e-5), high=np.log10(1e3)), 'transform': 'log'},
    'Kriv_Wiesackerbach': {'default': 10.0, 'distribution': spotpy.parameter.Uniform('Kriv_Wiesackerbach', low=np.log10(1e-5), high=np.log10(1e3)), 'transform': 'log'},
    
    # Recharge multipliers (dimensionless)
    'D_rch1': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('D_rch1', low=0.1, high=3.0), 'transform': 'linear'},
    'D_rch2': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('D_rch2', low=0.1, high=3.0), 'transform': 'linear'},
}

# =============================================================================
# SELECT UNCERTAIN PARAMETERS - MODIFY THIS LIST FOR DIFFERENT EXPERIMENTS
# =============================================================================

# Example configurations:
# For quick tests (3-5 parameters):
UNCERTAIN_PARAMS_QUICK = ['hk3', 'hk4', 'hk5']

# For medium complexity (8-10 parameters):
UNCERTAIN_PARAMS_MEDIUM = ['hk1', 'hk2', 'hk3', 'sy1', 'sy2', 'D_Isar', 'Kriv_Isar', 'D_rch1', 'D_rch2']

# For full complexity (all 19 parameters):
UNCERTAIN_PARAMS_FULL = list(ALL_PARAMETERS.keys())

# For hydraulics focus:
UNCERTAIN_PARAMS_HK = ['hk1', 'hk2', 'hk3', 'hk4', 'hk5']

# For rivers focus:
UNCERTAIN_PARAMS_RIVERS = ['Kriv_Isar', 'Kriv_Muhlbach', 'Kriv_Giessen', 'D_Isar']

# For recharge focus:
UNCERTAIN_PARAMS_RECHARGE = ['D_rch1', 'D_rch2', 'sy1', 'sy2']

# =============================================================================
# CHOOSE YOUR EXPERIMENT - MODIFY THIS LINE
# =============================================================================
SELECTED_UNCERTAIN_PARAMS = UNCERTAIN_PARAMS_QUICK  # RIVERS experiment

# Create the final parameter setup
names = SELECTED_UNCERTAIN_PARAMS
param_distros = [ALL_PARAMETERS[param]['distribution'] for param in names]

# Create default values dictionary for fixed parameters
default_values = {}
for param_name, param_info in ALL_PARAMETERS.items():
    if param_name not in names:  # Fixed parameter
        default_values[param_name] = param_info['default']

print(f"DREAM Experiment Configuration:")
print(f"  Uncertain parameters ({len(names)}): {names}")
print(f"  Fixed parameters ({len(default_values)}): {list(default_values.keys())}")
print(f"  Default values for fixed parameters: {default_values}")

# Number of repetitions and chains (auto-adjust based on number of parameters)
numParams = len(param_distros)

# Suggested settings based on number of parameters
if numParams <= 5:
    # Quick experiments - DREAM needs at least 2*numParams+1 chains
    rep = 500
    convEvals = 100
    nChains = max(2*numParams + 1, 5)  # Ensure sufficient chains for DREAM
    experiment_type = "QUICK"
elif numParams <= 10:
    # Medium experiments
    rep = 5000
    convEvals = 200
    nChains = max(2*numParams + 1, 6)  # Ensure sufficient chains
    experiment_type = "MEDIUM"
else:
    # Full experiments
    rep = 10000
    convEvals = 300
    nChains = max(2*numParams + 1, 8)  # Ensure sufficient chains
    experiment_type = "FULL"

# Allow manual override by setting custom values here:
# rep = 2000  # Uncomment and modify to override
# convEvals = 150  # Uncomment and modify to override
# nChains = 4  # Uncomment and modify to override

numSamples = rep

print(f"  Experiment type: {experiment_type}")
print(f"  DREAM settings: {rep} iterations, {nChains} chains, {convEvals} post-convergence")

# Generate sample arrays for initialization
samples = np.zeros((numSamples, numParams))
for i in range(numSamples):
    gen_samples = spotpy.parameter.generate(param_distros)
    for j in range(numParams):
        samples[i,j] = gen_samples[j][0]        

flag = 'dream-gwm-r%d-c%d-s%d' % (rep, convEvals, my_seed)