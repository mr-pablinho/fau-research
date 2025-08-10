# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

DREAM initial values for new model with dynamic parameter selection
"""

# %% Import libraries

import spotpy
import numpy as np
from datetime import datetime

# %% Load deterministic parameter values from params.py

# Import deterministic values and ranges from params.py
try:
    import params
    
    # Extract parameter definitions: (default_value, [min_range, max_range])
    param_definitions = {
        'hk1': params.hk1,
        'hk2': params.hk2, 
        'hk3': params.hk3,
        'hk4': params.hk4,
        'hk5': params.hk5,
        'sy1': params.sy1,
        'sy2': params.sy2,
        'sy3': params.sy3,
        'sy4': params.sy4,
        'sy5': params.sy5,
        'D_Isar': params.D_Isar,
        'Kriv_Isar': params.Kriv_Isar,
        'Kriv_Muhlbach': params.Kriv_Muhlbach,
        'Kriv_Giessen': params.Kriv_Giessen,
        'Kriv_Griesbach': params.Kriv_Griesbach,
        'Kriv_Schwabinger_Bach': params.Kriv_Schwabinger_Bach,
        'Kriv_Wiesackerbach': params.Kriv_Wiesackerbach,
        'D_rch1': params.D_rch1,
        'D_rch2': params.D_rch2
    }
except ImportError:
    print("Warning: params.py not found, using default parameter definitions")
    # Fallback parameter definitions if params.py is not available
    param_definitions = {
        'hk1': (1000, [1, 10000]),
        'hk2': (1000, [1, 10000]),
        'hk3': (1000, [1, 10000]),
        'hk4': (1000, [1, 10000]),
        'hk5': (1000, [1, 10000]),
        'sy1': (0.20, [0.05, 0.35]),
        'sy2': (0.20, [0.05, 0.35]),
        'sy3': (0.20, [0.05, 0.35]),
        'sy4': (0.20, [0.05, 0.35]),
        'sy5': (0.20, [0.05, 0.35]),
        'D_Isar': (0.0, [-0.5, 0.5]),
        'Kriv_Isar': (100, [10, 1000]),
        'Kriv_Muhlbach': (100, [10, 1000]),
        'Kriv_Giessen': (100, [10, 1000]),
        'Kriv_Griesbach': (100, [10, 1000]),
        'Kriv_Schwabinger_Bach': (100, [10, 1000]),
        'Kriv_Wiesackerbach': (100, [10, 1000]),
        'D_rch1': (1.0, [1, 3]),
        'D_rch2': (1.0, [1, 3])
    }

# %% Dynamic parameter selection

# CONFIGURE WHICH PARAMETERS TO CALIBRATE HERE
CALIBRATE_PARAMS = [
    'hk1',
    'hk2',
    'hk3',
    'hk4',
    'hk5',
    'Kriv_Isar',
    'Kriv_Muhlbach',
]

print(f"DREAM will calibrate {len(CALIBRATE_PARAMS)} parameters: {CALIBRATE_PARAMS}")

# %% Setup random state for the whole process

# set the random state
my_seed = 246
np.random.seed(my_seed)

# %% Create parameter distributions dynamically

# Parameters that use log-transformation (conductivities)
LOG_TRANSFORM_PARAMS = [
    'hk1', 'hk2', 'hk3', 'hk4', 'hk5', 
    'Kriv_Isar', 'Kriv_Muhlbach', 'Kriv_Giessen', 
    'Kriv_Griesbach', 'Kriv_Schwabinger_Bach', 'Kriv_Wiesackerbach'
    ]

# Build parameter names and distributions for only the calibrated parameters
names = []
param_distros = []
deterministic_values = {}

for param_name in CALIBRATE_PARAMS:
    default_val, (min_val, max_val) = param_definitions[param_name]
    
    names.append(param_name)
    
    if param_name in LOG_TRANSFORM_PARAMS:
        # Use log-uniform distribution for conductivity parameters
        low_bound = np.log10(min_val)
        high_bound = np.log10(max_val)
        param_distros.append(spotpy.parameter.Uniform(param_name, low=low_bound, high=high_bound))
        print(f"  {param_name}: log-uniform [{min_val:.1e}, {max_val:.1e}] m/d (default: {default_val:.1e})")
    else:
        # Use linear distribution for other parameters
        param_distros.append(spotpy.parameter.Uniform(param_name, low=min_val, high=max_val))
        print(f"  {param_name}: uniform [{min_val}, {max_val}] (default: {default_val})")

# Store deterministic values for all parameters (both calibrated and fixed)
for param_name, (default_val, _) in param_definitions.items():
    deterministic_values[param_name] = default_val

print(f"\nDeterministic values for fixed parameters:")
fixed_params = [p for p in param_definitions.keys() if p not in CALIBRATE_PARAMS]
for param_name in fixed_params:
    default_val = deterministic_values[param_name]
    if param_name in LOG_TRANSFORM_PARAMS:
        print(f"  {param_name}: {default_val:.1e}")
    else:
        print(f"  {param_name}: {default_val}")

# %% DREAM algorithm settings

rep =  5000  
numSamples = rep
convEvals = 300
numParams = len(param_distros)
nChains = max(7, 2*numParams + 1)

print(f"\nDREAM settings:")
print(f"  Parameters to calibrate: {numParams}")
print(f"  Repetitions: {rep}")
print(f"  Chains: {nChains} (minimum required: {2*numParams + 1})")
print(f"  Convergence evaluations: {convEvals}")

# Generate initial samples for parameter space exploration
# samples = np.zeros((numSamples, numParams))
# for i in range(numSamples):
#     gen_samples = spotpy.parameter.generate(param_distros)
#     for j in range(numParams):
#         samples[i,j] = gen_samples[j][0]        

# Create unique identifier for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
flag = 'dream-r%d-c%d-p%d-s%d-%s' % (rep, convEvals, numParams, my_seed, timestamp)
