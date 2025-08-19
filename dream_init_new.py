# -*- coding: utf-8 -*-
"""
DREAM initial values for new model with dynamic parameter selection
"""

import spotpy
import numpy as np
from datetime import datetime


# Set random state
my_seed = 246
np.random.seed(my_seed)

# Load parameter values
param_definitions = {
    'hk1': (885.8667904, [1, 10000]),
    'hk2': (1128.837892, [1, 10000]),
    'hk3': (127.4274986, [1, 10000]),
    'hk4': (127.4274986, [1, 10000]),
    'hk5': (2335.721469, [1, 10000]),
    'sy1': (0.113449022, [0.05, 0.35]),
    'sy2': (0.075315676, [0.05, 0.35]),
    'sy3': (0.061365983, [0.05, 0.35]),
    'sy4': (0.139238215, [0.05, 0.35]),
    'sy5': (0.125683887, [0.05, 0.35]),
    'D_Isar': (-0.184210526, [-0.5, 0.5]),
    'Kriv_Isar': (233.5721469, [10, 1000]),
    'Kriv_Muhlbach': (88.58667904, [10, 1000]),
    'Kriv_Giessen': (112.8837892, [10, 1000]),
    'Kriv_Griesbach': (11.28837892, [10, 1000]),
    'Kriv_Schwabinger_Bach': (1000, [10, 1000]),
    'Kriv_Wiesackerbach': (12.74274986, [10, 1000]),
    'D_rch1': (1.421052632, [1, 3]),
    'D_rch2': (1.105263158, [1, 3])
}

# Parameters that use log-transformation (conductivities)
LOG_TRANSFORM_PARAMS = [
    'hk1', 'hk2', 'hk3', 'hk4', 'hk5', 
    'Kriv_Isar', 'Kriv_Muhlbach', 'Kriv_Giessen', 
    'Kriv_Griesbach', 'Kriv_Schwabinger_Bach', 'Kriv_Wiesackerbach'
    ]

# Dynamic parameter selection
CALIBRATE_PARAMS = [
    # 'hk1',
    # 'hk2',
    # 'hk3',
    'hk4',
    'hk5',
    # 'Kriv_Isar',
    # 'Kriv_Muhlbach',
]

print(f"DREAM will calibrate {len(CALIBRATE_PARAMS)} parameters: {CALIBRATE_PARAMS}")

# Build parameter names and distributions
names = []
param_distros = []
deterministic_values = {}

for param_name in CALIBRATE_PARAMS:
    default_val, (min_val, max_val) = param_definitions[param_name]
    names.append(param_name)
    if param_name in LOG_TRANSFORM_PARAMS:
        # Log-uniform distribution for conductivity parameters
        low_bound = np.log10(min_val)
        high_bound = np.log10(max_val)
        param_distros.append(spotpy.parameter.Uniform(param_name, low=low_bound, high=high_bound))
        print(f"  {param_name}: log-uniform [{min_val:.1e}, {max_val:.1e}] m/d (default: {default_val:.1e})")
    else:
        # Use linear distribution for other parameters
        param_distros.append(spotpy.parameter.Uniform(param_name, low=min_val, high=max_val))
        print(f"  {param_name}: uniform [{min_val}, {max_val}] (default: {default_val})")

# Store values for all parameters (both calibrated and fixed)
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

# DREAM algorithm settings
rep =  10000  # number of maximum repetitions
numSamples = rep
convEvals = 20  # number of runs after convergence
numParams = len(param_distros)
nChains = max(7, 2*numParams + 1)  # number of chains
convergence_limit = (1.0)  # maximum Gelman–Rubin diagnostic across all model parameters
ato = 6  # acceptance test option (6: adaptive with covariance and parallel chains with crossover and adaptive step size)
nCr = 4  # number of crossover values
epsilon = 1e-5  # tolerance threshold (Turner & Sederberg, 2012)

print(f"\nDREAM settings:")
print(f"  Parameters to calibrate: {numParams}")
print(f"  Repetitions: {rep}")
print(f"  Chains: {nChains} (minimum required: {2*numParams + 1})")
print(f"  Convergence evaluations: {convEvals}")    

# Create unique identifier for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
flag = 'dream-r%d-c%d-p%d-s%d-%s' % (rep, convEvals, numParams, my_seed, timestamp)
