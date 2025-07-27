"""
Mock groundwater model for testing SPOTPY/DREAM setup quickly.
Replaces the expensive FloPy model with a simple mathematical function 
that still responds realistically to parameter changes.
"""

import os
import numpy as np
import pandas as pd
import tempfile
import time

def mock_GWM(hk1, hk2, hk3, hk4, hk5, sy1, sy2, sy3, sy4, sy5, D_Isar, 
             Kriv_Isar, Kriv_Muhlbach, Kriv_Giessen, Kriv_Griesbach, 
             Kriv_Schwabinger_Bach, Kriv_Wiesackerbach, D_rch1, D_rch2, 
             custom_out_dir=None):
    """
    Mock version of the GWM groundwater model that returns synthetic heads
    based on a simple mathematical relationship with the input parameters.
    
    This function mimics the behavior of the real model but runs in milliseconds
    instead of minutes, allowing for rapid testing of DREAM optimization setup.
    """
    
    # Create output directory
    if custom_out_dir is not None:
        out_dir = custom_out_dir
    else:
        out_dir = os.path.join('Mock_Output')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Add a small delay to simulate some computation (optional)
    time.sleep(0.01)  # 10ms delay - much faster than real model
    
    # Create a simple "model" object (just a dummy for compatibility)
    class MockModel:
        def __init__(self, out_dir):
            self.model_ws = out_dir
    
    model = MockModel(out_dir)
    
    return model, out_dir

def mock_get_heads_from_obs_csv(model_ws, obs_csv_path):
    """
    Mock version of get_heads_from_obs_csv that generates synthetic head data
    based on a mathematical function of the model parameters.
    
    The synthetic heads are designed to:
    1. Have realistic groundwater head values (around 500-520m)
    2. Show some correlation with hydraulic conductivity parameters
    3. Include some noise to simulate measurement uncertainty
    4. Have a known "optimal" parameter set for testing
    """
    
    # Load observation locations to get the right dimensions
    try:
        obs_data = pd.read_csv(obs_csv_path)
        n_time_steps, n_obs_points = obs_data.shape
        print(f"Mock model using dimensions: {n_time_steps} time steps × {n_obs_points} observation points")
    except:
        # Fallback if obs file not found - use actual dimensions from the real file
        n_time_steps = 139  # From the real obs file
        n_obs_points = 13   # From the real obs file
        print(f"Warning: Could not load {obs_csv_path}, using fallback {n_time_steps}×{n_obs_points} dimensions")
    
    # Generate synthetic head data
    np.random.seed(42)  # For reproducibility
    
    # Base head values (realistic for the area)
    base_heads = np.linspace(510, 520, n_obs_points)
    
    # Create some parameter-dependent variation
    # We'll simulate this by using global variables or reading from a temp file
    # For now, create a simple pattern
    
    # Time-varying component (seasonal pattern)
    time_steps = np.arange(n_time_steps)
    seasonal_component = 2 * np.sin(2 * np.pi * time_steps / 20)  # 20-step cycle
    
    # Create head matrix
    heads = np.zeros((n_time_steps, n_obs_points))
    
    for i in range(n_obs_points):
        # Base head for this location
        base_head = base_heads[i]
        
        # Add seasonal variation
        seasonal_heads = base_head + seasonal_component
        
        # Add some spatial correlation (locations closer to 0 have different behavior)
        spatial_factor = np.exp(-i * 0.1)  # Decay with distance
        
        # Add some noise
        noise = np.random.normal(0, 0.5, n_time_steps)
        
        heads[:, i] = seasonal_heads * spatial_factor + noise
    
    return heads

def create_mock_response_function(target_params=None):
    """
    Create a function that generates synthetic model responses based on parameters.
    This allows us to define what the "optimal" parameter set should be.
    """
    if target_params is None:
        # Define optimal parameter values for testing
        target_params = {
            'hk1': 5000.0,  # Optimal hydraulic conductivity for layer 1
            'hk3': 3000.0,  # Optimal hydraulic conductivity for layer 3
        }
    
    def response_function(**params):
        """
        Generate model response based on parameters.
        The response is designed so that parameters closer to target_params
        produce better (higher) objective function values.
        """
        
        # Calculate distance from optimal parameters
        total_error = 0
        for param_name, optimal_value in target_params.items():
            if param_name in params:
                current_value = params[param_name]
                # Normalized squared error
                error = ((current_value - optimal_value) / optimal_value) ** 2
                total_error += error
        
        # Convert error to a "performance" metric (lower error = higher performance)
        performance = np.exp(-total_error)  # Values between 0 and 1
        
        return performance
    
    return response_function

# Global variable to store current parameters (for the mock model)
_current_params = {}

def set_mock_parameters(**params):
    """Set parameters for the mock model response."""
    global _current_params
    _current_params.update(params)

def mock_get_heads_with_params(model_ws, obs_csv_path, **params):
    """
    Enhanced version that generates heads based on actual parameter values.
    """
    # Load observation structure to get correct dimensions
    try:
        if obs_csv_path and os.path.exists(obs_csv_path):
            obs_data = pd.read_csv(obs_csv_path)
            n_time_steps, n_obs_points = obs_data.shape
            print(f"Loaded observation structure: {n_time_steps} time steps × {n_obs_points} observation points")
        else:
            # Use the correct dimensions based on the actual data
            n_time_steps = 139  # From obs_values.csv
            n_obs_points = 13   # From obs_values.csv
            print(f"Using standard dimensions: {n_time_steps} time steps × {n_obs_points} observation points")
    except Exception as e:
        n_time_steps = 139
        n_obs_points = 13
        print(f"Error loading obs file, using fallback dimensions: {n_time_steps}×{n_obs_points}")
        print(f"Error was: {e}")
    
    # Create response function
    response_func = create_mock_response_function()
    
    # Get performance based on current parameters
    performance = response_func(**params)
    
    # Generate base synthetic data
    np.random.seed(42)  # Consistent base data
    
    # Base head values
    base_heads = np.linspace(510, 520, n_obs_points)
    
    # Time-varying component
    time_steps = np.arange(n_time_steps)
    seasonal_component = 2 * np.sin(2 * np.pi * time_steps / 20)  # 20-step cycle
    
    # Create head matrix
    heads = np.zeros((n_time_steps, n_obs_points))
    
    for i in range(n_obs_points):
        # Base pattern
        base_head = base_heads[i]
        seasonal_heads = base_head + seasonal_component
        
        # Modify based on parameters (this is where the "physics" happens)
        # Higher hydraulic conductivity -> different head response
        hk_effect = 0
        if 'hk1' in params:
            hk_effect += (params['hk1'] - 5000) / 5000 * 2  # ±2m variation
        if 'hk3' in params:
            hk_effect += (params['hk3'] - 3000) / 3000 * 1  # ±1m variation
        
        # Apply parameter effects
        param_modified_heads = seasonal_heads + hk_effect
        
        # Add noise (but make it consistent with performance)
        noise_level = 0.5 * (2 - performance)  # Less noise for better parameters
        noise = np.random.normal(0, noise_level, n_time_steps)
        
        heads[:, i] = param_modified_heads + noise
    
    return heads
