# -*- coding: utf-8 -*-
"""
Test script for DREAM setup with new GWM model
Quick test to verify the setup works before running full DREAM
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dream_init_new as di
from dream_modflow_new import spot_setup
from GWM_model_run import get_heads_from_obs_csv
import tempfile
import os

def test_single_simulation():
    """Test a single model simulation with visualization"""
    print("Testing single GWM simulation...")
    
    # Create spot_setup instance
    setup = spot_setup('dream')
    
    # Generate a random parameter set
    params = setup.parameters()
    
    # Debug: check parameter format
    print(f"Parameter format debug: {type(params)}")
    if len(params) > 0:
        print(f"First parameter: {params[0]}, type: {type(params[0])}")
    
    # Extract parameter values - SPOTPY returns numpy structured array
    param_values = []
    for p in params:
        # SPOTPY parameter format: (value, name, step, optguess, minbound, maxbound, as_int)
        if hasattr(p, '__getitem__') and len(p) > 0:
            param_values.append(float(p[0]))  # First element is the parameter value
        else:
            param_values.append(float(p))
    
    print(f"Testing with parameters:")
    for name, value in zip(di.names, param_values):
        print(f"  {name}: {value:.4f}")
    
    # Test simulation
    try:
        sim_result = setup.simulation(param_values)
        print(f"Simulation successful! Result shape: {np.array(sim_result).shape}")
        print(f"First few simulation values: {sim_result[:5]}")
        
        # Store simulation results for plotting
        global test_sim_result
        test_sim_result = np.array(sim_result)
        
        return True
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False

def test_evaluation():
    """Test loading of evaluation data"""
    print("\nTesting evaluation data loading...")
    
    setup = spot_setup('dream')
    
    try:
        obs_data = setup.evaluation()
        print(f"Evaluation data loaded successfully! Shape: {np.array(obs_data).shape}")
        print(f"First few observation values: {obs_data[:5]}")
        
        # Store observation data for plotting
        global test_obs_data
        test_obs_data = np.array(obs_data)
        
        return True
    except Exception as e:
        print(f"Evaluation loading failed: {e}")
        return False

def test_likelihood():
    """Test likelihood calculation"""
    print("\nTesting likelihood calculation...")
    
    setup = spot_setup('dream')
    
    # Create dummy simulation and evaluation data
    sim_data = np.random.normal(500, 10, 100)
    obs_data = np.random.normal(500, 10, 100)
    
    try:
        likelihood = setup.objectivefunction(sim_data, obs_data)
        print(f"Likelihood calculation successful! Value: {likelihood:.4f}")
        return True
    except Exception as e:
        print(f"Likelihood calculation failed: {e}")
        return False

def plot_test_results():
    """Create visualization plots of test results"""
    print("\nCreating test result plots...")
    
    try:
        # Check if we have both simulation and observation data
        if 'test_sim_result' not in globals() or 'test_obs_data' not in globals():
            print("No simulation or observation data available for plotting")
            return False
        
        # Load observation metadata
        obs_df = pd.read_csv('Output1_Input2/obs.csv')
        obs_names = obs_df.iloc[:, 0].values  # First column contains observation point names
        
        # Reshape data for plotting (assuming obs_values.csv format: rows=time, cols=obs_points)
        obs_values_df = pd.read_csv('Output1_Input2/obs_values.csv')
        n_time_steps = obs_values_df.shape[0]
        n_obs_points = obs_values_df.shape[1]
        
        # Reshape simulation and observation data
        sim_reshaped = test_sim_result.reshape(n_time_steps, n_obs_points)
        obs_reshaped = test_obs_data.reshape(n_time_steps, n_obs_points)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DREAM Setup Test Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series for first few observation points
        ax1 = axes[0, 0]
        n_plot = min(3, n_obs_points)  # Plot first 3 observation points
        time_steps = np.arange(n_time_steps)
        
        for i in range(n_plot):
            obs_name = obs_names[i] if i < len(obs_names) else f'Obs_{i+1}'
            ax1.plot(time_steps, obs_reshaped[:, i], 'o-', label=f'Observed {obs_name}', markersize=3)
            ax1.plot(time_steps, sim_reshaped[:, i], 's-', label=f'Simulated {obs_name}', markersize=2, alpha=0.7)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Head (m)')
        ax1.set_title('Time Series Comparison (First 3 Points)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot (simulated vs observed)
        ax2 = axes[0, 1]
        ax2.scatter(obs_reshaped.flatten(), sim_reshaped.flatten(), alpha=0.6, s=10)
        
        # Add 1:1 line
        min_val = min(obs_reshaped.min(), sim_reshaped.min())
        max_val = max(obs_reshaped.max(), sim_reshaped.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')
        
        ax2.set_xlabel('Observed Head (m)')
        ax2.set_ylabel('Simulated Head (m)')
        ax2.set_title('Simulated vs Observed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate and add R² and RMSE
        correlation_matrix = np.corrcoef(obs_reshaped.flatten(), sim_reshaped.flatten())
        r_squared = correlation_matrix[0, 1]**2
        rmse = np.sqrt(np.mean((obs_reshaped.flatten() - sim_reshaped.flatten())**2))
        ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}\nRMSE = {rmse:.2f} m', 
                transform=ax2.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Residuals over time
        ax3 = axes[1, 0]
        residuals = sim_reshaped - obs_reshaped
        mean_residuals = np.mean(residuals, axis=1)
        std_residuals = np.std(residuals, axis=1)
        
        ax3.plot(time_steps, mean_residuals, 'b-', label='Mean residual')
        ax3.fill_between(time_steps, mean_residuals - std_residuals, 
                        mean_residuals + std_residuals, alpha=0.3, label='±1 std')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Residual (m)')
        ax3.set_title('Residuals Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residuals histogram
        ax4 = axes[1, 1]
        ax4.hist(residuals.flatten(), bins=30, alpha=0.7, density=True, edgecolor='black')
        ax4.set_xlabel('Residual (m)')
        ax4.set_ylabel('Density')
        ax4.set_title('Residuals Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics to histogram
        mean_res = np.mean(residuals.flatten())
        std_res = np.std(residuals.flatten())
        ax4.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.2f}')
        ax4.text(0.05, 0.95, f'Mean: {mean_res:.2f} m\nStd: {std_res:.2f} m', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('test_plots', exist_ok=True)
        plt.savefig('test_plots/dream_test_results.png', dpi=300, bbox_inches='tight')
        print("Test results plot saved to: test_plots/dream_test_results.png")
        
        # Show the plot
        plt.show()
        
        # Print summary statistics
        print(f"\n=== TEST RESULTS SUMMARY ===")
        print(f"Number of observation points: {n_obs_points}")
        print(f"Number of time steps: {n_time_steps}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"RMSE: {rmse:.2f} m")
        print(f"Mean residual: {mean_res:.2f} m")
        print(f"Std residual: {std_res:.2f} m")
        print(f"Observed heads range: {obs_reshaped.min():.1f} - {obs_reshaped.max():.1f} m")
        print(f"Simulated heads range: {sim_reshaped.min():.1f} - {sim_reshaped.max():.1f} m")
        
        return True
        
    except Exception as e:
        print(f"Plotting failed: {e}")
        return False

if __name__ == "__main__":
    print("DREAM Setup Test for New GWM Model")
    print("=" * 50)
    
    # Test parameter initialization
    print(f"Number of parameters: {len(di.param_distros)}")
    print(f"Parameter names: {di.names}")
    print(f"Random seed: {di.my_seed}")
    
    # Run tests
    tests = [
        ("Single Simulation", test_single_simulation),
        ("Evaluation Data", test_evaluation),
        ("Likelihood Calculation", test_likelihood)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        print(f"Running {test_name} test...")
        results[test_name] = test_func()
    
    # Run visualization if simulation and evaluation tests passed
    if results.get("Single Simulation", False) and results.get("Evaluation Data", False):
        print(f"\n{'='*20}")
        print("Creating visualization plots...")
        results["Visualization"] = plot_test_results()
    
    print(f"\n{'='*50}")
    print("TEST SUMMARY:")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed! DREAM setup is ready.")
        print("Check 'test_plots/dream_test_results.png' for visualization")
        print("You can now run: python dream_run_new.py")
    else:
        print("\nSome tests failed. Please check the setup before running DREAM.")