# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

Evaluate DREAM results for new model
"""

# %% Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dream_init_new as di
from scipy.stats import skew, kurtosis
import os
import re
from datetime import datetime

# %% Parameter name mappings for plotting

# Short names for plot titles
params_names_short = ['hk1', 'hk2', 'hk3', 'hk4', 'hk5', 
                     'sy1', 'sy2', 'sy3', 'sy4', 'sy5',
                     'D_Isar', 'Kriv_Isar', 'Kriv_Muhlbach', 'Kriv_Giessen',
                     'Kriv_Griesbach', 'Kriv_Schwabinger_Bach', 'Kriv_Wiesackerbach',
                     'D_rch1', 'D_rch2']

# Full names for plot labels
params_names_full = ['Hydraulic Conductivity Zone 1 (m/d)', 
                    'Hydraulic Conductivity Zone 2 (m/d)',
                    'Hydraulic Conductivity Zone 3 (m/d)',
                    'Hydraulic Conductivity Zone 4 (m/d)',
                    'Hydraulic Conductivity Zone 5 (m/d)',
                    'Specific Yield Zone 1 (-)',
                    'Specific Yield Zone 2 (-)',
                    'Specific Yield Zone 3 (-)',
                    'Specific Yield Zone 4 (-)',
                    'Specific Yield Zone 5 (-)',
                    'Isar Stage Adjustment (m)',
                    'Isar River Conductance (m²/d)',
                    'Muhlbach River Conductance (m²/d)',
                    'Giessen River Conductance (m²/d)',
                    'Griesbach River Conductance (m²/d)',
                    'Schwabinger Bach Conductance (m²/d)',
                    'Wiesackerbach River Conductance (m²/d)',
                    'Background Recharge Multiplier (-)',
                    'Urban Recharge Multiplier (-)']

# Color definitions
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# %% Evaluate results

def analyze_dream_results(results_file='dream_GWM_new.csv', convergence_evals=1000):
    """
    Analyze DREAM results for the new GWM model
    
    Parameters:
    - results_file: CSV file containing DREAM results
    - convergence_evals: Number of final samples to consider as converged
    """
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Please run the DREAM algorithm first.")
        return None, None, None
    
    # Extract timestamp from filename for output files
    timestamp_match = re.search(r'dream_GWM_(\d{8}_\d{6})\.csv', results_file)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        output_suffix = f"_{timestamp}"
    else:
        # Fallback to current timestamp if no timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_suffix = f"_{timestamp}"
    
    print(f"Output files will use timestamp: {timestamp}")
    
    # Set random seed for reproducibility
    np.random.seed(di.my_seed) 
    
    print(f"Analyzing DREAM results from {results_file}")
    print(f"Using last {convergence_evals} samples for convergence analysis")
    
    # Import DREAM results
    data_results = pd.read_csv(results_file)
    print(f"Total samples in results: {len(data_results)}")
    print(f"All columns: {data_results.columns.tolist()}")
    
    # Identify parameter columns - ONLY actual parameters, not simulation results
    param_columns = []
    likelihood_col = None
    
    for col in data_results.columns:
        if 'like' in col.lower():
            likelihood_col = col
        elif col.startswith('par') and not col.startswith('simulation'):
            param_columns.append(col)
    
    # Also check for specific parameter names we know
    for col in ['parhk3', 'parhk4']:
        if col in data_results.columns and col not in param_columns:
            param_columns.append(col)
    
    if likelihood_col is None:
        print("Warning: No likelihood column found, using first column")
        likelihood_col = data_results.columns[0]
    
    if not param_columns:
        print("Error: No parameter columns found!")
        return None, None, None
    
    print(f"Likelihood column: {likelihood_col}")
    print(f"Parameter columns ({len(param_columns)}): {param_columns}")
    
    # Extract only the parameter data we want to analyze
    numParams = len(param_columns)
    results_array = np.zeros((len(data_results), numParams))
    for i, col in enumerate(param_columns):
        results_array[:,i] = data_results[col].values
        
    likelihood = data_results[likelihood_col].values
    
    # Extract estimated parameters (all samples vs converged samples)
    params_new_all = results_array
    # Adjust convergence_evals if we have fewer samples than requested
    actual_convergence_evals = min(convergence_evals, len(data_results))
    params_new_con = params_new_all[-actual_convergence_evals:,:]
    
    print(f"Using last {actual_convergence_evals} samples for convergence analysis (requested: {convergence_evals})")
    
    # Compute statistics
    params_stats = {}
    params_stats['mean_all'] = np.mean(params_new_all, axis=0)
    params_stats['std_all'] = np.std(params_new_all, axis=0)
    params_stats['mean_con'] = np.mean(params_new_con, axis=0)
    params_stats['std_con'] = np.std(params_new_con, axis=0)
    params_stats['skew_con'] = skew(params_new_con, axis=0)
    params_stats['kurt_con'] = kurtosis(params_new_con, axis=0)
    
    # Find best parameter set
    best_set_loc = data_results[likelihood_col].idxmax()
    best_set = np.array([data_results.loc[best_set_loc, col] for col in param_columns])
    best_likelihood = data_results.loc[best_set_loc, likelihood_col]
    
    # Use the actual parameter names we found
    param_names_to_use = param_columns
    
    print(f"\\nBest likelihood: {best_likelihood:.4f}")
    print("Best parameter set:")
    for i, (name, value) in enumerate(zip(param_names_to_use, best_set)):
        if 'hk' in name or 'Kriv' in name:  # Log-transformed parameters
            print(f"  {name}: {10**value:.6f} (log: {value:.4f})")
        else:
            print(f"  {name}: {value:.4f}")
    
    # Create results summary table
    create_results_table(params_stats, param_names_to_use, actual_convergence_evals, output_suffix)
    
    # Plot parameter evolution (traces)
    plot_parameter_traces(results_array, likelihood, param_names_to_use, output_suffix)
    
    # Plot parameter distributions (prior vs posterior)
    plot_parameter_distributions(params_new_con, param_names_to_use, actual_convergence_evals, output_suffix)
    
    # Plot likelihood evolution
    plot_likelihood_evolution(likelihood, output_suffix)
    
    # Plot parameter correlations
    plot_parameter_correlations(params_new_con, param_names_to_use, output_suffix)
    
    return params_stats, best_set, likelihood

def create_results_table(params_stats, param_names, convergence_evals, output_suffix=""):
    """Create a summary table of parameter statistics"""
    
    results_df = pd.DataFrame({
        'Parameter': param_names,
        'Mean_All': params_stats['mean_all'],
        'Std_All': params_stats['std_all'],
        'Mean_Converged': params_stats['mean_con'],
        'Std_Converged': params_stats['std_con'],
        'Skewness': params_stats['skew_con'],
        'Kurtosis': params_stats['kurt_con']
    })
    
    # Save table with timestamp
    filename = f'dream_parameters_summary{output_suffix}.csv'
    results_df.to_csv(filename, index=False)
    print(f"\\nParameter summary saved to '{filename}'")
    
    return results_df

def plot_parameter_traces(results_array, likelihood, param_names, output_suffix=""):
    """Plot parameter evolution over DREAM iterations"""
    
    numParams = results_array.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(numParams / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if numParams == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(numParams):
        axes[i].plot(results_array[:,i], alpha=0.7, linewidth=0.8, color=colors[i%len(colors)])
        axes[i].set_title(f'{param_names[i]}', fontsize=10)
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Parameter Value')
        axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(numParams, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    filename = f'dream_parameter_traces{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def plot_parameter_distributions(params_converged, param_names, convergence_evals, output_suffix=""):
    """Plot parameter distributions comparing prior and posterior"""
    
    numParams = params_converged.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(numParams / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if numParams == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    bars = 20
    alpha = 0.65
    
    for i in range(numParams):
        # Try to get prior samples if available
        try:
            import dream_init_new as di
            if hasattr(di, 'samples') and i < len(di.samples[0]):
                prior_samples = di.samples[:convergence_evals, i]
                range_bounds = (di.param_distros[i].minbound, di.param_distros[i].maxbound)
                # Plot prior histogram
                axes[i].hist(prior_samples, bars, alpha=alpha+0.2, color='orange', 
                            label="Prior", density=True, range=range_bounds)
            else:
                range_bounds = None
        except (ImportError, AttributeError, IndexError):
            print(f"Warning: Could not load prior samples for parameter {param_names[i]}")
            range_bounds = None
        
        # Plot posterior histogram
        axes[i].hist(params_converged[:,i], bars, alpha=alpha, color='blue', 
                    label="Posterior", density=True, range=range_bounds)
        
        axes[i].set_title(f'{param_names[i]}', fontsize=10)
        axes[i].set_xlabel('Parameter Value')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(numParams, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    filename = f'dream_parameter_distributions{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def plot_likelihood_evolution(likelihood, output_suffix=""):
    """Plot likelihood evolution"""
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(likelihood, alpha=0.8, linewidth=1.0, color='darkred')
    plt.title('Likelihood Evolution', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, alpha=0.3)
    
    # Running maximum
    plt.subplot(2, 1, 2)
    running_max = np.maximum.accumulate(likelihood)
    plt.plot(running_max, alpha=0.9, linewidth=1.5, color='darkgreen')
    plt.title('Running Maximum Likelihood', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Max Log-Likelihood')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'dream_objective_evolution{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def plot_parameter_correlations(params_converged, param_names, output_suffix=""):
    """Plot parameter correlation matrix"""
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(params_converged.T)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot correlation matrix
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add correlation values as text
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    plt.title('Parameter Correlation Matrix', fontsize=14)
    plt.tight_layout()
    filename = f'dream_parameter_correlation{output_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# %% Main execution

if __name__ == "__main__":
    
    # Look for the most recent DREAM results file
    import glob
    dream_files = glob.glob('dream_GWM_*.csv')
    if not dream_files:
        print("No DREAM results files found. Please run the DREAM algorithm first (dream_run_new.py)")
        exit(1)
    
    # Sort by modification time and get the most recent
    most_recent_file = max(dream_files, key=os.path.getmtime)
    print(f"Using most recent results file: {most_recent_file}")
    
    # Extract timestamp for output directory
    timestamp_match = re.search(r'dream_GWM_(\d{8}_\d{6})\.csv', most_recent_file)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        output_dir = f'dream_plots_{timestamp}'
    else:
        output_dir = 'dream_plots_new'
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Analyze results (adjust convergence_evals based on your needs)
    convergence_evals = 1000  # Number of final samples to consider converged
    
    try:
        stats, best_params, likelihood = analyze_dream_results(
            results_file=f'../{most_recent_file}', 
            convergence_evals=convergence_evals
        )
        
        print("\\nAnalysis completed successfully!")
        print(f"Plots saved in '{output_dir}' directory")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure to run the DREAM algorithm first (dream_run_new.py)")
    
    os.chdir('..')  # Return to main directory
