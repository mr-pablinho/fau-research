# -*- coding: utf-8 -*-
"""
DREAM results analysis and visualization for new GWM model
Adapted from bayes2022 DREAM experiment
@author: Pablo Merchán-Rivera (adapted)

Analysis and visualization of DREAM results for new GWM groundwater model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import spotpy
import os
import dream_init_new as di

def load_dream_results(dbname='dream_GWM'):
    """Load DREAM results from CSV file"""
    try:
        results = pd.read_csv(f'{dbname}.csv')
        return results
    except FileNotFoundError:
        print(f"Results file {dbname}.csv not found. Run DREAM algorithm first.")
        return None

def analyze_convergence(results, param_names, burnin_fraction=0.5):
    """Analyze convergence of DREAM chains"""
    if results is None:
        return
    
    # Remove burn-in period
    burnin_idx = int(len(results) * burnin_fraction)
    results_converged = results.iloc[burnin_idx:]
    
    print(f"Total simulations: {len(results)}")
    print(f"After burn-in ({burnin_fraction*100}%): {len(results_converged)}")
    
    # Calculate basic statistics for each parameter
    param_stats = {}
    for i, param_name in enumerate(param_names):
        param_col = f'par{param_name}'
        if param_col in results_converged.columns:
            param_data = results_converged[param_col].values
            param_stats[param_name] = {
                'mean': np.mean(param_data),
                'std': np.std(param_data),
                'median': np.median(param_data),
                'q025': np.percentile(param_data, 2.5),
                'q975': np.percentile(param_data, 97.5)
            }
    
    return param_stats, results_converged

def plot_parameter_traces(results, param_names, save_dir='dream_plots_gwm'):
    """Plot parameter traces for convergence diagnostics"""
    if results is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i, param_name in enumerate(param_names):
        param_col = f'par{param_name}'
        if param_col in results.columns:
            axes[i].plot(results[param_col].values, alpha=0.7, linewidth=0.5)
            axes[i].set_title(f'{param_name}')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel('Parameter value')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(param_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_traces.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_distributions(results, param_names, param_stats, save_dir='dream_plots_gwm'):
    """Plot posterior parameter distributions"""
    if results is None or param_stats is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i, param_name in enumerate(param_names):
        param_col = f'par{param_name}'
        if param_col in results.columns and param_name in param_stats:
            param_data = results[param_col].values
            
            # Plot histogram
            axes[i].hist(param_data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add vertical lines for statistics
            stats_data = param_stats[param_name]
            axes[i].axvline(stats_data['mean'], color='red', linestyle='--', label=f"Mean: {stats_data['mean']:.3f}")
            axes[i].axvline(stats_data['median'], color='green', linestyle='--', label=f"Median: {stats_data['median']:.3f}")
            axes[i].axvline(stats_data['q025'], color='orange', linestyle=':', alpha=0.7, label=f"95% CI")
            axes[i].axvline(stats_data['q975'], color='orange', linestyle=':', alpha=0.7)
            
            axes[i].set_title(f'{param_name}')
            axes[i].set_xlabel('Parameter value')
            axes[i].set_ylabel('Density')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(param_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_likelihood_evolution(results, save_dir='dream_plots_gwm'):
    """Plot evolution of likelihood values"""
    if results is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    if 'like1' in results.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(results['like1'].values, alpha=0.7, linewidth=0.8)
        plt.title('Likelihood Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/likelihood_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_parameter_correlation(results, param_names, save_dir='dream_plots_gwm'):
    """Plot parameter correlation matrix"""
    if results is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract parameter columns
    param_cols = [f'par{param_name}' for param_name in param_names if f'par{param_name}' in results.columns]
    param_data = results[param_cols]
    
    # Rename columns for better visualization
    param_data.columns = [col.replace('par', '') for col in param_data.columns]
    
    # Calculate correlation matrix
    corr_matrix = param_data.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_table(param_stats, save_dir='dream_plots_gwm'):
    """Generate and save parameter summary table"""
    if param_stats is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to DataFrame for easy handling
    summary_df = pd.DataFrame(param_stats).T
    summary_df = summary_df.round(4)
    
    print("\nParameter Summary Statistics:")
    print("=" * 80)
    print(summary_df.to_string())
    
    # Save to CSV
    summary_df.to_csv(f'{save_dir}/parameter_summary.csv')
    print(f"\nSummary table saved to {save_dir}/parameter_summary.csv")

def main():
    """Main function to run complete DREAM results analysis"""
    print("Loading DREAM results for GWM model...")
    
    # Load results
    results = load_dream_results('dream_GWM')
    if results is None:
        return
    
    # Analyze convergence
    param_stats, results_converged = analyze_convergence(results, di.names)
    
    if param_stats is None:
        print("Could not analyze convergence. Check your results file.")
        return
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_parameter_traces(results, di.names)
    plot_parameter_distributions(results_converged, di.names, param_stats)
    plot_likelihood_evolution(results)
    plot_parameter_correlation(results_converged, di.names)
    
    # Generate summary table
    generate_summary_table(param_stats)
    
    print("\nDREAM results analysis completed!")
    print("Check the 'dream_plots_gwm' directory for all generated plots and summary.")

if __name__ == "__main__":
    main()