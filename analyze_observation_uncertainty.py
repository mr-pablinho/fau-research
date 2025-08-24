# -*- coding: utf-8 -*-
"""
Analyze observation point uncertainty from DREAM post-convergence results
Creates statistical summaries and uncertainty plots for each observation point
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
import re

warnings.filterwarnings("ignore", message="'mode' parameter is deprecated", category=DeprecationWarning)

# Observation point names (from obs_values.csv header)
OBS_POINT_NAMES = [
    'MSB_5', 'U6_333', 'GWM_3', 'GAR_05', 'U6_305', 'U8_1045', 'MPG_53', 
    'Poststrasse', 'U6_310', 'MPG_40', 'MPG_26', 'MPG_42', 'OBS_13'
]

def find_most_recent_dream_file():
    """Find the most recent DREAM results file"""
    dream_files = [f for f in os.listdir('.') if f.startswith('dream_GWM_') and f.endswith('.csv')]
    if not dream_files:
        raise FileNotFoundError("No DREAM result files found!")
    
    dream_files.sort(reverse=True)
    return dream_files[0]

def extract_timestamp_from_filename(filename):
    """Extract timestamp from DREAM CSV filename"""
    timestamp_pattern = r"(\d{8}_\d{6})"
    match = re.search(timestamp_pattern, filename)
    return match.group(1) if match else None

def read_convergence_info_from_log(dream_timestamp):
    """Read convergence information from DREAM log file"""
    if not dream_timestamp:
        return {'converged': None, 'runs_after_convergence': None, 'log_file': None}
    
    log_file = f'logs/log_dream_{dream_timestamp}.txt'
    
    convergence_info = {
        'converged': None,
        'runs_after_convergence': None,
        'log_file': log_file
    }
    
    if not os.path.exists(log_file):
        print(f"⚠️  Log file not found: {log_file}")
        return convergence_info
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        if "Convergence reached:............. yes" in content:
            convergence_info['converged'] = True
        elif "Convergence reached:............. no" in content:
            convergence_info['converged'] = False
            
        conv_match = re.search(r"Runs after convergence:.......... (\d+)", content)
        if conv_match:
            convergence_info['runs_after_convergence'] = int(conv_match.group(1))
            
    except Exception as e:
        print(f"⚠️  Error reading log file {log_file}: {e}")
    
    return convergence_info

def load_observed_data():
    """Load observed head values from obs_values.csv"""
    try:
        obs_df = pd.read_csv('Output1_Input2/obs_values.csv', header=0)
        obs_array = obs_df.iloc[:, :13].values.astype(np.float64)  # Take only first 13 columns
        obs_array = np.where(obs_array == 'NA', np.nan, obs_array)
        obs_array = obs_array.astype(np.float64)
        
        print(f"📊 Loaded observations with shape {obs_array.shape}")
        
        return obs_array
        
    except Exception as e:
        print(f"⚠️  Error loading observations: {e}")
        return None

def extract_simulation_results_from_dream(dream_csv_path, fallback_last_n=100):
    """
    Extract simulation results directly from DREAM CSV file (NO re-running!)
    
    Returns:
    - ensemble_heads: array of shape [n_post_conv_runs, n_stress_periods, n_obs_points]
    """
    print(f"📊 Loading DREAM results from: {dream_csv_path}")
    
    dream_timestamp = extract_timestamp_from_filename(dream_csv_path)
    dream_results = pd.read_csv(dream_csv_path)
    print(f"   Total DREAM runs: {len(dream_results)}")
    
    # Check convergence status
    convergence_info = read_convergence_info_from_log(dream_timestamp)
    
    # Find simulation columns
    sim_cols = [col for col in dream_results.columns if col.startswith('simulation_')]
    sim_cols.sort(key=lambda x: int(x.split('_')[1]))  
    
    if not sim_cols:
        raise ValueError("No simulation columns found in DREAM results! Make sure save_sim=True was used.")
    
    print(f"   Found {len(sim_cols)} simulation columns")
    
    # Expected dimensions
    n_stress_periods = 139
    n_obs_points = 13
    expected_sims = n_stress_periods * n_obs_points
    
    if len(sim_cols) != expected_sims:
        print(f"   ⚠️  Warning: Expected {expected_sims} simulation columns, found {len(sim_cols)}")
    
    if convergence_info['converged'] is True and convergence_info['runs_after_convergence'] is not None:
        n_post_conv = convergence_info['runs_after_convergence']
        post_conv_data = dream_results[sim_cols].tail(n_post_conv)
        print(f"   ✅ Using {n_post_conv} post-convergence samples (from log file)")
    elif convergence_info['converged'] is False:
        post_conv_data = dream_results[sim_cols].tail(fallback_last_n)
        print(f"   ⚠️  No convergence reached, using last {fallback_last_n} samples")
    else:
        post_conv_data = dream_results[sim_cols].tail(fallback_last_n)
        print(f"   ⚠️  Convergence status unknown, using last {fallback_last_n} samples")
    
    # Convert to numpy array and reshape
    sim_array = post_conv_data.values  # Shape: [n_runs, n_simulations]
    
    # Remove rows with any NaN values
    valid_rows = ~np.isnan(sim_array).any(axis=1)
    sim_array = sim_array[valid_rows]
    
    print(f"   Kept {sim_array.shape[0]} runs with valid simulation data")
    
    if sim_array.shape[0] == 0:
        raise ValueError("No valid simulation data found!")
    
    # Reshape to [n_runs, n_stress_periods, n_obs_points]
    # The simulation columns are flattened as: [sp0_obs0, sp0_obs1, ..., sp0_obs12, sp1_obs0, ...]
    n_runs = sim_array.shape[0]
    
    try:
        ensemble_heads = sim_array.reshape(n_runs, n_stress_periods, n_obs_points)
        print(f"   Reshaped to: {ensemble_heads.shape} [runs, stress_periods, obs_points]")
    except ValueError as e:
        print(f"   ❌ Reshape failed: {e}")
        print(f"   Trying to fit available data...")
        
        # Calculate what dimensions we can actually support
        total_sims = sim_array.shape[1]
        possible_periods = total_sims // n_obs_points
        
        if possible_periods > 0:
            # Trim to fit
            usable_sims = possible_periods * n_obs_points
            ensemble_heads = sim_array[:, :usable_sims].reshape(n_runs, possible_periods, n_obs_points)
            print(f"   Adjusted to: {ensemble_heads.shape} [runs, {possible_periods}_periods, obs_points]")
        else:
            raise ValueError(f"Cannot reshape simulation data. Total simulations: {total_sims}")
    
    return ensemble_heads

def calculate_observation_statistics(ensemble_heads):
    """Calculate statistics for each observation point and stress period"""
    print(f"📊 Calculating statistics...")
    
    n_runs, n_stress_periods, n_obs_points = ensemble_heads.shape
    
    # Initialize statistics arrays
    stats = {
        'mean': np.full((n_stress_periods, n_obs_points), np.nan),
        'median': np.full((n_stress_periods, n_obs_points), np.nan),
        'std': np.full((n_stress_periods, n_obs_points), np.nan),
        'q25': np.full((n_stress_periods, n_obs_points), np.nan),
        'q75': np.full((n_stress_periods, n_obs_points), np.nan),
        'min': np.full((n_stress_periods, n_obs_points), np.nan),
        'max': np.full((n_stress_periods, n_obs_points), np.nan),
        'count': np.zeros((n_stress_periods, n_obs_points), dtype=int)
    }
    
    # Calculate statistics for each stress period and observation point
    for sp in range(n_stress_periods):
        for obs in range(n_obs_points):
            # Get valid (non-NaN) values across all runs
            values = ensemble_heads[:, sp, obs]
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                stats['mean'][sp, obs] = np.mean(valid_values)
                stats['median'][sp, obs] = np.median(valid_values)
                stats['std'][sp, obs] = np.std(valid_values)
                stats['q25'][sp, obs] = np.percentile(valid_values, 25)
                stats['q75'][sp, obs] = np.percentile(valid_values, 75)
                stats['min'][sp, obs] = np.min(valid_values)
                stats['max'][sp, obs] = np.max(valid_values)
                stats['count'][sp, obs] = len(valid_values)
    
    print(f"   Statistics calculated for {n_stress_periods} stress periods and {n_obs_points} observation points")
    return stats

def save_observation_statistics(stats, timestamp):
    """Save statistics to separate CSV files for each observation point"""
    print(f"💾 Saving statistics to CSV files...")
    
    output_dir = f"obs_uncertainty_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    n_stress_periods, n_obs_points = stats['mean'].shape
    
    # Save individual files for each observation point
    for obs_idx in range(n_obs_points):
        obs_name = OBS_POINT_NAMES[obs_idx] if obs_idx < len(OBS_POINT_NAMES) else f"OBS_{obs_idx+1}"
        
        # Create DataFrame for this observation point
        obs_df = pd.DataFrame({
            'Stress_Period': range(1, n_stress_periods + 1),
            'Mean_Head': stats['mean'][:, obs_idx],
            'Median_Head': stats['median'][:, obs_idx],
            'Std_Head': stats['std'][:, obs_idx],
            'Q25_Head': stats['q25'][:, obs_idx],
            'Q75_Head': stats['q75'][:, obs_idx],
            'Min_Head': stats['min'][:, obs_idx],
            'Max_Head': stats['max'][:, obs_idx],
            'Valid_Runs': stats['count'][:, obs_idx]
        })
        
        # Save to CSV
        filename = os.path.join(output_dir, f"{obs_name}_uncertainty_stats.csv")
        obs_df.to_csv(filename, index=False)
        print(f"   Saved: {filename}")
    
    # Also save a combined summary file
    summary_stats = []
    for obs_idx in range(n_obs_points):
        obs_name = OBS_POINT_NAMES[obs_idx] if obs_idx < len(OBS_POINT_NAMES) else f"OBS_{obs_idx+1}"
        
        # Calculate overall statistics across all time periods
        valid_means = stats['mean'][:, obs_idx][~np.isnan(stats['mean'][:, obs_idx])]
        valid_stds = stats['std'][:, obs_idx][~np.isnan(stats['std'][:, obs_idx])]
        
        if len(valid_means) > 0:
            summary_stats.append({
                'Observation_Point': obs_name,
                'Avg_Mean_Head': np.mean(valid_means),
                'Avg_Std_Head': np.mean(valid_stds),
                'Max_Uncertainty': np.max(valid_stds) if len(valid_stds) > 0 else np.nan,
                'Min_Uncertainty': np.min(valid_stds) if len(valid_stds) > 0 else np.nan,
                'Valid_Time_Periods': len(valid_means)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_filename = os.path.join(output_dir, "uncertainty_summary.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"   Saved summary: {summary_filename}")
    
    return output_dir

def create_uncertainty_plots(stats, output_dir, observed_data=None, std_multiplier=2.0):
    """Create uncertainty plots for each observation point with mean ± std and observations
    
    Parameters:
    - stats: Dictionary with statistical arrays
    - output_dir: Output directory for plots
    - observed_data: Array with observed data (optional)
    - std_multiplier: Multiplier for standard deviation bands (default: 2.0 for ±2σ)
    """
    print(f"📈 Creating uncertainty plots (±{std_multiplier}σ bands)...")
    
    n_stress_periods, n_obs_points = stats['mean'].shape
    
    # Create time index using stress periods (1 to n_stress_periods)
    time_index = np.arange(1, n_stress_periods + 1)
    print(f"   📅 Stress periods: 1 to {n_stress_periods}")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create individual plots for each observation point
    for obs_idx in range(n_obs_points):
        obs_name = OBS_POINT_NAMES[obs_idx] if obs_idx < len(OBS_POINT_NAMES) else f"OBS_{obs_idx+1}"
        
        # Get data for this observation point
        mean_vals = stats['mean'][:, obs_idx]
        std_vals = stats['std'][:, obs_idx]
        
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot uncertainty bands
        valid_mask = ~np.isnan(mean_vals)
        valid_time = time_index[valid_mask]
        
        if np.any(valid_mask):
            # If std_multiplier is 2.0, show both ±1σ and ±2σ bands
            if std_multiplier == 2.0:
                # Calculate both 1σ and 2σ bounds
                upper_1sigma = mean_vals + std_vals
                lower_1sigma = mean_vals - std_vals
                upper_2sigma = mean_vals + (2.0 * std_vals)
                lower_2sigma = mean_vals - (2.0 * std_vals)
                
                # Plot 2σ band (lighter, wider)
                ax.fill_between(valid_time, 
                               lower_2sigma[valid_mask], 
                               upper_2sigma[valid_mask], 
                               alpha=0.2, color='lightblue', 
                               label='Mean ± 2σ (~95% confidence)')
                
                # Plot 1σ band (darker, narrower)
                ax.fill_between(valid_time, 
                               lower_1sigma[valid_mask], 
                               upper_1sigma[valid_mask], 
                               alpha=0.4, color='skyblue', 
                               label='Mean ± 1σ (~68% confidence)')
            else:
                # For other multipliers, show single band
                upper_bound = mean_vals + (std_multiplier * std_vals)
                lower_bound = mean_vals - (std_multiplier * std_vals)
                
                ax.fill_between(valid_time, 
                               lower_bound[valid_mask], 
                               upper_bound[valid_mask], 
                               alpha=0.3, color='lightblue', 
                               label=f'Mean ± {std_multiplier}σ')
            
            # Plot mean line
            ax.plot(valid_time, mean_vals[valid_mask], 'b-', 
                   linewidth=2, label='Predicted Mean')
            
            # Plot median line
            median_vals = stats['median'][:, obs_idx]
            ax.plot(valid_time, median_vals[valid_mask], 'g--', 
                   linewidth=2, label='Predicted Median')
        
        # Plot observed data if available
        if observed_data is not None:
            obs_vals = observed_data[:, obs_idx]
            obs_valid_mask = ~np.isnan(obs_vals)
            
            if np.any(obs_valid_mask):
                obs_valid_time = time_index[obs_valid_mask]
                ax.plot(obs_valid_time, obs_vals[obs_valid_mask], 
                       'ro', markersize=4, alpha=0.8, 
                       label='Observed Data')
        
        ax.set_title(f'{obs_name} - Predicted vs Observed Groundwater Heads', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Stress Period', fontsize=12)
        ax.set_ylabel('Groundwater Head (m)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(output_dir, f"{obs_name}_uncertainty_plot.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved plot: {plot_filename}")
    
    # Create a summary plot showing uncertainty for all observation points
    create_summary_uncertainty_plot(stats, output_dir, time_index)

def create_summary_uncertainty_plot(stats, output_dir, time_index):
    """Create summary plots comparing uncertainty across all observation points"""
    
    n_stress_periods, n_obs_points = stats['mean'].shape
    
    # Plot 1: Average uncertainty by observation point
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate average statistics for each observation point
    avg_std = []
    avg_mean = []
    obs_names = []
    
    for obs_idx in range(n_obs_points):
        obs_name = OBS_POINT_NAMES[obs_idx] if obs_idx < len(OBS_POINT_NAMES) else f"OBS_{obs_idx+1}"
        obs_names.append(obs_name)
        
        valid_std = stats['std'][:, obs_idx][~np.isnan(stats['std'][:, obs_idx])]
        valid_mean = stats['mean'][:, obs_idx][~np.isnan(stats['mean'][:, obs_idx])]
        
        avg_std.append(np.mean(valid_std) if len(valid_std) > 0 else 0)
        avg_mean.append(np.mean(valid_mean) if len(valid_mean) > 0 else 0)
    
    # Bar plot of average uncertainty
    bars1 = ax1.bar(range(len(obs_names)), avg_std, color='coral', alpha=0.7)
    ax1.set_title('Average Prediction Uncertainty by Observation Point', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Observation Point', fontsize=12)
    ax1.set_ylabel('Average Standard Deviation (m)', fontsize=12)
    ax1.set_xticks(range(len(obs_names)))
    ax1.set_xticklabels(obs_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_std):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Heatmap of uncertainty over time and observation points
    std_matrix = stats['std'].T  # Transpose to have obs points on y-axis
    im = ax2.imshow(std_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax2.set_title('Uncertainty Heatmap (Std Dev over Time)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Stress Period', fontsize=12)
    ax2.set_ylabel('Observation Point', fontsize=12)
    ax2.set_yticks(range(len(obs_names)))
    ax2.set_yticklabels(obs_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Standard Deviation (m)', rotation=270, labelpad=15)
    
    # Time series of ensemble spread (average across all observation points)
    valid_periods = []
    avg_uncertainty = []
    
    for sp in range(n_stress_periods):
        period_stds = stats['std'][sp, :][~np.isnan(stats['std'][sp, :])]
        if len(period_stds) > 0:
            valid_periods.append(sp)
            avg_uncertainty.append(np.mean(period_stds))
    
    if valid_periods:
        valid_time = time_index[valid_periods]
        ax3.plot(valid_time, avg_uncertainty, 'purple', linewidth=2, marker='o', markersize=4)
        ax3.fill_between(valid_time, 0, avg_uncertainty, alpha=0.3, color='purple')
    
    ax3.set_title('Average Uncertainty Across All Observation Points', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Average Standard Deviation (m)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Box plot of uncertainties
    uncertainty_data = []
    uncertainty_labels = []
    
    for obs_idx in range(n_obs_points):
        obs_name = OBS_POINT_NAMES[obs_idx] if obs_idx < len(OBS_POINT_NAMES) else f"OBS_{obs_idx+1}"
        valid_std = stats['std'][:, obs_idx][~np.isnan(stats['std'][:, obs_idx])]
        
        if len(valid_std) > 0:
            uncertainty_data.append(valid_std)
            uncertainty_labels.append(obs_name)
    
    if uncertainty_data:
        box_plot = ax4.boxplot(uncertainty_data, labels=uncertainty_labels, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("Set3", len(box_plot['boxes']))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.set_title('Distribution of Uncertainties by Observation Point', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Observation Point', fontsize=12)
    ax4.set_ylabel('Standard Deviation (m)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_filename = os.path.join(output_dir, "uncertainty_summary_plots.png")
    plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved summary plots: {summary_plot_filename}")

def main(std_multiplier=2.0):
    """Main function to run the observation uncertainty analysis
    
    Parameters:
    - std_multiplier: Multiplier for standard deviation bands (default: 2.0 for ±2σ)
    """
    print("🔍 Starting Observation Point Uncertainty Analysis")
    print("=" * 60)
    
    try:
        # Find most recent DREAM results
        dream_file = find_most_recent_dream_file()
        timestamp = extract_timestamp_from_filename(dream_file) or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"📁 Using DREAM results: {dream_file}")
        print(f"🕐 Timestamp: {timestamp}")
        print(f"📊 Using ±{std_multiplier}σ uncertainty bands")
        
        # Extract simulation results directly from DREAM CSV (NO re-running!)
        ensemble_heads = extract_simulation_results_from_dream(dream_file)
        print(f"📊 Extracted simulation results: {ensemble_heads.shape}")
        
        # Load observed data
        observed_data = load_observed_data()
        if observed_data is not None:
            print(f"📊 Loaded observed data: {observed_data.shape}")
        else:
            print("⚠️  No observed data available")
        
        # Calculate statistics
        stats = calculate_observation_statistics(ensemble_heads)
        
        # Save results
        output_dir = save_observation_statistics(stats, timestamp)
        
        # Create plots with observations and configurable std bands
        create_uncertainty_plots(stats, output_dir, observed_data, std_multiplier)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        print(f"📊 Individual CSV files created for each observation point")
        print(f"📈 Uncertainty plots created with ±{std_multiplier}σ bands")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()