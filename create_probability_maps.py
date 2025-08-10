# -*- coding: utf-8 -*-
"""
Comprehensive script to:
1. Extract post-convergence parameters from DREAM results
2. Run GWM model for each parameter set
3. Save all head values for each run  
4. Create probability maps for groundwater reaching surface

Enhanced Features:
- Uses DREAM timestamp for organized output structure
- Simple progress messages (no tqdm dependency)
- Multiple depth threshold analysis
- Configurable testing mode for limited runs

Output Structure:
- ensemble_results/ensemble_YYYYMMDD_HHMMSS/
  ├── head_arrays/           # Pickled head arrays for each model run  
  ├── model_workspaces/      # Individual MODFLOW workspaces (optional cleanup)
  ├── probability_maps/      # Probability maps for each stress period and depth
  ├── plots/                 # Visualization plots
  └── results_summary.pkl    # Complete run metadata
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import flopy
import re
from GWM_model_run import GWM, get_heads_from_obs_csv
from dream_init_new import (
    CALIBRATE_PARAMS, deterministic_values, LOG_TRANSFORM_PARAMS, 
    param_definitions, runs_after_convergence, flag
)

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from DREAM CSV filename
    Expected format: dream_GWM_YYYYMMDD_HHMMSS.csv
    
    Returns:
    - timestamp string (YYYYMMDD_HHMMSS) or None if not found
    """
    # Try to extract timestamp pattern YYYYMMDD_HHMMSS
    timestamp_pattern = r'(\d{8}_\d{6})'
    match = re.search(timestamp_pattern, filename)
    
    if match:
        return match.group(1)
    
    # If no timestamp found, return None
    return None

def extract_post_convergence_parameters(dream_csv_path, convergence_point=None):
    """
    Extract post-convergence parameter sets from DREAM CSV file
    
    Parameters:
    - dream_csv_path: Path to DREAM results CSV
    - convergence_point: Number of runs to skip (pre-convergence). If None, uses last 500 runs
    
    Returns:
    - post_conv_params: DataFrame with post-convergence parameter sets
    """
    print(f"📊 Loading DREAM results from: {dream_csv_path}")
    
    # Load DREAM results
    dream_results = pd.read_csv(dream_csv_path)
    print(f"   Total DREAM runs: {len(dream_results)}")
    
    # Extract parameter columns (skip 'like1' and simulation columns)
    param_cols = [col for col in dream_results.columns 
                  if col.startswith('par') or col in CALIBRATE_PARAMS]
    
    # If no explicit parameter columns, use first few columns excluding likelihood and simulations
    if not param_cols:
        # Assume first columns after 'like1' are parameters
        non_param_cols = ['like1'] + [col for col in dream_results.columns if col.startswith('simulation')] + ['chain']
        param_cols = [col for col in dream_results.columns if col not in non_param_cols]
    
    print(f"   Parameter columns found: {param_cols}")
    
    # Determine convergence point
    if convergence_point is None:
        convergence_point = max(0, len(dream_results) - runs_after_convergence)
    
    # Extract post-convergence runs
    post_conv_params = dream_results.iloc[convergence_point:][param_cols].copy()
    print(f"   Post-convergence runs: {len(post_conv_params)} (from row {convergence_point})")
    
    # Map parameter names if needed (handle 'par' prefix)
    param_mapping = {}
    for col in param_cols:
        if col.startswith('par'):
            # Extract parameter name after 'par'
            clean_name = col.replace('par', '')
            # Try to match with CALIBRATE_PARAMS
            for param in CALIBRATE_PARAMS:
                if param.lower().replace('_', '') == clean_name.lower().replace('_', ''):
                    param_mapping[col] = param
                    break
        else:
            param_mapping[col] = col
    
    # Rename columns
    post_conv_params = post_conv_params.rename(columns=param_mapping)
    
    # Transform log parameters back to linear scale
    for param in post_conv_params.columns:
        if param in LOG_TRANSFORM_PARAMS:
            post_conv_params[param] = 10 ** post_conv_params[param]
            print(f"   Transformed {param} from log to linear scale")
    
    return post_conv_params

def prepare_full_parameter_sets(post_conv_params):
    """
    Combine post-convergence calibrated parameters with fixed parameters
    
    Returns:
    - full_param_sets: List of dictionaries, each containing all parameters for one model run
    """
    print(f"🔧 Preparing full parameter sets...")
    
    full_param_sets = []
    
    for idx, row in post_conv_params.iterrows():
        param_set = deterministic_values.copy()  # Start with default/fixed values
        
        # Update with calibrated values
        for param in CALIBRATE_PARAMS:
            if param in row.index:
                param_set[param] = row[param]
            else:
                print(f"⚠️  Warning: {param} not found in post-convergence results")
        
        full_param_sets.append(param_set)
    
    print(f"   Created {len(full_param_sets)} complete parameter sets")
    return full_param_sets

def run_ensemble_models(full_param_sets, timestamp=None):
    """
    Run GWM model for each parameter set and save head arrays
    
    Parameters:
    - full_param_sets: List of parameter dictionaries
    - timestamp: Timestamp string to use for output directory (if None, generates new one)
    
    Returns:
    - results_summary: Dictionary with run metadata
    """
    print(f"🚀 Running ensemble of {len(full_param_sets)} GWM models...")
    
    # Create output directory with specified or generated timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = f"ensemble_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   📁 Output directory: {output_dir}")
    
    # Create subdirectories
    heads_dir = os.path.join(output_dir, "head_arrays")
    models_dir = os.path.join(output_dir, "model_workspaces")
    os.makedirs(heads_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    results_summary = {
        'output_dir': output_dir,
        'timestamp': timestamp,
        'total_runs': len(full_param_sets),
        'successful_runs': 0,
        'failed_runs': 0,
        'head_files': [],
        'run_metadata': []
    }
    
    # Run models
    print(f"   🔧 Starting model runs...")
    for i, param_set in enumerate(full_param_sets):
        # Print progress for all runs if ≤10 runs, otherwise every 10 runs + last run
        if len(full_param_sets) <= 10 or i % 10 == 0 or i == len(full_param_sets) - 1:
            print(f"   🔄 Running model {i+1}/{len(full_param_sets)} ({100*(i+1)/len(full_param_sets):.1f}%)")
        
        try:
            run_id = f"run_{i:04d}"
            model_workspace = os.path.join(models_dir, run_id)
            
            # Run GWM model
            model, model_ws = GWM(
                hk1=param_set['hk1'], hk2=param_set['hk2'], hk3=param_set['hk3'], 
                hk4=param_set['hk4'], hk5=param_set['hk5'],
                sy1=param_set['sy1'], sy2=param_set['sy2'], sy3=param_set['sy3'], 
                sy4=param_set['sy4'], sy5=param_set['sy5'],
                D_Isar=param_set['D_Isar'],
                Kriv_Isar=param_set['Kriv_Isar'], Kriv_Muhlbach=param_set['Kriv_Muhlbach'],
                Kriv_Giessen=param_set['Kriv_Giessen'], Kriv_Griesbach=param_set['Kriv_Griesbach'],
                Kriv_Schwabinger_Bach=param_set['Kriv_Schwabinger_Bach'], 
                Kriv_Wiesackerbach=param_set['Kriv_Wiesackerbach'],
                D_rch1=param_set['D_rch1'], D_rch2=param_set['D_rch2'],
                custom_out_dir=model_workspace
            )
            
            # Extract and save all head arrays for all time steps
            hds_path = os.path.join(model_workspace, 'Garching_model.hds')
            if os.path.exists(hds_path):
                hds = flopy.utils.HeadFile(hds_path)
                
                # Get all available time steps
                available_times = hds.get_kstpkper()
                
                # Save head arrays for all time steps
                head_data = {}
                for kstpkper in available_times:
                    head_array = hds.get_data(kstpkper=kstpkper)
                    head_data[f"stress_period_{kstpkper[1]}"] = head_array[0]  # Only layer 0
                
                # Save to pickle file
                head_file = os.path.join(heads_dir, f"{run_id}_heads.pkl")
                with open(head_file, 'wb') as f:
                    pickle.dump(head_data, f)
                
                results_summary['head_files'].append(head_file)
                results_summary['successful_runs'] += 1
                
                # Save run metadata
                run_metadata = {
                    'run_id': run_id,
                    'parameters': param_set.copy(),
                    'model_workspace': model_workspace,
                    'head_file': head_file,
                    'time_steps': len(available_times)
                }
                results_summary['run_metadata'].append(run_metadata)
                
                # Clean up model workspace (optional - comment out to keep all files)
                # import shutil
                # shutil.rmtree(model_workspace)
                
            else:
                print(f"⚠️  Warning: Head file not found for {run_id}")
                results_summary['failed_runs'] += 1
                
        except Exception as e:
            print(f"❌ Error in {run_id}: {e}")
            results_summary['failed_runs'] += 1
    
    # Save results summary
    summary_file = os.path.join(output_dir, "results_summary.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"✅ Ensemble runs completed:")
    print(f"   Successful: {results_summary['successful_runs']}")
    print(f"   Failed: {results_summary['failed_runs']}")
    print(f"   Results saved in: {output_dir}")
    
    return results_summary

def create_probability_maps(results_summary, target_stress_periods=None, depth_thresholds=[0.0]):
    """
    Create probability maps for groundwater reaching specified depths below surface
    
    Parameters:
    - results_summary: Results from run_ensemble_models
    - target_stress_periods: List of stress periods to analyze (None = all)
    - depth_thresholds: List of depth thresholds below surface (e.g., [0.0, 1.0, 2.0])
                       0.0 = reaches surface, 1.0 = within 1m of surface, etc.
    """
    print(f"📊 Creating probability maps for depth thresholds: {depth_thresholds} meters below surface...")
    
    output_dir = results_summary['output_dir']
    head_files = results_summary['head_files']
    
    # Load topography (from model setup)
    topo_path = os.path.join('Output1_Input2', 'Cell_Top_ly1.csv')
    if os.path.exists(topo_path):
        topography = np.loadtxt(topo_path, delimiter=',')
        print(f"   Loaded topography from: {topo_path}")
    else:
        print("⚠️  Warning: Topography file not found, using placeholder")
        # Use a placeholder - this should be replaced with actual topography
        topography = np.full((183, 163), 480.0)  # Approximate grid size and elevation
    
    # Initialize probability arrays
    if not head_files:
        print("❌ No head files found!")
        return
    
    print(f"📊 Creating probability maps for {len(head_files)} model runs...")
    
    # Load first file to get dimensions and available time steps
    with open(head_files[0], 'rb') as f:
        sample_data = pickle.load(f)
    
    stress_periods = list(sample_data.keys())
    if target_stress_periods is None:
        target_stress_periods = stress_periods
    else:
        target_stress_periods = [sp for sp in target_stress_periods if sp in stress_periods]
    
    print(f"   Analyzing {len(target_stress_periods)} stress periods")
    
    nrows, ncols = sample_data[stress_periods[0]].shape
    print(f"   Grid dimensions: {nrows} x {ncols}")
    
    # Ensure topography matches grid dimensions
    if topography.shape != (nrows, ncols):
        print(f"   Resizing topography from {topography.shape} to {(nrows, ncols)}")
        # Simple resize - in practice, you might want more sophisticated interpolation
        from scipy.ndimage import zoom
        zoom_factors = (nrows / topography.shape[0], ncols / topography.shape[1])
        topography = zoom(topography, zoom_factors, order=1)
    
    # Process each stress period
    probability_maps = {}
    
    for sp in target_stress_periods:
        print(f"   Processing {sp}...")
        
        # Initialize probability maps for each depth threshold
        sp_results = {}
        
        for depth_threshold in depth_thresholds:
            # Initialize counters for this depth threshold
            exceed_count = np.zeros((nrows, ncols))
            valid_count = np.zeros((nrows, ncols))
            
            # Process each model run
            total_files = len(head_files)
            for file_idx, head_file in enumerate(head_files):
                # Print progress for each depth threshold
                if file_idx % max(1, total_files // 10) == 0 or file_idx == total_files - 1:
                    print(f"      📊 Processing {sp} (depth {depth_threshold}m): {file_idx+1}/{total_files} ({100*(file_idx+1)/total_files:.1f}%)")
                
                try:
                    with open(head_file, 'rb') as f:
                        head_data = pickle.load(f)
                    
                    if sp in head_data:
                        head_array = head_data[sp]
                        
                        # Check where head is above (topography - depth_threshold)
                        # For depth_threshold = 1.0: head > (topography - 1.0) means within 1m of surface
                        valid_mask = ~np.isnan(head_array)
                        threshold_elevation = topography - depth_threshold
                        exceed_mask = (head_array > threshold_elevation) & valid_mask
                        
                        exceed_count[exceed_mask] += 1
                        valid_count[valid_mask] += 1
                
                except Exception as e:
                    print(f"⚠️  Error processing {head_file}: {e}")
            
            # Calculate probability (avoid division by zero)
            probability = np.zeros((nrows, ncols))
            valid_cells = valid_count > 0
            probability[valid_cells] = exceed_count[valid_cells] / valid_count[valid_cells]
            
            # Store results for this depth threshold
            threshold_key = f"depth_{depth_threshold:.1f}m"
            sp_results[threshold_key] = {
                'probability': probability,
                'valid_count': valid_count,
                'exceed_count': exceed_count,
                'depth_threshold': depth_threshold,
                'threshold_elevation': threshold_elevation
            }
        
        probability_maps[sp] = sp_results
    
    # Save probability maps
    prob_maps_file = os.path.join(output_dir, "probability_maps.pkl")
    with open(prob_maps_file, 'wb') as f:
        pickle.dump({
            'probability_maps': probability_maps,
            'topography': topography,
            'grid_shape': (nrows, ncols),
            'stress_periods': target_stress_periods,
            'depth_thresholds': depth_thresholds
        }, f)
    
    # Create visualizations
    create_probability_plots(probability_maps, topography, output_dir, depth_thresholds)
    
    print(f"✅ Probability maps created and saved to: {prob_maps_file}")
    return probability_maps

def create_probability_plots(probability_maps, topography, output_dir, depth_thresholds):
    """Create and save probability map visualizations for multiple depth thresholds"""
    
    plots_dir = os.path.join(output_dir, "probability_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot parameters
    from matplotlib import cm
    cmap = cm.get_cmap('RdYlBu_r')  # Red = high probability, Blue = low probability
    
    # Create plots for each stress period and depth threshold
    for sp, sp_data in probability_maps.items():
        
        # Create subplot grid based on number of depth thresholds
        n_thresholds = len(depth_thresholds)
        if n_thresholds <= 2:
            fig, axes = plt.subplots(1, n_thresholds, figsize=(8*n_thresholds, 6))
        else:
            ncols = min(3, n_thresholds)
            nrows = (n_thresholds + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
        
        # Handle single subplot case
        if n_thresholds == 1:
            axes = [axes]
        elif n_thresholds > 1 and hasattr(axes, 'flatten'):
            axes = axes.flatten()
        
        # Plot each depth threshold
        for i, depth_threshold in enumerate(depth_thresholds):
            threshold_key = f"depth_{depth_threshold:.1f}m"
            
            if threshold_key in sp_data:
                data = sp_data[threshold_key]
                probability = data['probability']
                
                # Create probability map
                im = axes[i].imshow(probability, cmap=cmap, vmin=0, vmax=1, origin='upper')
                
                if depth_threshold == 0.0:
                    title = f'P(GW Reaches Surface)\n{sp.replace("_", " ").title()}'
                else:
                    title = f'P(GW Within {depth_threshold:.1f}m of Surface)\n{sp.replace("_", " ").title()}'
                
                axes[i].set_title(title)
                axes[i].set_xlabel('Column')
                axes[i].set_ylabel('Row')
                plt.colorbar(im, ax=axes[i], label='Probability', shrink=0.8)
        
        # Hide unused subplots
        for i in range(len(depth_thresholds), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"probability_map_{sp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create summary statistics plots
    create_summary_plots(probability_maps, plots_dir, depth_thresholds)
    
    print(f"   Probability plots saved to: {plots_dir}")

def create_summary_plots(probability_maps, plots_dir, depth_thresholds):
    """Create summary statistics plots for multiple depth thresholds"""
    
    from matplotlib import cm
    
    # Collect statistics for each depth threshold
    all_stats_data = {}
    
    for depth_threshold in depth_thresholds:
        threshold_key = f"depth_{depth_threshold:.1f}m"
        stats_data = []
        
        for sp, sp_data in probability_maps.items():
            if threshold_key in sp_data:
                data = sp_data[threshold_key]
                prob = data['probability']
                valid_mask = data['valid_count'] > 0
                
                if np.any(valid_mask):
                    stats = {
                        'stress_period': sp,
                        'mean_probability': np.mean(prob[valid_mask]),
                        'max_probability': np.max(prob[valid_mask]),
                        'cells_with_risk': np.sum(prob[valid_mask] > 0.1),  # >10% probability
                        'high_risk_cells': np.sum(prob[valid_mask] > 0.5),  # >50% probability
                        'total_valid_cells': np.sum(valid_mask),
                        'depth_threshold': depth_threshold
                    }
                    stats_data.append(stats)
        
        if stats_data:
            all_stats_data[threshold_key] = pd.DataFrame(stats_data)
    
    if not all_stats_data:
        return
    
    # Create comparison plots for different depth thresholds
    n_thresholds = len(depth_thresholds)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, n_thresholds))
    
    # Plot mean probability comparison
    for i, (threshold_key, stats_df) in enumerate(all_stats_data.items()):
        axes[0, 0].plot(range(len(stats_df)), stats_df['mean_probability'], 
                       'o-', label=threshold_key, color=colors[i], alpha=0.8)
    axes[0, 0].set_title('Mean Probability Comparison')
    axes[0, 0].set_xlabel('Stress Period')
    axes[0, 0].set_ylabel('Mean Probability')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot maximum probability comparison
    for i, (threshold_key, stats_df) in enumerate(all_stats_data.items()):
        axes[0, 1].plot(range(len(stats_df)), stats_df['max_probability'], 
                       'o-', label=threshold_key, color=colors[i], alpha=0.8)
    axes[0, 1].set_title('Maximum Probability Comparison')
    axes[0, 1].set_xlabel('Stress Period')
    axes[0, 1].set_ylabel('Maximum Probability')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot cells at risk (>10% probability)
    for i, (threshold_key, stats_df) in enumerate(all_stats_data.items()):
        axes[1, 0].plot(range(len(stats_df)), stats_df['cells_with_risk'], 
                       'o-', label=threshold_key, color=colors[i], alpha=0.8)
    axes[1, 0].set_title('Number of Cells at Risk (>10% probability)')
    axes[1, 0].set_xlabel('Stress Period')
    axes[1, 0].set_ylabel('Cell Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot percentage at risk
    for i, (threshold_key, stats_df) in enumerate(all_stats_data.items()):
        risk_pct = 100 * stats_df['cells_with_risk'] / stats_df['total_valid_cells']
        axes[1, 1].plot(range(len(stats_df)), risk_pct, 
                       'o-', label=threshold_key, color=colors[i], alpha=0.8)
    axes[1, 1].set_title('Percentage of Cells at Risk (>10% probability)')
    axes[1, 1].set_xlabel('Stress Period')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_file = os.path.join(plots_dir, "probability_summary_comparison.png")
    plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics CSV for each depth threshold
    for threshold_key, stats_df in all_stats_data.items():
        stats_csv_file = os.path.join(plots_dir, f"probability_statistics_{threshold_key}.csv")
        stats_df.to_csv(stats_csv_file, index=False)
    
    print(f"   Summary plots and statistics saved for {len(depth_thresholds)} depth thresholds")

def main():
    """Main function to run the complete analysis"""
    
    print("🌊 GROUNDWATER PROBABILITY MAPPING ANALYSIS")
    print("=" * 50)
    
    # ==================== CONFIGURATION ====================
    dream_csv_path = "dream_GWM_20250807_220325.csv"  # Update this path to your DREAM results
    
    # TESTING: Set to small number for testing (e.g., 4), None for all runs
    max_runs = 4  # Change to None to run all post-convergence parameter sets
    
    # Specific stress periods for analysis (None for all stress periods)
    target_stress_periods = ["stress_period_5", "stress_period_300", "stress_period_800"]  # Set to None for all stress periods
    
    # PROBABILITY THRESHOLDS: Depth below surface to consider "at risk"
    depth_thresholds = [0.0, 1.0, 2.0]  # meters below topography
    # 0.0 = reaches surface, 1.0 = within 1m of surface, 2.0 = within 2m of surface
    # ======================================================
    
    # Check if DREAM results file exists
    if not os.path.exists(dream_csv_path):
        print(f"❌ DREAM results file not found: {dream_csv_path}")
        print("Available CSV files:")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'dream' in f.lower()]
        for f in csv_files:
            print(f"   - {f}")
        return
    
    try:
        # Extract timestamp from DREAM CSV filename for consistent output organization
        dream_timestamp = extract_timestamp_from_filename(dream_csv_path)
        if dream_timestamp:
            print(f"📅 Using DREAM timestamp: {dream_timestamp}")
        else:
            print(f"⚠️  No timestamp found in filename, will generate new timestamp")
        
        # Step 1: Extract post-convergence parameters
        post_conv_params = extract_post_convergence_parameters(dream_csv_path)
        
        # Step 2: Prepare full parameter sets
        full_param_sets = prepare_full_parameter_sets(post_conv_params)
        
        # Apply run limit for testing
        if max_runs is not None and max_runs > 0:
            original_count = len(full_param_sets)
            full_param_sets = full_param_sets[:max_runs]
            print(f"🧪 TESTING MODE: Limited to {len(full_param_sets)} runs (out of {original_count} available)")
        else:
            print(f"📊 FULL ANALYSIS: Running all {len(full_param_sets)} parameter sets")
        
        # Display stress period selection
        if target_stress_periods:
            print(f"📊 Analyzing specific stress periods: {target_stress_periods}")
        else:
            print(f"📊 Analyzing all available stress periods")
        
        # Step 3: Run ensemble of models (with DREAM timestamp)
        results_summary = run_ensemble_models(full_param_sets, timestamp=dream_timestamp)
        
        # Step 4: Create probability maps
        if results_summary['successful_runs'] > 0:
            probability_maps = create_probability_maps(results_summary, target_stress_periods=target_stress_periods, 
                                                     depth_thresholds=depth_thresholds)
            
            print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"   Output directory: {results_summary['output_dir']}")
            print(f"   Successful model runs: {results_summary['successful_runs']}")
            if target_stress_periods:
                print(f"   Probability maps created for stress periods: {target_stress_periods}")
            else:
                print(f"   Probability maps created for all stress periods")
            
        else:
            print("❌ No successful model runs - cannot create probability maps")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
