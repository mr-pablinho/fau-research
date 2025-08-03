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
from datetime import datetime
import dream_init_new as di

def create_timestamped_dir(base_name='dream_plots_gwm'):
    """Create a directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{base_name}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def find_latest_dream_results():
    """Find the most recent DREAM results file"""
    import glob
    csv_files = glob.glob('dream_GWM_*.csv')
    if not csv_files:
        return None
    # Sort by modification time, most recent first
    latest_file = max(csv_files, key=os.path.getctime)
    return latest_file.replace('.csv', '')

def load_dream_results(dbname=None):
    """Load DREAM results from CSV file"""
    if dbname is None:
        # Try to find the latest results file automatically
        dbname = find_latest_dream_results()
        if dbname is None:
            print("No DREAM results files found. Run DREAM algorithm first.")
            return None
        print(f"🔍 Auto-detected latest results: {dbname}.csv")
    
    try:
        results = pd.read_csv(f'{dbname}.csv')
        print(f"✅ Loaded results from: {dbname}.csv")
        print(f"📊 Results shape: {results.shape[0]} iterations × {results.shape[1]} columns")
        
        # Validate that results contain expected parameters
        import dream_init_new as di
        expected_params = [f'par{name}' for name in di.names]
        missing_params = [p for p in expected_params if p not in results.columns]
        if missing_params:
            print(f"⚠️  Warning: Missing expected parameters: {missing_params}")
        else:
            print(f"✅ All expected parameters found: {di.names}")
            
        # Show first few iterations info
        if 'like1' in results.columns:
            best_like_idx = results['like1'].idxmax()
            print(f"📈 Best likelihood: {results.loc[best_like_idx, 'like1']:.2f} (iteration {best_like_idx + 1})")
        
        return results
    except FileNotFoundError:
        print(f"❌ Results file {dbname}.csv not found.")
        # Show available files
        import glob
        csv_files = glob.glob('dream_GWM_*.csv')
        if csv_files:
            print(f"Available DREAM result files:")
            for f in sorted(csv_files, key=os.path.getctime, reverse=True):
                mod_time = datetime.fromtimestamp(os.path.getctime(f)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  - {f} (modified: {mod_time})")
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

# === Additional Visualizations (from old version) ===
def plot_prior_posterior_histograms(prior_samples, posterior_samples, param_names, param_distros, save_dir='dream_plots_gwm'):
    """Overlay prior and posterior histograms for each parameter"""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    bars = 15
    alpha = 0.60
    for i, param_name in enumerate(param_names):
        plt.figure(f'histogram_prior_posterior_{param_name}', figsize=(3,3))
        # Prior
        plt.hist(prior_samples[:,i], bars, alpha=alpha+0.2, color='orange', label="Prior distribution",
                 range=(param_distros[i].minbound, param_distros[i].maxbound), density=False, stacked=True)
        # Posterior
        plt.hist(posterior_samples[:,i], bars, alpha=alpha, color='blue', label="Posterior distribution",
                 range=(param_distros[i].minbound, param_distros[i].maxbound), density=False, stacked=True)
        plt.ylabel('Frequency', fontsize=12)
        plt.xlim(min(prior_samples[:,i].min(), posterior_samples[:,i].min()),
                 max(prior_samples[:,i].max(), posterior_samples[:,i].max()))
        plt.title(f'Histogram - {param_name}')
        plt.legend(edgecolor='black', fancybox=False, fontsize=8.5,
                    borderpad=0.8, handletextpad=0.9, labelspacing=0.65)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/histogram_prior_posterior_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_predictive_uncertainty(sims_all, observations, obs_wells, save_dir='dream_plots_gwm'):
    """Plot predictive mean ± std bands for model outputs vs. observations"""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    sp_coord = np.arange(sims_all.shape[2])
    sims_all_exp = np.mean(sims_all, axis=0)
    sims_all_std = np.std(sims_all, axis=0)
    colors = ['black', 'deepskyblue', 'red']
    for i, well in enumerate(obs_wells):
        plt.figure(f'uncertainty_{well}', figsize=[6,3])
        plt.title(f'{well}', loc='left', fontsize=15)
        plt.plot(sp_coord, sims_all_exp[i,:], color='blue', alpha=1, label=r'$\mu_o$')
        plt.fill_between(sp_coord, sims_all_exp[i,:], sims_all_exp[i,:] - sims_all_std[i,:],
                          color=colors[1], alpha=0.55, edgecolor='None', label=r'[$\mu_o$ + $\sigma_o$, $\mu_o$ - $\sigma_o$]')
        plt.fill_between(sp_coord, sims_all_exp[i,:], sims_all_exp[i,:] + sims_all_std[i,:],
                          color=colors[1], alpha=0.55, edgecolor='None')
        plt.plot(sp_coord, observations[i,:], linestyle=':', color=colors[0], alpha=0.95, linewidth=2.5, label='Observations')
        plt.ylabel('Groundwater head [m a.s.l]', fontsize=12)
        plt.xlabel('Stress period', fontsize=12)
        plt.xlim(0, sims_all.shape[2]-1)
        plt.legend(edgecolor='black', fancybox=False, fontsize=8.5,
                    borderpad=0.8, handletextpad=0.9, labelspacing=0.65)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/uncertainty_{well}.png', dpi=300, bbox_inches='tight')
        plt.close()
def plot_violin_prior_posterior(sims_all, pro_prior, obs_wells, save_dir='dream_plots_gwm'):
    pass

def plot_prior_posterior_density(prior_samples, posterior_samples, param_names, save_dir='dream_plots_gwm'):
    """Plot density histograms and KDE curves for prior and posterior distributions."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)
    for i, param_name in enumerate(param_names):
        plt.figure(f'density_prior_posterior_{param_name}', figsize=(4,3))
        # Density histograms
        plt.hist(prior_samples[:,i], bins=30, density=True, alpha=0.4, color='orange', label="Prior (hist)")
        plt.hist(posterior_samples[:,i], bins=30, density=True, alpha=0.4, color='blue', label="Posterior (hist)")
        # KDE curves
        sns.kdeplot(prior_samples[:,i], color='orange', lw=2, label="Prior (KDE)")
        sns.kdeplot(posterior_samples[:,i], color='blue', lw=2, label="Posterior (KDE)")
        plt.ylabel('Density', fontsize=12)
        plt.xlabel(param_name)
        plt.title(f'Density & KDE - {param_name}')
        plt.legend(edgecolor='black', fancybox=False, fontsize=8.5,
                    borderpad=0.8, handletextpad=0.9, labelspacing=0.65)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/density_prior_posterior_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
def main(dbname=None):
    """Main function to run complete DREAM results analysis"""

    print("=" * 60)
    print("DREAM Results Analysis for GWM Model")
    print("=" * 60)
    
    # Ensure di is imported in local scope
    import dream_init_new as di

    # Load results (auto-detect latest if dbname not provided)
    results = load_dream_results(dbname)
    
    # Store the actual dbname used for later reference
    if dbname is None:
        dbname = find_latest_dream_results()
    if results is None:
        return

    # Analyze convergence
    param_stats, results_converged = analyze_convergence(results, di.names)

    if param_stats is None:
        print("Could not analyze convergence. Check your results file.")
        return

    print("\nGenerating visualizations...")
    
    # Create timestamped directory for plots
    save_dir = create_timestamped_dir('dream_plots_gwm')
    print(f"Saving plots to: {save_dir}")

    # Generate all plots
    plot_parameter_traces(results, di.names, save_dir)
    plot_parameter_distributions(results_converged, di.names, param_stats, save_dir)
    plot_likelihood_evolution(results, save_dir)
    plot_parameter_correlation(results_converged, di.names, save_dir)

    # Improved prior/posterior density+KDE plots (now after results_converged is defined)
    try:
        param_distros = di.param_distros
        numParams = len(param_distros)
        rep = di.rep
        np.random.seed(di.my_seed)
        prior_samples = np.zeros((rep, numParams))
        for i in range(rep):
            gen_samples = spotpy.parameter.generate(param_distros)
            for j in range(numParams):
                prior_samples[i,j] = gen_samples[j][0]
        posterior_samples = results_converged[[f'par{n}' for n in di.names]].values
        plot_prior_posterior_density(prior_samples, posterior_samples, di.names, save_dir)
    except Exception as e:
        print(f"[Optional] Could not plot prior/posterior density+KDE: {e}")

    # Generate summary table
    generate_summary_table(param_stats, save_dir)

    print("\nDREAM results analysis completed!")
    print(f"Check the '{save_dir}' directory for all generated plots and summary.")

    # === Additional visualizations (if data available) ===
    # 1. Overlayed prior/posterior histograms
    # 2. Predictive uncertainty plots
    # 3. Violin plots for prior/posterior predictive
    try:
        # Load prior samples and parameter distributions
        import dream_init_new as di
        # Generate prior samples (same shape as posterior)
        param_distros = di.param_distros
        numParams = len(param_distros)
        rep = di.rep
        np.random.seed(di.my_seed)
        prior_samples = np.zeros((rep, numParams))
        for i in range(rep):
            gen_samples = spotpy.parameter.generate(param_distros)
            for j in range(numParams):
                prior_samples[i,j] = gen_samples[j][0]
        # Posterior samples (after burn-in)
        posterior_samples = results_converged[[f'par{n}' for n in di.names]].values
        plot_prior_posterior_histograms(prior_samples, posterior_samples, di.names, param_distros)
    except Exception as e:
        print(f"[Optional] Could not plot prior/posterior histograms: {e}")

    # Predictive uncertainty and violin plots (if model outputs and observations available)
    try:
        # Example: load model outputs and observations (user must adapt paths/format as needed)
        # Here, we assume model outputs are in the DREAM results CSV file after parameter columns
        # and observations in './Output1_Input2/obs_values.csv'
        import pandas as pd
        data_results = pd.read_csv(f'{dbname}.csv')
        # Assume parameter columns are first len(di.names), outputs next, likelihood last
        numParams = len(di.names)
        # Example: outputs shape (n_samples, n_wells * n_times)
        # User must adapt n_wells and n_times to their case - updated for new GWM model
        n_wells = 13  # Updated: 13 observation points in new model
        n_times = 139  # Updated: 139 time steps in new model
        sims_all = data_results.iloc[:, numParams:numParams+n_wells*n_times].to_numpy().reshape(-1, n_wells, n_times)
        # Observations
        obs_values = pd.read_csv('./Output1_Input2/obs_values.csv', header=None).to_numpy().reshape(n_wells, n_times)
        # Use actual observation point IDs from the GWM model
        obs_point_ids = [8988, 10131, 9632, 13230, 14210, 15355, 16013, 15997, 16988, 18466, 18473, 19779, 19928]
        obs_labels = [f'ObsPoint_{obs_id}' for obs_id in obs_point_ids]
        plot_predictive_uncertainty(sims_all, obs_values, obs_labels)
        # Prior predictive (user must adapt path/format)
        # Example: load prior predictive from npy file if available
        import os
        prior_pred_path = './inputs/prior/stored_sim_prior_random_from0_to250_seed1234.npy'
        if os.path.exists(prior_pred_path):
            pro_prior = np.load(prior_pred_path, allow_pickle=True)[:,1]
            # This is just an example; user must adapt to match shape (n_prior, n_wells, n_times)
            pro_prior = pro_prior.reshape(-1, n_wells, n_times)
            plot_violin_prior_posterior(sims_all, pro_prior, obs_labels)
    except Exception as e:
        print(f"[Optional] Could not plot predictive uncertainty/violin plots: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        dbname = sys.argv[1]
        print(f"Using specified database: {dbname}")
        main(dbname)
    else:
        print("No database specified - will auto-detect latest results")
        main()