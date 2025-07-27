#!/usr/bin/env python3
"""
Comprehensive visualization script for DREAM optimization results
Analyzes and plots parameter traces, convergence, and distributions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DREAMResultsVisualizer:
    def __init__(self, csv_file):
        """Initialize with DREAM results CSV file"""
        self.csv_file = csv_file
        self.results = None
        self.param_names = []
        self.load_results()
    
    def load_results(self):
        """Load and parse DREAM results from CSV"""
        print(f"📊 Loading results from: {self.csv_file}")
        
        try:
            # Load the CSV file
            self.results = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.results)} optimization iterations")
            
            # Identify parameter columns (usually start with 'par')
            self.param_names = [col for col in self.results.columns if col.startswith('par')]
            print(f"📋 Parameters found: {self.param_names}")
            
            # Print basic statistics
            print(f"🎯 Best objective value: {self.results['like1'].max():.6f}")
            print(f"📈 Objective range: {self.results['like1'].min():.6f} to {self.results['like1'].max():.6f}")
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            raise
    
    def plot_parameter_traces(self, save_path=None):
        """Plot parameter evolution over iterations"""
        print("📈 Creating parameter trace plots...")
        
        n_params = len(self.param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        
        if n_params == 1:
            axes = [axes]
        
        for i, param in enumerate(self.param_names):
            axes[i].plot(self.results[param], alpha=0.7, linewidth=1)
            axes[i].set_ylabel(f'{param}')
            axes[i].set_title(f'Parameter Trace: {param}')
            axes[i].grid(True, alpha=0.3)
            
            # Add best value line
            best_value = self.results.loc[self.results['like1'].idxmax(), param]
            axes[i].axhline(y=best_value, color='red', linestyle='--', 
                           label=f'Best: {best_value:.2f}')
            axes[i].legend()
        
        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Parameter traces saved to: {save_path}")
        
        plt.show()
    
    def plot_objective_evolution(self, save_path=None):
        """Plot objective function evolution"""
        print("🎯 Creating objective function evolution plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Raw objective values
        ax1.plot(self.results['like1'], alpha=0.7, linewidth=1, color='blue')
        ax1.set_ylabel('Objective Function (Log-Likelihood)')
        ax1.set_title('Objective Function Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Add best value line
        best_value = self.results['like1'].max()
        ax1.axhline(y=best_value, color='red', linestyle='--', 
                   label=f'Best: {best_value:.6f}')
        ax1.legend()
        
        # Running maximum (convergence)
        running_max = np.maximum.accumulate(self.results['like1'])
        ax2.plot(running_max, color='green', linewidth=2, label='Running Maximum')
        ax2.set_ylabel('Best Objective So Far')
        ax2.set_xlabel('Iteration')
        ax2.set_title('Convergence Plot (Running Maximum)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Objective evolution saved to: {save_path}")
        
        plt.show()
    
    def plot_parameter_distributions(self, save_path=None):
        """Plot parameter posterior distributions"""
        print("📊 Creating parameter distribution plots...")
        
        n_params = len(self.param_names)
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for i, param in enumerate(self.param_names):
            # Histogram
            axes[i].hist(self.results[param], bins=20, alpha=0.7, density=True, 
                        color=f'C{i}', edgecolor='black')
            
            # Add best value line
            best_value = self.results.loc[self.results['like1'].idxmax(), param]
            axes[i].axvline(x=best_value, color='red', linestyle='--', linewidth=2,
                           label=f'Best: {best_value:.2f}')
            
            # Add mean line
            mean_value = self.results[param].mean()
            axes[i].axvline(x=mean_value, color='blue', linestyle=':', linewidth=2,
                           label=f'Mean: {mean_value:.2f}')
            
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Posterior Distribution: {param}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Parameter distributions saved to: {save_path}")
        
        plt.show()
    
    def plot_parameter_correlation(self, save_path=None):
        """Plot parameter correlation matrix and scatter plots"""
        if len(self.param_names) < 2:
            print("⚠️  Need at least 2 parameters for correlation analysis")
            return
        
        print("🔗 Creating parameter correlation plots...")
        
        # Correlation matrix
        param_data = self.results[self.param_names]
        correlation_matrix = param_data.corr()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1)
        ax1.set_title('Parameter Correlation Matrix')
        
        # Scatter plot for first two parameters
        param1, param2 = self.param_names[0], self.param_names[1]
        scatter = ax2.scatter(self.results[param1], self.results[param2], 
                             c=self.results['like1'], cmap='viridis', alpha=0.7)
        ax2.set_xlabel(param1)
        ax2.set_ylabel(param2)
        ax2.set_title(f'{param1} vs {param2} (colored by objective)')
        
        # Add best point
        best_idx = self.results['like1'].idxmax()
        ax2.scatter(self.results.loc[best_idx, param1], 
                   self.results.loc[best_idx, param2],
                   color='red', s=100, marker='*', label='Best')
        ax2.legend()
        
        # Colorbar
        plt.colorbar(scatter, ax=ax2, label='Objective Function')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Parameter correlation saved to: {save_path}")
        
        plt.show()
    
    def plot_chain_diagnostics(self, save_path=None):
        """Plot DREAM chain diagnostics if chain information is available"""
        print("⛓️  Creating chain diagnostic plots...")
        
        # Check if chain information is available
        if 'chain' not in self.results.columns:
            print("⚠️  No chain information found in results - skipping chain diagnostics")
            return
        
        n_chains = self.results['chain'].nunique()
        n_params = len(self.param_names)
        
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        if n_params == 1:
            axes = [axes]
        
        # Use a colormap instead of indexing
        colors = plt.cm.Set3(np.linspace(0, 1, n_chains))
        
        for i, param in enumerate(self.param_names):
            for j, chain_id in enumerate(sorted(self.results['chain'].unique())):
                chain_data = self.results[self.results['chain'] == chain_id]
                axes[i].plot(chain_data.index, chain_data[param], 
                           alpha=0.7, linewidth=1, color=colors[j],
                           label=f'Chain {chain_id}')
            
            axes[i].set_ylabel(param)
            axes[i].set_title(f'Chain Traces: {param}')
            axes[i].grid(True, alpha=0.3)
            if i == 0:  # Only show legend on first subplot
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Chain diagnostics saved to: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("📋 DREAM OPTIMIZATION SUMMARY REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"📊 Total Iterations: {len(self.results)}")
        print(f"🎯 Best Objective: {self.results['like1'].max():.6f}")
        print(f"📈 Objective Improvement: {self.results['like1'].max() - self.results['like1'].min():.6f}")
        
        # Best parameters
        best_idx = self.results['like1'].idxmax()
        print(f"\n🏆 BEST PARAMETER SET (Iteration {best_idx}):")
        for param in self.param_names:
            best_val = self.results.loc[best_idx, param]
            mean_val = self.results[param].mean()
            std_val = self.results[param].std()
            print(f"   {param}: {best_val:.4f} (mean: {mean_val:.4f} ± {std_val:.4f})")
        
        # Convergence analysis
        running_max = np.maximum.accumulate(self.results['like1'])
        last_improvement = np.where(np.diff(running_max) > 0)[0]
        if len(last_improvement) > 0:
            last_improvement_iter = last_improvement[-1] + 1
            print(f"\n📈 Last Improvement: Iteration {last_improvement_iter}")
            convergence_pct = (len(self.results) - last_improvement_iter) / len(self.results) * 100
            print(f"📊 Convergence: {convergence_pct:.1f}% of runs after last improvement")
        
        # Parameter statistics
        print(f"\n📊 PARAMETER STATISTICS:")
        param_stats = self.results[self.param_names].describe()
        print(param_stats)
        
        print("="*60)
    
    def create_all_plots(self, output_dir="dream_plots"):
        """Create all visualization plots and save them"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        print(f"📁 Creating plots in directory: {output_dir}")
        
        # Generate all plots
        self.plot_parameter_traces(f"{output_dir}/parameter_traces.png")
        self.plot_objective_evolution(f"{output_dir}/objective_evolution.png")
        self.plot_parameter_distributions(f"{output_dir}/parameter_distributions.png")
        
        if len(self.param_names) >= 2:
            self.plot_parameter_correlation(f"{output_dir}/parameter_correlation.png")
        
        self.plot_chain_diagnostics(f"{output_dir}/chain_diagnostics.png")
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\n✅ All visualizations completed! Check the '{output_dir}' folder.")

def main():
    """Main function to run visualization"""
    # Look for DREAM results files
    possible_files = [
        "DREAM_GWM_test.csv",
        # "DREAM_GWM_run.csv",
        # "MOCK_DREAM_GWM_test.csv"
    ]
    
    results_file = None
    for file in possible_files:
        if os.path.exists(file) and os.path.getsize(file) > 0:
            results_file = file
            break
    
    if results_file is None:
        print("❌ No DREAM results files found!")
        print("Looking for:", possible_files)
        return
    
    print(f"🎯 Using results file: {results_file}")
    
    # Create visualizer and run all plots
    try:
        visualizer = DREAMResultsVisualizer(results_file)
        visualizer.create_all_plots()
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
