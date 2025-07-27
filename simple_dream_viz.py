#!/usr/bin/env python3
"""
Simple visualization script for DREAM optimization results
Shows parameter traces, objective evolution and summary statistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_dream_results(csv_file="DREAM_GWM_test.csv"):
    """Visualize DREAM optimization results"""
    
    print(f"📊 Loading results from: {csv_file}")
    
    # Load results
    try:
        results = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(results)} optimization iterations")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Identify parameter columns
    param_cols = [col for col in results.columns if col.startswith('par')]
    print(f"📋 Parameters found: {param_cols}")
    
    # Create plots
    print("📈 Creating visualization plots...")
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Parameter traces
    n_params = len(param_cols)
    for i, param in enumerate(param_cols):
        ax = plt.subplot(3, n_params, i + 1)
        plt.plot(results[param], 'b-', alpha=0.7, linewidth=1)
        
        # Add best value line
        best_value = results.loc[results['like1'].idxmax(), param]
        plt.axhline(y=best_value, color='red', linestyle='--', 
                   label=f'Best: {best_value:.2f}')
        
        plt.title(f'Parameter Trace: {param}')
        plt.ylabel(param)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if i == n_params - 1:  # Last parameter
            plt.xlabel('Iteration')
    
    # 2. Parameter distributions
    for i, param in enumerate(param_cols):
        ax = plt.subplot(3, n_params, n_params + i + 1)
        plt.hist(results[param], bins=15, alpha=0.7, color=f'C{i}', edgecolor='black')
        
        # Add best value line
        best_value = results.loc[results['like1'].idxmax(), param]
        plt.axvline(x=best_value, color='red', linestyle='--', linewidth=2,
                   label=f'Best: {best_value:.2f}')
        
        # Add mean line
        mean_value = results[param].mean()
        plt.axvline(x=mean_value, color='blue', linestyle=':', linewidth=2,
                   label=f'Mean: {mean_value:.2f}')
        
        plt.title(f'Distribution: {param}')
        plt.xlabel(param)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Objective function evolution (spans both columns)
    ax = plt.subplot(3, 1, 3)
    
    # Plot objective values
    plt.plot(results['like1'], 'b-', alpha=0.7, linewidth=1, label='Objective')
    
    # Plot running maximum
    running_max = np.maximum.accumulate(results['like1'])
    plt.plot(running_max, 'g-', linewidth=2, label='Running Maximum')
    
    # Add best value line
    best_value = results['like1'].max()
    plt.axhline(y=best_value, color='red', linestyle='--', 
               label=f'Best: {best_value:.6f}')
    
    plt.title('Objective Function Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function (Log-Likelihood)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dream_results_summary.png', dpi=300, bbox_inches='tight')
    print("💾 Summary plot saved to: dream_results_summary.png")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📋 DREAM OPTIMIZATION SUMMARY")
    print("="*60)
    
    print(f"📊 Total Iterations: {len(results)}")
    print(f"🎯 Best Objective: {results['like1'].max():.6f}")
    print(f"📈 Objective Range: {results['like1'].min():.6f} to {results['like1'].max():.6f}")
    
    # Best parameters
    best_idx = results['like1'].idxmax()
    print(f"\n🏆 BEST PARAMETER SET (Iteration {best_idx}):")
    for param in param_cols:
        best_val = results.loc[best_idx, param]
        mean_val = results[param].mean()
        std_val = results[param].std()
        print(f"   {param}: {best_val:.4f} (mean: {mean_val:.4f} ± {std_val:.4f})")
    
    # Convergence analysis
    running_max = np.maximum.accumulate(results['like1'])
    improvements = np.where(np.diff(running_max) > 0)[0]
    if len(improvements) > 0:
        last_improvement = improvements[-1] + 1
        print(f"\n📈 Last Improvement: Iteration {last_improvement}")
        convergence_pct = (len(results) - last_improvement) / len(results) * 100
        print(f"📊 Convergence: {convergence_pct:.1f}% of runs after last improvement")
    
    print("="*60)
    
    # Create correlation plot if multiple parameters
    if len(param_cols) >= 2:
        print("🔗 Creating parameter correlation plot...")
        
        plt.figure(figsize=(10, 5))
        
        # Correlation scatter plot
        param1, param2 = param_cols[0], param_cols[1]
        scatter = plt.scatter(results[param1], results[param2], 
                            c=results['like1'], cmap='viridis', alpha=0.7)
        
        # Add best point
        best_idx = results['like1'].idxmax()
        plt.scatter(results.loc[best_idx, param1], 
                   results.loc[best_idx, param2],
                   color='red', s=100, marker='*', label='Best', zorder=5)
        
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'{param1} vs {param2} (colored by objective)')
        plt.colorbar(scatter, label='Objective Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_correlation.png', dpi=300, bbox_inches='tight')
        print("💾 Correlation plot saved to: parameter_correlation.png")
        plt.show()

def main():
    """Main function"""
    # Look for results files
    possible_files = [
        "DREAM_GWM_test.csv",
        "DREAM_GWM_run.csv",
        "MOCK_DREAM_GWM_test.csv"
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
    visualize_dream_results(results_file)

if __name__ == "__main__":
    main()
