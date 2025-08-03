# -*- coding: utf-8 -*-
"""
Test script to demonstrate the new timestamped output functionality
@author: Pablo Merchán-Rivera

This script shows how the modified DREAM files now save outputs with datetime stamps
"""

import os
from datetime import datetime
import dream_init_new as di

def show_timestamp_functionality():
    """Demonstrate the timestamped output functionality"""
    
    print("=== DREAM Timestamped Output Functionality ===\n")
    
    # Show the timestamp that would be used
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Current timestamp format: {current_timestamp}")
    
    # Show how the flag is created in dream_init_new.py
    print(f"\nDREAM run identifier: {di.flag}")
    
    # Show what the output files would be named
    print(f"\nWith timestamp {current_timestamp}, the output files will be:")
    print(f"  - CSV results file: dream_GWM_{current_timestamp}.csv")
    print(f"  - Log file: logs/log_dream_{current_timestamp}.txt")
    print(f"  - Plots directory: dream_plots_{current_timestamp}/")
    print(f"  - Parameter summary: dream_parameters_summary_{current_timestamp}.csv")
    print(f"  - Parameter traces plot: dream_parameter_traces_{current_timestamp}.png")
    print(f"  - Likelihood evolution: dream_objective_evolution_{current_timestamp}.png")
    print(f"  - Parameter distributions: dream_parameter_distributions_{current_timestamp}.png")
    print(f"  - Parameter correlations: dream_parameter_correlation_{current_timestamp}.png")
    
    print(f"\n=== Benefits of Timestamped Outputs ===")
    print("✓ Each DREAM run creates unique output files")
    print("✓ No risk of overwriting previous results")
    print("✓ Easy to compare different runs")
    print("✓ Organized file management")
    print("✓ Automatic identification of run sequence")
    
    print(f"\n=== How to Use ===")
    print("1. Run: python dream_run_new.py")
    print("   - Creates timestamped CSV and log files")
    print("2. Run: python dream_results_new.py")
    print("   - Automatically finds most recent results file")
    print("   - Creates timestamped plots and analysis")
    
    # Check if there are any existing timestamped files
    csv_files = [f for f in os.listdir('.') if f.startswith('dream_GWM_') and f.endswith('.csv')]
    if csv_files:
        print(f"\n=== Existing DREAM Result Files ===")
        for file in sorted(csv_files):
            timestamp = file.replace('dream_GWM_', '').replace('.csv', '')
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                print(f"  - {file} (Created: {dt.strftime('%Y-%m-%d %H:%M:%S')})")
            except ValueError:
                print(f"  - {file} (Legacy format)")
    else:
        print(f"\nNo existing DREAM result files found.")

if __name__ == "__main__":
    show_timestamp_functionality()
