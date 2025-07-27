#!/usr/bin/env python3
"""
Analyze DREAM optimization results from saved files
"""

import spotpy
import pandas as pd
import numpy as np

def analyze_hdf5_results(filename):
    """Analyze results from HDF5 file"""
    print(f"📊 Analyzing results from: {filename}")
    
    try:
        # Read the HDF5 database
        results = spotpy.analyser.load_csv_results(filename)
        
        print(f"📈 Total iterations: {len(results)}")
        print(f"📋 Parameters optimized: {list(results.columns[:2])}")  # First 2 columns are parameters
        
        # Get best parameters
        best_idx = np.argmax(results['like1'])  # Best likelihood
        best_params = results.iloc[best_idx]
        
        print("\n🎯 Best Parameters Found:")
        print(f"   hk1: {best_params['par0']:.4f}")
        print(f"   hk3: {best_params['par1']:.4f}")
        print(f"   Objective: {best_params['like1']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return None

def analyze_csv_results(filename):
    """Analyze results from CSV file"""
    print(f"📊 Analyzing results from: {filename}")
    
    try:
        # Read the CSV database
        df = pd.read_csv(filename)
        
        print(f"📈 Total iterations: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Get best parameters
        if 'like1' in df.columns:
            best_idx = np.argmax(df['like1'])
            best_params = df.iloc[best_idx]
            
            print("\n🎯 Best Parameters Found:")
            if 'par0' in df.columns:
                print(f"   hk1: {best_params['par0']:.4f}")
            if 'par1' in df.columns:
                print(f"   hk3: {best_params['par1']:.4f}")
            print(f"   Objective: {best_params['like1']:.4f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return None

if __name__ == "__main__":
    # Try to analyze existing results
    import os
    
    files_to_check = [
        "MOCK_DREAM_GWM_test.hdf5",
        "MOCK_DREAM_GWM_test.csv", 
        "DREAM_GWM_test.csv"
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"\n{'='*50}")
            if filename.endswith('.hdf5'):
                analyze_hdf5_results(filename)
            elif filename.endswith('.csv'):
                analyze_csv_results(filename)
        else:
            print(f"⚠️  File not found: {filename}")
