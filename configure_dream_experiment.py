# -*- coding: utf-8 -*-
"""
Research: A Bayesian Framework to Assess and Create Maps of Groundwater Flooding
Dataset and algorithms for the probabilistic assessment of groundwater flooding occurrence
Adapted for new Garching GWM model (2025)
@author: Pablo Merchán-Rivera

Configuration and setup for DREAM experiment
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

def check_dependencies():
    """Check if all required packages are installed"""
    
    required_packages = ['spotpy', 'flopy', 'numpy', 'pandas', 'matplotlib', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    if missing_packages:
        print(f"\\nPlease install missing packages: {', '.join(missing_packages)}")
        print("Use: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_files():
    """Check if all required data files exist"""
    
    required_files = [
        'Output1_Input2/obs.csv',
        'Output1_Input2/obs_values.csv',
        'Output1_Input2/RCH_data.csv',
        'Output1_Input2/RCH_cellid.csv',
        'MODFLOW-NWT_64.exe',
        'GWM_model_run.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} is missing")
    
    if missing_files:
        print(f"\\nMissing required files: {missing_files}")
        return False
    
    return True

def validate_observation_data():
    """Validate observation data structure"""
    
    try:
        # Check obs.csv
        obs_df = pd.read_csv('Output1_Input2/obs.csv')
        n_obs_points = len(obs_df)
        print(f"✓ Observation points: {n_obs_points}")
        
        # Check obs_values.csv
        obs_vals_df = pd.read_csv('Output1_Input2/obs_values.csv', header=None)
        n_stress_periods, n_obs_cols = obs_vals_df.shape
        print(f"✓ Stress periods: {n_stress_periods}")
        print(f"✓ Observation columns: {n_obs_cols}")
        
        if n_obs_cols != n_obs_points:
            print(f"⚠️  Warning: Number of observation points ({n_obs_points}) doesn't match observation columns ({n_obs_cols})")
        
        # Check for missing values
        missing_vals = obs_vals_df.isnull().sum().sum()
        if missing_vals > 0:
            print(f"⚠️  Warning: {missing_vals} missing values in observation data")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating observation data: {e}")
        return False

def test_model_run():
    """Test if the GWM model can run with default parameters"""
    
    print("\\nTesting model run with default parameters...")
    
    try:
        import GWM_model_run as gwm
        
        # Test parameters (reasonable default values)
        test_params = {
            'hk1': 10.0, 'hk2': 5.0, 'hk3': 1.0, 'hk4': 0.5, 'hk5': 0.1,
            'sy1': 0.2, 'sy2': 0.15, 'sy3': 0.1, 'sy4': 0.08, 'sy5': 0.05,
            'D_Isar': 0.0,
            'Kriv_Isar': 1e-3, 'Kriv_Muhlbach': 1e-4, 'Kriv_Giessen': 1e-4,
            'Kriv_Griesbach': 1e-4, 'Kriv_Schwabinger_Bach': 1e-4, 'Kriv_Wiesackerbach': 1e-4,
            'D_rch1': 1.0, 'D_rch2': 1.0
        }
        
        # Create test output directory
        test_dir = 'test_output'
        os.makedirs(test_dir, exist_ok=True)
        
        # Run model
        model, out_dir = gwm.GWM(**test_params, custom_out_dir=test_dir)
        
        # Test observation extraction
        sim_heads = gwm.get_heads_from_obs_csv(model_ws=test_dir, obs_csv_path='Output1_Input2/obs.csv')
        
        print(f"✓ Model run successful")
        print(f"✓ Simulation output shape: {sim_heads.shape}")
        print(f"✓ Expected flat size: {sim_heads.size}")
        
        # Cleanup
        import shutil
        try:
            shutil.rmtree(test_dir)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def estimate_runtime():
    """Estimate DREAM runtime based on test run"""
    
    print("\\nEstimating DREAM runtime...")
    
    try:
        import time
        import dream_init_new as di
        
        # Estimate single model run time
        start_time = time.time()
        success = test_model_run()
        single_run_time = time.time() - start_time
        
        if success:
            total_runs = di.rep * di.nChains
            estimated_time = single_run_time * total_runs
            
            hours = int(estimated_time // 3600)
            minutes = int((estimated_time % 3600) // 60)
            
            print(f"✓ Single model run time: {single_run_time:.2f} seconds")
            print(f"✓ Total model runs needed: {total_runs}")
            print(f"✓ Estimated total time: {hours}h {minutes}m")
            
            if estimated_time > 86400:  # More than 24 hours
                print("⚠️  This will take more than 24 hours. Consider reducing repetitions.")
        
        return success
        
    except Exception as e:
        print(f"✗ Runtime estimation failed: {e}")
        return False

def create_experiment_config():
    """Create configuration file for the experiment"""
    
    config = {
        'experiment_name': 'DREAM_GWM_Garching',
        'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'total_parameters': 19,
            'parameter_names': [
                'hk1', 'hk2', 'hk3', 'hk4', 'hk5',
                'sy1', 'sy2', 'sy3', 'sy4', 'sy5',
                'D_Isar', 'Kriv_Isar', 'Kriv_Muhlbach', 'Kriv_Giessen',
                'Kriv_Griesbach', 'Kriv_Schwabinger_Bach', 'Kriv_Wiesackerbach',
                'D_rch1', 'D_rch2'
            ],
            'log_transformed': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16]
        },
        'observations': {
            'observation_points': 13,
            'stress_periods': 139,
            'total_observations': 13 * 139
        },
        'dream_settings': {
            'max_repetitions': 10000,
            'chains': 6,
            'convergence_evaluations': 300,
            'random_seed': 246
        }
    }
    
    import json
    with open('dream_experiment_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("✓ Experiment configuration saved to 'dream_experiment_config.json'")

def main():
    """Main setup and validation function"""
    
    print("="*60)
    print("DREAM Algorithm Setup for GWM Garching Model")
    print("="*60)
    
    print("\\n1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\\n2. Checking data files...")
    files_ok = check_data_files()
    
    print("\\n3. Validating observation data...")
    obs_ok = validate_observation_data()
    
    print("\\n4. Testing model run...")
    model_ok = test_model_run()
    
    print("\\n5. Estimating runtime...")
    runtime_ok = estimate_runtime()
    
    print("\\n6. Creating experiment configuration...")
    create_experiment_config()
    
    print("\\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    all_ok = deps_ok and files_ok and obs_ok and model_ok
    
    if all_ok:
        print("✅ All checks passed! You can run the DREAM algorithm.")
        print("\\nTo start the calibration, run:")
        print("   python dream_run_new.py")
        print("\\nTo analyze results afterwards, run:")
        print("   python dream_results_new.py")
    else:
        print("❌ Some checks failed. Please fix the issues above before running DREAM.")
    
    print("\\nFiles created:")
    print("- dream_init_new.py       : Parameter definitions")
    print("- dream_modflow_new.py    : SPOTPY setup class")
    print("- dream_run_new.py        : Main DREAM execution")
    print("- dream_results_new.py    : Results analysis")
    print("- dream_experiment_config.json : Experiment configuration")
    
    return all_ok

if __name__ == "__main__":
    main()
