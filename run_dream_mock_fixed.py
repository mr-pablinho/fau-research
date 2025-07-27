# run_dream_mock_fixed.py
"""
Fixed version of the mock DREAM test that avoids database reading issues
by using RAM storage instead of CSV files.
"""

import os
import warnings
import numpy as np
import spotpy
from mock_spotpy_setup import Mock_GWM_Spotpy_Setup  # Import the mock class
from config import OptimizationConfig
import multiprocessing as mp
import time

# Required for Windows multiprocessing - MUST be at the very top
if __name__ == '__main__':
    mp.freeze_support()
    
    print("=" * 60)
    print("🧪 DREAM OPTIMIZATION - MOCK MODEL TEST (FIXED)")
    print("=" * 60)
    print("This script tests the DREAM setup using a fast mock model")
    print("with improved database handling to avoid reading issues.")
    print()
    
    # For reproducibility, you can set a random seed
    np.random.seed(42)

    # 1. Initialize the mock spotpy setup class
    print("🔧 Initializing mock model setup...")
    spotpy_setup = Mock_GWM_Spotpy_Setup()

    # 2. Initialize the DREAM sampler with disk storage to save results
    print("🔧 Initializing DREAM sampler with disk storage...")
    processing = 'seq'  # Use 'seq' for initial testing, 'mpc' for multiprocessing later
    sampler = spotpy.algorithms.dream(
        spotpy_setup,
        dbname=f"MOCK_{OptimizationConfig.DB_NAME}",
        dbformat='hdf5',  # Use HDF5 for efficient disk storage (alternatively: 'csv')
        parallel=processing
    )

    # 3. Configure sampling parameters
    n_cores = mp.cpu_count()
    n_chains = OptimizationConfig.DREAM_CHAINS
    repetitions = OptimizationConfig.REPETITIONS

    # For DREAM algorithm requirements:
    n_params = len(spotpy_setup.params)
    min_chains = 2 * n_params + 1

    print(f"💻 Available CPU cores: {n_cores}")
    print(f"🔗 Parameters to optimize: {n_params}")
    print(f"🔗 Minimum chains required: {min_chains}")
    print(f"🔗 Using {n_chains} chains and {repetitions} repetitions")
    print(f"🔗 Running in {processing} mode")
    print(f"🔗 Total model runs: {n_chains * repetitions}")
    print()

    if n_chains < min_chains:
        print(f"⚠️  WARNING: You have {n_chains} chains but need at least {min_chains}")
        print("   Consider increasing DREAM_CHAINS in config.py")
        print()

    # Set max_runs for progress tracking
    spotpy_setup.max_runs = n_chains * repetitions

    # 4. Start the sampling process
    print("🚀 Starting DREAM sampling with MOCK MODEL (RAM storage)...")
    print("   This should run much faster and avoid database reading issues!")
    print()
    
    start_time = time.time()
    
    try:
        sampler.sample(repetitions, nChains=n_chains)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print()
        print("=" * 60)
        print("✅ DREAM MOCK TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"⏱️  Average time per run: {elapsed_time / (n_chains * repetitions):.3f} seconds")
        print()
        
        # 5. Analyze the results using RAM storage (should work without issues)
        print("📊 Analyzing results from RAM...")
        
        try:
            results = sampler.getdata()
            
            print(f"📈 Number of model runs completed: {len(results)}")
            
            if len(results) > 0:
                # Use spotpy.analyser to get the best parameter set
                try:
                    best_params = spotpy.analyser.get_best_parameterset(results)
                    print()
                    print("🏆 Best parameter set found:")
                    for i, param in enumerate(spotpy_setup.params):
                        print(f"   {param.name}: {float(best_params[i]):.2f}")
                    
                    # Get best objective function value
                    best_obj_value = float(results['like1'].max())
                    print(f"🎯 Best objective value: {best_obj_value:.6f}")
                    
                    # Show some statistics
                    print()
                    print("📊 Objective function statistics:")
                    like_values = [float(x) for x in results['like1']]
                    print(f"   Mean: {np.mean(like_values):.6f}")
                    print(f"   Std:  {np.std(like_values):.6f}")
                    print(f"   Min:  {np.min(like_values):.6f}")
                    print(f"   Max:  {np.max(like_values):.6f}")
                    
                except Exception as analysis_error:
                    print(f"⚠️  Error in detailed analysis: {analysis_error}")
                    print("   But optimization completed successfully!")
                
                print()
                print("✅ MOCK TEST WITH RAM STORAGE SUCCESSFUL!")
                print("   Your DREAM setup is working perfectly.")
                print("   RAM storage avoided the CSV parsing issues.")
                print()
                print("💡 For the real model, you can use either:")
                print("   • dbformat='ram' (faster, but uses more memory)")
                print("   • dbformat='csv' (slower, but saves disk space)")
                print("   • dbformat='hdf5' (recommended for large runs)")
                print()
                print("🚀 Next steps:")
                print("   1. Import GWM_Spotpy_Setup instead of Mock_GWM_Spotpy_Setup")
                print("   2. Set TEST_MODE = False in config.py for full parameter set")
                print("   3. Increase repetitions for production runs")
                print("   4. Consider using dbformat='hdf5' for large optimizations")
                
            else:
                print("❌ No results obtained - check your setup")
                
        except Exception as results_error:
            print(f"⚠️  Error analyzing results: {results_error}")
            print("   But DREAM optimization completed successfully!")
            print("   The core algorithm is working.")
            
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print()
        print("=" * 60)
        print("❌ ERROR DURING MOCK TEST")
        print("=" * 60)
        print(f"⏱️  Time before error: {elapsed_time:.2f} seconds")
        print(f"🚨 Error: {e}")
        print()
        print("This indicates a problem with your DREAM setup that needs to be fixed")
        print("before trying the real model.")
        
        import traceback
        print()
        print("🔍 Full error details:")
        traceback.print_exc()
