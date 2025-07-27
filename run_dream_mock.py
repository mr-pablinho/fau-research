# run_dream_mock.py
"""
Test script to run DREAM optimization with the mock model.
This allows rapid testing of the DREAM setup before using the expensive real model.
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
    print("🧪 DREAM OPTIMIZATION - MOCK MODEL TEST")
    print("=" * 60)
    print("This script tests the DREAM setup using a fast mock model")
    print("instead of the expensive groundwater simulation.")
    print()
    
    # For reproducibility, you can set a random seed
    np.random.seed(42)

    # 1. Initialize the mock spotpy setup class
    print("🔧 Initializing mock model setup...")
    spotpy_setup = Mock_GWM_Spotpy_Setup()

    # 2. Initialize the DREAM sampler 
    print("🔧 Initializing DREAM sampler...")
    processing = 'seq'  # Use 'seq' for initial testing, 'mpc' for multiprocessing later
    sampler = spotpy.algorithms.dream(
        spotpy_setup,
        dbname=f"MOCK_{OptimizationConfig.DB_NAME}",
        dbformat=OptimizationConfig.DB_FORMAT,
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
    print("🚀 Starting DREAM sampling with MOCK MODEL...")
    print("   This should run much faster than the real model!")
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
        
        # 5. Analyze the results - Use our own method to avoid SPOTPY database issues
        print("📊 Analyzing results...")
        
        # Read the database file directly to avoid SPOTPY's problematic getdata() method
        try:
            import pandas as pd
            import numpy as np
            
            db_filename = f"MOCK_{OptimizationConfig.DB_NAME}.csv"
            results_df = pd.read_csv(db_filename)
            
            print(f"📈 Number of model runs completed: {len(results_df)}")
            
            if len(results_df) > 0:
                # Find the best result
                like_col = 'like1'
                if like_col in results_df.columns:
                    # Convert to numeric, handling any string issues
                    like_values = pd.to_numeric(results_df[like_col], errors='coerce')
                    best_idx = like_values.idxmax()
                    best_obj_value = like_values.max()
                    
                    print()
                    print("🏆 Best parameter set found:")
                    param_cols = [col for col in results_df.columns if col.startswith('par')]
                    for col in param_cols:
                        if col in results_df.columns:
                            param_name = col[3:]  # Remove 'par' prefix
                            best_val = results_df.loc[best_idx, col]
                            if pd.notna(best_val):
                                print(f"   {param_name}: {best_val:.2f}")
                    
                    print(f"🎯 Best objective value: {best_obj_value:.6f}")
                    
                    # Show some statistics
                    print()
                    print("📊 Objective function statistics:")
                    valid_likes = like_values.dropna()
                    if len(valid_likes) > 0:
                        print(f"   Mean: {valid_likes.mean():.6f}")
                        print(f"   Std:  {valid_likes.std():.6f}")
                        print(f"   Min:  {valid_likes.min():.6f}")
                        print(f"   Max:  {valid_likes.max():.6f}")
                    
                    print()
                    print("✅ MOCK TEST SUCCESSFUL!")
                    print("   Your DREAM setup appears to be working correctly.")
                    print("   You can now try running the real model with confidence.")
                    print()
                    print("💡 To run with the real model:")
                    print("   1. Import GWM_Spotpy_Setup instead of Mock_GWM_Spotpy_Setup")
                    print("   2. Set TEST_MODE = False in config.py for full parameter set")
                    print("   3. Increase repetitions for production runs")
                else:
                    print("⚠️  Could not find objective function column in results")
            else:
                print("❌ No results obtained - check your setup")
                
        except Exception as db_error:
            print(f"⚠️  Could not read database file directly: {db_error}")
            print("   But DREAM optimization completed successfully!")
            print("   The core algorithm is working - just a results reading issue.")
            
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
        
        # Try to get any available results
        try:
            if hasattr(sampler, 'datawriter') and sampler.datawriter is not None:
                results = sampler.getdata()
                print(f"📊 Partial results available: {len(results)} runs completed")
            else:
                print("📊 No results available - sampling may not have started properly")
        except:
            print("📊 Could not retrieve any results")
            
        import traceback
        print()
        print("🔍 Full error details:")
        traceback.print_exc()
