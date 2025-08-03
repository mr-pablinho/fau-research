#!/usr/bin/env python3
"""
Diagnostic script to understand what's happening with DREAM sampling
"""

import numpy as np
import spotpy
import dream_init_new as di
from dream_modflow_new import spot_setup
from datetime import datetime

def diagnose_dream_sampling():
    """Diagnose why DREAM is only running 10 iterations instead of the requested amount"""
    
    print("=== DREAM Sampling Diagnostic ===")
    
    # Check configuration
    print(f"Configuration from dream_init_new:")
    print(f"  di.rep (total iterations): {di.rep}")
    print(f"  di.nChains: {di.nChains}")
    print(f"  di.convEvals: {di.convEvals}")
    print(f"  di.names: {di.names}")
    print(f"  Number of parameters: {len(di.names)}")
    
    # Set up DREAM exactly like in batch runner
    random_state = di.my_seed
    np.random.seed(random_state)
    
    spot_setup_instance = spot_setup(_used_algorithm='dream')
    
    # Test database name
    dbname = f'test_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    print(f"\nCreating DREAM sampler with dbname: {dbname}")
    
    # Create sampler
    sampler = spotpy.algorithms.dream(
        spot_setup_instance, 
        dbname=dbname, 
        dbformat='csv',
        db_precision=np.float32, 
        save_sim=True,
        random_state=random_state
    )
    
    # DREAM parameters - same as in batch runner
    nChains = di.nChains
    convergence_limit = 1.3
    runs_after_convergence = di.convEvals
    epsilon = 0.001
    ato = 6
    nCr = min(2, (nChains - 1) // 2)
    
    print(f"\nDREAM parameters:")
    print(f"  nChains: {nChains}")
    print(f"  nCr: {nCr}")
    print(f"  convergence_limit: {convergence_limit}")
    print(f"  runs_after_convergence: {runs_after_convergence}")
    print(f"  epsilon: {epsilon}")
    print(f"  ato: {ato}")
    
    # Test different batch sizes
    test_batch_sizes = [4, 10, 20]
    
    for batch_size in test_batch_sizes:
        print(f"\n" + "="*50)
        print(f"Testing batch_size = {batch_size}")
        print(f"Expected result: {batch_size} iterations")
        
        test_dbname = f"{dbname}_batch{batch_size}"
        
        try:
            print(f"Calling sampler.sample({batch_size}, nChains={nChains}, ...)")
            
            r_hat = sampler.sample(
                batch_size,
                nChains=nChains,
                convergence_limit=convergence_limit,
                runs_after_convergence=runs_after_convergence,
                eps=epsilon,
                acceptance_test_option=ato,
                nCr=nCr
            )
            
            # Check actual results
            import pandas as pd
            csv_file = f"{test_dbname}.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                actual_iterations = len(df)
                print(f"✅ SUCCESS: {actual_iterations} iterations completed (requested: {batch_size})")
                
                # Clean up
                import os
                os.remove(csv_file)
            else:
                print(f"❌ No CSV file created: {csv_file}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
        # Reset sampler for next test by creating a new one
        sampler = spotpy.algorithms.dream(
            spot_setup_instance, 
            dbname=test_dbname, 
            dbformat='csv',
            db_precision=np.float32, 
            save_sim=True,
            random_state=random_state
        )

if __name__ == "__main__":
    import os
    os.chdir(r"c:\Users\PMR\Documents\Projects\FAU-Garching\fau-research")
    diagnose_dream_sampling()
