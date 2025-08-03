#!/usr/bin/env python3
"""
Test script to verify DREAM can initialize properly
"""

import numpy as np
import spotpy
import dream_init_new as di
from dream_modflow_new import spot_setup
from datetime import datetime

def test_dream_initialization():
    """Test if DREAM can initialize without the chain error"""
    
    print("=== Testing DREAM Initialization ===")
    
    # Set up DREAM exactly like in the batch runner
    random_state = di.my_seed
    np.random.seed(random_state)
    
    spot_setup_instance = spot_setup(_used_algorithm='dream')
    
    # Test database name
    dbname = f'test_dream_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    print(f"Creating DREAM sampler...")
    print(f"  Parameters: {di.names}")
    print(f"  Number of parameters: {len(di.names)}")
    print(f"  Chains: {di.nChains}")
    
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
    # Dynamic nCr calculation
    nCr = min(2, (nChains - 1) // 2)
    
    print(f"  DREAM parameters:")
    print(f"    nChains: {nChains}")
    print(f"    nCr: {nCr}")
    print(f"    nCr constraint satisfied: {nChains >= 2*nCr + 1}")
    print(f"    convergence_limit: {convergence_limit}")
    print(f"    epsilon: {epsilon}")
    print(f"    ato: {ato}")
    
    try:
        print(f"\nAttempting to run 1 iteration to test initialization...")
        
        # Try to run just 1 iteration to test
        r_hat = sampler.sample(
            1,  # Just 1 iteration for test
            nChains=nChains,
            convergence_limit=convergence_limit,
            runs_after_convergence=runs_after_convergence,
            eps=epsilon,
            acceptance_test_option=ato,
            nCr=nCr
        )
        
        print(f"✅ SUCCESS! DREAM initialized and ran 1 iteration")
        print(f"   R-hat convergence diagnostic: {r_hat}")
        
        # Clean up test file
        import os
        test_file = f"{dbname}.csv"
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"   Cleaned up test file: {test_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED! Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dream_initialization()
    if success:
        print(f"\n🎉 DREAM configuration is working correctly!")
        print(f"   You can now run: python dream_run_batch.py")
    else:
        print(f"\n💥 DREAM configuration still has issues.")
        print(f"   Check the error above for details.")
