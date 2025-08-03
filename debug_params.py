#!/usr/bin/env python3
"""
Debug script to check DREAM parameter configuration
"""

try:
    import dream_init_new as di
    
    print("=== DREAM Parameter Configuration Debug ===")
    print(f"Parameters: {di.names}")
    print(f"Number of parameters: {len(di.names)}")
    print(f"Chains configured: {di.nChains}")
    print(f"Required chains (2*n+1): {2*len(di.names)+1}")
    print(f"Parameters meet requirement: {di.nChains >= 2*len(di.names)+1}")
    
    # Check the DREAM algorithm requirements more specifically
    print(f"\n=== DREAM Requirements ===")
    n_params = len(di.names)
    min_chains_basic = 2 * n_params + 1
    min_chains_safe = 3 * n_params  # Often recommended for stability
    
    print(f"Minimum chains (basic): {min_chains_basic}")
    print(f"Recommended chains (safe): {min_chains_safe}")
    print(f"Current chains: {di.nChains}")
    
    if di.nChains < min_chains_basic:
        print(f"❌ INSUFFICIENT CHAINS! Need at least {min_chains_basic}, have {di.nChains}")
        print(f"Recommendation: Set nChains = {min_chains_safe}")
    else:
        print(f"✅ Chains requirement satisfied")
        
    print(f"\n=== Parameter Details ===")
    for i, name in enumerate(di.names):
        print(f"  {i+1}. {name}")
        
except Exception as e:
    print(f"Error importing or running: {e}")
    import traceback
    traceback.print_exc()
