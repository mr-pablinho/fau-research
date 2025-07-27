import numpy as np
import spotpy
from spotpy_setup import GWM_Spotpy_Setup
from config import OptimizationConfig
import multiprocessing as mp

if __name__ == '__main__':
    mp.freeze_support()
    np.random.seed(42)

    spotpy_setup = GWM_Spotpy_Setup()
    
    processing = 'seq'  # 'mpc' for multiprocessing later (still don't manage to get it working)
    sampler = spotpy.algorithms.dream(
        spotpy_setup,
        dbname=OptimizationConfig.DB_NAME,
        dbformat='csv',
        parallel=processing
    )

    n_cores = mp.cpu_count()
    n_chains = OptimizationConfig.DREAM_CHAINS
    repetitions = OptimizationConfig.REPETITIONS

    print(f"Available CPU cores: {n_cores}")
    print(f"Starting DREAM with {n_chains} chains and {repetitions} repetitions")
    print(f"Running in {processing} mode for initial testing")
    print(f"Total model runs: {n_chains * repetitions}")

    n_params = len(spotpy_setup.params)
    min_chains = 2 * n_params + 1 
    if n_chains < min_chains:
        print(f"❌ ERROR: DREAM needs at least {min_chains} chains for {n_params} parameters")
        print(f"Current setting: {n_chains} chains")
        print("Updating chains to minimum requirement...")
        n_chains = min_chains
        print(f"Using {n_chains} chains instead")

    print(f"Final configuration: {n_chains} chains × {repetitions} repetitions = {n_chains * repetitions} total runs")
    
    spotpy_setup.max_runs = n_chains * repetitions

    try:
        print("🚀 Starting DREAM optimization...")
        sampler.sample(repetitions, nChains=n_chains)
        
        print("✅ DREAM run finished successfully!")
        
        results = sampler.getdata()
        
        if results is not None and len(results) > 0:
            best_params = spotpy.analyser.get_best_parameterset(results)
            best_objective = results['like1'].max()
            print(f"📊 Analysis of {len(results)} completed runs:")
            print("🎯 Best parameter set found:")
            print(best_params)
            print(f"🏆 Best objective value: {best_objective}")
            print(f"✅ Successfully completed {len(results)} model runs")
            print(f"💾 Results saved to: {OptimizationConfig.DB_NAME}.csv")
        else:
            print("⚠️ No results were obtained from the optimization")
        
    except Exception as e:
        print(f"❌ Error during DREAM execution: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔍 Possible causes:")
        print("1. Array shape mismatch (now fixed with flattening)")
        print("2. Insufficient number of chains for DREAM algorithm")
        print("3. Model execution failures")
        print("4. Multiprocessing issues")
        
        try:
            if hasattr(sampler, 'datawriter') and sampler.datawriter is not None:
                results = sampler.getdata()
                if results is not None and len(results) > 0:
                    print(f"📊 Partial results retrieved: {len(results)} runs completed")
                    
                    try:
                        best_partial = spotpy.analyser.get_best_parameterset(results)
                        print("🎯 Best partial result:")
                        print(best_partial)
                        print(f"🏆 Best partial objective: {results['like1'].max()}")
                    except Exception as inner_e:
                        print(f"Could not analyze partial results: {inner_e}")
                else:
                    print("📊 No valid results available")
            else:
                print("📊 No results available - sampling may not have started properly")
        except Exception as retrieval_error:
            print(f"📊 Could not retrieve any results: {retrieval_error}")