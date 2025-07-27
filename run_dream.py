import numpy as np
import spotpy
from spotpy_setup import GWM_Spotpy_Setup
from config import OptimizationConfig
import multiprocessing as mp

# Required for Windows multiprocessing - MUST be at the very top
if __name__ == '__main__':
    mp.freeze_support()
    
    # For reproducibility, you can set a random seed
    np.random.seed(42)

    # 1. Initialize the spotpy setup class for your model
    spotpy_setup = GWM_Spotpy_Setup()

    # 2. Initialize the DREAM sampler with CSV storage (HDF5 not available)
    # CSV format works fine now that we've fixed the array flattening issue
    processing = 'seq'  # Use 'mpc' for multiprocessing later
    sampler = spotpy.algorithms.dream(
        spotpy_setup,
        dbname=OptimizationConfig.DB_NAME,
        dbformat='csv',  # Back to CSV since HDF5 not available
        parallel=processing  # Use 'seq' for initial testing, 'mpc' for multiprocessing
    )

    # 3. Start the sampling process with parallel execution
    # Get settings from configuration
    n_cores = mp.cpu_count()
    n_chains = OptimizationConfig.DREAM_CHAINS
    repetitions = OptimizationConfig.REPETITIONS

    print(f"Available CPU cores: {n_cores}")
    print(f"Starting DREAM with {n_chains} chains and {repetitions} repetitions")
    print(f"Running in {processing} mode for initial testing")
    print(f"Total model runs: {n_chains * repetitions}")

    # For DREAM algorithm requirements:
    # - Need at least 2*n_parameters + 1 chains for proper sampling
    n_params = len(spotpy_setup.params)
    min_chains = 2 * n_params + 1
    if n_chains < min_chains:
        print(f"❌ ERROR: DREAM needs at least {min_chains} chains for {n_params} parameters")
        print(f"Current setting: {n_chains} chains")
        print("Updating chains to minimum requirement...")
        n_chains = min_chains
        print(f"Using {n_chains} chains instead")

    print(f"Final configuration: {n_chains} chains × {repetitions} repetitions = {n_chains * repetitions} total runs")
    
    # Set max_runs for progress tracking
    spotpy_setup.max_runs = n_chains * repetitions

    try:
        print("🚀 Starting DREAM optimization...")
        sampler.sample(repetitions, nChains=n_chains)
        
        print("✅ DREAM run finished successfully!")
        
        # 4. Analyze the results
        results = sampler.getdata()
        
        if results is not None and len(results) > 0:
            print(f"📊 Analysis of {len(results)} completed runs:")
            
            # Use the spotpy.analyser to get the best parameter set from the results
            best_params = spotpy.analyser.get_best_parameterset(results)
            print("🎯 Best parameter set found:")
            print(best_params)

            # You can also get the corresponding best objective function value
            best_objective = results['like1'].max()
            print(f"🏆 Best objective value: {best_objective}")
            
            print(f"✅ Successfully completed {len(results)} model runs")
            
            # Save results summary
            print(f"💾 Results saved to: {OptimizationConfig.DB_NAME}.csv")
        else:
            print("⚠️ No results were obtained from the optimization")

        # You can also use spotpy's built-in plotting tools (commented out for now)
        # spotpy.analyser.plot_parameter_trace(results)
        # spotpy.analyser.plot_posterior_parameter_histogram(results)
        
    except Exception as e:
        print(f"❌ Error during DREAM execution: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔍 Possible causes:")
        print("1. Array shape mismatch (now fixed with flattening)")
        print("2. Insufficient number of chains for DREAM algorithm")
        print("3. Model execution failures")
        print("4. Multiprocessing issues")
        
        # Try to get any available results
        try:
            if hasattr(sampler, 'datawriter') and sampler.datawriter is not None:
                results = sampler.getdata()
                if results is not None and len(results) > 0:
                    print(f"📊 Partial results retrieved: {len(results)} runs completed")
                    
                    # Show best partial result
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