# -*- coding: utf-8 -*-
"""
DREAM algorithm with checkpoint/resume functionality for new GWM model
Simple batch runner that allows stopping and resuming long DREAM runs

Usage:
    python dream_run_batch.py              # Start new run
    python dream_run_batch.py --resume     # Resume from last checkpoint

@author: Pablo Merchán-Rivera (adapted)
"""

import numpy as np
import spotpy
import dream_init_new as di
from dream_modflow_new import spot_setup
from datetime import datetime
import os
import sys
import pickle
import glob
import pandas as pd
import shutil

# Create directories
os.makedirs('./logs', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

def save_checkpoint(dbname, iteration, batch_count):
    """Save checkpoint for resuming DREAM run later"""
    checkpoint_file = f'./checkpoints/{dbname}_checkpoint_{iteration}.pkl'
    checkpoint_data = {
        'dbname': dbname,
        'iteration': iteration,
        'batch_count': batch_count,
        'timestamp': datetime.now()
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"✓ Checkpoint saved: {iteration} iterations completed")

def find_resume_info():
    """Find the most recent checkpoint to resume from"""
    checkpoints = glob.glob('./checkpoints/dream_GWM_*_checkpoint_*.pkl')
    if not checkpoints:
        return None, None, None
    
    # Find the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    
    with open(latest_checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    return checkpoint_data['dbname'], checkpoint_data['iteration'], checkpoint_data['batch_count']

def get_actual_iterations_from_db(dbname):
    """Get the actual number of iterations completed from the database"""
    csv_file = f"{dbname}.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return len(df)
        except Exception as e:
            print(f"Warning: Could not read database {csv_file}: {e}")
    return 0

def run_dream_batch(dbname, batch_size, completed_iterations=0):
    """Run a single batch of DREAM iterations"""
    
    # Set up DREAM
    random_state = di.my_seed
    np.random.seed(random_state)
    
    spot_setup_instance = spot_setup(_used_algorithm='dream')
    
    # Check if this is a continuation of an existing run
    csv_file = f"{dbname}.csv"
    backup_file = f"{dbname}_backup.csv"
    is_continuation = os.path.exists(csv_file) and completed_iterations > 0
    
    if is_continuation:
        # Backup existing results before running new batch
        print(f"📁 Backing up existing results to {backup_file}")
        shutil.copy2(csv_file, backup_file)
    
    # Create sampler
    sampler = spotpy.algorithms.dream(
        spot_setup_instance, 
        dbname=dbname, 
        dbformat='csv',
        db_precision=np.float32, 
        save_sim=True,
        random_state=random_state
    )
    
    # DREAM parameters
    nChains = di.nChains
    convergence_limit = 1.3
    runs_after_convergence = di.convEvals
    epsilon = 0.001
    ato = 6
    # nCr must satisfy: nChains >= 2*nCr + 1
    # With nChains=10, max nCr = (10-1)/2 = 4.5, so nCr=4 should work
    # With nChains=5, max nCr = (5-1)/2 = 2, so use nCr=2
    nCr = min(2, (nChains - 1) // 2)  # Ensure nCr satisfies chain requirement
    
    print(f"Running {batch_size} iterations...")
    print(f"Chains: {nChains}, Convergence limit: {convergence_limit}")
    
    # Get iterations before running this batch
    iterations_before = get_actual_iterations_from_db(dbname)
    
    # Run the batch
    r_hat = sampler.sample(
        batch_size,
        nChains=nChains,
        convergence_limit=convergence_limit,
        runs_after_convergence=runs_after_convergence,
        eps=epsilon,
        acceptance_test_option=ato,
        nCr=nCr
    )
    
    # Get actual iterations after running this batch
    new_results_count = get_actual_iterations_from_db(dbname)
    
    # If this was a continuation, we need to merge the backup with new results
    if is_continuation and os.path.exists(backup_file):
        print(f"🔗 Merging backup results with new batch results...")
        
        # Load backup data and new data
        backup_df = pd.read_csv(backup_file)
        new_df = pd.read_csv(csv_file)
        
        # Combine the dataframes
        combined_df = pd.concat([backup_df, new_df], ignore_index=True)
        
        # Save combined results
        combined_df.to_csv(csv_file, index=False)
        print(f"✅ Merged {len(backup_df)} previous + {len(new_df)} new = {len(combined_df)} total iterations")
        
        # Clean up backup file
        os.remove(backup_file)
        
        actual_iterations = len(combined_df)
        iterations_completed_this_batch = len(new_df)
    else:
        actual_iterations = new_results_count
        iterations_completed_this_batch = actual_iterations - iterations_before
    
    print(f"📊 Batch completed: {iterations_completed_this_batch} iterations (requested: {batch_size})")
    if iterations_completed_this_batch > batch_size:
        print(f"   ℹ️  DREAM achieved convergence and ran additional post-convergence samples")
    
    return r_hat, sampler, actual_iterations

if __name__ == "__main__":
    
    print("=" * 60)
    print("DREAM Algorithm Batch Runner for GWM Model")
    print("=" * 60)
    
    # Check for resume flag
    resume_flag = '--resume' in sys.argv
    
    # Initialize variables
    dbname = None
    completed_iterations = 0
    batch_count = 0
    
    if resume_flag:
        dbname, completed_iterations, batch_count = find_resume_info()
        if dbname:
            print(f"📂 Resuming from checkpoint:")
            print(f"   Database: {dbname}")
            print(f"   Completed iterations: {completed_iterations}")
            print(f"   Batch count: {batch_count}")
        else:
            print("⚠️  No checkpoint found. Starting new run.")
            resume_flag = False
    
    if not resume_flag:
        # Start new run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dbname = f'dream_GWM_{timestamp}'
        completed_iterations = 0
        batch_count = 0
        print(f"🚀 Starting new DREAM run:")
        print(f"   Database: {dbname}")
    
    # Configuration
    total_iterations = di.rep
    batch_size = 4
    
    print(f"\n📊 Run Configuration:")
    print(f"   Total target iterations: {total_iterations}")
    print(f"   Batch size: {batch_size} iterations")
    print(f"   Estimated batches needed: {(total_iterations - completed_iterations + batch_size - 1) // batch_size}")
    print(f"   Parameters: {di.names}")
    
    # Log start
    with open('./logs/log_dream_gwm.txt', 'a') as f:
        f.write(f"\n--- Batch run started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} ---\n")
        f.write(f"Database: {dbname}\n")
        f.write(f"Starting from iteration: {completed_iterations}\n")
    
    print(f"\n🎯 Press Ctrl+C anytime to safely stop and save progress")
    print("=" * 60)
    
    try:
        while completed_iterations < total_iterations:
            remaining = total_iterations - completed_iterations
            current_batch_size = min(batch_size, remaining)
            batch_count += 1
            
            print(f"\n🔄 BATCH {batch_count}")
            print(f"   Running: {current_batch_size} iterations")
            print(f"   Progress: {completed_iterations}/{total_iterations} ({100*completed_iterations/total_iterations:.1f}%)")
            
            start_time = datetime.now()
            
            # Run the batch
            iterations_before_batch = completed_iterations
            r_hat, sampler, actual_iterations = run_dream_batch(dbname, current_batch_size, completed_iterations)
            
            # Update progress with actual iterations completed
            completed_iterations = actual_iterations
            
            # Check if DREAM achieved convergence and ran post-convergence samples
            iterations_this_batch = actual_iterations - iterations_before_batch
            dream_converged = iterations_this_batch > current_batch_size
            
            # Save checkpoint
            save_checkpoint(dbname, completed_iterations, batch_count)
            
            # Show progress
            elapsed = datetime.now() - start_time
            print(f"✅ Batch {batch_count} completed in {elapsed}")
            print(f"📈 Total progress: {completed_iterations}/{total_iterations} ({100*completed_iterations/total_iterations:.1f}%)")
            
            # If DREAM converged, we should stop even if we haven't reached target iterations
            if dream_converged:
                print(f"🎯 DREAM algorithm has achieved convergence and completed post-convergence sampling!")
                print(f"   Total iterations completed: {completed_iterations}")
                print(f"   Stopping early (target was {total_iterations} iterations)")
                break
            
            if completed_iterations < total_iterations:
                print(f"💾 Safe to stop now with Ctrl+C. Next batch will run {min(batch_size, total_iterations - completed_iterations)} iterations.")
        
        print(f"\n🎉 DREAM run completed!")
        print(f"   Total iterations: {completed_iterations}")
        print(f"   Database: {dbname}.csv")
        
        # Clean up checkpoints for this run
        checkpoint_pattern = f'./checkpoints/{dbname}_checkpoint_*.pkl'
        for checkpoint_file in glob.glob(checkpoint_pattern):
            os.remove(checkpoint_file)
        print(f"🧹 Cleaned up checkpoint files")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Run paused by user")
        print(f"   Progress saved: {completed_iterations}/{total_iterations} iterations")
        print(f"   To resume: python dream_run_batch.py --resume")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"   Progress saved: {completed_iterations}/{total_iterations} iterations")
        print(f"   To resume: python dream_run_batch.py --resume")
        raise
    
    print("\n" + "=" * 60)