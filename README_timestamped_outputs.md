# DREAM Timestamped Output Feature

## Overview

The DREAM algorithm files have been enhanced to automatically save all output files with datetime stamps. This prevents overwriting previous results and makes it easy to manage multiple DREAM runs.

## Changes Made

### 1. dream_init_new.py
- Added `datetime` import
- Modified the `flag` variable to include timestamp: `dream-r50-c20-p4-s246-20250803_201657`

### 2. dream_run_new.py
- Added timestamped filenames for CSV results and log files
- Format: `dream_GWM_YYYYMMDD_HHMMSS.csv` and `logs/log_dream_YYYYMMDD_HHMMSS.txt`

### 3. dream_results_new.py
- Automatically detects and uses the most recent DREAM results file
- Extracts timestamp from filename and uses it for all output files
- Creates timestamped plot directory: `dream_plots_YYYYMMDD_HHMMSS/`
- All plot files include timestamp suffix

## File Naming Convention

### Main Output Files
- **CSV Results**: `dream_GWM_20250803_201657.csv`
- **Log File**: `logs/log_dream_20250803_201657.txt`
- **Plots Directory**: `dream_plots_20250803_201657/`

### Analysis Output Files
- **Parameter Summary**: `dream_parameters_summary_20250803_201657.csv`
- **Parameter Traces**: `dream_parameter_traces_20250803_201657.png`
- **Likelihood Evolution**: `dream_objective_evolution_20250803_201657.png`
- **Parameter Distributions**: `dream_parameter_distributions_20250803_201657.png`
- **Parameter Correlations**: `dream_parameter_correlation_20250803_201657.png`

## How to Use

### Running DREAM Algorithm
```bash
python dream_run_new.py
```
This will create:
- Timestamped CSV results file
- Timestamped log file

### Analyzing Results
```bash
python dream_results_new.py
```
This will:
- Automatically find the most recent results file
- Create timestamped analysis plots and summary
- Save everything in a timestamped directory

## Benefits

1. **No Overwriting**: Each run creates unique files
2. **Easy Comparison**: Compare results from different runs
3. **Organized Management**: Files are clearly dated and organized
4. **Automatic Detection**: Analysis script automatically finds latest results
5. **Backward Compatibility**: Still works with existing non-timestamped files

## Timestamp Format

- Format: `YYYYMMDD_HHMMSS`
- Example: `20250803_201657` (August 3, 2025 at 20:16:57)

## Testing

Run the test script to see the functionality:
```bash
python test_timestamp_functionality.py
```

This will show:
- Current timestamp format
- Example output filenames
- Benefits of the new system
- Usage instructions
- List of existing DREAM result files

## Backward Compatibility

The modified scripts maintain compatibility with existing non-timestamped files. If no timestamped files are found, the system will fall back to the old naming convention.
