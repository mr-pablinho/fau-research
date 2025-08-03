# DREAM Algorithm Implementation for GWM Garching Model

This implementation adapts the DREAM (DiffeRential Evolution Adaptive Metropolis) algorithm from your 2022 Bayesian study to work with your new Garching groundwater model.

## 📁 Files Created

### Core DREAM Files
- **`dream_init_new.py`** - Parameter definitions and initialization
- **`dream_modflow_new.py`** - SPOTPY setup class for model interface  
- **`dream_run_new.py`** - Main DREAM execution script
- **`dream_results_new.py`** - Results analysis and visualization

### Helper Files
- **`configure_dream_experiment.py`** - Setup validation and configuration
- **`simple_convergence_test.py`** - Quick test script
- **`README_DREAM.md`** - This documentation

## 🔧 Setup and Usage

### 1. Initial Setup and Validation
```bash
python configure_dream_experiment.py
```
This script will:
- Check if all required packages are installed
- Validate data files exist
- Test observation data structure
- Run a test model simulation
- Estimate runtime
- Create experiment configuration

### 2. Quick Test (Optional but Recommended)
```bash
python simple_convergence_test.py
```
Runs a few iterations to verify everything works before the full run.

### 3. Run Full DREAM Algorithm
```bash
python dream_run_new.py
```
This will run the complete DREAM calibration. **Warning**: This can take many hours!

### 4. Analyze Results
```bash
python dream_results_new.py
```
Creates plots and statistical analysis of the calibration results.

## 📊 Model Parameters (19 total)

### Hydraulic Parameters
- **hk1-hk5**: Hydraulic conductivity for zones 1-5 (m/d) [log-transformed]
- **sy1-sy5**: Specific yield for zones 1-5 (-) [linear]

### River Parameters  
- **D_Isar**: Isar river stage adjustment (m) [linear]
- **Kriv_Isar**: Isar river bed conductance (m²/d) [log-transformed]
- **Kriv_Muhlbach**: Muhlbach river conductance (m²/d) [log-transformed]
- **Kriv_Giessen**: Giessen river conductance (m²/d) [log-transformed]
- **Kriv_Griesbach**: Griesbach river conductance (m²/d) [log-transformed]
- **Kriv_Schwabinger_Bach**: Schwabinger Bach conductance (m²/d) [log-transformed]
- **Kriv_Wiesackerbach**: Wiesackerbach conductance (m²/d) [log-transformed]

### Recharge Parameters
- **D_rch1**: Background recharge multiplier (-) [linear]
- **D_rch2**: Urban area recharge multiplier (-) [linear]

## 📈 Observation Data Structure

- **Observation Points**: 13 monitoring wells
- **Stress Periods**: 139 time steps  
- **Total Observations**: 1,807 head measurements
- **Data Files**: 
  - `Output1_Input2/obs.csv` - observation point locations
  - `Output1_Input2/obs_values.csv` - observed head values

## ⚙️ DREAM Algorithm Settings

- **Maximum Iterations**: 10,000
- **Chains**: 6 parallel chains
- **Convergence Evaluations**: 300 (samples after convergence)
- **Convergence Criterion**: Gelman-Rubin R̂ < 1.3
- **Random Seed**: 246 (for reproducibility)

## 📋 Key Differences from 2022 Implementation

### Parameters
- **2022**: 11 parameters (HK_SA, HK_SB, SY_SA, SY_SB, RCH, CON_CHA1, CON_CHA2, CON_RIV1, CON_RIV2, CON_RIV3, STA)
- **2025**: 19 parameters (5 HK zones, 5 SY zones, 7 river conductances, 2 recharge multipliers, 1 stage adjustment)

### Model Structure  
- **2022**: Called `f_runModflow_da.runModflow()`
- **2025**: Calls `GWM_model_run.GWM()` with temporary directories for parallel execution

### Observations
- **2022**: 4 observation points × 300 stress periods = 1,200 observations
- **2025**: 13 observation points × 139 stress periods = 1,807 observations

### Output Format
- **2022**: Used `.txt` files and specific array reshaping
- **2025**: Uses CSV files and flattened arrays for SPOTPY compatibility

## 🔍 Results Analysis

The `dream_results_new.py` script generates:

1. **Parameter Traces**: Evolution of parameters over iterations
2. **Parameter Distributions**: Prior vs posterior distributions  
3. **Likelihood Evolution**: Convergence monitoring
4. **Parameter Correlations**: Correlation matrix
5. **Summary Statistics**: Mean, std, skewness, kurtosis
6. **Best Parameter Set**: Maximum likelihood parameter values

## 📁 Output Files

- **`dream_GWM_new.csv`** - Complete DREAM results
- **`dream_parameters_summary.csv`** - Statistical summary
- **`logs/log_dream_new.txt`** - Execution log
- **`dream_plots_new/`** - Generated plots and figures
- **`dream_experiment_config.json`** - Experiment configuration

## ⚡ Performance Tips

1. **Runtime Estimation**: Use `configure_dream_experiment.py` to estimate total runtime
2. **Parallel Execution**: DREAM uses multiple chains but model runs are sequential
3. **Memory Management**: Temporary directories are created/cleaned for each simulation
4. **Monitoring**: Check `logs/log_dream_new.txt` for progress updates

## 🐛 Troubleshooting

### Common Issues:
1. **Missing Dependencies**: Run `pip install spotpy flopy numpy pandas matplotlib scipy`
2. **Data File Errors**: Check that all files in `Output1_Input2/` exist
3. **MODFLOW Execution**: Ensure `MODFLOW-NWT_64.exe` is in the correct location
4. **Memory Issues**: Reduce `rep` value in `dream_init_new.py` for shorter runs

### Debug Steps:
1. Run `configure_dream_experiment.py` first
2. Use `simple_convergence_test.py` for quick validation  
3. Check log files in `logs/` directory
4. Verify observation data dimensions match expectations

## 📞 Support

If you encounter issues:
1. Check the log files for detailed error messages
2. Verify all input files are correctly formatted
3. Test with smaller parameter ranges or fewer iterations
4. Compare with the working 2022 implementation structure

## 🔄 Migration from 2022 Implementation

The new structure maintains the same DREAM algorithm principles but adapts to:
- More complex parameter space (19 vs 11 parameters)
- Different model interface (GWM function vs runModflow)
- Updated observation handling (CSV format vs text files)
- Improved error handling and temporary file management

This implementation preserves the scientific rigor of your 2022 Bayesian approach while scaling to the increased complexity of your new Garching model.
