# DREAM Experiment Quick Start Guide

## Overview
This DREAM setup allows you to easily run Bayesian calibration experiments with different numbers of uncertain parameters, automatically adjusting the computational settings.

## Quick Usage

### 1. Configure Your Experiment
Choose one of these methods:

**Method A: Interactive Configuration**
```bash
python configure_dream_experiment.py
```

**Method B: Direct Edit**
Edit line 79 in `dream_init_new.py`:
```python
SELECTED_UNCERTAIN_PARAMS = UNCERTAIN_PARAMS_QUICK  # Change this line
```

### 2. Run DREAM
```bash
python dream_run_batch.py
```

### 3. Analyze Results
```bash
python dream_results_new.py
```

## Available Experiment Configurations

| Experiment | Parameters | Count | Est. Time* | Use Case |
|------------|------------|-------|------------|----------|
| `QUICK` | hk1, sy1, D_rch1 | 3 | ~1 hour | Fast testing |
| `MEDIUM` | hk1-3, sy1-2, D_Isar, Kriv_Isar, D_rch1-2 | 9 | ~4 hours | Balanced analysis |
| `FULL` | All parameters | 19 | ~12 hours | Complete uncertainty |
| `HK` | hk1, hk2, hk3, hk4, hk5 | 5 | ~2 hours | Hydraulic conductivity focus |
| `RIVERS` | Kriv_Isar, Kriv_Muhlbach, Kriv_Giessen, D_Isar | 4 | ~1.5 hours | River interaction focus |
| `RECHARGE` | D_rch1, D_rch2, sy1, sy2 | 4 | ~1.5 hours | Recharge/storage focus |

*Estimated times depend on model complexity and hardware

## Auto-Adjusted Settings

The algorithm automatically adjusts computational settings based on parameter count:

- **≤5 parameters**: 1000 iterations, 3 chains
- **6-10 parameters**: 5000 iterations, 4 chains  
- **>10 parameters**: 10000 iterations, 6 chains

## Parameter Descriptions

### Hydraulic Conductivity (hk1-hk5)
- **Units**: m/day
- **Range**: 0.01 - 100 m/day (log-uniform)
- **Description**: Hydraulic conductivity for different geological zones

### Specific Yield (sy1-sy5)  
- **Units**: Dimensionless
- **Range**: 0.05 - 0.35
- **Description**: Specific yield for unconfined aquifer zones

### River Stage (D_Isar)
- **Units**: meters
- **Range**: -2.0 to +2.0 m
- **Description**: Vertical adjustment to Isar river stage

### River Conductance (Kriv_*)
- **Units**: m²/day
- **Range**: 1e-5 - 1000 m²/day (log-uniform)
- **Description**: River bed conductance for each river system

### Recharge Multipliers (D_rch1, D_rch2)
- **Units**: Dimensionless
- **Range**: 0.1 - 3.0
- **Description**: Multipliers for recharge rates (general area, urban area)

## File Structure

```
├── dream_init_new.py          # Parameter configuration
├── dream_modflow_new.py       # Model interface  
├── dream_run_batch.py         # Main DREAM execution (batch mode)
├── dream_results_new.py       # Results analysis
├── configure_dream_experiment.py  # Easy configuration
├── test_dream_setup.py        # Setup testing
└── logs/
    └── log_dream_gwm.txt      # Algorithm log
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all required packages are installed:
   ```bash
   pip install spotpy flopy pandas matplotlib seaborn numpy scipy
   ```

2. **Model Failure**: Run the test first:
   ```bash
   python test_dream_setup.py
   ```

3. **Memory Issues**: Reduce the number of parameters or iterations

4. **Slow Performance**: Start with `QUICK` experiment to test setup

### Custom Parameter Ranges

To modify parameter ranges, edit the `ALL_PARAMETERS` dictionary in `dream_init_new.py`:

```python
'hk1': {'default': 1.0, 'distribution': spotpy.parameter.Uniform('hk1', low=np.log10(0.1), high=np.log10(10)), 'transform': 'log'},
```

## Output Files

- `dream_GWM.csv`: DREAM chain results
- `logs/log_dream_gwm.txt`: Algorithm log
- `dream_plots_gwm/`: Result visualizations
- `dream_plots_gwm/parameter_summary.csv`: Statistics table

## Tips for Efficient Experiments

1. **Start Small**: Begin with `QUICK` experiment to test setup
2. **Parameter Selection**: Focus on most sensitive parameters first
3. **Convergence**: Check R-hat values in log file (should be < 1.3)
4. **Parallel Processing**: For faster runs, consider cluster computing
5. **Results Analysis**: Use `dream_results_new.py` for comprehensive analysis