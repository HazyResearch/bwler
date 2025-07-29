# PDE Experiment Sweep System

This directory contains scripts to run comprehensive sweeps across different PDE problems and methods.

## Overview

The sweep system generates job scripts for:
- **Problems**: convection, reaction, wave
- **Methods**: 
  - MLP (Adam + L-BFGS alternating)
  - MLP (SSBroyden)
  - PirateNet (SSBroyden)
  - BWLer-hat (SSBroyden)
  - Explicit BWLer (SSBroyden)

## Quick Start

1. **Run the complete setup**:
   ```bash
   python scripts/pdes/run_sweep.py
   ```

2. **Submit all jobs**:
   ```bash
   cd scripts/pdes/generated_jobs
   bash submit_all.sh
   ```

3. **Monitor jobs**:
   ```bash
   squeue -u $USER
   ```

## Generated Files

After running the setup, you'll find:
- `scripts/pdes/generated_jobs/` - Individual SLURM job scripts
- `scripts/pdes/generated_jobs/submit_all.sh` - Master script to submit all jobs
- `scripts/pdes/generated_jobs/job_list.txt` - List of all commands for reference

## Job Structure

Each job is named as `{problem}_{method}`:
- `convection_mlp_adam_lbfgs`
- `convection_mlp_ssbroyden`
- `convection_piratenet`
- `convection_bwler_hat`
- `convection_explicit_bwler`
- (and similar for reaction and wave)

## Customization

### Modifying Parameters

Edit `scripts/pdes/sweep_manager.py` to change:
- Problem-specific parameters (c, rho, beta, etc.)
- Model hyperparameters (layers, hidden dim, etc.)
- Training parameters (epochs, evaluation frequency, etc.)

### Adding New Methods

1. Add the method to the `method_mapping` in `SweepManager._build_command()`
2. Add model parameters to `ProblemConfig.model_params`
3. Ensure the PDE files support the new model type

### Adding New Problems

1. Add a new `ProblemConfig` to `SweepManager._define_problems()`
2. Ensure the corresponding PDE file exists in `src/experiments/pdes/simple/`

## Results

Results will be saved in `plots/pdes/{problem}/{method}/` with timestamps.

## Troubleshooting

- **PirateNet not found**: Ensure `src/models/piratenet.py` exists
- **Job failures**: Check SLURM output files in `scripts/pdes/generated_jobs/`
- **Parameter errors**: Verify all required arguments are provided in the sweep manager 