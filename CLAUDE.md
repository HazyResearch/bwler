# Claude Code Project Notes

## Project Overview
- Working directory: `/scr-ssd/cdeng/bwler`
- Current branch: `constr-exp`
- Main branch: `main`

## Recent Work
- Added embedding runs
- Added k-sweeps
- Updated functionality for weight evals during training
- Added MLP interpolation+deriv runner

## Modified Files
- `scripts/interpolation/generate_mlp_size_k_sweep.sh`

## Generated Files/Directories
- `plots/`
- `scripts/interpolation/mlp_size_k_sweeps/`
- Error logs: `mlp_layers2_hdim16_adam_bary_M8_k_*.err`

## Current Analysis Focus
- Investigating why Adam optimizer performs worse than SSBroyden/SGD
- Analyzing SVD evolution plots showing spikes in singular values
- Understanding relationship between weight evolution and frequency content

## Useful Commands
(Add commands here as we discover them)

## Key Observations
- SVD plots show evolution of weight matrix singular values during training
- Current data: 2-layer MLP with hidden_dim=16, various embeddings (none, theta, cheb, bary)
- Multiple optimizers tested: Adam, SGD, SSBroyden
- Wavenumber sweep: k ∈ {1,2,4,8,16,32,64}

## Planned Analysis Studies

### 1. Function Approximation Decomposition
- Track how model builds up sine function over training
- Compute residuals: `residual(x) = target(x) - model(x)` at each epoch
- FFT residuals to see which frequency components learned first
- Plot "learning timeline" showing when each Fourier mode gets captured

### 4. Frequency-Specific Learning Rates
- Project model output onto Fourier basis at each epoch
- Track convergence rate of each Fourier coefficient
- Identify if Adam struggles with specific frequency ranges
- Compare with target frequency (wavenumber k)

### 5. Weight Specialization Analysis
- Analyze individual weight evolution patterns
- Correlate weight changes with input/output frequency content
- Track weight "specialization" to specific frequencies
- Compare weight role evolution between optimizers

### 8. Information Flow Analysis
- Track gradient flow through network in frequency domain
- Analyze gradient components by frequency
- Identify which network parts respond to which frequencies
- Compare information bottlenecks between optimizers

### 9. Layer-by-Layer Frequency Content Analysis (TODO)
- Track how frequency content changes as signals flow through network layers
- Identify which layers act as bottlenecks for high-frequency information
- Compare Adam vs SGD: does Adam create early low-pass filtering behavior?
- Hypothesis: Adam networks lose high-freq content in shallow layers, SGD preserves it deeper
- Implementation: Pass high-freq inputs through each layer, measure FFT of layer outputs
- Could reveal spectral bias mechanism at the architectural level

## New Script: run_freq_sweeps.py

Created comprehensive frequency analysis script that follows the same structure as run_mlp_k_sweep.py:

**Usage:**
```bash
python scripts/interpolation/run_freq_sweeps.py \
    --n_layers 2 --hidden_dim 16 \
    --optimizer adam --embedding bary --embedding_M 8 \
    --k_values "1.0,4.0,16.0" \
    --save_dir analysis/freq_study_adam_bary
```

**Key Features:**
- Same argument structure as existing sweep scripts
- Integrates with existing training infrastructure via `weight_evals` parameter
- Provides 3 analysis functions as weight evaluators:
  1. `function_decomposition_analyzer` - tracks residual/learned FFT evolution
  2. `frequency_learning_rates_analyzer` - monitors top frequency components
  3. `weight_specialization_analyzer` - analyzes weight matrix spectral properties

**Output:**
- `freq_evolution_timeline_k_*.png` - shows what frequencies learned when
- `weight_specialization_k_*.png` - weight matrix analysis over training
- Training logs with frequency data for post-analysis

## Notes
(Add important project context here)