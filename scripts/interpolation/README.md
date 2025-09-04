# Interpolation Experiments

This folder contains scripts for running 1D interpolation experiments with neural networks, including derivative supervision studies.

## 🚀 Quick Start: Derivative Sweep Experiment

### Overview
The derivative sweep experiment systematically tests how derivative supervision affects MLP performance on `sin(2x)` interpolation. It sweeps over different numbers of derivative training points while keeping value training points fixed.

### What It Tests
- **Fixed MLP architectures**: 2-3 layers, 32-128 hidden dimensions
- **Derivative supervision levels**: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 points
- **Optimizer comparison**: Adam vs SSBroyden
- **Performance metrics**: L2 Relative Error (L2RE), RMSE, weight norms

### 1. Generate Sweep Scripts
```bash
cd /pscratch/sd/j/jwl50/bwler
bash scripts/interpolation/generate_mlp_size_deriv_sweep.sh
```

This creates **12 sbatch scripts** (6 MLP configurations × 2 optimizers):
- **Adam scripts**: `mlp_layers2_hdim32_adam_deriv_sweep.sbatch`, etc.
- **SSBroyden scripts**: `mlp_layers2_hdim32_ssbroyden_deriv_sweep.sbatch`, etc.

### 2. Submit Jobs

#### Submit All Adam Jobs (Default)
```bash
cd scripts/interpolation/mlp_size_deriv_sweeps
for script in mlp_*_adam_deriv_sweep.sbatch; do
    sbatch $script
done
```

#### Submit All SSBroyden Jobs
```bash
cd scripts/interpolation/mlp_size_deriv_sweeps
for script in mlp_*_ssbroyden_deriv_sweep.sbatch; do
    sbatch $script
done
```

#### Submit Individual Jobs
```bash
# Example: 2-layer, 32-hidden dim MLP with Adam
sbatch mlp_layers2_hdim32_adam_deriv_sweep.sbatch

# Example: 3-layer, 64-hidden dim MLP with SSBroyden
sbatch mlp_layers3_hdim64_ssbroyden_deriv_sweep.sbatch
```

### 3. Experiment Configuration

#### MLP Architectures Tested
- **Layers**: 2, 3
- **Hidden Dimensions**: 32, 64, 128
- **Total Configurations**: 6 (2×3)

#### Training Parameters
- **Target Function**: `sin(2x)` on domain [-1, 1]
- **Training Epochs**: 10,000
- **Evaluation Frequency**: Every 1,000 epochs
- **0th Order Points**: Fixed at 100 (function values)
- **1st Order Points**: Variable [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
- **Test Points**: 1,000 equispaced points
- **Loss Weights**: α=1.0 (value), β=1.0 (derivative)

#### Optimizers
- **Adam**: Learning rate scheduling, gradient clipping
- **SSBroyden**: Second-order optimization, fixed training points

### 4. Output Structure

Each job creates a directory structure:
```
plots/interpolation/mlp_size_sweep_{optimizer}/layers{N}_hdim{M}/
├── deriv_sweep_results.json          # All numerical results
├── deriv_sweep_l2re.png             # L2RE vs derivative points plot
├── weight_norms_n1st_0.png          # Weight norms (no derivatives)
├── weight_norms_n1st_1.png          # Weight norms (1 derivative point)
├── weight_norms_n1st_2.png          # Weight norms (2 derivative points)
├── ...                              # etc.
├── solution_n1st_0.png              # Custom plots (no derivatives)
├── solution_n1st_1.png              # Custom plots (1 derivative point)
├── ...                              # etc.
└── model_n1st_*.pt                  # Trained model states
```

### 5. Key Results Generated

#### Performance Analysis
- **L2RE vs Derivative Points**: Shows optimal level of derivative supervision
- **Convergence Comparison**: Adam vs SSBroyden performance
- **Architecture Impact**: How different MLP sizes respond to derivative supervision

#### Weight Analysis
- **Parameter Norms**: How derivative supervision affects learned representations
- **Layer-wise Analysis**: Impact on different network layers
- **Optimization Insights**: Understanding the learning process

#### Visualization
- **Custom Solution Plots**: 
  - 0th order points (red): Function value training data
  - 1st order points (orange): Derivative training data
  - MLP predictions (blue/purple): Model outputs
- **Training Progress**: Loss history, L2RE evolution
- **Error Analysis**: Prediction errors across domain

### 6. Monitoring and Analysis

#### Check Job Status
```bash
squeue -u $USER
```

#### View Output Logs
```bash
tail -f mlp_layers2_hdim32_adam_*.out
```

#### Expected Runtime
- **Adam jobs**: ~2-3 hours each
- **SSBroyden jobs**: ~1-2 hours each
- **Total sweep**: ~24-36 hours for all configurations

### 7. Post-Experiment Analysis

#### Results Summary
```bash
# View summary table in job output
grep "DERIVATIVE SWEEP RESULTS SUMMARY" *.out
```

#### Data Analysis
- **JSON Results**: Load `deriv_sweep_results.json` for further analysis
- **Performance Trends**: Analyze L2RE improvement vs derivative points
- **Optimizer Comparison**: Compare Adam vs SSBroyden across configurations

#### Publication Plots
- **Main Sweep Plot**: `deriv_sweep_l2re.png`
- **Weight Norm Analysis**: `weight_norms_n1st_*.png`
- **Solution Quality**: `solution_n1st_*.png`

## 🔧 Technical Details

### Architecture
- **SineTarget**: Simplified target class with built-in derivative supervision
- **Separate Point Sampling**: 0th and 1st order points can be different counts/locations
- **Unified Training**: Adam and SSBroyden training methods in one class
- **Logger Integration**: Consistent logging and checkpointing

### Key Innovations
- **No Wrapper Classes**: Direct derivative supervision in target
- **Flexible Point Counts**: Independent control over value vs derivative supervision
- **Built-in Training**: Training methods follow BasePDE patterns
- **Custom Plotting**: Visualizes both training point types distinctly

### Dependencies
- PyTorch
- NumPy
- Matplotlib
- Custom modules: `src.experiments.interpolation.*`

## 📊 Research Questions Addressed

1. **How much derivative supervision is optimal?**
2. **Does more derivative points always help?**
3. **How does derivative supervision affect learned representations?**
4. **Which optimizer benefits most from derivative supervision?**
5. **What's the computational cost vs performance trade-off?**

## 🎯 Expected Outcomes

- **Systematic understanding** of derivative supervision impact
- **Optimal point counts** for different MLP architectures
- **Optimizer recommendations** for physics-informed training
- **Publication-ready plots** showing derivative supervision benefits
- **Weight norm insights** into how physics constraints affect learning

## 🚨 Troubleshooting

### Common Issues
- **Directory not found**: Ensure `plots/interpolation/` exists
- **Job failures**: Check GPU availability and memory limits
- **Plot errors**: Verify matplotlib backend compatibility

### Debug Commands
```bash
# Check GPU status
nvidia-smi

# Monitor memory usage
htop

# View detailed job info
scontrol show job <job_id>
```

---

For questions or issues, check the job output logs or contact the development team.
