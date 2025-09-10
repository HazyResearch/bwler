#!/bin/bash
# Script to generate separate sbatch jobs for each MLP configuration
# Each job sweeps over wavenumbers (k) for a fixed model size
# Now supports Adam, SGD, and SSBroyden optimizers

# Configuration
TARGET="sine"
SEED=0
N_TRAIN_0TH=400  # points for value loss
N_TRAIN_1ST=0    # points for derivative loss (typically 0 for k sweep)
N_TEST=1000
N_EPOCHS=10000
EVAL_EVERY=1000
SAVE_ROOT="/scr/biggest/junmiaoh/bwler/plots/interpolation"
DERIV_ALPHA=1.0
DERIV_BETA=1.0

# MLP configurations to test
LAYERS_LIST=(2 3 4 5)
HDIM_LIST=(32 64 128)

# Wavenumbers to sweep over (comma-separated list)
K_VALUES="1.0,2.0,4.0,8.0,16.0,32.0"

# Create scripts directory if it doesn't exist
mkdir -p /scr/biggest/junmiaoh/bwler/scripts/interpolation/mlp_size_k_sweeps

# Function to generate sbatch script for a given optimizer
generate_sbatch_script() {
    local LAYERS=$1
    local HDIM=$2
    local OPTIMIZER=$3
    local OPTIMIZER_LOWER=$(echo $OPTIMIZER | tr '[:upper:]' '[:lower:]')
    
    # Set optimizer-specific training parameters
    if [ "$OPTIMIZER" = "SSBroyden" ]; then
        local N_EPOCHS_OPT=10000     # SSBroyden: 10000 epochs for full training
        local EVAL_EVERY_OPT=1000    # SSBroyden: log every 1000 epochs
        local JOB_TIME="4:00:00"     # SSBroyden: longer job time
        local PLOT_EVERY=1
        
    elif [ "$OPTIMIZER" = "SGD" ]; then
        local N_EPOCHS_OPT=50000     # SGD: 50000 epochs for full training (may need longer due to slower convergence)
        local EVAL_EVERY_OPT=1000    # SGD: log every 1000 epochs
        local JOB_TIME="5:00:00"     # SGD: longer job time for more epochs
        local PLOT_EVERY=2

    else
        local N_EPOCHS_OPT=40000     # Adam: 40000 epochs for full training
        local EVAL_EVERY_OPT=1000    # Adam: log every 1000 epochs
        local JOB_TIME="4:00:00"     # Adam: longer job time
        local PLOT_EVERY=2
    fi
    
    local CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
    local SAVE_DIR="${SAVE_ROOT}/mlp_size_k_sweep_${OPTIMIZER_LOWER}/${CONFIG_NAME}"
    
    # Create the sbatch script
    cat > "/scr/biggest/junmiaoh/bwler/scripts/interpolation/mlp_size_k_sweeps/mlp_${CONFIG_NAME}_${OPTIMIZER_LOWER}_k_sweep.sbatch" << EOF

#!/bin/bash
#SBATCH --job-name=mlp_${CONFIG_NAME}_${OPTIMIZER_LOWER}_k
#SBATCH --time=${JOB_TIME}
#SBATCH --account=hazy
#SBATCH --partition=hazy
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=mlp_${CONFIG_NAME}_${OPTIMIZER_LOWER}_k_%j.out
#SBATCH --error=mlp_${CONFIG_NAME}_${OPTIMIZER_LOWER}_k_%j.err

# Environment
source /scr/biggest/junmiaoh/bwler/.venv/bin/activate

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-32}
export PYTHONUNBUFFERED=1
export MPICH_GPU_SUPPORT_ENABLED=1

# Navigate to project
cd /scr/biggest/junmiaoh/bwler/
mkdir -p "${SAVE_DIR}"

echo "Starting MLP ${CONFIG_NAME} wavenumber sweep with ${OPTIMIZER}"
echo "Saving to: ${SAVE_DIR}"

# Run the wavenumber sweep for this MLP configuration
python scripts/interpolation/run_mlp_k_sweep.py \\
    --target "${TARGET}" \\
    --device cuda \\
    --seed "${SEED}" \\
    --n_train_0th "${N_TRAIN_0TH}" \\
    --n_train_1st "${N_TRAIN_1ST}" \\
    --n_test "${N_TEST}" \\
    --n_epochs "${N_EPOCHS_OPT}" \\
    --eval_every "${EVAL_EVERY_OPT}" \\
    --save_dir "${SAVE_DIR}" \\
    --n_layers "${LAYERS}" \\
    --hidden_dim "${HDIM}" \\
    --deriv_alpha "${DERIV_ALPHA}" \\
    --deriv_beta "${DERIV_BETA}" \\
    --optimizer "${OPTIMIZER_LOWER}" \\
    --k_values "${K_VALUES}" \\
    --svd_plot_every "${PLOT_EVERY}" \\

echo "Completed MLP ${CONFIG_NAME} wavenumber sweep with ${OPTIMIZER}"
EOF

    echo "Generated: mlp_${CONFIG_NAME}_${OPTIMIZER_LOWER}_k_sweep.sbatch (${N_EPOCHS_OPT} epochs, log every ${EVAL_EVERY_OPT})"
}

# Generate scripts for Adam (default)
echo "=== Generating Adam sweep scripts ==="
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        generate_sbatch_script $LAYERS $HDIM "Adam"
    done
done

# Generate scripts for SGD
echo ""
echo "=== Generating SGD sweep scripts ==="
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        generate_sbatch_script $LAYERS $HDIM "SGD"
    done
done

# Generate scripts for SSBroyden
echo ""
echo "=== Generating SSBroyden sweep scripts ==="
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        generate_sbatch_script $LAYERS $HDIM "SSBroyden"
    done
done

echo ""
echo "Generated $((${#LAYERS_LIST[@]} * ${#HDIM_LIST[@]} * 3)) total sbatch scripts:"
echo ""
echo "=== CONFIGURATION SUMMARY ==="
echo "Adam jobs:      40,000 epochs, log every 1,000 epochs, 4:00:00 time limit"
echo "SGD jobs:       50,000 epochs, log every 1,000 epochs, 5:00:00 time limit"
echo "SSBroyden jobs: 10,000 epochs, log every 1,000 epochs, 4:00:00 time limit"
echo "Wavenumbers:    ${K_VALUES}"
echo ""
echo "Adam scripts:"
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
        echo "  mlp_${CONFIG_NAME}_adam_k_sweep.sbatch"
    done
done

echo ""
echo "SGD scripts:"
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
        echo "  mlp_${CONFIG_NAME}_sgd_k_sweep.sbatch"
    done
done

echo ""
echo "SSBroyden scripts:"
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
        echo "  mlp_${CONFIG_NAME}_ssbroyden_k_sweep.sbatch"
    done
done

echo ""
echo "To submit all Adam jobs, run:"
echo "  cd /scr/biggest/junmiaoh/bwler/scripts/interpolation/mlp_size_k_sweeps"
echo "  for script in mlp_*_adam_k_sweep.sbatch; do"
echo "    sbatch \$script"
echo "  done"
echo ""
echo "To submit all SGD jobs, run:"
echo "  cd /scr/biggest/junmiaoh/bwler/scripts/interpolation/mlp_size_k_sweeps"
echo "  for script in mlp_*_sgd_k_sweep.sbatch; do"
echo "    sbatch \$script"
echo "  done"
echo ""
echo "To submit all SSBroyden jobs, run:"
echo "  cd /scr/biggest/junmiaoh/bwler/scripts/interpolation/mlp_size_k_sweeps"
echo "  for script in mlp_*_ssbroyden_k_sweep.sbatch; do"
echo "    sbatch \$script"
echo "  done"
echo ""
echo "Or submit individually:"
echo "Adam jobs:"
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
        echo "  sbatch mlp_${CONFIG_NAME}_adam_k_sweep.sbatch"
    done
done
echo ""
echo "SGD jobs:"
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
        echo "  sbatch mlp_${CONFIG_NAME}_sgd_k_sweep.sbatch"
    done
done
echo ""
echo "SSBroyden jobs:"
for LAYERS in "${LAYERS_LIST[@]}"; do
    for HDIM in "${HDIM_LIST[@]}"; do
        CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
        echo "  sbatch mlp_${CONFIG_NAME}_ssbroyden_k_sweep.sbatch"
    done
done
