#!/bin/bash
# Script to generate separate sbatch jobs for frequency analysis sweeps
# Each job sweeps over wavenumbers (k) for a fixed model size with comprehensive frequency analysis
# Focuses on 2-layer models with different hidden dimensions

# Configuration
TARGET="sine"
SEED=0
N_TRAIN_0TH=400  # points for value loss
N_TRAIN_1ST=0    # points for derivative loss (typically 0 for k sweep)
N_TEST=1000
N_EPOCHS=10000
EVAL_EVERY=1000
SAVE_ROOT="/scr/biggest/cdeng/bwler/analysis/freq_sweeps"
DERIV_ALPHA=1.0
DERIV_BETA=1.0

# Embedding configuration
EMBED_M=8  # Feature count for 'cheb' or degree for 'bary' embedding

# MLP configurations to test (focused on 2-layer with different hidden dims)
LAYERS_LIST=(2)
HDIM_LIST=(16 32 64)

# Embeddings to test
EMBEDDING_LIST=("none" "theta" "cheb" "bary")

# Wavenumbers to sweep over (comma-separated list)
K_VALUES="1.0,2.0,4.0,8.0,16.0,32.0,64.0"

# Create scripts directory if it doesn't exist
mkdir -p /scr/biggest/cdeng/bwler/scripts/interpolation/freq_sweeps

# Function to generate sbatch script for a given optimizer and embedding
generate_sbatch_script() {
    local LAYERS=$1
    local HDIM=$2
    local OPTIMIZER=$3
    local EMBEDDING=$4
    local OPTIMIZER_LOWER=$(echo $OPTIMIZER | tr '[:upper:]' '[:lower:]')

    # Set optimizer-specific training parameters
    if [ "$OPTIMIZER" = "SSBroyden" ]; then
        local N_EPOCHS_OPT=20000     # SSBroyden: 20000 epochs for full training
        local EVAL_EVERY_OPT=100    # SSBroyden: log every 100 epochs
        local JOB_TIME="5:00:00"     # SSBroyden: longer job time for frequency analysis

    elif [ "$OPTIMIZER" = "SGD" ]; then
        local N_EPOCHS_OPT=80000     # SGD: 80000 epochs for full training
        local EVAL_EVERY_OPT=1000    # SGD: log every 1000 epochs
        local JOB_TIME="6:00:00"     # SGD: longer job time for more epochs + analysis

    else
        local N_EPOCHS_OPT=80000     # Adam: 80000 epochs for full training
        local EVAL_EVERY_OPT=1000    # Adam: log every 1000 epochs
        local JOB_TIME="5:00:00"     # Adam: longer job time for frequency analysis
    fi

    local CONFIG_NAME="layers${LAYERS}_hdim${HDIM}"
    local EMBED_SUFFIX=""
    if [ "$EMBEDDING" != "none" ]; then
        EMBED_SUFFIX="_${EMBEDDING}"
        if [ "$EMBEDDING" = "cheb" ] || [ "$EMBEDDING" = "bary" ]; then
            EMBED_SUFFIX="${EMBED_SUFFIX}_M${EMBED_M}"
        fi
    fi

    local SAVE_DIR="${SAVE_ROOT}/freq_analysis_${OPTIMIZER_LOWER}${EMBED_SUFFIX}/${CONFIG_NAME}"

    # Set embedding arguments
    local EMBEDDING_ARGS=""
    if [ "$EMBEDDING" != "none" ]; then
        EMBEDDING_ARGS="--embedding \"${EMBEDDING}\""
        if [ "$EMBEDDING" = "cheb" ] || [ "$EMBEDDING" = "bary" ]; then
            EMBEDDING_ARGS="${EMBEDDING_ARGS} --embedding_M \"${EMBED_M}\""
        fi
    fi

    # Create the sbatch script
    cat > "/scr/biggest/cdeng/bwler/scripts/interpolation/freq_sweeps/freq_${CONFIG_NAME}_${OPTIMIZER_LOWER}${EMBED_SUFFIX}_sweep.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=freq_${CONFIG_NAME}_${OPTIMIZER_LOWER}${EMBED_SUFFIX}
#SBATCH --time=${JOB_TIME}
#SBATCH --account=hazy
#SBATCH --partition=hazy
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=freq_${CONFIG_NAME}_${OPTIMIZER_LOWER}${EMBED_SUFFIX}_%j.out
#SBATCH --error=freq_${CONFIG_NAME}_${OPTIMIZER_LOWER}${EMBED_SUFFIX}_%j.err

# Environment
source /scr/biggest/junmiaoh/bwler/.venv/bin/activate

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-32}
export PYTHONUNBUFFERED=1
export MPICH_GPU_SUPPORT_ENABLED=1

# Navigate to project
cd /scr/biggest/cdeng/bwler/
mkdir -p "${SAVE_DIR}"

echo "Starting frequency analysis sweep for MLP ${CONFIG_NAME} with ${OPTIMIZER} and ${EMBEDDING} embedding"
echo "Saving to: ${SAVE_DIR}"

# Run the frequency analysis sweep for this MLP configuration
python scripts/interpolation/run_freq_sweeps.py \\
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
    ${EMBEDDING_ARGS} \\

echo "Completed frequency analysis sweep for MLP ${CONFIG_NAME} with ${OPTIMIZER} and ${EMBEDDING} embedding"
EOF

    local embed_desc=""
    if [ "$EMBEDDING" = "none" ]; then
        embed_desc="no embedding"
    elif [ "$EMBEDDING" = "theta" ]; then
        embed_desc="theta embedding"
    else
        embed_desc="${EMBEDDING} embedding (M=${EMBED_M})"
    fi

    echo "Generated: freq_${CONFIG_NAME}_${OPTIMIZER_LOWER}${EMBED_SUFFIX}_sweep.sbatch (${N_EPOCHS_OPT} epochs, ${embed_desc})"
}

# Generate scripts for each optimizer and embedding combination
for OPTIMIZER in "Adam" "SGD" "SSBroyden"; do
    echo ""
    echo "=== Generating ${OPTIMIZER} frequency analysis scripts ==="
    for EMBEDDING in "${EMBEDDING_LIST[@]}"; do
        echo "  -- ${EMBEDDING} embedding --"
        for LAYERS in "${LAYERS_LIST[@]}"; do
            for HDIM in "${HDIM_LIST[@]}"; do
                generate_sbatch_script $LAYERS $HDIM $OPTIMIZER $EMBEDDING
            done
        done
    done
done

# Calculate total scripts
TOTAL_SCRIPTS=$((${#LAYERS_LIST[@]} * ${#HDIM_LIST[@]} * 3 * ${#EMBEDDING_LIST[@]}))

echo ""
echo "Generated ${TOTAL_SCRIPTS} total frequency analysis sbatch scripts:"
echo ""
echo "=== CONFIGURATION SUMMARY ==="
echo "Architecture:   2-layer MLPs only"
echo "Hidden dims:    ${HDIM_LIST[@]}"
echo "Embeddings:     ${EMBEDDING_LIST[@]} (M=${EMBED_M} for cheb/bary)"
echo "Adam jobs:      80000 epochs, log every 1,000 epochs, 5:00:00 time limit"
echo "SGD jobs:       80000 epochs, log every 1,000 epochs, 6:00:00 time limit"
echo "SSBroyden jobs: 20000 epochs, log every 100 epochs, 5:00:00 time limit"
echo "Wavenumbers:    ${K_VALUES}"
echo "Analysis:       Function decomposition, frequency learning rates, weight specialization"
echo ""

# Generate submission commands for each optimizer-embedding combination
for OPTIMIZER in "Adam" "SGD" "SSBroyden"; do
    OPTIMIZER_LOWER=$(echo $OPTIMIZER | tr '[:upper:]' '[:lower:]')
    echo "=== ${OPTIMIZER} frequency analysis submission commands ==="

    for EMBEDDING in "${EMBEDDING_LIST[@]}"; do
        EMBED_SUFFIX=""
        if [ "$EMBEDDING" != "none" ]; then
            EMBED_SUFFIX="_${EMBEDDING}"
            if [ "$EMBEDDING" = "cheb" ] || [ "$EMBEDDING" = "bary" ]; then
                EMBED_SUFFIX="${EMBED_SUFFIX}_M${EMBED_M}"
            fi
        fi

        echo ""
        echo "To submit all ${OPTIMIZER} jobs with ${EMBEDDING} embedding:"
        echo "  cd /scr/biggest/cdeng/bwler/scripts/interpolation/freq_sweeps"
        echo "  for script in freq_*_${OPTIMIZER_LOWER}${EMBED_SUFFIX}_sweep.sbatch; do"
        echo "    sbatch \$script"
        echo "  done"
    done
done

echo ""
echo "To submit ALL frequency analysis jobs at once:"
echo "  cd /scr/biggest/cdeng/bwler/scripts/interpolation/freq_sweeps"
echo "  for script in freq_*_sweep.sbatch; do"
echo "    sbatch \$script"
echo "  done"

echo ""
echo "=== WHAT THIS ANALYSIS PROVIDES ==="
echo "For each optimizer/embedding/hidden_dim combination, you'll get:"
echo "  - Detailed frequency evolution timelines showing WHEN specific frequencies are learned"
echo "  - Weight specialization analysis tracking how weight matrices evolve in frequency space"
echo "  - Comparison data to understand why Adam behaves differently from SSBroyden/SGD"
echo "  - Function decomposition showing residual vs learned components over training"
echo ""
echo "This will help answer:"
echo "  - Which frequencies does Adam struggle with?"
echo "  - How do weight matrices specialize differently across optimizers?"
echo "  - What is the timeline of frequency learning for each method?"
echo "  - How does embedding choice affect frequency learning dynamics?"