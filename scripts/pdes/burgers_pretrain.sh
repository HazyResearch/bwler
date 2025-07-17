#!/bin/bash
#SBATCH -o slurm_burgers.sh
#SBATCH --job-name burgers_training
#SBATCH -p gpu-dgx
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00     
#SBATCH --mem=200G

source /home/research/junmiaoh/CME391chris/bwler/.venv/bin/activate

DEVICE="0"
N_EPOCHS=1000
N_T=161
N_X=161
HIDDEN_DIM=64
N_LAYERS=3

# Training parameters
PRETRAIN_EPOCHS=50000
FINAL_EPOCHS=20000
PRETRAIN_EVAL_EVERY=100
FINAL_EVAL_EVERY=100

# Create timestamp for unique directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="plots/pdes/burgers_pretrain_${TIMESTAMP}"

echo "Starting Burgers pre-training workflow..."
echo "Configuration:"
echo "  - Grid size: ${N_T} x ${N_X}"
echo "  - Pre-training epochs: ${PRETRAIN_EPOCHS}"
echo "  - Final epochs: ${FINAL_EPOCHS}"
echo "  - Save directory: ${SAVE_DIR}"
echo ""

# Run the pre-training workflow
CUDA_VISIBLE_DEVICES=$DEVICE python src/experiments/pdes/benchmarks/burgers_pretrain_train.py \
    --n_t $N_T \
    --n_x $N_X \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --pretrain_optimizer adam \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --final_epochs $FINAL_EPOCHS \
    --final_optimizer ssbroyden \
    --pretrain_eval_every $PRETRAIN_EVAL_EVERY \
    --final_eval_every $FINAL_EVAL_EVERY \
    --save_dir $SAVE_DIR \
    --seed 0

echo ""
echo "Workflow completed! Results saved to: ${SAVE_DIR}"