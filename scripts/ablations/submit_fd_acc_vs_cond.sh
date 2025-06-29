#!/bin/bash

# Activate Python environment
conda init && conda activate bwler

# Base directory for saving results
# TODO: replace with your own path
BASE_SAVE_DIR="PATH/TO/interpolants-torch/plots/pdes/fd_acc_vs_cond"

# Common parameters
N_EPOCHS=100000
EVAL_EVERY=1000
DEVICE="cuda:1"
SEED=0

# Run experiments for each PDE and model type
for MODEL_TYPE in "interpolant" "mlpinterp"; do
    echo "Running experiments with ${MODEL_TYPE} model..."
    
    echo "Running Reaction equation experiments..."
    python -m src.experiments.pdes.simple.fd_acc_vs_cond \
        --pde reaction \
        --rho 5.0 \
        --n_epochs ${N_EPOCHS} \
        --eval_every ${EVAL_EVERY} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --deriv_type fd \
        --model_type ${MODEL_TYPE}

    python -m src.experiments.pdes.simple.fd_acc_vs_cond \
        --pde reaction \
        --rho 5.0 \
        --n_epochs ${N_EPOCHS} \
        --eval_every ${EVAL_EVERY} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --deriv_type spectral \
        --model_type ${MODEL_TYPE}

    echo "Running convection equation experiments..."
    python -m src.experiments.pdes.simple.fd_acc_vs_cond \
        --pde convection \
        --c 40 \
        --n_epochs ${N_EPOCHS} \
        --eval_every ${EVAL_EVERY} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --deriv_type fd \
        --model_type ${MODEL_TYPE}

    python -m src.experiments.pdes.simple.fd_acc_vs_cond \
        --pde convection \
        --c 40 \
        --n_epochs ${N_EPOCHS} \
        --eval_every ${EVAL_EVERY} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --deriv_type spectral \
        --model_type ${MODEL_TYPE}

    echo "Running Wave equation experiments..."
    python -m src.experiments.pdes.simple.fd_acc_vs_cond \
        --pde wave \
        --c 2 \
        --beta 5 \
        --n_epochs ${N_EPOCHS} \
        --eval_every ${EVAL_EVERY} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --deriv_type fd \
        --model_type ${MODEL_TYPE}

    python -m src.experiments.pdes.simple.fd_acc_vs_cond \
        --pde wave \
        --c 2 \
        --beta 5 \
        --n_epochs ${N_EPOCHS} \
        --eval_every ${EVAL_EVERY} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --deriv_type spectral \
        --model_type ${MODEL_TYPE}
done

echo "All experiments completed!"