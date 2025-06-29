#!/bin/bash

# Activate Python environment
conda init && conda activate bwler

# Base directory for saving results
BASE_SAVE_DIR="PATH/TO/interpolants-torch/plots/pdes/fd_stencil_sweep"

# Common parameters
N_EPOCHS=100000
EVAL_EVERY=1000
DEVICE="cuda:1"
SEED=0
METHOD="nys_newton"
NNCG_RANK=1000
NNCG_CGMAXITERS=100

# Stencil sizes to test
K_VALUES=(1 2 3 4 5)

# Create base save directory
mkdir -p ${BASE_SAVE_DIR}

# Run experiments for each PDE
for PDE in "burgers" "allen_cahn"; do
    echo "Running experiments for ${PDE} equation..."
    
    for K in "${K_VALUES[@]}"; do
        echo "Testing stencil size k=${K}..."
        
        # Run with FD in time dimension only
        python -m src.experiments.pdes.benchmarks.${PDE} \
            --model polynomial \
            --method ${METHOD} \
            --n_epochs ${N_EPOCHS} \
            --eval_every ${EVAL_EVERY} \
            --device ${DEVICE} \
            --seed ${SEED} \
            --nncg_rank ${NNCG_RANK} \
            --nncg_cgmaxiters ${NNCG_CGMAXITERS} \
            --fd_k_t ${K} \
            --hessian_every 1000 \
            --hessian_num_iter 100 \
            --hessian_num_run 1
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Successfully completed experiment for ${PDE} with k=${K}"
        else
            echo "Error running experiment for ${PDE} with k=${K}"
            exit 1
        fi
    done
done

echo "All experiments completed successfully!" 