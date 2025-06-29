#!/bin/bash

conda init && conda activate bwler

# Common parameters
N_EPOCHS=100000
EVAL_EVERY=100
METHOD="nys_newton"
SAMPLE_TYPE="standard"

# Function to run reaction equation experiments
run_reaction() {
    local rank=$1
    local device=$2
    
    CUDA_VISIBLE_DEVICES=$device python -m src.experiments.pdes.simple.reaction \
        --rho 5 \
        --n_t 41 \
        --n_x 41 \
        --n_epochs $N_EPOCHS \
        --method $METHOD \
        --sample_type $SAMPLE_TYPE \
        --model polynomial \
        --eval_every $EVAL_EVERY \
        --nncg_rank $rank \
        --nncg_cgmaxiters $rank &
}

# Function to run wave equation experiments
run_wave() {
    local rank=$1
    local device=$2
    
    CUDA_VISIBLE_DEVICES=$device python -m src.experiments.pdes.simple.wave \
        --c 2 \
        --beta 5 \
        --n_t 41 \
        --n_x 41 \
        --n_epochs $N_EPOCHS \
        --method $METHOD \
        --sample_type $SAMPLE_TYPE \
        --model polynomial \
        --eval_every $EVAL_EVERY \
        --nncg_rank $rank \
        --nncg_cgmaxiters $rank &
}

# Run reaction equation experiments
echo "Starting reaction equation experiments..."
for rank in 16 64 256 1024; do
    device=$((rank == 16 ? 0 : rank == 64 ? 1 : rank == 256 ? 2 : 3))
    run_reaction $rank $device
done

# Wait for reaction experiments to complete
wait

# Run wave equation experiments
echo "Starting wave equation experiments..."
for rank in 16 64 256 1024; do
    device=$((rank == 16 ? 0 : rank == 64 ? 1 : rank == 256 ? 2 : 3))
    run_wave $rank $device
done

# Wait for all experiments to complete
wait

echo "All experiments completed!" 