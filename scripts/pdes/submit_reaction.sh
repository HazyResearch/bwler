#!/bin/bash

conda init && conda activate bwler

DEVICE="0"
N_EPOCHS=1000000
RHO=5
N_T=41
N_X=41

# MLP, Adam
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlp --eval_every 1000
# BWLer-hatted MLP, Adam
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000
# Explicit BWLer, Adam
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_epochs $N_EPOCHS --method adam --sample_type standard --model polynomial --eval_every 1000

# High-precision explicit BWLer
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 81 --n_x 81 --n_epochs 250000 --method nys_newton --sample_type standard --model polynomial --eval_every 1000 --nncg_rank 16 --nncg_cgmaxiters 16