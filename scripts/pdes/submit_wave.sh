#!/bin/bash

conda init && conda activate bwler

DEVICE="0"
N_EPOCHS=1000000
C=2
BETA=5
N_T=41
N_X=41

# MLP, Adam
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.wave --c $C --beta $BETA --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlp --eval_every 1000
# BWLer-hatted MLP, Adam
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.wave --c $C --beta $BETA --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000
# Explicit BWLer, Adam
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.wave --c $C --beta $BETA --n_t $N_T --n_x $N_X --n_epochs $N_EPOCHS --method adam --sample_type standard --model polynomial --eval_every 1000

# High-precision explicit BWLer
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.wave --c $C --beta $BETA --n_t 41 --n_x 41 --n_epochs 200 --method nys_newton --sample_type standard --model polynomial --eval_every 10 --nncg_rank 1000 --nncg_cgmaxiters 1000