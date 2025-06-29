#!/bin/bash

conda init && conda activate bwler

DEVICE="0"
N_EPOCHS=51000

# High-precision explicit BWLer
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.poisson_2d_cg --n_x 51 --n_y 51 --n_epochs 51000 --optimizer nys_newton --eval_every 100 --nncg_rank 1000 --nncg_maxiters 64