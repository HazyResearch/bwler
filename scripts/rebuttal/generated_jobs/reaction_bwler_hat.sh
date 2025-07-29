#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_reaction_bwler_hat.out
#SBATCH --job-name reaction_bwler_hat

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.simple.reaction --rho 5 --n_epochs 20000 --eval_every 50 --sample_type standard --n_t 41 --n_x 41 --n_layers 3 --hidden_dim 32 --activation tanh --method ssbroyden --model mlpinterp --seed 0
