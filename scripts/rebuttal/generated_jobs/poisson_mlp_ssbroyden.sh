#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_poisson_mlp_ssbroyden.out
#SBATCH --job-name poisson_mlp_ssbroyden

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.poisson_2d_cg --n_epochs 1000000 --eval_every 50 --n_layers 3 --hidden_dim 32 --activation tanh --method ssbroyden --model mlp
