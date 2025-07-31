#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_allen_cahn_explicit_bwler.out
#SBATCH --job-name allen_cahn_explicit_bwler

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 1000000 --eval_every 50 --sample_type standard --n_t 161 --n_x 161 --method ssbroyden --model polynomial --seed 0
# CUDA_VISIBLE_DEVICES=2 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 100000 --eval_every 50 --sample_type standard --n_t 321 --n_x 321 --method nys_newton --model polynomial_fd --seed 0 --nncg_rank 2000 --nncg_cgmaxiters 2000 --fd_k_t 1
