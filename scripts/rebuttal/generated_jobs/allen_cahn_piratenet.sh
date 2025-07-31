#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_allen_cahn_piratenet.out
#SBATCH --job-name allen_cahn_piratenet

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 3000000 --eval_every 50 --sample_type standard --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs 1000000 --eval_every 1000 --method adam --method adam --model piratenet --seed 0
# CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 1000000 --eval_every 50 --sample_type standard --n_layers 3 --hidden_dim 256 --activation tanh --method ssbroyden --model piratenet --seed 0