#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_allen_cahn_mlpinterp_temporal.out
#SBATCH --job-name allen_cahn_mlpinterp_temporal

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 1000000 --eval_every 50 --sample_type standard --n_t 321 --n_x 321 --n_layers 3 --hidden_dim 64 --activation tanh --method ssbroyden --model mlpinterp_temporal --seed 0
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 1000000 --eval_every 50 --sample_type standard --n_t 641 --n_x 641 --n_layers 3 --hidden_dim 64 --activation tanh --method ssbroyden --model mlpinterp_temporal --seed 0
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs 1000000 --eval_every 50 --sample_type standard --n_t 1281 --n_x 1281 --n_layers 3 --hidden_dim 64 --activation tanh --method ssbroyden --model mlpinterp_temporal --seed 0