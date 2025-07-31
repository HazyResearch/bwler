#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_convection_piratenet.out
#SBATCH --job-name convection_piratenet

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.simple.convection --c 40 --n_epochs 20000 --eval_every 50 --sample_type standard --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs 1000000 --eval_every 1000 --method adam --method adam --model piratenet --seed 0
# CUDA_VISIBLE_DEVICES=1 python -m src.experiments.pdes.simple.convection --c 40 --n_epochs 1000000 --eval_every 50 --sample_type standard --n_layers 3 --hidden_dim 64 --activation tanh --method ssbroyden --model piratenet --seed 0