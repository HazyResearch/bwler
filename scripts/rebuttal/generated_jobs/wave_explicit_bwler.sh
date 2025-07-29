#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_wave_explicit_bwler.out
#SBATCH --job-name wave_explicit_bwler

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

CUDA_VISIBLE_DEVICES=0 python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_epochs 20000 --eval_every 50 --sample_type standard --n_t 41 --n_x 41 --method ssbroyden --model polynomial
