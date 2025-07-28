#!/bin/bash
#SBATCH -o slurm_poisson.sh
#SBATCH --job-name poisson_training
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00     
#SBATCH --mem=100G

source /home/research/junmiaoh/CME391chris/bwler/.venv/bin/activate

DEVICE="0"
N_EPOCHS=51000

# High-precision explicit BWLer
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.poisson_2d_cg --n_x 51 --n_y 51 --n_epochs 51000 --optimizer nys_newton --eval_every 100 --nncg_rank 1000 --nncg_maxiters 64

CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.poisson_2d_cg --n_x 51 --n_y 51 --n_epochs 50000 --optimizer ssbroyden --eval_every 100 --nncg_rank 1000 --nncg_maxiters 64