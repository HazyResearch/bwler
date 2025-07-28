#!/bin/bash
#SBATCH -o slurm_reaction.sh
#SBATCH --job-name reaction_training
#SBATCH -p gpu-volta
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00     
#SBATCH --mem=10G

source /home/research/junmiaoh/CME391chris/bwler/.venv/bin/activate

DEVICE="0"
N_EPOCHS=100000
RHO=5
N_T=41
N_X=41

# MLP, Adam
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlp --eval_every 1000
# MLP, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_layers 3 --hidden_dim 64 --activation tanh --n_epochs $N_EPOCHS --method ssbroyden --sample_type standard --model mlp --eval_every 50

# BWLer-hatted MLP, Adam
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000

# BWLer-hatted MLP, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_layers 3 --hidden_dim 64 --activation tanh --n_epochs $N_EPOCHS --method ssbroyden --sample_type standard --model mlpinterp --eval_every 50

# Explicit BWLer, Adam
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_epochs $N_EPOCHS --method adam --sample_type standard --model polynomial --eval_every 1000

# Explicit BWLer, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 81 --n_x 81 --n_epochs $N_EPOCHS --method ssbroyden --sample_type standard --model polynomial --eval_every 50
