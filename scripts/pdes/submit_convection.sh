#!/bin/bash
#SBATCH -o slurm_convection.sh
#SBATCH --job-name convection_training
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00     
#SBATCH --mem=100G

source /home/research/junmiaoh/CME391chris/bwler/.venv/bin/activate

DEVICE="0"
N_EPOCHS=50000
C=40
N_T=81
N_X=80

# MLP, Adam
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlp --eval_every 1000

# MLP, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_layers 3 --hidden_dim 64 --activation tanh --n_epochs $N_EPOCHS --method ssbroyden --sample_type standard --model mlp --eval_every 50

# BWLer-hatted MLP, Adam
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000

# BWLer-hatted MLP, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 64 --activation tanh --n_epochs $N_EPOCHS --method ssbroyden --sample_type standard --model mlpinterp --eval_every 50

# Explicit BWLer, Adam
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_t $N_T --n_x $N_X --n_epochs $N_EPOCHS --method adam --sample_type standard --model polynomial --eval_every 1000

# Explicit BWLer, SSBroyden
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_t $N_T --n_x $N_X --n_epochs $N_EPOCHS --method ssbroyden --sample_type standard --model polynomial --eval_every 50

# Ablations: test BWLer-hatted MLP for forward pass and derivatives
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000 --use_mlp_for_forward
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c $C --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000 --use_mlp_for_derivatives

# High-precision explicit BWLer
# C = 40
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c 40 --n_t 81 --n_x 80 --n_epochs 350 --method nys_newton --sample_type standard --model polynomial --eval_every 10 --nncg_rank 1000 --nncg_cgmaxiters 100
#SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c 40 --n_t 81 --n_x 80 --n_epochs 350 --method ssbroyden --sample_type standard --model polynomial --eval_every 50


# C = 80
# CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c 80 --n_t 161 --n_x 160 --n_epochs 2500 --method nys_newton --sample_type standard --model polynomial --eval_every 100 --nncg_rank 1000 --nncg_cgmaxiters 100
#SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.simple.convection --c 80 --n_t 161 --n_x 160 --n_epochs 2500 --method ssbroyden --sample_type standard --model polynomial --eval_every 50