#!/bin/bash
#SBATCH -o slurm_burgers.sh
#SBATCH --job-name burgers_training
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00     
#SBATCH --mem=100G

source /home/research/junmiaoh/CME391chris/bwler/.venv/bin/activate

DEVICE="0"
N_EPOCHS=1000
N_T=321
N_X=321

# MLP, Adam
#CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlp --eval_every 1000
# BWLer-hatted MLP, Adam
#CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_t $N_T --n_x $N_X --n_layers 3 --hidden_dim 256 --activation tanh --n_epochs $N_EPOCHS --method adam --sample_type standard --model mlpinterp --eval_every 1000
# Explicit BWLer, Adam
#CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_t $N_T --n_x $N_X --n_epochs $N_EPOCHS --method adam --sample_type standard --model polynomial --eval_every 1000

# MLP, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_layers 3 --hidden_dim 64 --activation tanh --n_epochs 1000 --method ssbroyden --sample_type standard --model mlp --eval_every 50
# BWLer-hatted MLP, SSBroyden
CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_t 121 --n_x 121 --n_layers 3 --hidden_dim 64 --activation tanh --n_epochs 50000 --method ssbroyden --sample_type standard --model mlpinterp --eval_every 100
# Explicit BWLer, SSBroyden
#CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_t 161 --n_x 161 --n_epochs 50000 --method ssbroyden --sample_type standard --model polynomial_fd --eval_every 100 --fd_k 1

# High-precision explicit BWLer
#CUDA_VISIBLE_DEVICES=$DEVICE python -m src.experiments.pdes.benchmarks.burgers --n_t 321 --n_x 321 --n_epochs 850 --method nys_newton --sample_type standard --model polynomial_fd --eval_every 10 --from_pretrained --nncg_rank 1000 --nncg_cgmaxiters 2000 --fd_k 1