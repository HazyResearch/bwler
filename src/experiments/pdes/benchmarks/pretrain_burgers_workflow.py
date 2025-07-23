"""
Pre-training workflow for Burgers equation:
1. Train BWLer-hatted MLP (MLPSpectralInterpolationND) 
2. Extract values at interpolation nodes
3. Initialize explicit polynomial BWLer (SpectralInterpolationND) with these values
4. Continue training the polynomial model

Usage:
    python scripts/pdes/pretrain_burgers_workflow.py --nu 0.01 --n_t 81 --n_x 81 --pretrain_epochs 5000 --final_epochs 1000
"""

import argparse
import os
import torch
import torch.nn as nn
from datetime import datetime
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.experiments.pdes.benchmarks.burgers import Burgers
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger
from src.optimizers.ssbroyden import SSBroyden2


def transfer_mlp_to_polynomial(
    mlp_model: MLPSpectralInterpolationND, 
    poly_model: SpectralInterpolationND
) -> None:
    """
    Transfer learned values from BWLer-hatted MLP to explicit polynomial BWLer.
    
    Args:
        mlp_model: Trained MLPSpectralInterpolationND model
        poly_model: SpectralInterpolationND model to initialize
    """
    print("Transferring values from MLP model to polynomial model...")
    
    # Extract values at interpolation nodes from the MLP model
    with torch.no_grad():
        # Evaluate the MLP at its own interpolation nodes
        node_values = mlp_model.interpolate(mlp_model.nodes)
        
        # Set the polynomial model's values
        poly_model.values.data = node_values.detach().clone()
    
    print(f"Transferred values with shape: {poly_model.values.shape}")
    print(f"Value range: [{poly_model.values.min().item():.4f}, {poly_model.values.max().item():.4f}]")


def pretrain_mlp_model(
    pde: Burgers,
    n_t: int,
    n_x: int,
    n_epochs: int,
    hidden_dim: int = 64,
    n_layers: int = 3,
    optimizer_type: str = "adam",
    save_dir: str = None,
    eval_every: int = 100,
) -> MLPSpectralInterpolationND:
    """
    Pre-train a BWLer-hatted MLP model for Burgers equation.
    
    Returns:
        Trained MLPSpectralInterpolationND model
    """
    print(f"Pre-training BWLer-hatted MLP model for {n_epochs} epochs...")
    
    # Initialize BWLer-hatted MLP model
    bases = ["chebyshev", "chebyshev"]
    mlp_model = MLPSpectralInterpolationND(
        Ns=[n_t, n_x],
        bases=bases,
        domains=pde.domain,
        device=pde.device,
        hidden_layers=(hidden_dim,) * n_layers,
        activation=torch.tanh,
    )
    
    # Setup optimizer
    if optimizer_type == "ssbroyden":
        optimizer = SSBroyden2(
            mlp_model.parameters(),
            lr=1.0,
            init_scale=True,
            c1=1e-4,
            c2=0.9,
            max_ls=20,
        )
    else:
        optimizer = pde.get_optimizer(mlp_model, optimizer_type)
    
    # Setup logger
    if save_dir:
        logger = Logger(path=os.path.join(save_dir, "pretrain_logger.json"))
    else:
        logger = None
    
    # Training parameters
    n_t_train = 161
    n_x_train = 161
    n_ic_train = 161
    ic_weight = 10
    
    # Define samplers
    def pde_sampler():
        t_nodes = pde.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis=bases[0],
            type="standard",
        )
        x_nodes = pde.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis=bases[1],
            type="standard",
        )
        return [t_nodes, x_nodes]
    
    def ic_sampler():
        ic_nodes = pde.sample_domain_1d(
            n_samples=n_ic_train,
            dim=1,
            basis=bases[1],
            type="standard",
        )
        return [torch.tensor([0.0], device=pde.device, requires_grad=True), ic_nodes]
    
    def eval_sampler():
        t_eval = torch.linspace(pde.domain[0][0], pde.domain[0][1], 200, device=pde.device, requires_grad=True)
        x_eval = torch.linspace(pde.domain[1][0], pde.domain[1][1], 200, device=pde.device, requires_grad=True)
        return t_eval, x_eval
    
    eval_metrics = [l2_error, max_error, l2_relative_error]
    
    # Train the model
    print("Starting pre-training...")
    pde.train(
        model=mlp_model,
        n_epochs=n_epochs,
        optimizer=optimizer,
        pde_sampler=pde_sampler,
        ic_sampler=ic_sampler,
        ic_weight=ic_weight,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
        lr_schedule=True,
        gradient_clip=1.0,
    )
    
    print("Pre-training completed!")
    return mlp_model


def train_polynomial_model(
    pde: Burgers,
    poly_model: SpectralInterpolationND,
    n_epochs: int,
    optimizer_type: str = "ssbroyden",
    save_dir: str = None,
    eval_every: int = 50,
) -> SpectralInterpolationND:
    """
    Train the polynomial model starting from pre-trained initialization.
    
    Returns:
        Trained SpectralInterpolationND model
    """
    print(f"Training polynomial model for {n_epochs} epochs...")
    
    # Setup optimizer
    if optimizer_type == "ssbroyden":
        optimizer = SSBroyden2(
            poly_model.parameters(),
            lr=1.0,
            init_scale=True,
            c1=1e-4,
            c2=0.9,
            max_ls=20,
        )
    elif optimizer_type == "nys_newton":
        from src.optimizers.nys_newton_cg import NysNewtonCG
        optimizer = NysNewtonCG(
            poly_model.parameters(),
            lr=1,
            rank=1000,
            cg_max_iters=100,
            mu=1e-2,
            cg_tol=1e-16,
            line_search_fn="armijo",
        )
    else:
        optimizer = pde.get_optimizer(poly_model, optimizer_type)
    
    # Setup logger
    if save_dir:
        logger = Logger(path=os.path.join(save_dir, "final_logger.json"))
    else:
        logger = None
    
    # Training parameters
    bases = ["chebyshev", "chebyshev"]
    n_t_train = 161
    n_x_train = 161
    n_ic_train = 161
    ic_weight = 10
    
    # Define samplers
    def pde_sampler():
        t_nodes = pde.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis=bases[0],
            type="standard",
        )
        x_nodes = pde.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis=bases[1],
            type="standard",
        )
        return [t_nodes, x_nodes]
    
    def ic_sampler():
        ic_nodes = pde.sample_domain_1d(
            n_samples=n_ic_train,
            dim=1,
            basis=bases[1],
            type="standard",
        )
        return [torch.tensor([0.0], device=pde.device, requires_grad=True), ic_nodes]
    
    def eval_sampler():
        t_eval = torch.linspace(pde.domain[0][0], pde.domain[0][1], 200, device=pde.device, requires_grad=True)
        x_eval = torch.linspace(pde.domain[1][0], pde.domain[1][1], 200, device=pde.device, requires_grad=True)
        return t_eval, x_eval
    
    eval_metrics = [l2_error, max_error, l2_relative_error]
    
    # Train the model
    print("Starting final training...")
    pde.train(
        model=poly_model,
        n_epochs=n_epochs,
        optimizer=optimizer,
        pde_sampler=pde_sampler,
        ic_sampler=ic_sampler,
        ic_weight=ic_weight,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
    )
    
    print("Final training completed!")
    return poly_model


def main():
    parser = argparse.ArgumentParser(description="Pre-training workflow for Burgers equation")
    
    # Problem parameters
    parser.add_argument("--nu", type=float, default=0.01, help="Viscosity parameter")
    parser.add_argument("--n_t", type=int, default=81, help="Number of time nodes")
    parser.add_argument("--n_x", type=int, default=81, help="Number of space nodes")
    
    # Pre-training parameters
    parser.add_argument("--pretrain_epochs", type=int, default=5000, help="Pre-training epochs")
    parser.add_argument("--pretrain_optimizer", type=str, default="adam", choices=["adam", "ssbroyden"], help="Pre-training optimizer")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for MLP")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of MLP layers")
    
    # Final training parameters
    parser.add_argument("--final_epochs", type=int, default=1000, help="Final training epochs")
    parser.add_argument("--final_optimizer", type=str, default="ssbroyden", choices=["adam", "ssbroyden", "nys_newton"], help="Final optimizer")
    
    # Evaluation parameters
    parser.add_argument("--pretrain_eval_every", type=int, default=100, help="Pre-training evaluation frequency")
    parser.add_argument("--final_eval_every", type=int, default=50, help="Final evaluation frequency")
    
    # Output parameters
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    # Set up
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"plots/pdes/burgers_pretrain_{timestamp}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("BURGERS EQUATION PRE-TRAINING WORKFLOW")
    print("="*80)
    print(f"Problem parameters: nu={args.nu}, grid=({args.n_t}, {args.n_x})")
    print(f"Pre-training: {args.pretrain_epochs} epochs with {args.pretrain_optimizer}")
    print(f"Final training: {args.final_epochs} epochs with {args.final_optimizer}")
    print(f"Device: {device}")
    print(f"Save directory: {args.save_dir}")
    print("="*80)
    
    # Initialize PDE
    pde = Burgers(nu=args.nu, device=device)
    
    # Stage 1: Pre-train BWLer-hatted MLP
    print("\n" + "="*50)
    print("STAGE 1: PRE-TRAINING BWLER-HATTED MLP")
    print("="*50)
    
    pretrain_save_dir = os.path.join(args.save_dir, "pretrain")
    os.makedirs(pretrain_save_dir, exist_ok=True)
    
    mlp_model = pretrain_mlp_model(
        pde=pde,
        n_t=args.n_t,
        n_x=args.n_x,
        n_epochs=args.pretrain_epochs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        optimizer_type=args.pretrain_optimizer,
        save_dir=pretrain_save_dir,
        eval_every=args.pretrain_eval_every,
    )
    
    # Save pre-trained model
    torch.save(mlp_model.state_dict(), os.path.join(pretrain_save_dir, "pretrained_mlp.pt"))
    
    # Stage 2: Initialize polynomial model
    print("\n" + "="*50)
    print("STAGE 2: INITIALIZING POLYNOMIAL MODEL")
    print("="*50)
    
    # Create polynomial model with same configuration
    bases = ["chebyshev", "chebyshev"]
    poly_model = SpectralInterpolationND(
        Ns=[args.n_t, args.n_x],
        bases=bases,
        domains=pde.domain,
        device=pde.device,
    )
    
    # Transfer values from MLP to polynomial
    transfer_mlp_to_polynomial(mlp_model, poly_model)
    
    # Stage 3: Train polynomial model
    print("\n" + "="*50)
    print("STAGE 3: FINAL TRAINING OF POLYNOMIAL MODEL")
    print("="*50)
    
    final_save_dir = os.path.join(args.save_dir, "final")
    os.makedirs(final_save_dir, exist_ok=True)
    
    poly_model = train_polynomial_model(
        pde=pde,
        poly_model=poly_model,
        n_epochs=args.final_epochs,
        optimizer_type=args.final_optimizer,
        save_dir=final_save_dir,
        eval_every=args.final_eval_every,
    )
    
    # Save final model
    torch.save(poly_model.state_dict(), os.path.join(final_save_dir, "final_polynomial.pt"))
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Pre-trained MLP saved to: {pretrain_save_dir}/pretrained_mlp.pt")
    print(f"Final polynomial saved to: {final_save_dir}/final_polynomial.pt")
    print(f"Logs and plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()