import argparse
import os
import torch
import numpy as np
from typing import List, Dict, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.experiments.pdes.simple.reaction import Reaction
from src.experiments.pdes.simple.convection import convection
from src.experiments.pdes.simple.wave import Wave
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger
from src.optimizers.nys_newton_cg import NysNewtonCG


def run_fd_sweep(
    pde: Union[Reaction, convection, Wave],
    n_t: int,
    n_x: int,
    h_values: List[float],
    n_epochs: int = 100000,
    eval_every: int = 1000,
    device: str = "cuda",
    save_dir: str = None,
    seed: int = 0,
    deriv_type: str = "fd",
    model_type: str = "interpolant",
):
    """Run experiments with different FD step sizes.

    Args:
        pde: PDE object (Reaction, convection, or Wave)
        n_t: Number of time points
        n_x: Number of spatial points
        h_values: List of FD step sizes to try
        n_epochs: Number of training epochs
        eval_every: How often to evaluate
        device: Device to use
        save_dir: Directory to save results
        seed: Random seed
        deriv_type: Type of derivatives to use ("fd" or "spectral")
        model_type: Type of model to use ("interpolant" or "mlpinterp")
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model
    if isinstance(pde, convection):
        bases = ["chebyshev", "fourier"]
    else:  # Reaction or Wave
        bases = ["chebyshev", "chebyshev"]

    # Run experiments for each h value
    for h in h_values:
        print(f"\nRunning experiment with h = {h}")

        # Initialize model with appropriate derivative settings
        if model_type == "interpolant":
            model = SpectralInterpolationND(
                Ns=[n_t, n_x],
                bases=bases,
                domains=pde.domain,
                device=device,
                fd_h=[h, h] if deriv_type == "fd" else [None, None],
                fd_scheme=(
                    ["central", "central"] if deriv_type == "fd" else [None, None]
                ),
            )
        else:  # mlpinterp
            model = MLPSpectralInterpolationND(
                Ns=[n_t, n_x],
                bases=bases,
                domains=pde.domain,
                device=device,
                hidden_layers=(256,) * 3,  # 3 layers, 256 hidden dim
                activation=torch.tanh,
                fd_h=[h, h] if deriv_type == "fd" else [None, None],
                fd_scheme=(
                    ["central", "central"] if deriv_type == "fd" else [None, None]
                ),
            )

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Set up samplers
        n_t_train = 2 * n_t
        n_x_train = 2 * n_x
        n_ic_train = 2 * n_x
        ic_weight = 10

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
            return [torch.tensor([0.0], requires_grad=True, device=device), ic_nodes]

        # Set up evaluation
        n_eval = 200
        t_eval = torch.linspace(
            pde.domain[0][0],
            pde.domain[0][1],
            n_eval,
            device=device,
            requires_grad=True,
        )
        x_eval = torch.linspace(
            pde.domain[1][0],
            pde.domain[1][1],
            n_eval,
            device=device,
            requires_grad=True,
        )

        def eval_sampler():
            return [t_eval, x_eval]

        eval_metrics = [l2_error, max_error, l2_relative_error]

        # Create logger
        h_str = f"h={h:.2e}" if h is not None else "spectral"
        h_save_dir = os.path.join(save_dir, h_str) if save_dir is not None else None
        logger = (
            Logger(path=os.path.join(h_save_dir, "logger.json")) if h_save_dir else None
        )

        # Train using PDE's built-in train method
        pde.train(
            model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=eval_every,
            save_dir=h_save_dir,
            logger=logger,
        )


def plot_results(results: Dict, save_dir: str, deriv_type: str):
    """Plot results of FD sweep."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Error vs h
    ax1.loglog(results["h_values"], results["train_errors"], "o-", label="Train")
    ax1.loglog(results["h_values"], results["test_errors"], "o--", label="Test")
    ax1.set_xlabel("Step size h")
    ax1.set_ylabel("Error")
    ax1.set_title("Error vs Step Size")
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # Plot 2: Condition number vs h
    ax2.loglog(results["h_values"], results["condition_numbers"], "o-")
    ax2.set_xlabel("Step size h")
    ax2.set_ylabel("Condition Number")
    ax2.set_title("Condition Number vs Step Size")
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 3: Convergence rate
    if deriv_type == "fd" and len(results["convergence_rates"]) > 0:
        ax3.semilogx(results["h_values"][1:], results["convergence_rates"], "o-")
        ax3.axhline(y=2, color="r", linestyle="--", label="2nd order")
        ax3.set_xlabel("Step size h")
        ax3.set_ylabel("Convergence Rate")
        ax3.set_title("Convergence Rate vs Step Size")
        ax3.grid(True, which="both", ls="-", alpha=0.2)
        ax3.legend()

    # Plot 4: Error vs Condition number
    ax4.loglog(
        results["condition_numbers"], results["train_errors"], "o-", label="Train"
    )
    ax4.loglog(
        results["condition_numbers"], results["test_errors"], "o--", label="Test"
    )
    ax4.set_xlabel("Condition Number")
    ax4.set_ylabel("Error")
    ax4.set_title("Error vs Condition Number")
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fd_sweep_analysis_{deriv_type}.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pde",
        type=str,
        default="reaction",
        choices=["reaction", "convection", "wave"],
        help="PDE to solve",
    )
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--n_t", type=int, default=None)
    parser.add_argument("--n_x", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deriv_type",
        type=str,
        default="fd",
        choices=["fd", "spectral"],
        help="Type of derivative to use: 'fd' for finite difference or 'spectral' for spectral derivatives",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="interpolant",
        choices=["interpolant", "mlpinterp"],
        help="Type of model to use: 'interpolant' for standard interpolant or 'mlpinterp' for MLP+Interpolant",
    )
    args = parser.parse_args()

    # Set default n_t and n_x based on PDE type
    if args.n_t is None:
        if args.pde == "convection":
            args.n_t = 81
        elif args.pde == "wave":
            args.n_t = 41
        else:  # reaction
            args.n_t = 41

    if args.n_x is None:
        if args.pde == "convection":
            args.n_x = 80
        elif args.pde == "wave":
            args.n_x = 41
        else:  # reaction
            args.n_x = 41

    # Define FD step sizes to sweep
    h_values = np.logspace(-2, -5, 4)  # h from 0.01 to 1e-5

    # Base save directory
    base_save_dir = os.getenv(
        "BASE_SAVE_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "plots/pdes/fd_acc_vs_cond",
        ),
    )
    # Create PDE-specific directory name
    if args.pde == "reaction":
        pde_dir = f"pde=reaction_rho={args.rho}"
    elif args.pde == "convection":
        pde_dir = f"pde=convection_c={args.c}"
    else:  # wave
        pde_dir = f"pde=wave_c={args.c}_beta={args.beta}"

    # Initialize appropriate PDE
    if args.pde == "reaction":
        pde = Reaction(rho=args.rho, device=args.device)
    elif args.pde == "convection":
        pde = convection(c=args.c, device=args.device)
    else:  # wave
        pde = Wave(c=args.c, beta=args.beta, device=args.device)

    # For FD derivatives, we'll create a subdirectory for each h value
    if args.deriv_type == "fd":
        for h in h_values:
            save_dir = os.path.join(
                base_save_dir,
                f"{pde_dir}_nt={args.n_t}_nx={args.n_x}_{args.deriv_type}/model={args.model_type}",
            )
            os.makedirs(save_dir, exist_ok=True)
            run_fd_sweep(
                pde=pde,
                n_t=args.n_t,
                n_x=args.n_x,
                h_values=[
                    h
                ],  # Pass single h value since we're creating a directory for each
                n_epochs=args.n_epochs,
                eval_every=args.eval_every,
                device=args.device,
                save_dir=save_dir,
                seed=args.seed,
                deriv_type=args.deriv_type,
                model_type=args.model_type,
            )
    else:  # spectral
        save_dir = os.path.join(
            base_save_dir,
            f"{pde_dir}_nt={args.n_t}_nx={args.n_x}_{args.deriv_type}/model={args.model_type}",
        )
        os.makedirs(save_dir, exist_ok=True)
        run_fd_sweep(
            pde=pde,
            n_t=args.n_t,
            n_x=args.n_x,
            h_values=[None],  # Pass None for spectral
            n_epochs=args.n_epochs,
            eval_every=args.eval_every,
            device=args.device,
            save_dir=save_dir,
            seed=args.seed,
            deriv_type=args.deriv_type,
            model_type=args.model_type,
        )
