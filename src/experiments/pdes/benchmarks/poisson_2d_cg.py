import os
import sys
import argparse

# Add the project root to the Python path. Might need to comment out depending on the cluster HPC environment.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(project_root)

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP

"""
Laplace equation in 2D with complex geometry:

∇²u = 0 in Ω
u = 1 on ∂Ω_outer
u = 0 on ∂Ω_inner

where:
- Ω is the unit square [0,1]×[0,1] with four circular holes
- The circular holes are centered at (±0.3, ±0.3) with radius 0.1
- The boundary condition is 1 on the outer square boundary
- The boundary condition is 0 on the inner circular boundaries

This benchmark tests the ability of neural networks to approximate solutions on domains
with complex geometry where traditional mesh-based methods might be challenging.
"""


class Poisson2DCG(BasePDE):
    valid_points = None

    def __init__(
        self,
        device: str = "cpu",
        loss_weight_update_policy: str = "grad_norm",
        loss_weight_update_interval: int = -1,
        debug: bool = False,
    ):
        # Domain is [-1,1] x [-1,1] with holes
        super().__init__(
            name="poisson_2d_cg",
            domain=[(-0.5, 0.5), (-0.5, 0.5)],
            device=device,
            loss_weight_update_policy=loss_weight_update_policy,
            loss_weight_update_interval=loss_weight_update_interval,
        )

        self.debug = debug

        # Load reference solution
        ref_path = os.path.join(
            project_root, "src/experiments/pdes/benchmarks/ref/poisson2d_cg_data.dat"
        )
        self.ref_points, self.ref_values = self._load_reference_solution(ref_path)

        # Move reference solution to device
        self.ref_points = self.ref_points.to(self.device)
        self.ref_values = self.ref_values.to(self.device)
        self.valid_points = None

    def _load_reference_solution(
        self, filepath: Path
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load reference solution from file"""
        # Load data, skipping comment lines starting with %
        data = np.loadtxt(filepath, comments="%")

        # Extract coordinates and values
        points = torch.tensor(
            data[:, :2], dtype=torch.float32, device=self.device
        )  # x, y coordinates
        values = torch.tensor(
            data[:, 2], dtype=torch.float32, device=self.device
        )  # solution values

        return points, values

    def plot_reference_solution(self, save_path: str = None):
        """Plot the reference solution"""
        # Convert to numpy for plotting
        points = self.ref_points.cpu().numpy()
        values = self.ref_values.cpu().numpy()

        # Print grid size information
        if self.debug:
            print(f"Total number of points: {len(points)}")
            print(
                f"Points shape: {points.shape}, x size: {points[:,0].shape}, y size: {points[:,1].shape}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left subplot: Solution values
        scatter = ax1.scatter(
            points[:, 0], points[:, 1], c=values, cmap="viridis", s=10
        )
        fig.colorbar(scatter, ax=ax1, label="Solution")
        ax1.set_title("Poisson Equation Reference Solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.axis("equal")

        # Right subplot: Node distribution
        ax2.scatter(points[:, 0], points[:, 1], s=10, alpha=0.7)
        ax2.set_title("Reference Solution Node Distribution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        # First we consider the case that we have a SpectralInterpolationND model.
        if isinstance(model, SpectralInterpolationND):

            # Compute the solution and its derivatives based on current model parameter values.
            u_xx = model.derivative(pde_nodes, k=(2, 0))
            u_yy = model.derivative(pde_nodes, k=(0, 2))

            # Compute the value of the function at the boundary nodes from initial conditions
            u_boundary = model.forward(ic_nodes)

            # Split boundary nodes into exterior square boundary and interior circular boundaries
            # Assuming ic_nodes contains metadata or structure to identify which boundary is which
            # For example, first n_square_boundary points are for square, rest for circles
            n_square_boundary = kwargs.get("n_square_boundary", 0)

            if n_square_boundary > 0:
                # Square exterior boundary (Dirichlet boundary condition = 1)
                u_square_boundary = u_boundary[:n_square_boundary]
                square_boundary_values = torch.ones_like(u_square_boundary)
                square_boundary_loss = torch.mean(
                    (u_square_boundary - square_boundary_values) ** 2
                )

                # Circular interior boundaries (Dirichlet boundary condition = 0)
                u_circle_boundary = u_boundary[n_square_boundary:]
                circle_boundary_loss = torch.mean(u_circle_boundary**2)

            else:
                # If boundary separation is not provided, treat all as one type
                # This is a fallback and should be avoided in practice
                boundary_values = kwargs.get(
                    "boundary_values", torch.zeros_like(u_boundary)
                )
                boundary_loss = torch.mean((u_boundary - boundary_values) ** 2)
                square_boundary_loss = boundary_loss
                circle_boundary_loss = 0.0

            pde_residual = u_xx + u_yy
            pde_loss = torch.mean(pde_residual**2)
            boundary_loss = square_boundary_loss + circle_boundary_loss

            loss_names = ["pde_loss", "boundary_loss"]

            return dict(zip(loss_names, [pde_loss, boundary_loss]))

        else:
            raise ValueError(f"Model type {type(model)} not supported")

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes, **kwargs)

        pde_weight = self.loss_weights.get("pde_loss_weight", 1.0)
        ic_weight = self.loss_weights.get("ic_loss_weight", ic_weight)

        loss = (pde_weight * loss_dict["pde_loss"]) + (
            ic_weight * loss_dict["boundary_loss"]
        )

        return loss, loss_dict["pde_loss"], loss_dict["boundary_loss"]

    def plot_solution(
        self,
        model: nn.Module,
        nodes: List[torch.Tensor],
        u: torch.Tensor,
        save_path: str = None,
    ):
        """Plot the predicted solution, ground truth, error as scatter plots, and a pcolormesh as a new column."""
        import matplotlib

        # nodes: [x, y], u: predicted solution at those nodes
        x = nodes[:, 0].detach().cpu().numpy()
        y = nodes[:, 1].detach().cpu().numpy()
        u_pred = u.detach().cpu()
        u_true = self.get_solution(nodes).detach().cpu()
        error = u_pred - u_true
        u_pred = u_pred.numpy()
        u_true = u_true.numpy()
        error = error.numpy()

        # --- Prepare dense grid for pcolormesh ---
        grid_res = 200
        xg = np.linspace(-0.5, 0.5, grid_res)
        yg = np.linspace(-0.5, 0.5, grid_res)
        Xg, Yg = np.meshgrid(xg, yg)
        grid_points = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
        grid_points_torch = torch.tensor(
            grid_points, dtype=nodes.dtype, device=nodes.device
        )
        with torch.no_grad():
            u_grid = u.new_zeros(grid_points_torch.shape[0])
            u_grid = model.forward(grid_points_torch).detach().cpu().numpy()
        U_grid = u_grid.reshape(grid_res, grid_res)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        # First subplot: just the (x,y) points
        axes[0].scatter(x, y, s=10, c="blue")
        axes[0].set_title("Sample Points")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].axis("equal")
        # Plot the circular holes
        centers = [(0.3, 0.3), (0.3, -0.3), (-0.3, 0.3), (-0.3, -0.3)]
        radius = 0.1
        for center in centers:
            circle = matplotlib.patches.Circle(
                center, radius, fill=False, color="red", linewidth=1
            )
            axes[0].add_patch(circle)
        # Second subplot: predicted solution
        sc1 = axes[1].scatter(x, y, c=u_pred, cmap="viridis", s=10)
        plt.colorbar(sc1, ax=axes[1], label="Predicted Solution")
        axes[1].set_title("Predicted Solution")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].axis("equal")
        # Third subplot: ground truth
        sc2 = axes[2].scatter(x, y, c=u_true, cmap="viridis", s=10)
        plt.colorbar(sc2, ax=axes[2], label="Ground Truth")
        axes[2].set_title("Ground Truth")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].axis("equal")
        # Fourth subplot: error
        sc3 = axes[3].scatter(x, y, c=error, cmap="inferno", s=10)
        plt.colorbar(sc3, ax=axes[3], label="Absolute Error")
        l2_error = np.sqrt(np.mean(error**2))
        rel_l2_error = l2_error / np.sqrt(np.mean(u_true**2))
        axes[3].set_title(f"Absolute Error (mean={rel_l2_error:.3e})")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        axes[3].axis("equal")
        # Fifth subplot: pcolormesh of predicted solution on dense grid
        pcm = axes[4].pcolormesh(Xg, Yg, U_grid, cmap="viridis", shading="auto")
        plt.colorbar(pcm, ax=axes[4], label="Predicted Solution (Grid)")
        axes[4].set_title("Predicted Solution (Dense Grid)")
        axes[4].set_xlabel("x")
        axes[4].set_ylabel("y")
        axes[4].axis("equal")
        for center in centers:
            circle = matplotlib.patches.Circle(
                center, radius, fill=False, color="red", linewidth=1
            )
            axes[4].add_patch(circle)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def get_solution(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        """Get the reference solution values at the given nodes by nearest neighbor lookup."""
        # nodes: [x, y], each of shape (N,)
        x, y = nodes[:, 0], nodes[:, 1]
        ref_x = self.ref_points[:, 0]
        ref_y = self.ref_points[:, 1]
        # For each (x, y), find the closest point in ref_points
        # Compute squared distances (broadcasted)
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        ref_x = ref_x.view(1, -1)
        ref_y = ref_y.view(1, -1)
        dists = (x - ref_x) ** 2 + (y - ref_y) ** 2  # shape (N, n_ref)
        nn_indices = torch.argmin(dists, dim=1)  # shape (N,)
        # Return the reference values at these indices
        return self.ref_values[nn_indices]

    def plot_sampled_points(self, sample_x, sample_y, save_path=None, n_points=None):
        """Plot sampled points in the domain with holes and save or show the plot."""
        if n_points is None:
            n_points = len(sample_x)
        plt.figure(figsize=(8, 8))
        plt.scatter(
            sample_x.detach().cpu().numpy(),
            sample_y.detach().cpu().numpy(),
            s=10,
            alpha=0.6,
            c="blue",
            label="Interior points",
        )
        # Plot the domain boundary
        boundary_x = torch.linspace(-0.5, 0.5, 100, device=sample_x.device)
        boundary_y_top = torch.ones_like(boundary_x) * 0.5
        boundary_y_bottom = torch.ones_like(boundary_x) * -0.5
        plt.plot(
            boundary_x.cpu().numpy(), boundary_y_top.cpu().numpy(), "k-", linewidth=2
        )
        plt.plot(
            boundary_x.cpu().numpy(), boundary_y_bottom.cpu().numpy(), "k-", linewidth=2
        )
        boundary_y = torch.linspace(-0.5, 0.5, 100, device=sample_x.device)
        boundary_x_left = torch.ones_like(boundary_y) * -0.5
        boundary_x_right = torch.ones_like(boundary_y) * 0.5
        plt.plot(
            boundary_x_left.cpu().numpy(), boundary_y.cpu().numpy(), "k-", linewidth=2
        )
        plt.plot(
            boundary_x_right.cpu().numpy(), boundary_y.cpu().numpy(), "k-", linewidth=2
        )
        # Plot the circular holes
        centers = [(0.3, 0.3), (0.3, -0.3), (-0.3, 0.3), (-0.3, -0.3)]
        radius = 0.1
        for center in centers:
            circle = plt.Circle(center, radius, fill=True, color="red", alpha=0.3)
            plt.gca().add_patch(circle)
            circle = plt.Circle(center, radius, fill=False, color="red", linewidth=2)
            plt.gca().add_patch(circle)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Sampled Points for Poisson 2D with Complex Geometry (n={n_points})")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def ic_sampler(self, n_per_side=250, n_per_circle=250):
        # Outer square boundary: 4 sides
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5

        # Sample points on each side (excluding corners to avoid duplicates)
        # Bottom: x from x_min to x_max, y = y_min
        x_bottom = torch.linspace(x_min, x_max, n_per_side, device=self.device)
        y_bottom = torch.full_like(x_bottom, y_min)
        # Top: x from x_min to x_max, y = y_max
        x_top = torch.linspace(x_min, x_max, n_per_side, device=self.device)
        y_top = torch.full_like(x_top, y_max)
        # Left: y from y_min to y_max, x = x_min
        y_left = torch.linspace(y_min, y_max, n_per_side, device=self.device)
        x_left = torch.full_like(y_left, x_min)
        # Right: y from y_min to y_max, x = x_max
        y_right = torch.linspace(y_min, y_max, n_per_side, device=self.device)
        x_right = torch.full_like(y_right, x_max)

        # Stack all square boundary points
        x_square = torch.cat([x_bottom, x_top, x_left, x_right])
        y_square = torch.cat([y_bottom, y_top, y_left, y_right])

        # Inner circular boundaries (4 circles)
        centers = [(0.3, 0.3), (0.3, -0.3), (-0.3, 0.3), (-0.3, -0.3)]
        radius = 0.1
        theta = torch.linspace(0, 2 * np.pi, n_per_circle, device=self.device)

        x_circles = []
        y_circles = []
        for cx, cy in centers:
            x_c = cx + radius * torch.cos(theta)
            y_c = cy + radius * torch.sin(theta)
            x_circles.append(x_c)
            y_circles.append(y_c)
        x_circles = torch.cat(x_circles)
        y_circles = torch.cat(y_circles)

        # Combine all boundary points into a single tensor with shape [N_points, 2]
        boundary_points = torch.stack(
            [torch.cat([x_square, x_circles]), torch.cat([y_square, y_circles])], dim=1
        ).requires_grad_(True)

        return boundary_points

    def pde_sampler(self, n_samples=3000, oversample_factor=2):

        if self.valid_points is not None:
            return self.valid_points

        # Sample points uniformly in the domain
        n_oversample = int(n_samples * oversample_factor)
        # Generate uniform random points in the square domain
        x = torch.rand(n_oversample, device=self.device) - 0.5  # [-0.5, 0.5]
        y = torch.rand(n_oversample, device=self.device) - 0.5  # [-0.5, 0.5]
        # Check if points are outside the circular holes
        centers = torch.tensor(
            [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]], device=self.device
        )
        radius = 0.1
        points = torch.stack([x, y], dim=1)  # Shape: [n_oversample, 2]
        valid_mask = torch.ones(n_oversample, dtype=torch.bool, device=self.device)
        for center in centers:
            distances = torch.norm(points - center, dim=1)
            valid_mask &= distances > radius
        valid_points = points[valid_mask]
        if valid_points.shape[0] > n_samples:
            indices = torch.randperm(valid_points.shape[0], device=self.device)[
                :n_samples
            ]
            valid_points = valid_points[indices]
        if valid_points.shape[0] < n_samples:
            print(
                f"Warning: Only generated {valid_points.shape[0]} valid points out of {n_samples} requested"
            )
        # Return the valid points as a single tensor with shape [N_points, 2]

        self.valid_points = valid_points.clone().requires_grad_(True)
        return self.valid_points

    def pde_sampler_boundary_biased(
        self, n_samples=3000, oversample_factor=5, power=2, epsilon=1e-4
    ):
        """
        Sample points in the domain with density increasing quadratically as points approach the box boundaries.
        Points inside the central circles are excluded.
        Args:
            n_samples: Number of valid points to return
            oversample_factor: How many more points to sample for rejection sampling
            power: The power for the boundary bias (2 for quadratic)
            epsilon: Small value to avoid division by zero
        Returns:
            valid_points: [n_samples, 2] tensor of sampled points
        """
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5
        n_oversample = int(n_samples * oversample_factor)
        device = self.device

        # Sample uniformly in the square
        x = torch.rand(n_oversample, device=device) * (x_max - x_min) + x_min
        y = torch.rand(n_oversample, device=device) * (y_max - y_min) + y_min
        points = torch.stack([x, y], dim=1)  # [n_oversample, 2]

        # Compute distance to nearest boundary for each point
        dist_left = torch.abs(points[:, 0] - x_min)
        dist_right = torch.abs(points[:, 0] - x_max)
        dist_bottom = torch.abs(points[:, 1] - y_min)
        dist_top = torch.abs(points[:, 1] - y_max)
        dist_to_boundary = torch.min(
            torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=1), dim=1
        )[0]

        # Compute acceptance probability: higher near boundary, quadratic
        # p ~ 1/(d^power + epsilon)
        probs = 1.0 / (dist_to_boundary**power + epsilon)
        probs = probs / probs.max()  # Normalize to [0,1]

        # Exclude points inside the circles
        centers = torch.tensor(
            [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]], device=device
        )
        radius = 0.1
        valid_mask = torch.ones(n_oversample, dtype=torch.bool, device=device)
        for center in centers:
            distances = torch.norm(points - center, dim=1)
            valid_mask &= distances > radius

        # Rejection sampling
        rand_vals = torch.rand(n_oversample, device=device)
        accept_mask = (rand_vals < probs) & valid_mask
        valid_points = points[accept_mask]
        if valid_points.shape[0] > n_samples:
            indices = torch.randperm(valid_points.shape[0], device=device)[:n_samples]
            valid_points = valid_points[indices]
        if valid_points.shape[0] < n_samples:
            print(
                f"Warning: Only generated {valid_points.shape[0]} valid points out of {n_samples} requested (boundary-biased)"
            )
        return valid_points.clone().requires_grad_(True)

    def eval_sampler(self):
        return self.ref_points


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Poisson 2D CG Spectral Interpolant Trainer"
    )
    parser.add_argument(
        "--n_x", type=int, default=51, help="Number of Chebyshev points in x-direction"
    )
    parser.add_argument(
        "--n_y", type=int, default=51, help="Number of Chebyshev points in y-direction"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10000, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_every", type=int, default=100, help="Evaluation frequency (in epochs)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(project_root, "plots/pdes/poisson_holes"),
        help="Base directory for saving results",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="nys_newton",
        choices=["nys_newton", "adam", "ssbroyden", "lssbroyden", "adam_ssbroyden"],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--n_adam_epochs",
        type=int,
        default=5000,
        help="Number of Adam epochs for adam_ssbroyden method",
    )
    parser.add_argument(
        "--n_per_side",
        type=int,
        default=250,
        help="Number of points per side for square boundary in ic_sampler",
    )
    parser.add_argument(
        "--n_per_circle",
        type=int,
        default=250,
        help="Number of points per circle for circular boundaries in ic_sampler",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of PDE points to sample in pde_sampler",
    )
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=2.0,
        help="Oversampling factor for pde_sampler",
    )
    parser.add_argument(
        "--nncg_rank", type=int, default=1000, help="Rank for NysNewtonCG optimizer"
    )
    parser.add_argument(
        "--nncg_maxiters",
        type=int,
        default=16,
        help="Max iterations for NysNewtonCG optimizer",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    # Print hyperparameters
    print("\033[94m\n===== HYPERPARAMETERS =====\033[0m")
    print(f"\033[94m{vars(args)}\033[0m")
    print("\033[94m==========================\033[0m\n")

    # Print GPU info
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        print(f"Using GPU: {device_name} (device index: {device_idx})\n")
    else:
        print("CUDA is not available. Using CPU.\n")

    # ========================
    # Reference Solution
    # ========================

    pde = Poisson2DCG(device="cuda", debug=True)

    # ========================
    # Spectral Interpolant Model Setup
    # ========================
    print("\033[94m" + "=" * 80 + "\033[0m")
    print("\033[94m TRAINING SPECTRAL INTERPOLANT ON POISSON 2D CG DOMAIN... \033[0m")
    print("\033[94m" + "=" * 80 + "\033[0m")

    # Model configuration
    n_x = args.n_x
    n_y = args.n_y
    bases = ["chebyshev", "chebyshev"]

    # Initialize model
    model_spec = SpectralInterpolationND(
        Ns=[n_x, n_y], bases=bases, domains=pde.domain, device=pde.device
    )
    model_spec.values.weight = 0.5 * torch.ones_like(model_spec.values)

    # Optimizer and training parameters
    if args.optimizer == "nys_newton":
        from src.optimizers.nys_newton_cg import NysNewtonCG

        optimizer_spec = NysNewtonCG(
            model_spec.parameters(),
            lr=1.0,
            rank=args.nncg_rank,
            mu=1e-2,
            line_search_fn="armijo",
            cg_tol=1e-16,
            cg_max_iters=args.nncg_maxiters,
            verbose=False,
        )
    elif args.optimizer == "adam":
        optimizer_spec = pde.get_optimizer(model_spec, "adam")
    elif args.optimizer == "ssbroyden":
        from src.optimizers.ssbroyden import SSBroyden2
        
        optimizer_spec = SSBroyden2(
            model_spec.parameters(),
            lr=1.0,
            init_scale=True,
            c1=1e-4,
            c2=0.9,
            max_ls=20,
        )
    elif args.optimizer == "lssbroyden":
        from src.optimizers.Lssbroyden import L_SSBroyden
        
        optimizer_spec = L_SSBroyden(
            model_spec.parameters(),
            lr=1.0,
            history_size=10,
            init_scale=True,
            c1=1e-4,
            c2=0.9,
            max_ls=20,
        )
    elif args.optimizer == "adam_ssbroyden":
        # For adam_ssbroyden, we'll use a placeholder optimizer and call the special training method
        optimizer_spec = None  # Will be handled in the training call
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    ic_weight = 10.0
    n_square_boundary = args.n_per_side * 4  # Default n_per_side in ic_sampler

    # Setup directories and logger
    from src.loggers.logger import Logger

    # Compose a directory name that reflects all relevant hyperparameters
    spectral_save_dir = os.path.join(
        base_dir,
        f"nx={n_x}_ny={n_y}_epochs={args.n_epochs}_evalevery={args.eval_every}_opt={args.optimizer}"
        f"_nside={args.n_per_side}_ncircle={args.n_per_circle}_nsamp={args.n_samples}_over={args.oversample_factor}"
        f"_rank={getattr(args, 'nncg_rank', 'NA')}_cgmax={getattr(args, 'nncg_maxiters', 'NA')}",
    )
    weights_dir = os.path.join(spectral_save_dir, "weights/")
    images_dir = os.path.join(spectral_save_dir, "images/")
    os.makedirs(spectral_save_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    logger_spec = Logger(path=os.path.join(spectral_save_dir, "logger.json"))

    # Define samplers
    def pde_sampler():
        return pde.pde_sampler(
            n_samples=args.n_samples, oversample_factor=args.oversample_factor
        )

    def ic_sampler():
        return pde.ic_sampler(
            n_per_side=args.n_per_side, n_per_circle=args.n_per_circle
        )

    def eval_sampler():
        return pde.ref_points

    # Define evaluation metrics
    from src.utils.metrics import l2_error, max_error, l2_relative_error

    eval_metrics = [l2_error, max_error, l2_relative_error]

    # Train the model
    hyperparam_str = f"spectral_nx={n_x}_ny={n_y}_epochs={args.n_epochs}_eval_every={args.eval_every}_sample=uniform_optimizer={args.optimizer}"

    # Handle different training methods
    if args.optimizer == "adam_ssbroyden":
        # Use the special adam_ssbroyden training method
        pde.train_adam_ssbroyden(
            model=model_spec,
            n_epochs=args.n_epochs,
            n_adam_epochs=args.n_adam_epochs,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=args.eval_every,
            save_dir=spectral_save_dir,
            logger=logger_spec,
            n_square_boundary=n_square_boundary,
        )
    else:
        # Use the standard training method
        pde.train(
            model=model_spec,
            n_epochs=args.n_epochs,
            optimizer=optimizer_spec,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=args.eval_every,
            save_dir=spectral_save_dir,
            logger=logger_spec,
            n_square_boundary=n_square_boundary,
            hessian_every=-1,  # Add explicit hessian parameters
            hessian_num_iter=100,
            hessian_num_run=1,
        )

    # Evaluate and plot the final solution
    print("\033[92m" + "=" * 80 + "\033[0m")
    print("\033[92m TRAINING COMPLETE! \033[0m")
    print("\033[92m" + "=" * 80 + "\033[0m")
