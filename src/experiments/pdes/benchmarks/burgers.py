import argparse
import os
import scipy.io
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.models.mlp import MLP

from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.optimizers.nys_newton_cg import NysNewtonCG
from src.loggers.logger import Logger


def make_grid(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Create a meshgrid from t and x tensors.

    Args:
        t: Tensor of shape (n_t,)
        x: Tensor of shape (n_x,)

    Returns:
        Tensor of shape (n_t, n_x, 2)
    """
    mesh = torch.meshgrid(t, x, indexing="ij")
    return torch.stack(mesh, dim=-1)  # (t, x, 2)


"""
1D Burgers equation:
u_t + u * u_x - nu * u_xx = 0
t in [0, 1]
x in [-1, 1]
u(t=0, x) = -sin(pi*x)
u(t, x=-1) = u(t, x=1) = 0
"""


class Burgers(BasePDE):
    def __init__(
        self,
        nu: float = 0.01 / torch.pi,
        device: str = "cpu",
        t_final: float = 1.0,
        sample_type: str = "uniform",
        n_t: int = 200,
        n_x: int = 512,
        **base_kwargs,
    ):
        super().__init__(
            "burgers", [(0, t_final), (-1, 1)], device=device, **base_kwargs
        )
        self.t_final = t_final
        self.nu = nu
        self.u_0 = lambda x: -torch.sin(torch.pi * x)

        # Load reference solution
        self.sample_type = sample_type
        self.n_t = int(n_t * self.t_final)
        self.n_x = n_x
        self.ref_u, self.ref_t, self.ref_x = self.load_ref_solution()
        self.full_grid = make_grid(self.ref_t, self.ref_x)
        # Subsample grid for training and evaluation
        self._set_train_and_eval_nodes(self.n_t, self.n_x)

    def load_ref_solution(
        self,
        path: str = None,
    ):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "ref",
                "burgers_1d_dirichlet.mat",
            )
        mat = scipy.io.loadmat(path)
        u_ref = torch.tensor(mat["usol"], dtype=torch.float64, device=self.device)
        t = torch.tensor(mat["t"][0], dtype=torch.float64, device=self.device)
        x = torch.tensor(mat["x"][0], dtype=torch.float64, device=self.device)
        t_len = len(t)
        return u_ref[: int(t_len * self.t_final)], t[: int(t_len * self.t_final)], x

    def _set_train_and_eval_nodes(self, n_t: int, n_x: int):
        """Set up training and evaluation nodes based on the sampling type"""
        if self.sample_type == "uniform":
            # Subsample training nodes uniformly
            self.train_t = self.ref_t[:: (max(1, self.ref_t.shape[0] // n_t))]
            self.train_x = self.ref_x[:: (max(1, self.ref_x.shape[0] // n_x))]
            # Set eval nodes to be the full grid
            self.eval_t = self.ref_t
            self.eval_x = self.ref_x
        elif self.sample_type == "chebyshev":
            # Sample Chebyshev nodes
            cheb_t = (
                torch.cos(torch.linspace(0, 1, n_t, device=self.device) * torch.pi) / 2
                + 0.5
            )  # (0, 1)
            cheb_x = torch.cos(
                torch.linspace(0, 1, n_x, device=self.device) * torch.pi
            )  # (-1, 1)
            # Round to nearest grid values we have
            self.train_t = torch.round(cheb_t * (self.ref_t.shape[0] - 1)) / (
                self.ref_t.shape[0] - 1
            )
            self.train_x = torch.round(cheb_x * (self.ref_x.shape[0] - 1)) / (
                self.ref_x.shape[0] - 1
            )
            # Set eval nodes to be the full grid
            self.eval_t = self.ref_t
            self.eval_x = self.ref_x
        else:
            raise ValueError(f"Invalid sampling type: {self.sample_type}")

    def nodes_to_indices(self, nodes: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert nodes to indices in the reference solution grid.

        Args:
            nodes: List of tensors [t, x] with node coordinates

        Returns:
            List of tensors [t_indices, x_indices] with corresponding indices
        """
        # Nodes are always linspace. So shift and rescale
        t_nodes = (nodes[0] - self.domain[0][0]) / (
            self.domain[0][1] - self.domain[0][0]
        )
        x_nodes = (nodes[1] - self.domain[1][0]) / (
            self.domain[1][1] - self.domain[1][0]
        )
        t_indices = (t_nodes * (len(self.ref_t) - 1)).round().long()
        x_indices = (x_nodes * (len(self.ref_x) - 1)).round().long()
        return t_indices, x_indices

    def get_solution(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        """
        Get the reference solution values at the given nodes.

        Args:
            nodes: List of tensors [t, x] with node coordinates

        Returns:
            Tensor with reference solution values
        """
        indices = self.nodes_to_indices(nodes)
        u_pred = self.ref_u[indices[0][:, None], indices[1][None, :]]
        return u_pred

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        n_t, n_x = pde_nodes[0].shape[0], pde_nodes[1].shape[0]
        n_ic = ic_nodes[1].shape[0]

        if isinstance(model, SpectralInterpolationND):
            # PDE
            u = model.interpolate(pde_nodes)
            u_t = model.derivative(pde_nodes, k=(1, 0))
            u_x = model.derivative(pde_nodes, k=(0, 1))
            u_xx = model.derivative(pde_nodes, k=(0, 2))
            # IC
            u_ic = model.interpolate(ic_nodes)[0]
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][0]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )
            u_periodic_t1 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][1]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )
        else:
            # PDE
            grid = model.make_grid(pde_nodes)  # (N_t*N_x, 2)
            u = model.forward_grid(grid).reshape(n_t, n_x)  # (N_t, N_x)
            grads = torch.autograd.grad(
                u.sum(), grid, create_graph=True
            )  # (N_t*N_x, 2)
            # First derivatives
            u_t = grads[0][..., 0].reshape(n_t, n_x)
            u_x = grads[0][..., 1].reshape(n_t, n_x)
            # Second derivative: u_xx
            grad_xx = torch.autograd.grad(u_x.sum(), grid, create_graph=True)
            u_xx = grad_xx[0][..., 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes)[0]
            # Enforce Dirichlet boundary conditions at t nodes
            u_periodic_t0 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][0]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )
            u_periodic_t1 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][1]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )

        # PDE loss
        pde_residual = u_t + u * u_x - self.nu * u_xx
        pde_loss = torch.mean(pde_residual**2)

        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2)

        # Dirichlet boundary conditions loss
        pbc_loss = torch.mean(u_periodic_t0**2) + torch.mean(u_periodic_t1**2)

        loss_names = ["pde_loss", "ic_loss", "pbc_loss"]
        return dict(zip(loss_names, [pde_loss, ic_loss, pbc_loss]))

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)

        pde_weight = self.loss_weights.get("pde_loss_weight", 1.0)
        ic_weight = self.loss_weights.get("ic_loss_weight", ic_weight)
        pbc_weight = self.loss_weights.get("pbc_loss_weight", ic_weight)

        loss = (
            (pde_weight * loss_dict["pde_loss"])
            + (ic_weight * ic_weight * loss_dict["ic_loss"])
            + (ic_weight * pbc_weight * loss_dict["pbc_loss"])
        )

        return (
            loss,
            loss_dict["pde_loss"],
            loss_dict["ic_loss"] + loss_dict["pbc_loss"],
        )

    def get_least_squares(self, model: SpectralInterpolationND):
        raise NotImplementedError("Least squares not implemented for Burgers equation")

    def fit_least_squares(self, model: SpectralInterpolationND):
        raise NotImplementedError("Least squares not implemented for Burgers equation")

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,  # (N_t, N_x)
        save_path: str = None,
        **kwargs,
    ):
        self._plot_solution_default(
            nodes=nodes,
            u=u,
            save_path=save_path,
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--nu", type=float, default=0.01 / torch.pi)
    args.add_argument(
        "--n_t", type=int, default=161
    )  # Number of time nodes in interpolant
    args.add_argument(
        "--n_x", type=int, default=161
    )  # Number of space nodes in interpolant
    args.add_argument("--n_layers", type=int, default=3)  # Number of layers in MLP
    args.add_argument(
        "--hidden_dim", type=int, default=256
    )  # Number of hidden nodes in MLP
    args.add_argument("--activation", type=str, default="tanh")
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    # NNCG-specific args
    args.add_argument("--nncg_rank", type=int, default=1000)
    args.add_argument("--nncg_cgmaxiters", type=int, default=100)

    args.add_argument("--n_epochs", type=int, default=100000)
    args.add_argument("--eval_every", type=int, default=1000)
    args.add_argument("--model", type=str, default=None)
    args.add_argument("--t_final", type=float, default=1.0)
    args.add_argument("--from_pretrained", action="store_true")

    # lwup is one of [grad_norm, none].
    args.add_argument("--loss_weight_update_policy", "-lwup", type=str, default="none")
    args.add_argument("--loss_weight_update_interval", "-lwui", type=int, default=1000)

    # Add learning rate scheduling and gradient clipping arguments
    args.add_argument(
        "--lr_schedule", action="store_true", help="Enable learning rate scheduling"
    )
    args.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (0 to disable)",
    )
    args.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing",
    )
    args.add_argument(
        "--lr_max",
        type=float,
        default=1e-3,
        help="Maximum learning rate for cosine annealing",
    )

    args.add_argument(
        "--fd_k_t",
        type=int,
        default=None,
        help="Stencil size for FD in time dimension (None for spectral)",
    )

    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    nu = args.nu
    pde = Burgers(
        nu=nu,
        device=device,
        t_final=args.t_final,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
    )

    base_save_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "plots/pdes/burgers/nu={nu}_tfinal={args.t_final}",
    )

    # Add timestamp to base save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(base_save_dir, timestamp)

    # Evaluation setup
    eval_every = args.eval_every
    n_eval = 200
    t_eval = pde.ref_t.requires_grad_(True)
    x_eval = pde.ref_x.requires_grad_(True)

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    #########################################################
    # 1. Neural network
    #########################################################
    if args.model is None or args.model == "mlp":
        save_dir = os.path.join(
            base_save_dir,
            f"mlp/method={args.method}_nlayers={args.n_layers}_hdim={args.hidden_dim}_sample={args.sample_type}",
        )

        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        model_mlp = MLP(
            n_dim=2,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            activation=torch.tanh,
            device=device,
        )

        # Training setup
        n_epochs = args.n_epochs
        # lr = 1e-3
        if args.method == "nys_newton":
            from src.optimizers.nys_newton_cg import NysNewtonCG

            optimizer = NysNewtonCG(
                model_mlp.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden":
            from src.optimizers.ssbroyden import SSBroyden2
            
            optimizer = SSBroyden2(
                model_mlp.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        else:
            optimizer = pde.get_optimizer(model_mlp, args.method)

        n_t_train = 321
        n_x_train = 321
        n_ic_train = 321
        ic_weight = 10

        def pde_sampler():
            t_nodes = pde.sample_domain_1d(
                n_samples=n_t_train,
                dim=0,
                basis="fourier",
                type=args.sample_type,
            )
            x_nodes = pde.sample_domain_1d(
                n_samples=n_x_train,
                dim=1,
                basis="fourier",
                type=args.sample_type,
            )
            return [t_nodes, x_nodes]

        def ic_sampler():
            ic_nodes = pde.sample_domain_1d(
                n_samples=n_ic_train,
                dim=1,
                basis="fourier",
                type=args.sample_type,
            )
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"Training MLP with {args.method} optimizer...")
        pde.train(
            model_mlp,
            n_epochs=args.n_epochs,
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

    #########################################################
    # 2. Polynomial interpolation
    #########################################################
    if args.model is None or args.model == "polynomial":
        save_dir = os.path.join(
            base_save_dir,
            f"polynomial/method={args.method}_nt={args.n_t}_nx={args.n_x}_sample={args.sample_type}_nncgrank={args.nncg_rank}_nncgcgmaxiters={args.nncg_cgmaxiters}",
        )

        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        n_t = args.n_t
        n_x = args.n_x
        bases = ["chebyshev", "chebyshev"]
        model = SpectralInterpolationND(
            Ns=[n_t, n_x],
            bases=bases,
            domains=pde.domain,
            device=device,
        )

        # Training setup
        n_epochs = args.n_epochs
        if args.method == "nys_newton":
            from src.optimizers.nys_newton_cg import NysNewtonCG

            optimizer = NysNewtonCG(
                model.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden":
            from src.optimizers.ssbroyden import SSBroyden2
            
            optimizer = SSBroyden2(
                model.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        else:
            optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 321
        n_x_train = 321
        n_ic_train = 321
        ic_weight = 10

        def pde_sampler():
            t_nodes = pde.sample_domain_1d(
                n_samples=n_t_train,
                dim=0,
                basis=bases[0],
                type=args.sample_type,
            )
            x_nodes = pde.sample_domain_1d(
                n_samples=n_x_train,
                dim=1,
                basis=bases[1],
                type=args.sample_type,
            )
            return [t_nodes, x_nodes]

        def ic_sampler():
            ic_nodes = pde.sample_domain_1d(
                n_samples=n_ic_train,
                dim=1,
                basis=bases[1],
                type=args.sample_type,
            )
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"Training Polynomial Interpolant with {args.method} optimizer...")
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
            save_dir=save_dir,
            logger=logger,
        )

    #########################################################
    # 2b. Polynomial interpolation with finite differences
    #########################################################
    if args.model is None or args.model == "polynomial_fd":
        save_dir = os.path.join(
            base_save_dir,
            f"polynomial/expt=fd/method={args.method}_nt={args.n_t}_nx={args.n_x}_sample={args.sample_type}_fdkt={args.fd_k_t}_nncgrank={args.nncg_rank}_nncgcgmaxiters={args.nncg_cgmaxiters}",
        )

        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        n_t = args.n_t
        n_x = args.n_x
        bases = ["chebyshev", "chebyshev"]
        model = SpectralInterpolationND(
            Ns=[n_t, n_x],
            bases=bases,
            domains=pde.domain,
            device=device,
            fd_k=[args.fd_k_t, None],  # Use FD in time dimension only
        )

        # Training setup
        n_epochs = args.n_epochs
        if args.method == "nys_newton":
            from src.optimizers.nys_newton_cg import NysNewtonCG

            optimizer = NysNewtonCG(
                model.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden":
            from src.optimizers.ssbroyden import SSBroyden2
            
            optimizer = SSBroyden2(
                model.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        else:
            optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 321
        n_x_train = 321
        n_ic_train = 321
        ic_weight = 10

        def pde_sampler():
            t_nodes = pde.sample_domain_1d(
                n_samples=n_t_train,
                dim=0,
                basis=bases[0],
                type=args.sample_type,
            )
            x_nodes = pde.sample_domain_1d(
                n_samples=n_x_train,
                dim=1,
                basis=bases[1],
                type=args.sample_type,
            )
            return [t_nodes, x_nodes]

        def ic_sampler():
            ic_nodes = pde.sample_domain_1d(
                n_samples=n_ic_train,
                dim=1,
                basis=bases[1],
                type=args.sample_type,
            )
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"Training Polynomial Interpolant with FD in time dimension...")
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
            save_dir=save_dir,
            logger=logger,
        )

    #########################################################
    # 3. MLP Interpolant
    #########################################################
    if args.model is None or args.model == "mlpinterp":
        save_dir = os.path.join(
            base_save_dir,
            f"mlpinterp/method={args.method}_nt={args.n_t}_nx={args.n_x}_nlayers={args.n_layers}_hdim={args.hidden_dim}_activation={args.activation}_sample={args.sample_type}",
        )
        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        n_t = args.n_t
        n_x = args.n_x
        bases = ["chebyshev", "chebyshev"]
        try:
            activation = getattr(torch, args.activation)
        except AttributeError:
            raise ValueError(f"Invalid activation function: {args.activation}")

        model = MLPSpectralInterpolationND(
            Ns=[n_t, n_x],
            bases=bases,
            domains=pde.domain,
            device=device,
            hidden_layers=(args.hidden_dim,) * args.n_layers,
            activation=activation,
        )

        # Training setup
        n_epochs = args.n_epochs
        if args.method == "nys_newton":
            from src.optimizers.nys_newton_cg import NysNewtonCG

            optimizer = NysNewtonCG(
                model.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden":
            from src.optimizers.ssbroyden import SSBroyden2
            
            optimizer = SSBroyden2(
                model.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        else:
            optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 321
        n_x_train = 321
        n_ic_train = 321
        ic_weight = 10

        def pde_sampler():
            t_nodes = pde.sample_domain_1d(
                n_samples=n_t_train,
                dim=0,
                basis=bases[0],
                type=args.sample_type,
            )
            x_nodes = pde.sample_domain_1d(
                n_samples=n_x_train,
                dim=1,
                basis=bases[1],
                type=args.sample_type,
            )
            return [t_nodes, x_nodes]

        def ic_sampler():
            ic_nodes = pde.sample_domain_1d(
                n_samples=n_ic_train,
                dim=1,
                basis=bases[1],
                type=args.sample_type,
            )
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"\nTraining MLP Interpolant with {args.method} optimizer...")
        pde.train(
            model,
            n_epochs=args.n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=eval_every,
            save_dir=save_dir,
            logger=logger,
            lr_schedule=args.lr_schedule,
            gradient_clip=args.gradient_clip,
        )
