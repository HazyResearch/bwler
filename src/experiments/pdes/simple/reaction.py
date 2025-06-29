import argparse
import os
import torch
import torch.nn as nn
from typing import List, Callable, Tuple, Dict
from datetime import datetime

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger

"""
1D Reaction equation:
u_t - rho * u(1-u) = 0
t in [0, t_final] (default: t_final = 1)
x in [0, 2*pi]
u(t=0, x) = u_0(x) (default: u_0(x) = exp(-(x - pi)**2 / (2 * (pi / 4)**2)))
u(t, x=0) = u(t, x=2*pi)

Solution:
u(t, x) = u_0(x) exp(rho * t) / [u_0(x) exp(rho * t) + (1 - u_0(x))]
"""


class Reaction(BasePDE):
    def __init__(
        self,
        rho: float,
        t_final: float = 1,
        u_0: Callable = None,
        device: str = "cpu",
        **base_kwargs,
    ):
        super().__init__(
            "reaction", [(0, 1), (0, 2 * torch.pi)], device=device, **base_kwargs
        )
        self.rho = rho
        self.t_final = t_final
        if u_0 is None:
            self.u_0 = lambda x: torch.exp(
                -((x - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2)
            )
        else:
            self.u_0 = u_0
        self.exact_solution = (
            lambda t, x: self.u_0(x)
            * torch.exp(self.rho * t)
            / (self.u_0(x) * torch.exp(self.rho * t) + (1 - self.u_0(x)))
        )

    def get_solution(self, nodes: List[torch.Tensor]):
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        return self.exact_solution(t_mesh, x_mesh)

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        n_t, n_x = pde_nodes[0].shape[0], pde_nodes[1].shape[0]
        n_ic = ic_nodes[1].shape[0]

        if isinstance(model, SpectralInterpolationND) or isinstance(
            model, MLPSpectralInterpolationND
        ):
            # PDE
            u = model.interpolate(pde_nodes)
            u_t = model.derivative(pde_nodes, k=(1, 0))  # (N_t, N_x)
            # IC
            u_ic = model.interpolate(ic_nodes)[0]  # (N_ic)
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
            grid = model.make_grid(pde_nodes)
            u = model.forward_grid(grid).reshape(n_t, n_x)
            grads = torch.autograd.grad(
                u.sum(), grid, create_graph=True
            )  # (N_t*N_x, 2)
            u_t = grads[0][:, :, 0].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes).reshape(n_ic)
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model(
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
            u_periodic_t1 = model(
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
        pde_residual = u_t - self.rho * u * (1 - u)
        pde_loss = torch.mean(pde_residual**2)

        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2)

        # Periodic boundary conditions loss
        pbc_loss = torch.mean((u_periodic_t0 - u_periodic_t1) ** 2)

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

    # Get the least squares problem equivalent to a spectral solve
    # Since this problem is nonlinear, we perform Picard iteration using the model's current guess for u
    def get_least_squares(self, model: SpectralInterpolationND):
        n_t, n_x = model.nodes[0].shape[0], model.nodes[1].shape[0]

        # PDE operator
        D_t = model.derivative_matrix(k=(1, 0))  # (N_t*N_x, N_t*N_x)
        L = D_t

        # Initial condition: extract t=0 values
        IC = torch.zeros(n_x, n_t * n_x, device=model.device, dtype=model.values.dtype)
        for i in range(n_x):
            IC[i, n_x * (n_t - 1) + i] = 1  # Set t=0 value to 1 for each x

        # Right hand side
        b = torch.zeros(n_t * n_x + n_x, device=model.device, dtype=model.values.dtype)
        # Picard iteration: calculate rho * u_0(x) * (1 - u_0(x))
        L_rhs = self.rho * model(model.nodes) * (1 - model(model.nodes))  # (N_t*N_x)
        b[: n_t * n_x] = L_rhs
        # IC
        b[n_t * n_x :] = self.u_0(model.nodes[1])

        # Full system
        A = torch.cat([L, IC], dim=0)
        return A, b

    def fit_least_squares(self, model: SpectralInterpolationND, max_iter: int = 100):
        # Set model values to initial condition
        IC = self.u_0(model.nodes[1])
        model.values.data = IC[None, :]
        # Perform Picard iteration
        for i in range(self.max_iter):
            A, b = self.get_least_squares(model)
            u = torch.linalg.lstsq(A, b).solution
            model.values.data = u.reshape(
                model.nodes[0].shape[0], model.nodes[1].shape[0]
            )
        return model

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
    args.add_argument("--rho", type=float, default=5)
    args.add_argument(
        "--n_t", type=int, default=81
    )  # Number of time nodes in interpolant
    args.add_argument(
        "--n_x", type=int, default=81
    )  # Number of space nodes in interpolant
    args.add_argument("--n_layers", type=int, default=3)  # Number of layers in MLP
    args.add_argument(
        "--hidden_dim", type=int, default=256
    )  # Number of hidden nodes in MLP
    args.add_argument("--activation", type=str, default="tanh")
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=100000)
    args.add_argument("--eval_every", type=int, default=1000)
    args.add_argument("--model", type=str, default=None)  # If None, run all models

    # lwup is one of [grad_norm, none].
    args.add_argument("--loss_weight_update_policy", "-lwup", type=str, default="none")
    args.add_argument("--loss_weight_update_interval", "-lwui", type=int, default=-1)

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

    # Add Hessian tracking argument
    args.add_argument(
        "--hessian_every",
        type=int,
        default=-1,
        help="How often to compute Hessian during training (-1 for never)",
    )

    # NNCG-specific args
    args.add_argument("--nncg_rank", type=int, default=1000)
    args.add_argument("--nncg_cgmaxiters", type=int, default=100)

    args.add_argument("--seed", type=int, default=0)

    args = args.parse_args()

    torch.random.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    rho = args.rho
    t_final = 1

    def u_0(x):
        return torch.exp(-((x - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2))

    pde = Reaction(
        rho=rho,
        t_final=t_final,
        u_0=u_0,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
        device=device,
    )

    base_save_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "plots/pdes/reaction/rho={rho}",
    )

    # Add timestamp to base save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(base_save_dir, timestamp)

    # Evaluation setup (shared for all methods)
    eval_every = args.eval_every
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
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    #########################################################
    # 1. Neural network
    #########################################################
    if args.model is None or args.model == "mlp":
        save_dir = os.path.join(
            base_save_dir,
            f"mlp/method={args.method}_nlayers={args.n_layers}_hdim={args.hidden_dim}_activation={args.activation}_sample={args.sample_type}",
        )
        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        try:
            activation = getattr(torch, args.activation)
        except AttributeError:
            raise ValueError(f"Invalid activation function: {args.activation}")

        model_mlp = MLP(
            n_dim=2,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            activation=activation,
            device=device,
        )

        # Training setup
        n_epochs = args.n_epochs
        optimizer = torch.optim.Adam(model_mlp.parameters(), lr=args.lr_max)
        optimizer = pde.get_optimizer(model_mlp, args.method)

        n_t_train = 161
        n_x_train = 161
        n_ic_train = 161
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
            lr_schedule=args.lr_schedule,
            gradient_clip=args.gradient_clip,
            hessian_every=args.hessian_every,
        )

    #########################################################
    # 2. Polynomial interpolation
    #########################################################
    if args.model is None or args.model == "polynomial":
        save_dir = os.path.join(
            base_save_dir,
            f"polynomial/method={args.method}_nt={args.n_t}_nx={args.n_x}_sample={args.sample_type}_nncgmaxiters={args.nncg_cgmaxiters}_nncgrank={args.nncg_rank}",
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
        else:
            optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 161
        n_x_train = 161
        n_ic_train = 161
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
            hessian_every=args.hessian_every,
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
        else:
            optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 161
        n_x_train = 161
        n_ic_train = 161
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
            hessian_every=args.hessian_every,
        )
