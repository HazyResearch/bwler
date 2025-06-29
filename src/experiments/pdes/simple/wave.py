import argparse
import os
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from datetime import datetime

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.models.mlp import MLP
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger

"""
1D Wave equation:
u_tt - c^2 * u_xx = 0
t in [0, 1]
x in [0, 1]
u(t=0, x) = sin(pi*x) + 1/2 * sin(beta*pi*x)
u_t(t=0, x) = 0
u(t, x=0) = u(t, x=1) = 0

Solution:
u(t, x) = sin(pi*x) cos(2*pi*t) + 1/2 * sin(beta*pi*x) cos(2*beta*pi*t)
"""


class Wave(BasePDE):
    def __init__(
        self,
        c: float = 2,
        beta: float = 5,
        device: str = "cpu",
        **base_kwargs,
    ):
        super().__init__("wave", [(0, 1), (0, 1)], device=device, **base_kwargs)
        self.c = c
        self.beta = beta
        self.u_0 = lambda x: torch.sin(torch.pi * x) + 0.5 * torch.sin(
            beta * torch.pi * x
        )
        self.u_0_t = lambda x: 0
        self.exact_solution = lambda t, x: torch.sin(torch.pi * x) * torch.cos(
            2 * torch.pi * t
        ) + 0.5 * torch.sin(beta * torch.pi * x) * torch.cos(2 * beta * torch.pi * t)

    def get_solution(self, nodes: List[torch.Tensor]):
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        return self.exact_solution(t_mesh, x_mesh)

    def init_at_noisy_solution(self, model: SpectralInterpolationND, eps: float = 1e-3):
        solution = self.get_solution(model.nodes)
        solution += eps * torch.randn_like(solution)
        model.values.data = solution
        return model

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
            u_tt = model.derivative(pde_nodes, k=(2, 0))
            u_xx = model.derivative(pde_nodes, k=(0, 2))
            # IC
            u_ic = model.interpolate(ic_nodes)[0]
            u_t_ic = model.derivative(ic_nodes, k=(1, 0))[0]
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
            # Second derivative: u_tt
            grad_tt = torch.autograd.grad(u_t.sum(), grid, create_graph=True)
            u_tt = grad_tt[0][..., 0].reshape(n_t, n_x)
            # Second derivative: u_xx
            grad_xx = torch.autograd.grad(u_x.sum(), grid, create_graph=True)
            u_xx = grad_xx[0][..., 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes)[0]
            # First derivative: u_t_ic
            grad_t_ic = torch.autograd.grad(u_ic.sum(), ic_nodes, create_graph=True)
            u_t_ic = grad_t_ic[1]
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
        pde_residual = u_tt - self.c**2 * u_xx
        pde_loss = torch.mean(pde_residual**2)

        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_dt_residual = u_t_ic - self.u_0_t(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2) + torch.mean(ic_dt_residual**2)

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

        # Total loss
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
    def get_least_squares(self, model: SpectralInterpolationND):
        n_t, n_x = model.nodes[0].shape[0], model.nodes[1].shape[0]

        # PDE operator
        D_t = model.derivative_matrix(k=(1, 0))  # (N_t, N_t*N_x)
        D_tt = model.derivative_matrix(k=(2, 0))  # (N_t*N_x, N_t*N_x)
        D_xx = model.derivative_matrix(k=(0, 2))  # (N_t*N_x, N_t*N_x)
        L = D_tt - self.c**2 * D_xx

        # Initial condition: extract t=0 values
        IC = torch.zeros(n_x, n_t * n_x, device=model.device, dtype=model.values.dtype)
        for i in range(n_x):
            IC[i, n_x * (n_t - 1) + i] = 1  # Set t=0 value to 1 for each x
        D_t_IC = D_t[n_x * (n_t - 1) : n_x * n_t, :]

        dirichlet_BC1 = torch.zeros(
            n_t - 1, n_t * n_x, device=model.device, dtype=model.values.dtype
        )
        for i in range(n_t - 1):
            dirichlet_BC1[i, n_x * (i + 1)] = (
                1  # Set x=0 value to 1 for each t (except t=0)
            )

        dirichlet_BC2 = torch.zeros(
            n_t - 1, n_t * n_x, device=model.device, dtype=model.values.dtype
        )
        for i in range(n_t - 1):
            dirichlet_BC2[i, n_x * (i + 1) + n_x - 1] = (
                1  # Set x=1 value to 1 for each t (except t=0)
            )

        # Right hand side
        b = torch.zeros(
            n_t * n_x + n_x + n_x + 2 * (n_t - 1),
            device=model.device,
            dtype=model.values.dtype,
        )
        b[n_t * n_x : n_t * n_x + n_x] = self.u_0(model.nodes[1])
        b[n_t * n_x + n_x :] = self.u_0_t(model.nodes[1])

        # Full system
        A = torch.cat([L, IC, D_t_IC, dirichlet_BC1, dirichlet_BC2], dim=0)
        return A, b

    def fit_least_squares(self, model: SpectralInterpolationND):
        A, b = self.get_least_squares(model)
        # Solve on cpu for gelsd driver
        u = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gelsd").solution
        u = u.to(A.device)
        u = u.reshape(model.nodes[0].shape[0], model.nodes[1].shape[0])
        model.values.data = u
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
    args.add_argument("--c", type=float, default=2)
    args.add_argument("--beta", type=float, default=5)
    args.add_argument(
        "--n_t", type=int, default=41
    )  # Number of time nodes in interpolant
    args.add_argument(
        "--n_x", type=int, default=41
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
    args.add_argument("--loss_weight_max", "-lmw", type=float, default=100.0)

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

    # New args for finite differences
    args.add_argument("--fd_k_t", type=int, default=1)
    args.add_argument("--fd_k_x", type=int, default=1)

    args.add_argument("--seed", type=int, default=0)
    args = args.parse_args()

    torch.random.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    c = args.c
    beta = args.beta
    pde = Wave(
        c=c,
        beta=beta,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
        device=device,
    )

    base_save_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "plots",
        "pdes",
        "wave",
        f"c={c}_beta={beta}",
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
    # 1. Least Squares Polynomial Interpolant
    #########################################################
    if args.model is None or args.model == "least_squares":
        save_dir = os.path.join(base_save_dir, f"least_squares")

        # Model setup
        n_t_ls = args.n_t
        n_x_ls = args.n_x
        bases = ["chebyshev", "chebyshev"]
        model_ls = SpectralInterpolationND(
            Ns=[n_t_ls, n_x_ls],
            bases=bases,
            domains=pde.domain,
            device=device,
        )
        model_ls = pde.fit_least_squares(model_ls)
        pde.plot_solution(
            [t_eval, x_eval],
            model_ls.interpolate([t_eval, x_eval]),
            save_path=os.path.join(save_dir, "wave_ls_solution.png"),
        )

    #########################################################
    # 2. Neural network
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
    # 3. Polynomial interpolation
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
        model_ls = SpectralInterpolationND(
            Ns=[n_t, n_x],
            bases=bases,
            domains=pde.domain,
            device=device,
        )

        # Sanity check: fit least squares solution
        print("Fitting least squares solution...")
        pde.fit_least_squares(model_ls)
        pde.plot_solution(
            model_ls.nodes,
            model_ls.values,
            save_path=os.path.join(save_dir, "fit_least_squares.png"),
        )

        # Model setup
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
            lr_schedule=args.lr_schedule,
            gradient_clip=args.gradient_clip,
            hessian_every=args.hessian_every,
        )

    #########################################################
    # 4. MLP Interpolant
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
            lr_schedule=args.lr_schedule,
            gradient_clip=args.gradient_clip,
            hessian_every=args.hessian_every,
        )
