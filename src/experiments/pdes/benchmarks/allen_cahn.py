import argparse
import os
import scipy.io
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Dict, Callable
from datetime import datetime
import logging
from time import time

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.models.mlp import MLP
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.optimizers.nys_newton_cg import NysNewtonCG
from src.loggers.logger import Logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
1D Allen-Cahn equation:
u_t - eps * u_xx - 5u + 5u^3 = 0
t in [0, 1]
x in [-1, 1]
u(t=0, x) = x^2 cos(pi*x)
u(t, x=-1) = u(t, x=1) = 0
"""


class AllenCahn(BasePDE):
    def __init__(
        self,
        eps: float = 1e-4,
        device: str = "cpu",
        **base_kwargs,
    ):
        super().__init__("allen_cahn", [(0, 1), (-1, 1)], device=device, **base_kwargs)
        self.eps = eps
        self.u_0 = lambda x: x**2 * torch.cos(torch.pi * x)
        self.ref_u, self.ref_t, self.ref_x = self.load_ref_soln()

    def load_ref_soln(
        self,
        path: str = None,
    ):
        if path is None:
            # Use relative path like Burgers implementation
            path = os.path.join(
                os.path.dirname(__file__),
                "ref",
                "allen_cahn_d=1e-4.mat",
            )
        
        logger.info(f"Loading reference solution from: {path}")
        logger.info(f"File exists: {os.path.exists(path)}")
        
        try:
            mat = scipy.io.loadmat(path)
            u_ref = torch.tensor(mat["usol"], dtype=torch.float64, device=self.device)
            t = torch.tensor(mat["t"][0], dtype=torch.float64, device=self.device)
            x = torch.tensor(mat["x"][0], dtype=torch.float64, device=self.device)
            logger.info(f"Successfully loaded reference solution with shape: {u_ref.shape}")
            logger.info(f"Time range: [{t.min().item():.4f}, {t.max().item():.4f}]")
            logger.info(f"Space range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            return u_ref, t, x
        except Exception as e:
            logger.error(f"Failed to load reference solution: {e}")
            raise

    # Hack: assume t and x are the same as the reference solution
    def get_solution(self, nodes: List[torch.Tensor]):
        return self.ref_u

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        n_t, n_x = pde_nodes[0].shape[0], pde_nodes[1].shape[0]
        n_ic = ic_nodes[1].shape[0]

        if isinstance(model, SpectralInterpolationND):
            # PDE
            u = model.interpolate(pde_nodes)
            u_t = model.derivative(pde_nodes, k=(1, 0))
            u_xx = model.derivative(pde_nodes, k=(0, 2))
            # IC
            u_ic = model.interpolate(ic_nodes)
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model(
                [pde_nodes[0], torch.tensor([self.domain[1][0]], dtype=pde_nodes[1].dtype, device=model.device)]
            )
            u_periodic_t1 = model(
                [pde_nodes[0], torch.tensor([self.domain[1][1]], dtype=pde_nodes[1].dtype, device=model.device)]
            )
            # Periodic boundary derivatives
            u_x_periodic_t0 = model.derivative(
                [pde_nodes[0], torch.tensor([self.domain[1][0]], dtype=pde_nodes[1].dtype, device=model.device)], k=(0, 1)
            )
            u_x_periodic_t1 = model.derivative(
                [pde_nodes[0], torch.tensor([self.domain[1][1]], dtype=pde_nodes[1].dtype, device=model.device)], k=(0, 1)
            )
        else:
            # PDE
            grid = model.make_grid(pde_nodes)  # (N_t*N_x, 2)
            u = model.forward_grid(grid).reshape(n_t, n_x)  # (N_t, N_x)
            grads = torch.autograd.grad(u.sum(), grid, create_graph=True)[0]  # (N_t*N_x, 2)
            u_t = grads[..., 0].reshape(n_t, n_x)
            u_x = grads[..., 1].reshape(n_t, n_x)
            # Second derivative: u_xx
            grad_xx = torch.autograd.grad(u_x.sum(), grid, create_graph=True)[0]
            u_xx = grad_xx[..., 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes)
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model(
                [pde_nodes[0], torch.tensor([self.domain[1][0]], dtype=pde_nodes[1].dtype, device=model.device)]
            )
            u_periodic_t1 = model(
                [pde_nodes[0], torch.tensor([self.domain[1][1]], dtype=pde_nodes[1].dtype, device=model.device)]
            )
            # Periodic boundary derivatives
            grid_bc_t0 = model.make_grid([pde_nodes[0], torch.tensor([self.domain[1][0]], dtype=pde_nodes[1].dtype, device=model.device)])
            grid_bc_t1 = model.make_grid([pde_nodes[0], torch.tensor([self.domain[1][1]], dtype=pde_nodes[1].dtype, device=model.device)])
            u_bc_t0 = model.forward_grid(grid_bc_t0)
            u_bc_t1 = model.forward_grid(grid_bc_t1)
            u_x_periodic_t0 = torch.autograd.grad(u_bc_t0.sum(), grid_bc_t0, create_graph=True)[0][..., 1]
            u_x_periodic_t1 = torch.autograd.grad(u_bc_t1.sum(), grid_bc_t1, create_graph=True)[0][..., 1]

        # PDE loss
        pde_residual = u_t - self.eps * u_xx - 5 * u + 5 * u**3
        pde_loss = torch.mean(pde_residual**2)
        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2)
        # Periodic boundary conditions loss (both function values and derivatives)
        pbc_func_loss = torch.mean((u_periodic_t0 - u_periodic_t1) ** 2)
        pbc_deriv_loss = torch.mean((u_x_periodic_t0 - u_x_periodic_t1) ** 2)
        pbc_loss = pbc_func_loss + pbc_deriv_loss

        loss_names = ["pde_loss", "ic_loss", "pbc_loss"]
        return dict(zip(loss_names, [pde_loss, ic_loss, pbc_loss]))

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)

        pde_weight = self.loss_weights.get("pde_loss_weight", 1.0)
        ic_weight_adjusted = self.loss_weights.get("ic_loss_weight", ic_weight)
        pbc_weight = self.loss_weights.get("pbc_loss_weight", ic_weight)

        loss = (
            (pde_weight * loss_dict["pde_loss"])
            + (ic_weight_adjusted * ic_weight * loss_dict["ic_loss"])
            + (ic_weight * pbc_weight * loss_dict["pbc_loss"])
        )

        return (
            loss,
            loss_dict["pde_loss"],
            loss_dict["ic_loss"] + loss_dict["pbc_loss"],
        )

    def get_least_squares(self, model: SpectralInterpolationND):
        raise NotImplementedError(
            "Least squares not implemented for Allen-Cahn equation"
        )

    def fit_least_squares(self, model: SpectralInterpolationND):
        raise NotImplementedError(
            "Least squares not implemented for Allen-Cahn equation"
        )

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,  # (N_t, N_x)
        save_path: str = None,
    ):
        """Plot the solution."""
        self._plot_solution_default(nodes, u, save_path)


if __name__ == "__main__":
    # Test SSBroyden availability
    try:
        from src.optimizers.ssbroyden import SSBroyden2
        from src.optimizers.Lssbroyden import L_SSBroyden
        logger.info("SSBroyden optimizers are available")
        ssbroyden_available = True
    except ImportError as e:
        logger.warning(f"SSBroyden optimizers not available: {e}")
        ssbroyden_available = False

    args = argparse.ArgumentParser()
    args.add_argument("--eps", type=float, default=1e-4)
    args.add_argument("--n_t", type=int, default=81)
    args.add_argument("--n_x", type=int, default=80)
    args.add_argument("--n_layers", type=int, default=3)  # Number of layers in MLP
    args.add_argument("--hidden_dim", type=int, default=256)  # Number of hidden nodes in MLP
    args.add_argument("--activation", type=str, default="tanh")
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument(
        "--method",
        type=str,
        default="adam",
        choices=["adam", "lbfgs", "nys_newton", "ssbroyden", "lssbroyden", "mini_batch"],
        help="Optimization method",
    )
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
    args.add_argument("--lr_schedule", action="store_true", help="Enable learning rate scheduling")
    args.add_argument("--gradient_clip", type=float, default=1.0, help="Maximum gradient norm for clipping (0 to disable)")
    args.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    args.add_argument("--lr_max", type=float, default=1e-3, help="Maximum learning rate for cosine annealing")

    args.add_argument("--fd_k_t", type=int, default=None, help="Stencil size for FD in time dimension (None for spectral)")
    
    # Add pretraining parameters
    args.add_argument("--pretrain_mlp", action="store_true", help="Enable MLP pretraining for polynomial model")
    args.add_argument("--pretrain_epochs", type=int, default=5000, help="Number of pretraining epochs")
    args.add_argument("--pretrain_optimizer", type=str, default="ssbroyden", help="Optimizer for pretraining")
    args.add_argument("--pretrain_eval_every", type=int, default=100, help="Evaluation frequency during pretraining")
    
    # Mini-batching parameters
    args.add_argument("--use_mini_batch", action="store_true", help="Enable mini-batching for memory efficiency")
    args.add_argument("--batch_size", type=int, default=1000, help="Mini-batch size for PDE loss computation")
    args.add_argument("--accumulate_grads", action="store_true", help="Accumulate gradients across mini-batches")

    args.add_argument(
        "--use_mlp_for_derivatives",
        action="store_true",
        help="Use MLP autograd for derivatives in MLPSpectralInterpolationND",
    )
    
    # Add MLPInterpolant collocation control
    args.add_argument("--mlpinterp_random_collocation", action="store_true", 
                     help="Use random collocation points for MLPInterpolant (default: fixed BWLer nodes)")

    # Add pretrained MLP initialization
    args.add_argument("--pretrained_mlp_path", type=str, default=None,
                     help="Path to pretrained MLP checkpoint to initialize polynomial model")

    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    eps = args.eps
    pde = AllenCahn(
        eps=eps,
        device=device,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
    )

    base_save_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        f"plots/pdes/allen_cahn/eps={pde.eps}",
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
        if args.method == "nys_newton":
            optimizer = NysNewtonCG(
                model_mlp.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden" and ssbroyden_available:
            optimizer = SSBroyden2(
                model_mlp.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        elif args.method == "lssbroyden" and ssbroyden_available:
            optimizer = L_SSBroyden(
                model_mlp.parameters(),
                lr=1.0,
                history_size=10,
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
        if args.use_mini_batch:
            pde.train_model_mini_batch(
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
                batch_size=args.batch_size,
                accumulate_grads=args.accumulate_grads,
            )
        else:
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
        # Add pretrained info to save directory
        pretrained_suffix = "_from_mlp" if args.pretrained_mlp_path is not None else ""
        
        save_dir = os.path.join(
            base_save_dir,
            f"polynomial/method={args.method}_nt={args.n_t}_nx={args.n_x}_sample={args.sample_type}_nncgrank={args.nncg_rank}_nncgcgmaxiters={args.nncg_cgmaxiters}{pretrained_suffix}",
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

        # Initialize from pretrained MLP if specified
        if args.pretrained_mlp_path is not None:
            print(f"Initializing polynomial model from pretrained MLP: {args.pretrained_mlp_path}")
            
            # Load the pretrained MLP checkpoint
            checkpoint = torch.load(args.pretrained_mlp_path, map_location=device)
            
            # Create a temporary MLP with the same architecture to load the weights
            temp_mlp = MLP(
                n_dim=2,
                n_layers=3,  # Based on the path, it's 3 layers
                hidden_dim=64,  # Based on the path, it's 64 hidden dim
                activation=torch.tanh,
                device=device,
            )
            temp_mlp.load_state_dict(checkpoint)
            temp_mlp.eval()
            
            # Sample points on the polynomial grid to get MLP predictions
            with torch.no_grad():
                # Get the actual physical nodes of the polynomial model
                t_nodes = model.nodes[0]  # Physical t nodes
                x_nodes = model.nodes[1]  # Physical x nodes
                
                # Create meshgrid from the actual nodes
                t_mesh, x_mesh = torch.meshgrid(t_nodes, x_nodes, indexing="ij")
                
                # Flatten for MLP input
                grid_points = torch.stack([t_mesh.flatten(), x_mesh.flatten()], dim=1)
                
                # Get MLP predictions at the polynomial model's physical nodes
                mlp_predictions = temp_mlp(grid_points).reshape(t_nodes.shape[0], x_nodes.shape[0])
                
                # Set polynomial model values to MLP predictions
                model.values.data = mlp_predictions
            
            print(f"Successfully initialized polynomial model from MLP checkpoint")
            print(f"Polynomial model shape: {model.values.shape}")
            print(f"MLP predictions range: [{mlp_predictions.min().item():.6f}, {mlp_predictions.max().item():.6f}]")
            
            # Evaluate MLP performance on reference solution
            print("\n" + "="*50)
            print("EVALUATING PRETRAINED MLP PERFORMANCE")
            print("="*50)
            
            # Use the evaluation grid that matches reference solution
            eval_nodes = eval_sampler()  # This gives t_eval, x_eval
            ref_solution = pde.get_solution(eval_nodes)
            
            # Get MLP predictions on evaluation grid
            t_eval, x_eval = eval_nodes
            t_mesh_eval, x_mesh_eval = torch.meshgrid(t_eval, x_eval, indexing="ij")
            grid_points_eval = torch.stack([t_mesh_eval.flatten(), x_mesh_eval.flatten()], dim=1)
            mlp_predictions_eval = temp_mlp(grid_points_eval).reshape(t_eval.shape[0], x_eval.shape[0])
            
            # Evaluate MLP predictions
            mlp_l2_error = l2_error(mlp_predictions_eval, ref_solution)
            mlp_max_error = max_error(mlp_predictions_eval, ref_solution)
            mlp_l2_relative_error = l2_relative_error(mlp_predictions_eval, ref_solution)
            
            print(f"MLP L2 Error: {mlp_l2_error:.6e}")
            print(f"MLP Max Error: {mlp_max_error:.6e}")
            print(f"MLP L2 Relative Error: {mlp_l2_relative_error:.6e}")
            
            # Evaluate polynomial model performance (should be same as MLP initially)
            print("\n" + "="*50)
            print("EVALUATING POLYNOMIAL MODEL AFTER MLP INITIALIZATION")
            print("="*50)
            
            # Get polynomial predictions on evaluation grid
            poly_predictions_eval = model.interpolate(eval_nodes)
            
            # Evaluate polynomial predictions
            poly_l2_error = l2_error(poly_predictions_eval, ref_solution)
            poly_max_error = max_error(poly_predictions_eval, ref_solution)
            poly_l2_relative_error = l2_relative_error(poly_predictions_eval, ref_solution)
            
            print(f"Polynomial L2 Error: {poly_l2_error:.6e}")
            print(f"Polynomial Max Error: {poly_max_error:.6e}")
            print(f"Polynomial L2 Relative Error: {poly_l2_relative_error:.6e}")
            
            # Check if polynomial matches MLP (should be very close)
            poly_mlp_diff = torch.abs(poly_predictions_eval - mlp_predictions_eval)
            print(f"Max difference between polynomial and MLP: {poly_mlp_diff.max().item():.6e}")
            print(f"Mean difference between polynomial and MLP: {poly_mlp_diff.mean().item():.6e}")
            
            print("="*50 + "\n")

        # Training setup
        n_epochs = args.n_epochs
        if args.method == "nys_newton":
            optimizer = NysNewtonCG(
                model.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden" and ssbroyden_available:
            optimizer = SSBroyden2(
                model.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        elif args.method == "lssbroyden" and ssbroyden_available:
            optimizer = L_SSBroyden(
                model.parameters(),
                lr=1.0,
                history_size=10,
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

        print(f"\nTraining Polynomial Interpolant with {args.method} optimizer...")
        if args.use_mini_batch:
            pde.train_model_mini_batch(
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
                batch_size=args.batch_size,
                accumulate_grads=args.accumulate_grads,
            )
        else:
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
            optimizer = NysNewtonCG(
                model.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden" and ssbroyden_available:
            optimizer = SSBroyden2(
                model.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        elif args.method == "lssbroyden" and ssbroyden_available:
            optimizer = L_SSBroyden(
                model.parameters(),
                lr=1.0,
                history_size=10,
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

        # Check if pretraining is enabled
        if args.pretrain_mlp:
            print(f"Pre-training MLP with {args.pretrain_optimizer} optimizer...")
            
            # Create MLP model for pretraining
            mlp_model = MLP(
                n_dim=2,
                n_layers=args.n_layers,
                hidden_dim=args.hidden_dim,
                activation=torch.tanh,
                device=device,
            )
            
            # Setup pretraining optimizer
            if args.pretrain_optimizer == "ssbroyden" and ssbroyden_available:
                pretrain_optimizer = SSBroyden2(
                    mlp_model.parameters(),
                    lr=1.0,
                    init_scale=True,
                    c1=1e-4,
                    c2=0.9,
                    max_ls=20,
                )
            else:
                pretrain_optimizer = pde.get_optimizer(mlp_model, args.pretrain_optimizer)
            
            # Setup pretraining logger
            pretrain_save_dir = os.path.join(save_dir, "pretrain")
            os.makedirs(pretrain_save_dir, exist_ok=True)
            pretrain_logger = Logger(path=os.path.join(pretrain_save_dir, "pretrain_logger.json"))
            
            # Pretrain MLP
            pde.train(
                mlp_model,
                n_epochs=args.pretrain_epochs,
                optimizer=pretrain_optimizer,
                pde_sampler=pde_sampler,
                ic_sampler=ic_sampler,
                ic_weight=ic_weight,
                eval_sampler=eval_sampler,
                eval_metrics=eval_metrics,
                eval_every=args.pretrain_eval_every,
                save_dir=pretrain_save_dir,
                logger=pretrain_logger,
                lr_schedule=args.lr_schedule,
                gradient_clip=args.gradient_clip,
            )
            
            # Transfer MLP values to polynomial model
            print("Transferring MLP values to polynomial model...")
            model.load_values_from_model(mlp_model)
            print(f"Transferred values with shape: {model.values.shape}")
            print(f"Value range: [{model.values.min().item():.4f}, {model.values.max().item():.4f}]")
            
            # Save pretrained MLP
            torch.save(mlp_model.state_dict(), os.path.join(pretrain_save_dir, "pretrained_mlp.pt"))
            
            print(f"Training Polynomial Interpolant with FD in time dimension (from pretrained initialization)...")
        else:
            print(f"Training Polynomial Interpolant with FD in time dimension...")
            
        if args.use_mini_batch:
            pde.train_model_mini_batch(
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
                batch_size=args.batch_size,
                accumulate_grads=args.accumulate_grads,
            )
        else:
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
        # Add collocation type to save directory
        collocation_suffix = "_random_collocation" if args.mlpinterp_random_collocation else "_fixed_collocation"
        
        save_dir = os.path.join(
            base_save_dir,
            f"mlpinterp/method={args.method}_nt={args.n_t}_nx={args.n_x}_nlayers={args.n_layers}_hdim={args.hidden_dim}_activation={args.activation}_sample={args.sample_type}{collocation_suffix}",
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
            optimizer = NysNewtonCG(
                model.parameters(),
                lr=1,
                rank=args.nncg_rank,
                cg_max_iters=args.nncg_cgmaxiters,
                mu=1e-2,
                cg_tol=1e-16,
                line_search_fn="armijo",
            )
        elif args.method == "ssbroyden" and ssbroyden_available:
            optimizer = SSBroyden2(
                model.parameters(),
                lr=1.0,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        elif args.method == "lssbroyden" and ssbroyden_available:
            optimizer = L_SSBroyden(
                model.parameters(),
                lr=1.0,
                history_size=10,
                init_scale=True,
                c1=1e-4,
                c2=0.9,
                max_ls=20,
            )
        else:
            optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 321*2
        n_x_train = 321*2
        n_ic_train = 321*2
        ic_weight = 10

        def pde_sampler():
            # Determine sampling type based on argument
            if args.mlpinterp_random_collocation:
                sampling_type = "uniform"  # Random collocation points
                print(f"Using random collocation points for MLPInterpolant")
            else:
                sampling_type = "standard"  # Fixed BWLer nodes
                print(f"Using fixed BWLer nodes for MLPInterpolant")
            
            t_nodes = pde.sample_domain_1d(
                n_samples=n_t_train,
                dim=0,
                basis=bases[0],
                type=sampling_type,
            )
            x_nodes = pde.sample_domain_1d(
                n_samples=n_x_train,
                dim=1,
                basis=bases[1],
                type=sampling_type,
            )
            return [t_nodes, x_nodes]

        def ic_sampler():
            # Use same sampling type for IC nodes
            if args.mlpinterp_random_collocation:
                sampling_type = "uniform"
            else:
                sampling_type = "standard"
            
            ic_nodes = pde.sample_domain_1d(
                n_samples=n_ic_train,
                dim=1,
                basis=bases[1],
                type=sampling_type,
            )
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"\nTraining MLP Interpolant with {args.method} optimizer...")
        if args.use_mini_batch:
            pde.train_model_mini_batch(
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
                batch_size=args.batch_size,
                accumulate_grads=args.accumulate_grads,
            )
        else:
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
