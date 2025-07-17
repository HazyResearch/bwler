import matplotlib.pyplot as plt
import os
import torch
from typing import List, Tuple

from src.optimizers.nys_newton_cg import NysNewtonCG

from src.utils.metrics import l2_relative_error


# Base class for all functions, including interpolation and PDEs.
# Note that the primitive for eval points, both during sampling and during evaluation, is a list of tensors, which defines a grid of points.
class BaseFcn:
    def __init__(
        self,
        name: str,
        domain: List[Tuple[float, float]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype
        self.domain = domain
        self.n_dims = len(domain)
        self.domain_lengths = [domain[i][1] - domain[i][0] for i in range(self.n_dims)]
        # Set up domain mapping functions for each dimension, for both chebyshev and fourier
        # Default domains are [-1, 1] for chebyshev and [0, 2*pi] for fourier
        self._to_cheb = (
            lambda x, d: 2 * (x - self.domain[d][0]) / (self.domain_lengths[d]) - 1
        )
        self._from_cheb = (
            lambda x, d: self.domain[d][0] + (x + 1) * (self.domain_lengths[d]) / 2
        )
        self._to_fourier = (
            lambda x, d: 2
            * torch.pi
            * (x - self.domain[d][0])
            / (self.domain_lengths[d])
        )
        self._from_fourier = (
            lambda x, d: self.domain[d][0]
            + (x / (2 * torch.pi)) * self.domain_lengths[d]
        )

    def get_domain(self, dim: int) -> Tuple[float, float]:
        return self.domain[dim]

    def get_domain_lengths(self) -> List[float]:
        return self.domain_lengths

    def get_n_dims(self) -> int:
        return self.n_dims

    def get_name(self) -> str:
        return self.name

    # Input: nodes = [(n_1,), ..., (n_d,)]
    # Output: u = (n_1 * ... * n_d,)
    def get_solution(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    # Sample points from one dimension of the domain:
    # basis: "chebyshev" distributed or "fourier" (uniformly) distributed
    # type: "standard" (canonical nodes) or "uniform" (uniformly sampled)
    def sample_domain_1d(
        self, n_samples: int, dim: int, basis: str, type: str
    ) -> torch.Tensor:
        if type == "standard":
            if basis == "chebyshev":
                cheb_nodes = torch.cos(
                    torch.linspace(0, torch.pi, n_samples, device=self.device)
                )
                return self._from_cheb(cheb_nodes, dim)
            elif basis == "fourier" or basis == "fd":
                fourier_nodes = torch.linspace(
                    0, 2 * torch.pi, n_samples + 1, device=self.device
                )[:-1]
                return self._from_fourier(fourier_nodes, dim)
            else:
                raise ValueError(f"Invalid basis: {basis}")
        elif type == "uniform":
            if basis == "chebyshev":
                uniform_nodes = torch.cos(
                    torch.rand(n_samples, device=self.device) * torch.pi
                )
                return self._from_cheb(uniform_nodes, dim)
            elif basis == "fourier" or basis == "fd":
                uniform_nodes = torch.rand(n_samples, device=self.device) * 2 * torch.pi
                return self._from_fourier(uniform_nodes, dim)
            else:
                raise ValueError(f"Invalid basis: {basis}")
        else:
            raise ValueError(f"Invalid type: {type}")

    def sample_domain(
        self, n_samples: int, basis: List[str], type: List[str]
    ) -> List[torch.Tensor]:
        return [
            self.sample_domain_1d(n_samples, dim, basis[dim], type[dim])
            for dim in range(self.n_dims)
        ]

    # Default plot solution function for 1D/2D functions
    # Plots predicted, true, and error (log scale)
    def _plot_solution_default(
        self,
        nodes: List[torch.Tensor],  # (N_x) or (N_t, N_x)
        u: torch.Tensor,  # (N_x) or (N_t, N_x)
        save_path: str = None,
        u_true: torch.Tensor = None,
        **kwargs,
    ):
        u_cpu = u.detach().cpu()
        if u_true is None:
            u_true = self.get_solution(nodes).detach().cpu()
        errors = torch.abs(u_cpu - u_true)
        l2_relative_errors = l2_relative_error(u_cpu, u_true)
        if self.n_dims == 1:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            nodes_cpu = nodes[0].detach().cpu()

            # Plot 1: predicted solution
            ax1.plot(nodes_cpu, u_cpu, "b-", label="Predicted")
            ax1.plot(nodes_cpu, u_true, "k:", label="True")
            ax1.set_title("Predicted vs True")
            ax1.legend()
            ax1.grid(True)

            # Plot 2: error (log scale)
            ax2.semilogy(nodes_cpu, errors, "b-", label="Absolute Error")
            ax2.set_title("Absolute Error")
            ax2.legend()
            ax2.grid(True)

        elif self.n_dims == 2:
            # For 2D, we assume the solution is parameterized as (t, x) for temporal PDEs. So for plotting, we plot u.T.

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

            # Plot 1: predicted solution
            im1 = ax1.imshow(
                u_cpu.T,
                extent=[
                    self.domain[0][0],
                    self.domain[0][1],
                    self.domain[1][0],
                    self.domain[1][1],
                ],
                origin="lower",
                aspect="auto",
            )
            plt.colorbar(im1, ax=ax1)
            ax1.set_title("Predicted Solution")
            ax1.set_xlabel("t")
            ax1.set_ylabel("x")

            # Plot 2: true solution
            im2 = ax2.imshow(
                u_true.T,
                extent=[
                    self.domain[0][0],
                    self.domain[0][1],
                    self.domain[1][0],
                    self.domain[1][1],
                ],
                origin="lower",
                aspect="auto",
            )
            plt.colorbar(im2, ax=ax2)
            ax2.set_title("True Solution")
            ax2.set_xlabel("t")
            ax2.set_ylabel("x")

            # Plot 3: error (log scale)
            im3 = ax3.imshow(
                errors.T,
                extent=[
                    self.domain[0][0],
                    self.domain[0][1],
                    self.domain[1][0],
                    self.domain[1][1],
                ],
                origin="lower",
                aspect="auto",
                norm="log",
            )
            plt.colorbar(im3, ax=ax3)
            title = "Absolute Error (L2RE = {:.2e})".format(l2_relative_errors)
            ax3.set_title(title)
            ax3.set_xlabel("t")
            ax3.set_ylabel("x")

        else:
            raise ValueError(
                f"Invalid number of dimensions for _plot_solution_default: {self.n_dims}"
            )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()

    def get_optimizer(self, model, optimizer_name, **override_kwargs):
        optimizer_dict = {
            "adam": {
                "constructor": torch.optim.Adam,
                "kwargs": {"lr": 1e-3},
            },
            "lbfgs": {
                "constructor": torch.optim.LBFGS,
                "kwargs": {"history_size": 100, "tolerance_grad": 1e-16, "tolerance_change": 1e-16},
            },
            "nys_newton": {
                "constructor": NysNewtonCG,
                "kwargs": {
                    "lr": 1.0,
                    "rank": 1000,
                    "cg_max_iters": 1000,
                    # "cg_max_iters": 100,
                    "mu": 1e-2,
                    "line_search_fn": "armijo",
                    "cg_tol": 1e-16,
                    "cg_max_iters": 1000,
                },
            },
        }

        entry = optimizer_dict[optimizer_name]
        constructor = entry["constructor"]
        kwargs = entry["kwargs"]
        kwargs.update(override_kwargs)

        return constructor(model.parameters(), **kwargs)
