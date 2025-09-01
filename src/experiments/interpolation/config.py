"""
Configuration and parameter management for 1D interpolation experiments.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
import os


@dataclass
class ExperimentConfig:
    """Configuration for interpolation experiments."""

    # Training parameters
    n_train: int = 100
    n_test: int = 1000
    n_epochs: int = 10000
    eval_every: int = 100
    device: str = "cuda"
    dtype: torch.dtype = torch.float64
    seed: int = 0

    # Model architecture parameters
    hidden_dims: List[int] = None
    n_layers: List[int] = None
    cheb_node_sweep: List[int] = None

    # Target function parameters
    target_type: str = "sine"  # "gp", "sine", "exp"
    length_scales: List[float] = None  # for GP targets
    sine_frequencies: List[int] = None  # for sine targets

    # Experiment parameters
    model_type: str = "mlp"  # "mlp", "chebyshev", "mlpinterp", "all"
    sampling: str = "uniform"  # "uniform", "cheb"

    # Save directory
    save_dir: Optional[str] = None

    # Derivative supervision
    use_derivatives: bool = False
    deriv_alpha: float = 1.0  # weight on value loss
    deriv_beta: float = 1.0  # weight on derivative loss
    n_deriv_points: int | None = (
        None  # number of points to use for derivative term (None => all, 0 => disable)
    )

    def __post_init__(self):
        """Set default values if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [16, 32, 64, 128, 256]
        if self.n_layers is None:
            self.n_layers = [2, 3, 4, 5]  # Reduced from [2, 3, 4, 5, 6, 7, 8]
        if self.cheb_node_sweep is None:
            self.cheb_node_sweep = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 41, 61]
        if self.length_scales is None:
            self.length_scales = [0.1, 0.2, 0.4, 0.8]
        if self.sine_frequencies is None:
            self.sine_frequencies = [1, 2, 4, 8, 16, 32]
        if self.save_dir is None:
            # Default to a local outputs directory to avoid permission issues
            self.save_dir = os.path.join(os.getcwd(), "outputs")


@dataclass
class ModelConfig:
    """Configuration for individual model training."""

    hidden_dim: int
    n_layers: int
    activation = torch.tanh
    device: str = "cuda"
    dtype: torch.dtype = torch.float64

    @property
    def learning_rate(self) -> float:
        """Compute adaptive learning rate based on architecture."""
        return (
            0.05
            / (
                (
                    torch.sqrt(torch.tensor(2.0))
                    ** torch.log2(torch.tensor(self.n_layers / 2))
                )
                * (
                    torch.sqrt(torch.tensor(2.0))
                    ** torch.log2(torch.tensor(self.hidden_dim / 16))
                )
            ).item()
        )


@dataclass
class ChebyshevConfig:
    """Configuration for Chebyshev interpolation."""

    n_nodes: int
    domain: List[tuple] = None
    device: str = "cpu"

    def __post_init__(self):
        if self.domain is None:
            self.domain = [(-1, 1)]


@dataclass
class MLPInterpolantConfig:
    """Configuration for MLP-based interpolants."""

    n_nodes: int
    hidden_layers: tuple = (128, 128)
    activation = torch.nn.Tanh()
    domain: List[tuple] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        if self.domain is None:
            self.domain = [(-1, 1)]

    @property
    def learning_rate(self) -> float:
        """Compute adaptive learning rate for MLP interpolant."""
        return (
            0.05
            / (
                (torch.sqrt(torch.tensor(2.0)) ** torch.log2(torch.tensor(2 / 2)))
                * (torch.sqrt(torch.tensor(2.0)) ** torch.log2(torch.tensor(128 / 16)))
            ).item()
        )
