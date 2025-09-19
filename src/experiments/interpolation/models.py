"""
Model creation and training utilities for 1D interpolation experiments.
"""

import torch
import os
from typing import Tuple, List, Callable

from src.models.mlp import MLP
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp_interpolant_nd import MLPSpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger
from .config import ModelConfig, ChebyshevConfig, MLPInterpolantConfig


def create_mlp(config: ModelConfig) -> MLP:
    """Create an MLP model with given configuration."""
    return MLP(
        n_dim=1,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        activation=config.activation,
        device=config.device,
        dtype=config.dtype,
        embedding=config.embedding,
        embedding_M=config.embedding_M,
    )


def create_chebyshev_interpolant(config: ChebyshevConfig) -> SpectralInterpolationND:
    """Create a Chebyshev interpolant with given configuration."""
    return SpectralInterpolationND(
        Ns=[config.n_nodes],
        bases=["chebyshev"],
        domains=config.domain,
        device=config.device,
        dtype=config.dtype,
    )


def create_mlp_interpolant(config: MLPInterpolantConfig) -> MLPSpectralInterpolationND:
    """Create an MLP-based interpolant with given configuration."""
    return MLPSpectralInterpolationND(
        Ns=[config.n_nodes],
        bases=["chebyshev"],
        domains=config.domain,
        device=config.device,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        dtype=config.dtype,
    )


def train_and_evaluate_model(
    target,
    model,
    optimizer,
    n_epochs: int,
    eval_every: int,
    exp_dir: str,
    logger: Logger,
    l2_error,
    max_error,
    l2_relative_error,
    weight_evals: List[Callable] = [],
) -> Tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
    """
    Train and evaluate a model, returning metrics and predictions.

    Returns:
        Tuple of (rmse_train, l_inf_train, rmse_test, l_inf_test, u_pred_train, u_pred_test)
    """
    target.train(
        model=model,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=lambda: [target.train_points],
        eval_sampler=lambda: [target.test_points],
        eval_metrics=[l2_error, max_error, l2_relative_error],
        eval_every=eval_every,
        save_dir=exp_dir,
        logger=logger,
        weight_evals=weight_evals,
    )

    # Check if training terminated early due to SSBroyden NaN and load checkpoint if available
    optimizer_name = optimizer.__class__.__name__
    if "SSBroyden" in optimizer_name or "SSbroyden" in optimizer_name:
        # Look for the final early termination checkpoint first
        early_termination_checkpoint = os.path.join(exp_dir, "checkpoint_final_early_termination.pth")
        if os.path.exists(early_termination_checkpoint):
            print("Loading final early termination checkpoint for evaluation")
            model.load_state_dict(torch.load(early_termination_checkpoint, map_location=model.device if hasattr(model, 'device') else 'cpu'))
        else:
            # If no early termination checkpoint, look for the latest regular checkpoint
            import glob
            checkpoint_pattern = os.path.join(exp_dir, "checkpoint_*.pth")
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if checkpoint_files:
                # Extract epoch numbers and find the latest one (excluding early termination)
                regular_checkpoints = [f for f in checkpoint_files if "early_termination" not in f]
                if regular_checkpoints:
                    def extract_epoch(filename):
                        import re
                        match = re.search(r'checkpoint_(\d+)\.pth', filename)
                        return int(match.group(1)) if match else -1
                    
                    latest_checkpoint = max(regular_checkpoints, key=extract_epoch)
                    latest_epoch = extract_epoch(latest_checkpoint)
                    print(f"Loading latest regular checkpoint from epoch {latest_epoch}")
                    model.load_state_dict(torch.load(latest_checkpoint, map_location=model.device if hasattr(model, 'device') else 'cpu'))

    with torch.no_grad():
        u_pred_train = model([target.train_points])
        u_pred_test = model([target.test_points])

        rmse_train = torch.sqrt(
            torch.mean((u_pred_train - target.train_values) ** 2)
        ).item()
        l_inf_train = torch.max(torch.abs(u_pred_train - target.train_values)).item()
        rmse_test = torch.sqrt(
            torch.mean((u_pred_test - target.test_values) ** 2)
        ).item()
        l_inf_test = torch.max(torch.abs(u_pred_test - target.test_values)).item()

    return rmse_train, l_inf_train, rmse_test, l_inf_test, u_pred_train, u_pred_test


def run_mlp_experiment(
    target,
    config: ModelConfig,
    n_epochs: int,
    eval_every: int,
    exp_dir: str,
    optimizer_name: str = "adam",
    weight_evals: List[Callable] = [],
) -> dict:
    """Run MLP experiment for a single architecture."""
    embedding_str = f"_embed_{config.embedding}"
    if config.embedding_M is not None:
        embedding_str += f"_M{config.embedding_M}"
    
    print(
        f"Running MLP experiment (hidden_dim={config.hidden_dim}, n_layers={config.n_layers}, "
        f"embedding={config.embedding}{f', M={config.embedding_M}' if config.embedding_M else ''})"
    )

    # Create model and optimizer
    model = create_mlp(config)

    # Use the target's get_optimizer method to create the optimizer
    optimizer = target.get_optimizer(
        model,
        optimizer_name.lower(),
        lr=config.learning_rate,
    )

    # Create logger with original naming convention (no embedding suffix)
    logger = Logger(
        path=os.path.join(
            exp_dir, f"mlp_hdim{config.hidden_dim}_layers{config.n_layers}.json"
        )
    )

    # Create subdirectory for this specific model configuration
    model_subdir = os.path.join(
        exp_dir, f"mlp_hdim{config.hidden_dim}_layers{config.n_layers}{embedding_str}"
    )
    os.makedirs(model_subdir, exist_ok=True)

    # Train and evaluate
    results = train_and_evaluate_model(
        target,
        model,
        optimizer,
        n_epochs,
        eval_every,
        model_subdir,
        logger,
        l2_error,
        max_error,
        l2_relative_error,
        weight_evals,
    )

    rmse_train, l_inf_train, rmse_test, l_inf_test, u_pred_train, u_pred_test = results

    print(
        f"MLP (hdim={config.hidden_dim}, layers={config.n_layers}) "
        f"Train RMSE: {rmse_train:.2e}, L_inf: {l_inf_train:.2e} | "
        f"Test RMSE: {rmse_test:.2e}, L_inf: {l_inf_test:.2e}"
    )

    return {
        "rmse_train": rmse_train,
        "l_inf_train": l_inf_train,
        "rmse_test": rmse_test,
        "l_inf_test": l_inf_test,
        "u_pred_train": u_pred_train,
        "u_pred_test": u_pred_test,
        "model": model,
    }


def run_chebyshev_experiment(
    target,
    config: ChebyshevConfig,
    n_epochs: int,
    eval_every: int,
    exp_dir: str,
) -> dict:
    """Run Chebyshev interpolant experiment."""
    print(f"Running Chebyshev interpolant experiment (n_nodes={config.n_nodes})")

    # Create model and optimizer
    model = create_chebyshev_interpolant(config)
    optimizer = target.get_optimizer(model, "adam")

    # Create logger
    logger = Logger(
        path=os.path.join(exp_dir, f"chebyshev_interp_{config.n_nodes}.json")
    )

    # Train and evaluate
    results = train_and_evaluate_model(
        target,
        model,
        optimizer,
        n_epochs,
        eval_every,
        os.path.join(exp_dir, f"chebyshev_interp_{config.n_nodes}"),
        logger,
        l2_error,
        max_error,
        l2_relative_error,
    )

    rmse_train, l_inf_train, rmse_test, l_inf_test, u_pred_train, u_pred_test = results

    print(
        f"Chebyshev Interpolant (n={config.n_nodes}) "
        f"Train RMSE: {rmse_train:.2e}, L_inf: {l_inf_train:.2e} | "
        f"Test RMSE: {rmse_test:.2e}, L_inf: {l_inf_test:.2e}"
    )

    return {
        "rmse_train": rmse_train,
        "l_inf_train": l_inf_train,
        "rmse_test": rmse_test,
        "l_inf_test": l_inf_test,
        "u_pred_train": u_pred_train,
        "u_pred_test": u_pred_test,
        "model": model,
    }


def run_mlp_interpolant_experiment(
    target,
    config: MLPInterpolantConfig,
    n_epochs: int,
    eval_every: int,
    exp_dir: str,
) -> dict:
    """Run MLP interpolant experiment."""
    print(f"Running MLP Interpolant experiment (n_nodes={config.n_nodes})")

    # Create model and optimizer
    model = create_mlp_interpolant(config)
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=config.learning_rate)

    # Create logger
    logger = Logger(
        path=os.path.join(exp_dir, f"mlpinterp_interp_{config.n_nodes}.json")
    )

    # Train and evaluate
    results = train_and_evaluate_model(
        target,
        model,
        optimizer,
        n_epochs,
        eval_every,
        os.path.join(exp_dir, f"mlpinterp_interp_{config.n_nodes}"),
        logger,
        l2_error,
        max_error,
        l2_relative_error,
    )

    rmse_train, l_inf_train, rmse_test, l_inf_test, u_pred_train, u_pred_test = results

    print(
        f"MLP Interpolant (n={config.n_nodes}) "
        f"Train RMSE: {rmse_train:.2e}, L_inf: {l_inf_train:.2e} | "
        f"Test RMSE: {rmse_test:.2e}, L_inf: {l_inf_test:.2e}"
    )

    return {
        "rmse_train": rmse_train,
        "l_inf_train": l_inf_train,
        "rmse_test": rmse_test,
        "l_inf_test": l_inf_test,
        "u_pred_train": u_pred_train,
        "u_pred_test": u_pred_test,
        "model": model,
    }
