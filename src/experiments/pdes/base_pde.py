import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Callable
import numpy as np

from src.optimizers.nys_newton_cg import NysNewtonCG
from src.optimizers.ssbroyden import SSBroyden2
from src.optimizers.Lssbroyden import L_SSBroyden
from src.utils.pyhessian import HessianAnalyzer

from src.experiments.base_fcn import BaseFcn
from src.loggers.logger import Logger


class BasePDE(BaseFcn):
    def __init__(
        self,
        name: str,
        domain: List[Tuple[float, float]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        loss_weight_update_policy: str = "grad_norm",
        loss_weight_update_interval: int = -1,
    ):
        super().__init__(
            name=name,
            domain=domain,
            device=device,
            dtype=dtype,
        )
        self.loss_weights = {}
        self.loss_weight_update_policy = loss_weight_update_policy
        self.loss_weight_update_interval = loss_weight_update_interval
        self.loss_weight_ema_alpha = 0.9

    # Input: x = [(n_1,), ..., (n_d,)]
    # Output: u = (n_1 * ... * n_d,)
    def get_solution(self, x: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def sample_domain_1d(
        self, n_samples: int, dim: int, basis: str, type: str
    ) -> torch.Tensor:
        """Sample points from domain with requires_grad=True"""
        points = super().sample_domain_1d(n_samples, dim, basis, type)
        points.requires_grad_(True)
        return points

    def sample_domain(
        self, n_samples: int, dim: int, basis: str, type: str
    ) -> List[torch.Tensor]:
        """Sample points from domain with requires_grad=True"""
        points = super().sample_domain(n_samples, dim, basis, type)
        points = [p.requires_grad_(True) for p in points]
        return points

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ):
        raise NotImplementedError

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],
        ic_weight: float,
        n_square_boundary: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def plot_solution(
        self,
        model: nn.Module,
        nodes: List[torch.Tensor],
        u: torch.Tensor,
        save_path: str = None,
    ):
        """Plot the solution. To be implemented by subclasses."""
        raise NotImplementedError

    def update_loss_weights(self, epoch, model, optimizer, pde_nodes, ic_nodes):
        if self.loss_weight_update_policy != "grad_norm":
            return

        # Update loss weights every self.loss_weight_update_interval epochs
        if (
            self.loss_weight_update_interval == -1
            or epoch % self.loss_weight_update_interval != 0
        ):
            return

        # If loss weights are not initialized, initialize them to 1
        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)
        loss_names = loss_dict.keys()
        if self.loss_weights == {}:
            self.loss_weights = {
                f"{loss_name}_weight": 1 / len(loss_names) for loss_name in loss_names
            }
            return

        # Compute average gradient norm for each loss term
        self.latest_loss_weights = {}
        num_params_grad = sum(
            p.numel() for p in model.parameters() if p.grad is not None
        )
        grad_norm_sum = 0

        for loss_name in loss_names:
            optimizer.zero_grad()
            curr_loss = self.get_loss_dict(model, pde_nodes, ic_nodes)[loss_name]
            optimizer.zero_grad()
            curr_loss.backward(retain_graph=True)
            curr_loss_avg_grad_norm = (
                torch.sqrt(
                    sum(
                        [
                            torch.sum(p.grad**2)
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                / num_params_grad
            )
            grad_norm_sum += curr_loss_avg_grad_norm
            self.latest_loss_weights[f"{loss_name}_weight"] = curr_loss_avg_grad_norm

        print(grad_norm_sum)
        print(self.latest_loss_weights)

        for loss_name in loss_names:
            # if nan, set to 1
            prev_loss_weight = self.loss_weights[f"{loss_name}_weight"]
            self.loss_weights[f"{loss_name}_weight"] = (
                self.loss_weight_ema_alpha * prev_loss_weight
                + (1 - self.loss_weight_ema_alpha)
                * self.latest_loss_weights[f"{loss_name}_weight"]
                / grad_norm_sum
            )
            if torch.isnan(self.loss_weights[f"{loss_name}_weight"]):
                self.loss_weights[f"{loss_name}_weight"] = 1

        print(self.loss_weights)

    def train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 1000,
        save_dir: str = None,
        logger: Logger = None,
        lr_schedule: bool = True,
        gradient_clip: float = 1.0,
        hessian_every: int = -1,  # How often to compute Hessian (-1 for never)
        hessian_num_iter: int = 100,  # Number of iterations for Lanczos
        hessian_num_run: int = 1,  # Number of runs for Hessian computation
        n_square_boundary: int = 0,  # Number of points on square boundary
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Add learning rate scheduler if requested
        if lr_schedule and isinstance(optimizer, torch.optim.Adam):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=1e-6
            )
        else:
            scheduler = None

        print("Training model...")
        start_time = time()
        iter_times = []  # Track per-iteration times

        for epoch in tqdm(range(n_epochs)):
            iter_start_time = time()  # Start timing this iteration

            # Sample points
            pde_nodes = pde_sampler()
            ic_nodes = ic_sampler()

            # Update loss weights
            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

            # Train step
            optimizer.zero_grad()

            # Get PDE loss
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model,
                pde_nodes,
                ic_nodes,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )

            # Backprop
            loss.backward()

            # Gradient clipping if requested
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )

            # Update parameters
            optimizer.step()

            # Update learning rate if using scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                logger.log("learning_rate", current_lr, epoch)

            # Record iteration time
            iter_time = time() - iter_start_time
            iter_times.append(iter_time)
            logger.log("iter_time", iter_time, epoch)
            logger.log(
                "avg_iter_time",
                sum(iter_times[-eval_every:]) / len(iter_times[-eval_every:]),
                epoch,
            )

            # Log
            logger.log("loss", loss.item(), epoch)
            logger.log("train_pde_loss", pde_loss.item(), epoch)
            logger.log("train_ic_loss", ic_loss.item(), epoch)

            # Compute Hessian if requested
            if hessian_every > 0 and (epoch + 1) % hessian_every == 0:
                print(f"\nComputing Hessian at epoch {epoch + 1}...")
                print(f"Using {hessian_num_iter} iterations and {hessian_num_run} runs")
                hessian_start_time = time()

                # Create Hessian analyzer
                print("Creating Hessian analyzer...")
                hessian_analyzer = HessianAnalyzer(
                    model=model,
                    pde=self,
                    pde_sampler=pde_sampler,
                    ic_sampler=ic_sampler,
                    device=model.device,
                )

                # Compute spectral density
                print("Computing spectral density...")
                eigen_list_full, weight_list_full = hessian_analyzer.density(
                    num_iter=hessian_num_iter, num_run=hessian_num_run
                )

                # Save spectral density data
                if save_dir is not None:
                    print(f"Saving Hessian data to {save_dir}/hessian/")
                    hessian_dir = os.path.join(save_dir, "hessian")
                    os.makedirs(hessian_dir, exist_ok=True)

                    # Save raw data (convert complex to real)
                    np.savez(
                        os.path.join(
                            hessian_dir, f"spectral_density_epoch_{epoch}.npz"
                        ),
                        eigenvalues=np.array([np.real(e) for e in eigen_list_full]),
                        weights=weight_list_full,
                    )

                    # Create and save visualization
                    plt.figure(figsize=(10, 6))
                    for run_idx, (eigenvalues, weights) in enumerate(
                        zip(eigen_list_full, weight_list_full)
                    ):
                        # Convert complex eigenvalues to real
                        real_eigenvalues = np.real(eigenvalues)
                        plt.semilogy(
                            real_eigenvalues,
                            weights,
                            label=f"Run {run_idx+1}",
                            alpha=0.7,
                        )
                    plt.xlabel("Eigenvalue")
                    plt.ylabel("Density")
                    plt.title(f"Hessian Spectral Density at Epoch {epoch}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(hessian_dir, f"spectral_density_epoch_{epoch}.png")
                    )
                    plt.close()

                # Log eigenvalues and weights (convert complex to real)
                print("Logging eigenvalues and weights...")
                for run_idx, (eigenvalues, weights) in enumerate(
                    zip(eigen_list_full, weight_list_full)
                ):
                    # Debug print raw eigenvalues
                    print(f"\nRaw eigenvalues for run {run_idx + 1}:")
                    print(f"First 5 eigenvalues: {eigenvalues[:5]}")
                    print(f"Last 5 eigenvalues: {eigenvalues[-5:]}")

                    # Convert complex eigenvalues to real and ensure numpy arrays
                    real_eigenvalues = np.array(np.real(eigenvalues))
                    weights = np.array(weights)

                    # Debug print after conversion
                    print(f"\nAfter conversion to real:")
                    print(f"First 5 real eigenvalues: {real_eigenvalues[:5]}")
                    print(f"Last 5 real eigenvalues: {real_eigenvalues[-5:]}")

                    # Sort eigenvalues and weights
                    sorted_indices = np.argsort(real_eigenvalues)
                    sorted_eigenvalues = real_eigenvalues[sorted_indices]
                    sorted_weights = weights[sorted_indices]

                    # Debug print after sorting
                    print(f"\nAfter sorting:")
                    print(f"First 5 sorted eigenvalues: {sorted_eigenvalues[:5]}")
                    print(f"Last 5 sorted eigenvalues: {sorted_eigenvalues[-5:]}")

                    # Log top 10 eigenvalues
                    top_k = min(10, len(real_eigenvalues))
                    for i in range(top_k):
                        logger.log(
                            f"hessian_eigenvalue_{i+1}_run_{run_idx}",
                            float(sorted_eigenvalues[i]),
                            epoch,
                        )
                        logger.log(
                            f"hessian_weight_{i+1}_run_{run_idx}",
                            float(sorted_weights[i]),
                            epoch,
                        )

                    # Compute and log condition number metrics
                    max_eigenvalue = float(np.max(real_eigenvalues))
                    min_eigenvalue = float(np.min(real_eigenvalues))

                    # Debug print max/min eigenvalues
                    print(f"\nMax eigenvalue: {max_eigenvalue:.2e}")
                    print(f"Min eigenvalue: {min_eigenvalue:.2e}")

                    # Ensure we're using absolute values for condition number
                    condition_number = abs(max_eigenvalue) / (
                        abs(min_eigenvalue) + 1e-10
                    )

                    # Log summary statistics
                    logger.log(
                        f"hessian_max_eigenvalue_run_{run_idx}", max_eigenvalue, epoch
                    )
                    logger.log(
                        f"hessian_min_eigenvalue_run_{run_idx}", min_eigenvalue, epoch
                    )
                    logger.log(
                        f"hessian_condition_number_run_{run_idx}",
                        condition_number,
                        epoch,
                    )

                    # Log additional condition number metrics
                    logger.log(
                        f"hessian_condition_number_log10_run_{run_idx}",
                        np.log10(condition_number),
                        epoch,
                    )

                    # Log eigenvalue spread metrics
                    logger.log(
                        f"hessian_eigenvalue_range_run_{run_idx}",
                        max_eigenvalue - min_eigenvalue,
                        epoch,
                    )
                    logger.log(
                        f"hessian_eigenvalue_mean_run_{run_idx}",
                        float(np.mean(real_eigenvalues)),
                        epoch,
                    )
                    logger.log(
                        f"hessian_eigenvalue_std_run_{run_idx}",
                        float(np.std(real_eigenvalues)),
                        epoch,
                    )

                    # Print condition number information
                    print(f"\nHessian Condition Number Analysis (Run {run_idx + 1}):")
                    print(f"Max eigenvalue: {max_eigenvalue:.2e}")
                    print(f"Min eigenvalue: {min_eigenvalue:.2e}")
                    print(f"Condition number: {condition_number:.2e}")
                    print(f"Log10 condition number: {np.log10(condition_number):.2f}")

                hessian_time = time() - hessian_start_time
                print(f"Hessian computation took {hessian_time:.2f} seconds")
                logger.log("hessian_computation_time", hessian_time, epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Save checkpoint
                if save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                    )

                # Evaluate solution
                with torch.no_grad():
                    eval_nodes = eval_sampler()
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                # Get losses for history
                eval_weighted_loss, eval_bulk_loss, eval_ic_loss = self.get_pde_loss(
                    model,
                    eval_nodes,
                    ic_nodes,
                    ic_weight,
                    n_square_boundary=n_square_boundary,
                )

                eval_pde_loss = eval_bulk_loss + eval_ic_loss

                logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"Average iteration time: {sum(iter_times[-eval_every:]) / len(iter_times[-eval_every:]):.3f} seconds"
                )
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if scheduler is not None:
                    print(f"Learning rate: {current_lr:.2e}")
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                if scheduler is not None:
                    plt.semilogy(
                        logger.get_iters("learning_rate"),
                        logger.get_values("learning_rate"),
                        label="Learning Rate",
                    )
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

        # Log final timing metrics
        total_time = time() - start_time
        logger.log("total_runtime", total_time, n_epochs - 1)
        logger.log("avg_iter_time", sum(iter_times) / len(iter_times), n_epochs - 1)
        logger.log("min_iter_time", min(iter_times), n_epochs - 1)
        logger.log("max_iter_time", max(iter_times), n_epochs - 1)
        logger.save()

    def train_model_lbfgs(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.LBFGS,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        hessian_every: int = -1,  # Add hessian_every parameter
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        n_square_boundary: int = 0,  # Number of points on square boundary
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since L-BFGS works better with fixed points
        pde_nodes = pde_sampler()
        ic_nodes = ic_sampler()
        eval_nodes = eval_sampler()

        print("Training model with L-BFGS...")
        start_time = time()

        # Define closure for L-BFGS
        def closure():
            optimizer.zero_grad()
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model,
                pde_nodes,
                ic_nodes,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )
            loss.backward()
            return loss

        # Training loop
        for epoch in tqdm(range(n_epochs)):

            # Optimize
            loss = optimizer.step(closure)

            # Update loss weights
            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Save checkpoint
                if save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                    )

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model.interpolate(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    # Evaluate PDE losses
                    total_loss, pde_loss, ic_loss = self.get_pde_loss(
                        model,
                        pde_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model,
                        eval_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

    def train_model_nys_newton(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: NysNewtonCG,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 1e-10,
        hessian_every: int = -1,  # Add hessian_every parameter
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        n_square_boundary: int = 0,  # Number of points on square boundary
        **kwargs,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since Newton methods work better with fixed points
        pde_nodes = pde_sampler()
        ic_nodes = ic_sampler()
        eval_nodes = eval_sampler()

        # Early stopping setup
        best_error = float("inf")
        best_epoch = -1
        best_state_dict = None
        patience_counter = 0

        print("Training model with Nyström Newton-CG...")
        start_time = time()

        # Define closure for NysNewtonCG that returns both loss and gradient
        def closure():
            optimizer.zero_grad()
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model,
                pde_nodes,
                ic_nodes,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )
            return loss

        # Training loop
        for epoch in tqdm(range(n_epochs)):
            # Optimize
            loss = optimizer.step(closure)

            # Update loss weights
            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Save checkpoint
                if save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                    )

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)

                    # Calculate metrics
                    metrics_values = {}
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        metrics_values[eval_metric.__name__] = eval_metric_value
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    # Get losses for history
                    _, pde_loss, ic_loss = self.get_pde_loss(
                        model,
                        pde_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model,
                        eval_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                # Early stopping check (only if enabled)
                if early_stopping:
                    current_error = metrics_values.get(
                        "l2_relative_error", float("inf")
                    )

                    if current_error < best_error - early_stopping_min_delta:
                        # We found a better model
                        best_error = current_error
                        best_epoch = epoch
                        best_state_dict = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }
                        patience_counter = 0
                        print(f"New best model with relative error: {best_error:.3e}")
                    else:
                        # No improvement
                        patience_counter += 1
                        print(
                            f"No improvement for {patience_counter} evaluations. Best error: {best_error:.3e}"
                        )

                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        print(
                            f"Best model was at epoch {best_epoch} with error {best_error:.3e}"
                        )
                        # Restore best model
                        if best_state_dict is not None:
                            model.load_state_dict(best_state_dict)
                        break

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_relative_error"),
                    logger.get_values("eval_l2_relative_error"),
                    label="Eval L2 Relative Error",
                )
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

    def train_model_ssbroyden(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: SSBroyden2,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        *,
        eval_every: int = 100,
        save_dir: str | None = None,
        logger: Logger | None = None,
        hessian_every: int = -1,
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        n_square_boundary: int = 0,
    ):
        """
        Train **only** with the dense Self-Scaled Broyden-II optimiser that lives
        in `ssbroyden2_optimizer.py`.  Mirrors the signature & behaviour of
        `train_model_lbfgs` so it fits seamlessly into existing scripts.
        """

        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since Newton methods work better with fixed points
        pde_nodes = pde_sampler()
        ic_nodes = ic_sampler()
        eval_nodes = eval_sampler()

        start_time = time()

        def closure():
            optimizer.zero_grad()
            loss, *_ = self.get_pde_loss(
                model,
                pde_nodes,
                ic_nodes,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )
            loss.backward()
            return loss
        
        for epoch in tqdm(range(n_epochs)):
            # Optimize
            loss = optimizer.step(closure)

            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

            logger.log("loss", loss.item(), epoch)


            if (epoch + 1) % eval_every == 0:

                # Save checkpoint
                if save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                    )

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)

                    # Calculate metrics
                    metrics_values = {}
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        metrics_values[eval_metric.__name__] = eval_metric_value
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )
                with torch.enable_grad():
                    # Get losses for history
                    _, pde_loss, ic_loss = self.get_pde_loss(
                        model,
                        pde_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model,
                        eval_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_relative_error"),
                    logger.get_values("eval_l2_relative_error"),
                    label="Eval L2 Relative Error",
                )
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

    def train_model_Lssbroyden(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: L_SSBroyden,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        *,
        eval_every: int = 100,
        save_dir: str | None = None,
        logger: Logger | None = None,
        hessian_every: int = -1,
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        n_square_boundary: int = 0,
    ):
        """
        Train with the Limited-memory Self-Scaled Broyden optimizer.
        Mirrors the signature & behaviour of train_model_ssbroyden to fit 
        seamlessly into existing scripts.
        """

        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since Newton methods work better with fixed points
        pde_nodes = pde_sampler()
        ic_nodes = ic_sampler()
        eval_nodes = eval_sampler()

        start_time = time()

        def closure():
            optimizer.zero_grad()
            loss, *_ = self.get_pde_loss(
                model,
                pde_nodes,
                ic_nodes,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )
            loss.backward()
            return loss
        
        for epoch in tqdm(range(n_epochs)):
            # Optimize
            loss = optimizer.step(closure)

            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

            logger.log("loss", loss.item(), epoch)

            if (epoch + 1) % eval_every == 0:

                # Save checkpoint
                if save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                    )

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)

                    # Calculate metrics
                    metrics_values = {}
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        metrics_values[eval_metric.__name__] = eval_metric_value
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )
                with torch.enable_grad():
                    # Get losses for history
                    _, pde_loss, ic_loss = self.get_pde_loss(
                        model,
                        pde_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model,
                        eval_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_relative_error"),
                    logger.get_values("eval_l2_relative_error"),
                    label="Eval L2 Relative Error",
                )
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

    def train_adam_ssbroyden(
        self,
        model: nn.Module,
        n_epochs: int,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        *,
        n_adam_epochs: int = 5000,
        eval_every: int = 100,
        save_dir: str | None = None,
        logger: Logger | None = None,
        lr_schedule: bool = True,
        gradient_clip: float = 1.0,
        hessian_every: int = -1,
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        n_square_boundary: int = 0,
        **kwargs,
    ):
        """
        Train with Adam for n_adam_epochs, then switch to SSBroyden for the remaining epochs.
        This provides a good initialization with Adam before switching to the more aggressive SSBroyden.
        """
        from src.optimizers.ssbroyden import SSBroyden2

        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        print(f"Training model with Adam ({n_adam_epochs} epochs) + SSBroyden ({n_epochs - n_adam_epochs} epochs)...")
        
        # Create optimizers
        optimizer_adam = self.get_optimizer(model, "adam")
        optimizer_ssbroyden = SSBroyden2(
            model.parameters(),
            lr=1.0,
            init_scale=True,
            c1=1e-4,
            c2=0.9,
            max_ls=20,
        )

        # Add learning rate scheduler for Adam if requested
        if lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_adam, T_max=n_adam_epochs, eta_min=1e-6
            )
        else:
            scheduler = None

        start_time = time()
        iter_times = []

        # Sample points once for SSBroyden (will be used after Adam phase)
        pde_nodes_fixed = pde_sampler()
        ic_nodes_fixed = ic_sampler()
        eval_nodes = eval_sampler()

        # Define closure for SSBroyden
        def ssbroyden_closure():
            optimizer_ssbroyden.zero_grad()
            loss, *_ = self.get_pde_loss(
                model,
                pde_nodes_fixed,
                ic_nodes_fixed,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )
            loss.backward()
            return loss

        for epoch in tqdm(range(n_epochs)):
            iter_start_time = time()

            # Phase 1: Adam optimization
            if epoch < n_adam_epochs:
                # Sample fresh points for Adam
                pde_nodes = pde_sampler()
                ic_nodes = ic_sampler()

                # Update loss weights
                self.update_loss_weights(epoch, model, optimizer_adam, pde_nodes, ic_nodes)

                # Train step
                optimizer_adam.zero_grad()

                # Get PDE loss
                loss, pde_loss, ic_loss = self.get_pde_loss(
                    model,
                    pde_nodes,
                    ic_nodes,
                    ic_weight,
                    n_square_boundary=n_square_boundary,
                )

                # Backprop
                loss.backward()

                # Gradient clipping if requested
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=gradient_clip
                    )

                # Update parameters
                optimizer_adam.step()

                # Update learning rate if using scheduler
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    logger.log("learning_rate", current_lr, epoch)

            # Phase 2: SSBroyden optimization
            else:
                # Use fixed points for SSBroyden (better for second-order methods)
                pde_nodes = pde_nodes_fixed
                ic_nodes = ic_nodes_fixed

                # Switch message (only print once)
                if epoch == n_adam_epochs:
                    print(f"Switching to SSBroyden optimization at epoch {epoch}...")

                # Update loss weights
                self.update_loss_weights(epoch, model, optimizer_ssbroyden, pde_nodes, ic_nodes)

                # Optimize with SSBroyden
                loss = optimizer_ssbroyden.step(ssbroyden_closure)

            # Record iteration time
            iter_time = time() - iter_start_time
            iter_times.append(iter_time)
            logger.log("iter_time", iter_time, epoch)
            logger.log(
                "avg_iter_time",
                sum(iter_times[-eval_every:]) / len(iter_times[-eval_every:]),
                epoch,
            )

            # Log loss
            logger.log("loss", loss.item(), epoch)

            # Log which optimizer is being used
            optimizer_name = "adam" if epoch < n_adam_epochs else "ssbroyden"
            logger.log("optimizer", optimizer_name, epoch)

            # Evaluate, print, and plot progress
            if (epoch + 1) % eval_every == 0:
                # Save checkpoint
                if save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                    )

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                # Get losses for history (use current nodes)
                with torch.enable_grad():
                    _, pde_loss, ic_loss = self.get_pde_loss(
                        model,
                        pde_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model,
                        eval_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                optimizer_phase = "Adam" if epoch < n_adam_epochs else "SSBroyden"
                print(f"Epoch {epoch + 1} ({optimizer_phase}) completed in {current_time:.2f} seconds")
                print(
                    f"Average iteration time: {sum(iter_times[-eval_every:]) / len(iter_times[-eval_every:]):.3f} seconds"
                )
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if scheduler is not None and epoch < n_adam_epochs:
                    print(f"Learning rate: {current_lr:.2e}")

                # Plot solution
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                
                # Add vertical line to show transition from Adam to SSBroyden
                if n_adam_epochs < n_epochs:
                    plt.axvline(x=n_adam_epochs, color='red', linestyle='--', alpha=0.7, label='Adam→SSBroyden')
                
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

        # Log final timing metrics
        total_time = time() - start_time
        logger.log("total_runtime", total_time, n_epochs - 1)
        logger.log("avg_iter_time", sum(iter_times) / len(iter_times), n_epochs - 1)
        logger.log("min_iter_time", min(iter_times), n_epochs - 1)
        logger.log("max_iter_time", max(iter_times), n_epochs - 1)
        logger.save()

    # Optimizes with Adam for n_adam_epochs, then optimizes with NysNewtonCG for n_nys_newton_epochs
    def train_model_alternating(
        self,
        model: nn.Module,
        n_epochs: int,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        n_adam_epochs: int = 10000,
        n_nys_newton_epochs: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        hessian_every: int = -1,  # Add hessian_every parameter
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        n_square_boundary: int = 0,  # Number of points on square boundary
        **kwargs,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        print("Training model with alternating optimization...")
        optimizer_adam = self.get_optimizer(model, "adam")
        optimizer_nys_newton = self.get_optimizer(model, "nys_newton")
        start_time = time()

        # Define closure for NysNewtonCG that returns both loss and gradient
        def closure():
            optimizer_nys_newton.zero_grad()
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model,
                pde_nodes,
                ic_nodes,
                ic_weight,
                n_square_boundary=n_square_boundary,
            )
            # Compute gradient with create_graph=True for Hessian computation
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Make gradients contiguous and reshape them
            grads = [g.contiguous() for g in grads]

            # Update preconditioner every iteration
            optimizer_nys_newton.update_preconditioner(grads)

            return loss, grads

        # Training loop
        for epoch in tqdm(range(n_epochs)):

            # Sample points
            pde_nodes = pde_sampler()
            ic_nodes = ic_sampler()

            # Optimize with Adam
            if (epoch % (n_adam_epochs + n_nys_newton_epochs)) < n_adam_epochs:

                # Update loss weights
                self.update_loss_weights(
                    epoch, model, optimizer_adam, pde_nodes, ic_nodes
                )

                # Train step
                optimizer_adam.zero_grad()

                # Get PDE loss
                loss, pde_loss, ic_loss = self.get_pde_loss(
                    model,
                    pde_nodes,
                    ic_nodes,
                    ic_weight,
                    n_square_boundary=n_square_boundary,
                )

                # Backprop
                loss.backward()

                # Update parameters
                optimizer_adam.step()

            # Optimize with NysNewtonCG
            else:

                # Optimize
                loss, _ = optimizer_nys_newton.step(closure)

                # Update loss weights
                self.update_loss_weights(
                    epoch, model, optimizer_nys_newton, pde_nodes, ic_nodes
                )

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Save checkpoint
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                )

                # Evaluate solution
                with torch.no_grad():
                    eval_nodes = eval_sampler()
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    # Get losses for history
                    total_loss, pde_loss, ic_loss = self.get_pde_loss(
                        model,
                        pde_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model,
                        eval_nodes,
                        ic_nodes,
                        ic_weight,
                        n_square_boundary=n_square_boundary,
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                print(
                    f"Evaluation L2 relative error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                if self.__class__.__name__ == "Poisson2DCG":
                    self.plot_solution(
                        model,
                        eval_nodes,
                        u_eval,
                        save_path=(
                            os.path.join(save_dir, f"{self.name}_solution_{epoch}.png")
                            if save_dir is not None
                            else None
                        ),
                    )
                else:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(
                            save_dir, f"{self.name}_solution_{epoch}.png"
                        ),
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.semilogy(
                    logger.get_iters("train_pde_loss"),
                    logger.get_values("train_pde_loss"),
                    label="Train PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("train_ic_loss"),
                    logger.get_values("train_ic_loss"),
                    label="Train IC Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_pde_loss"),
                    logger.get_values("eval_pde_loss"),
                    label="Eval PDE Loss",
                )
                plt.semilogy(
                    logger.get_iters("eval_l2_error"),
                    logger.get_values("eval_l2_error"),
                    label="Eval L2 Error",
                )
                plt.semilogy(
                    logger.get_iters("eval_max_error"),
                    logger.get_values("eval_max_error"),
                    label="Eval Max Error",
                )
                plt.legend()
                plt.savefig(os.path.join(save_dir, "loss_history.png"))
                plt.close()

    def train(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        hessian_every: int = -1,  # Add hessian_every parameter
        hessian_num_iter: int = 100,
        hessian_num_run: int = 1,
        **kwargs,
    ):
        if isinstance(optimizer, NysNewtonCG):
            self.train_model_nys_newton(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every,
                save_dir,
                logger,
                hessian_every=hessian_every,
                hessian_num_iter=hessian_num_iter,
                hessian_num_run=hessian_num_run,
                **kwargs,
            )
        elif isinstance(optimizer, torch.optim.LBFGS):
            self.train_model_lbfgs(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every,
                save_dir,
                logger,
                hessian_every=hessian_every,
                hessian_num_iter=hessian_num_iter,
                hessian_num_run=hessian_num_run,
                n_square_boundary=kwargs.get(
                    "n_square_boundary", 0
                ),  # Pass through n_square_boundary
            )
        elif isinstance(optimizer, SSBroyden2):
            self.train_model_ssbroyden(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every=eval_every,
                save_dir=save_dir,
                logger=logger,
                hessian_every=hessian_every,
                hessian_num_iter=hessian_num_iter,
                hessian_num_run=hessian_num_run,
                n_square_boundary=kwargs.get(
                    "n_square_boundary", 0
                ),  # Pass through n_square_boundary
            )
        elif isinstance(optimizer, L_SSBroyden):
            self.train_model_Lssbroyden(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every=eval_every,
                save_dir=save_dir,
                logger=logger,
                hessian_every=hessian_every,
                hessian_num_iter=hessian_num_iter,
                hessian_num_run=hessian_num_run,
                n_square_boundary=kwargs.get(
                    "n_square_boundary", 0
                ),  # Pass through n_square_boundary
            )
        else:
            self.train_model(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every,
                save_dir,
                logger,
                hessian_every=hessian_every,
                hessian_num_iter=hessian_num_iter,
                hessian_num_run=hessian_num_run,
                n_square_boundary=kwargs.get(
                    "n_square_boundary", 0
                ),  # Pass through n_square_boundary
            )
