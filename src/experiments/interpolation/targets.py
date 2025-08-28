"""
Target function definitions for 1D interpolation experiments.
"""
import torch
import numpy as np
from typing import List, Callable
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm

from ..base_fcn import BaseFcn
from src.loggers.logger import Logger


class SineTarget(BaseFcn):
    """Sine function target with configurable frequency and built-in derivative supervision."""
    
    def __init__(
        self,
        n_train_0th: int,  # number of points for value loss
        n_train_1st: int,  # number of points for derivative loss
        n_test: int = 1000,
        domain: List[tuple] = [(-1, 1)],
        device: str = "cpu",
        sampling: str = "uniform",
        seed: int = None,
        k: int = 1,  # sin(kx)
        alpha: float = 1.0,  # weight for value loss
        beta: float = 1.0,   # weight for derivative loss
    ):
        # Initialize BaseFcn with basic parameters
        super().__init__(
            name="sine",
            domain=domain,
            device=device,
        )
        
        # Store the function and derivatives as instance attributes
        self.f = lambda x: torch.sin(self.k * x)
        self.derivative = lambda x: self.k * torch.cos(self.k * x)
        self.second_derivative = lambda x: -(self.k**2) * torch.sin(self.k * x)
        
        self.n_train_0th = n_train_0th
        self.n_train_1st = n_train_1st
        self.n_test = n_test
        self.sampling = sampling
        self.seed = seed
        self.k = k
        self.alpha = alpha
        self.beta = beta
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self._generate_points()
        self._compute_values()

    def _generate_points(self):
        """Generate separate training points for 0th and 1st order, plus test points."""
        a, b = self.domain[0]
        
        # Generate 0th order training points (for value loss)
        if self.sampling == "cheb":
            i = torch.arange(self.n_train_0th, device=self.device, dtype=torch.float64)
            cheb_nodes = torch.cos(np.pi * i / (self.n_train_0th - 1))
            train_points_0th = 0.5 * (cheb_nodes * (b - a) + (a + b))
        else:
            # Generate n_train_0th-2 interior points
            interior_points = torch.rand(
                self.n_train_0th-2, device=self.device
            ) * (b - a) + a
            # Add boundary points
            train_points_0th = torch.cat([
                torch.tensor([a], device=self.device),  
                interior_points,                               
                torch.tensor([b], device=self.device)   
            ])
        
        # Generate 1st order training points (for derivative loss) - can be different locations
        if self.sampling == "cheb":
            i = torch.arange(self.n_train_1st, device=self.device, dtype=torch.float64)
            cheb_nodes = torch.cos(np.pi * i / (self.n_train_1st - 1))
            train_points_1st = 0.5 * (cheb_nodes * (b - a) + (a + b))
        else:
            # Generate n_train_1st-2 interior points
            interior_points = torch.rand(
                self.n_train_1st-2, device=self.device
            ) * (b - a) + a
            # Add boundary points
            train_points_1st = torch.cat([
                torch.tensor([a], device=self.device),  
                interior_points,                               
                torch.tensor([b], device=self.device)   
            ])
            
        # Store both sets of training points
        self.train_points_0th = train_points_0th
        self.train_points_1st = train_points_1st
        
        # For backward compatibility, also store the 0th order points as train_points
        self.train_points = train_points_0th
        self.n_train = self.n_train_0th  # for backward compatibility
            
        # Test points are always equispaced
        self.test_points = torch.linspace(a, b, self.n_test, device=self.device)

    def _compute_values(self):
        """Compute function values at training and test points."""
        self.train_values_0th = self.f(self.train_points_0th)
        self.train_values_1st = self.f(self.train_points_1st)
        self.test_values = self.f(self.test_points)
        
        # For backward compatibility
        self.train_values = self.train_values_0th

    def get_function(self, nodes):
        """Get function values at given nodes."""
        if isinstance(nodes, list):
            x = nodes[0]
        else:
            x = nodes
        return self.f(x)

    def get_derivative(self, nodes):
        """Get derivative values at given nodes."""
        if isinstance(nodes, list):
            x = nodes[0]
        else:
            x = nodes
        return self.derivative(x)

    def get_solution(self, nodes):
        """Get solution values at given nodes (alias for get_function for BaseFcn compatibility)."""
        return self.get_function(nodes)

    def get_loss(self, model: torch.nn.Module, nodes: List[torch.Tensor]) -> torch.Tensor:
        """Compute combined value + derivative loss using separate point sets.
        
        Loss = alpha * ||u_pred(x_0th) - f(x_0th)||^2 + beta * ||du_pred/dx(x_1st) - f'(x_1st)||^2
        """
        # Expect 1D input: nodes = [x]
        x = nodes[0]
        
        # Value loss (0th order) - use 0th order training points
        u_pred_0th = model([self.train_points_0th])
        u_true_0th = self.train_values_0th
        value_loss = torch.mean((u_pred_0th - u_true_0th) ** 2)
        
        total_loss = self.alpha * value_loss
        
        # Derivative loss (1st order) - use 1st order training points
        if self.beta > 0:
            # Ensure input requires grad for derivative computation
            x_deriv = self.train_points_1st.clone().detach().requires_grad_(True)
            
            # Compute MLP output and its derivative
            u_pred_deriv = model([x_deriv])
            du_pred = torch.autograd.grad(
                u_pred_deriv.sum(), x_deriv, create_graph=True
            )[0]
            
            # Compute true derivative
            du_true = self.get_derivative([x_deriv])
            
            # Derivative loss
            derivative_loss = torch.mean((du_pred - du_true) ** 2)
            total_loss += self.beta * derivative_loss
            
            # Debug output (only once per training session)
            if not hasattr(self, '_debug_deriv_printed'):
                self._debug_deriv_printed = True
                print(f"[DERIV DEBUG] Using {self.n_train_0th} 0th order points and {self.n_train_1st} 1st order points")
                print(f"[DERIV DEBUG] Value loss: {value_loss.item():.6f}, Deriv loss: {derivative_loss.item():.6f}")
                print(f"[DERIV DEBUG] Total loss: {total_loss.item():.6f}")
        
        return total_loss

    def plot_solution(self, nodes, u, save_path=None):
        self._plot_solution_default(nodes, u, save_path=save_path)

    def train_model(
        self,
        model: torch.nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        lr_schedule: bool = True,
        gradient_clip: float = 1.0,
    ):
        """Train model using standard optimizers (Adam, SGD, etc.)."""
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
        iter_times = []

        for epoch in tqdm(range(n_epochs)):
            iter_start_time = time()

            # Sample training points
            train_nodes = train_sampler()

            # Training step
            optimizer.zero_grad()
            loss = self.get_loss(model, train_nodes)
            loss.backward()

            # Gradient clipping if requested
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

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

            # Log loss
            logger.log("loss", loss.item(), epoch)

            # Periodic evaluation
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
                    u_true = self.get_function(eval_nodes)
                    
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        # Ensure the metric value is a scalar for JSON serialization
                        if hasattr(eval_metric_value, 'item'):
                            eval_metric_value = eval_metric_value.item()
                        logger.log(f"eval_{eval_metric.__name__}", eval_metric_value, epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"Average iteration time: {sum(iter_times[-eval_every:]) / len(iter_times[-eval_every:]):.3f} seconds")
                print(f"Loss: {logger.get_most_recent_value('loss'):1.3e}")
                
                # Plot solution
                if save_dir is not None:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(save_dir, f"solution_{epoch}.png")
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
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

                # Plot L2RE history if available
                try:
                    if "eval_l2_relative_error" in logger.data and len(logger.get_values("eval_l2_relative_error")) > 0:
                        plt.figure()
                        plt.semilogy(
                            logger.get_iters("eval_l2_relative_error"), 
                            logger.get_values("eval_l2_relative_error"), 
                            label="L2 Relative Error"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("L2 Relative Error")
                        plt.title("L2 Relative Error History")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        if save_dir is not None:
                            plt.savefig(os.path.join(save_dir, "l2re_history.png"))
                        else:
                            plt.show()
                        plt.close()
                except (KeyError, AttributeError):
                    # Metric not available, skip plotting
                    pass

        # Log final timing metrics
        total_time = time() - start_time
        logger.log("total_runtime", total_time, n_epochs - 1)
        logger.log("avg_iter_time", sum(iter_times) / len(iter_times), n_epochs - 1)
        logger.log("min_iter_time", min(iter_times), n_epochs - 1)
        logger.log("max_iter_time", max(iter_times), n_epochs - 1)
        logger.save()

    def train_model_ssbroyden(
        self,
        model: torch.nn.Module,
        n_epochs: int,
        optimizer,  # SSBroyden optimizer
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
    ):
        """Train model using SSBroyden optimizer."""
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since SSBroyden works better with fixed points
        train_nodes = train_sampler()
        eval_nodes = eval_sampler()

        print("Training model with SSBroyden...")
        start_time = time()

        # Define closure for SSBroyden
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(model, train_nodes)
            loss.backward()
            return loss

        # Training loop
        for epoch in tqdm(range(n_epochs)):
            # Optimize
            try:
                loss = optimizer.step(closure)
            except RuntimeError as e:
                if "rho_k_minus is NaN" in str(e):
                    print(f"SSBroyden terminated due to NaN in rho_k_minus at epoch {epoch + 1}")
                    print("Running final evaluation before ending training...")
                    
                    # Run final evaluation
                    with torch.no_grad():
                        u_eval = model(eval_nodes)
                        u_true = self.get_function(eval_nodes)

                        for eval_metric in eval_metrics:
                            eval_metric_value = eval_metric(u_eval, u_true)
                            # Ensure the metric value is a scalar for JSON serialization
                            if hasattr(eval_metric_value, 'item'):
                                eval_metric_value = eval_metric_value.item()
                            logger.log(f"eval_{eval_metric.__name__}", eval_metric_value, epoch)

                    current_time = time() - start_time
                    print(f"Final evaluation at epoch {epoch + 1} (terminated early)")
                    
                    # Save final checkpoint
                    if save_dir is not None:
                        torch.save(
                            model.state_dict(),
                            os.path.join(save_dir, f"checkpoint_final_early_termination.pth"),
                        )
                        
                        # Plot final solution
                        self.plot_solution(
                            eval_nodes,
                            u_eval,
                            save_path=os.path.join(save_dir, f"solution_final_early_termination.png"),
                        )
                    
                    # Save history and exit
                    logger.save()
                    print("Training terminated early due to SSBroyden NaN issue")
                    return
                else:
                    # Re-raise other RuntimeErrors
                    raise e

            # Log loss
            logger.log("loss", loss.item(), epoch)

            # Periodic evaluation
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
                    u_true = self.get_function(eval_nodes)
                    
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        # Ensure the metric value is a scalar for JSON serialization
                        if hasattr(eval_metric_value, 'item'):
                            eval_metric_value = eval_metric_value.item()
                        logger.log(f"eval_{eval_metric.__name__}", eval_metric_value, epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"Loss: {logger.get_most_recent_value('loss'):1.3e}")

                # Plot solution
                if save_dir is not None:
                    self.plot_solution(
                        eval_nodes,
                        u_eval,
                        save_path=os.path.join(save_dir, f"solution_{epoch}.png")
                    )

                # Save history
                logger.save()

                # Plot loss history
                plt.figure()
                plt.semilogy(
                    logger.get_iters("loss"), logger.get_values("loss"), label="Loss"
                )
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, "loss_history.png"))
                else:
                    plt.show()
                plt.close()

                # Plot L2RE history if available
                try:
                    if "eval_l2_relative_error" in logger.data and len(logger.get_values("eval_l2_relative_error")) > 0:
                        plt.figure()
                        plt.semilogy(
                            logger.get_iters("eval_l2_relative_error"), 
                            logger.get_values("eval_l2_relative_error"), 
                            label="L2 Relative Error"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("L2 Relative Error")
                        plt.title("L2 Relative Error History")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        if save_dir is not None:
                            plt.savefig(os.path.join(save_dir, "l2re_history.png"))
                        else:
                            plt.show()
                        plt.close()
                except (KeyError, AttributeError):
                    # Metric not available, skip plotting
                    pass

    def train(
        self,
        model: torch.nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        lr_schedule: bool = True,
        gradient_clip: float = 1.0,
        **kwargs,
    ):
        """Unified training method that routes to appropriate training implementation."""
        # Check if it's SSBroyden optimizer
        optimizer_name = optimizer.__class__.__name__
        if "SSBroyden" in optimizer_name or "SSbroyden" in optimizer_name:
            print(f"Using SSBroyden training for {optimizer_name}")
            self.train_model_ssbroyden(
                model, n_epochs, optimizer, train_sampler, eval_sampler,
                eval_metrics, eval_every, save_dir, logger
            )
        else:
            print(f"Using standard training for {optimizer_name}")
            self.train_model(
                model, n_epochs, optimizer, train_sampler, eval_sampler,
                eval_metrics, eval_every, save_dir, logger, lr_schedule, gradient_clip
            )


def create_target(config, **kwargs):
    """Factory function to create target based on configuration."""
    if config.target_type == "sine":
        # Extract 0th and 1st order training point counts
        n_train_0th = kwargs.get('n_train_0th', config.n_train)
        n_train_1st = kwargs.get('n_train_1st', config.n_train)
        
        return SineTarget(
            n_train_0th=n_train_0th,
            n_train_1st=n_train_1st,
            n_test=config.n_test,
            domain=[(-1, 1)],
            device=config.device,
            sampling=config.sampling,
            seed=config.seed,
            k=kwargs.get('k', 1),
            alpha=kwargs.get('deriv_alpha', 1.0),
            beta=kwargs.get('deriv_beta', 1.0)
        )
    else:
        raise ValueError(f"Unknown target_type: {config.target_type}")
