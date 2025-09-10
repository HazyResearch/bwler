#!/usr/bin/env python3
"""
Script to run derivative sweep for a fixed MLP configuration using the new SineTarget.
Sweeps over different numbers of 1st order training points and analyzes weight norms.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import sys
import re

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.experiments.interpolation.config import ExperimentConfig, ModelConfig
from src.experiments.interpolation.targets import create_target
from src.experiments.interpolation.models import create_mlp, run_mlp_experiment
from src.utils.metrics import l2_relative_error, l2_error, max_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MLP derivative sweep for fixed configuration"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="sine",
        choices=["sine"],
        help="Target function type",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_train_0th",
        type=int,
        default=100,
        help="Number of 0th order training points",
    )
    parser.add_argument(
        "--n_train_1st",
        type=int,
        default=100,
        help="Number of 1st order training points",
    )
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument("--n_layers", type=int, required=True, help="Number of layers")
    parser.add_argument(
        "--hidden_dim", type=int, required=True, help="Hidden dimension"
    )
    parser.add_argument(
        "--deriv_alpha", type=float, default=1.0, help="Weight on value loss"
    )
    parser.add_argument(
        "--deriv_beta", type=float, default=1.0, help="Weight on derivative loss"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "ssbroyden"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=2.0,
        help="Frequency multiplier for sine target (i.e., sin(kx))",
    )
    parser.add_argument(
        "--svd_plot_every",
        type=int,
        default=2,
        help="Plot every k-th evaluation in SVD evolution plots to reduce clutter",
    )
    return parser.parse_args()


def compute_weight_norms(model):
    """Compute L2 norms of all weights and biases in the MLP."""
    norms = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Compute L2 norm
            norm = torch.norm(param.data, p=2).item()
            norms[name] = norm

    return norms


def plot_weight_norms(norms_dict, save_path):
    """Create bar chart of weight norms with log y-axis."""
    plt.figure(figsize=(12, 6))

    names = list(norms_dict.keys())
    norms = list(norms_dict.values())

    # Create bar plot
    bars = plt.bar(range(len(names)), norms)
    plt.yscale("log")

    # Customize plot
    plt.xlabel("Layer Parameters")
    plt.ylabel("L2 Norm")
    plt.title("MLP Weight and Bias Norms")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, norm) in enumerate(zip(bars, norms)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{norm:.2e}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def compute_svds(model):
    """Compute singular values of weight matrices in the MLP."""
    svd_dict = {}

    for name, param in model.named_parameters():
        if "weight" in name:
            # Compute SVD
            weight_matrix = param.data.cpu().numpy()
            u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
            svd_dict[name] = s  # Store singular values

    return svd_dict

def _sanitize_filename(s: str) -> str:
    """Make a safe filename chunk from parameter name."""
    # Replace anything not alnum, dash, underscore, or dot with underscore
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def plot_svd_norms(svd_dict, save_dir, run_tag=None, use_logy=True, json_file_path=None, every_k_eval=1):
    """
    For each weight matrix in svd_dict, create a figure that plots its
    singular values in descending order as a line plot, and save it.
    
    If json_file_path is provided, also plot the evolution of singular values
    over training iterations from the JSON log.

    Args:
        svd_dict: dict[str, np.ndarray] mapping parameter name -> singular values (final state)
        save_dir: directory to save PNGs
        run_tag: optional string to include in filename (e.g., 'n1st_32')
        use_logy: whether to use log scale for y-axis
        json_file_path: optional path to JSON file with training history
        every_k_eval: only plot every k-th evaluation from training history to reduce clutter
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load training history if provided
    training_data = None
    if json_file_path and os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                training_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load training data from {json_file_path}: {e}")
    
    for name, s in svd_dict.items():
        # Ensure descending order explicitly (should already be)
        s_sorted = np.sort(s)[::-1]
        plt.figure(figsize=(10, 6))
        
        # Plot training evolution if available
        if training_data:
            weight_key = f"weight_compute_svds_{name}"
            if weight_key in training_data:
                iterations = training_data[weight_key]['iter']
                values_history = training_data[weight_key]['value']  # Note: 'value' not 'vals'
                
                # Filter every k evaluations
                filtered_iterations = iterations[::every_k_eval]
                filtered_values = values_history[::every_k_eval]
                
                # Plot each iteration's singular values
                for i, (iteration, svd_values) in enumerate(zip(filtered_iterations, filtered_values)):
                    svd_sorted = np.sort(svd_values)[::-1]
                    alpha = 0.5  # Fade in over time

                    if iteration == -1:
                        label = "Initial (epoch -1)"
                        color = 'gray'
                        linewidth = 2
                    else:
                        label = f"Epoch {iteration}"
                        color = plt.cm.viridis(i / max(1, len(filtered_iterations) - 1))
                        linewidth = 1.5
                    
                    print(f"Debug: Plotting iteration {iteration} with {len(svd_sorted)} singular values")
                    plt.plot(np.arange(1, len(svd_sorted) + 1), svd_sorted, 
                            color=color, alpha=alpha, linewidth=linewidth, 
                            marker='o' if len(svd_sorted) <= 10 else None, 
                            markersize=3, label=label)
            else:
                print(f"Debug: Key '{weight_key}' NOT found in training data")
        
        # Plot final state (from svd_dict) with emphasis
        plt.plot(np.arange(1, len(s_sorted) + 1), s_sorted, 
                color='red', linewidth=2, marker='o', markersize=3, 
                label="Final state", alpha=0.9)
        
        if use_logy:
            plt.yscale("log")
            ylabel = "Singular value (log)"
        else:
            ylabel = "Singular value"
            
        plt.xlabel("Index (descending order)")
        plt.ylabel(ylabel)
        plt.title(f"SVD Evolution: {name}")
        plt.grid(True, alpha=0.3)
        
        # Handle legend - only show if we have training data
        if training_data and f"weight_compute_svds_{name}" in training_data:
            # Limit legend entries to avoid clutter
            handles, labels = plt.gca().get_legend_handles_labels()
            if len(handles) > 10:
                # Show initial, final, and a few intermediate points
                indices_to_show = [0] + list(range(1, len(handles)-1, max(1, (len(handles)-2)//5))) + [len(handles)-1]
                handles = [handles[i] for i in indices_to_show]
                labels = [labels[i] for i in indices_to_show]
            plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend()
            
        plt.tight_layout()

        tag = f"_{run_tag}" if run_tag else ""
        fname = f"svd_{_sanitize_filename(name)}{tag}.png"
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"SVD evolution plot saved to: {out_path}")


def plot_deriv_sweep_results(results, save_path):
    """Plot L2RE vs n_train_1st for the derivative sweep."""
    plt.figure(figsize=(10, 6))

    n_train_1st_list = [r["n_train_1st"] for r in results]
    l2re_values = [r["l2re_test"] for r in results]

    # Plot with log scale for both x-axis and y-axis
    plt.loglog(n_train_1st_list, l2re_values, "o-", linewidth=2, markersize=8)

    # Customize plot
    plt.xlabel("Number of 1st Order Training Points")
    plt.ylabel("L2 Relative Error")
    plt.title("L2RE vs Derivative Supervision Points")
    plt.grid(True, alpha=0.3)

    # Add horizontal line at n_train_1st=0 for reference
    if 0 in n_train_1st_list:
        zero_idx = n_train_1st_list.index(0)
        plt.axhline(
            y=l2re_values[zero_idx],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"No derivative supervision (L2RE={l2re_values[zero_idx]:.2e})",
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sine_target_solution(target, model, save_path, title_suffix=""):
    """Custom plotting function for SineTarget showing 0th and 1st order training points."""
    plt.figure(figsize=(15, 10))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training data fit (0th order points)
    axes[0, 0].scatter(
        target.train_points_0th.cpu(),
        target.train_values_0th.cpu(),
        c="red",
        s=30,
        alpha=0.8,
        label="0th Order Points (Value)",
    )
    with torch.no_grad():
        u_pred_train_0th = model([target.train_points_0th])
    axes[0, 0].scatter(
        target.train_points_0th.cpu(),
        u_pred_train_0th.cpu(),
        c="blue",
        s=20,
        alpha=0.6,
        label="MLP Predictions",
    )
    axes[0, 0].set_title(f"Training Data Fit - 0th Order{title_suffix}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("u(x)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Test data fit
    axes[0, 1].plot(
        target.test_points.cpu(),
        target.test_values.cpu(),
        "r-",
        linewidth=2,
        label="True Function",
    )
    with torch.no_grad():
        u_pred_test = model([target.test_points])
    axes[0, 1].plot(
        target.test_points.cpu(),
        u_pred_test.cpu(),
        "b--",
        linewidth=2,
        label="MLP Predictions",
    )
    axes[0, 1].set_title(f"Test Data Fit{title_suffix}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("u(x)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Error analysis
    error = u_pred_test - target.test_values
    axes[1, 0].plot(target.test_points.cpu(), error.cpu(), "g-", linewidth=1)
    axes[1, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1, 0].set_title(f"Prediction Error{title_suffix}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Derivative comparison (1st order points)
    if target.n_train_1st > 0:
        axes[1, 1].scatter(
            target.train_points_1st.cpu(),
            target.get_derivative([target.train_points_1st]).cpu(),
            c="orange",
            s=30,
            alpha=0.8,
            label="1st Order Points (Derivatives)",
        )
        x_deriv = target.train_points_1st.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            u_pred_deriv = model([x_deriv])
            du_pred = torch.autograd.grad(
                u_pred_deriv.sum(), x_deriv, create_graph=True
            )[0]
        axes[1, 1].scatter(
            target.train_points_1st.cpu(),
            du_pred.detach().cpu(),
            c="purple",
            s=20,
            alpha=0.6,
            label="MLP Derivatives",
        )
        axes[1, 1].set_title(f"Derivative Comparison - 1st Order{title_suffix}")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("du/dx")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # No derivative supervision
        axes[1, 1].text(
            0.5,
            0.5,
            "No Derivative\nSupervision",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        axes[1, 1].set_title(f"Derivative Comparison{title_suffix}")
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # 1st order training points to sweep (exponential spacing)
    n_train_1st_list = [0]  # Only train with 0 derivative points

    # Store results
    results = []

    print(
        f"Running derivative sweep for MLP: {args.n_layers} layers, {args.hidden_dim} hidden dim"
    )
    print(f"Target: {args.target}, Device: {args.device}, Optimizer: {args.optimizer}")
    print(f"Fixed 0th order points: {args.n_train_0th}")
    print(f"Sweeping over 1st order points: {n_train_1st_list}")

    for n_train_1st in n_train_1st_list:
        print(f"\n--- Testing n_train_1st = {n_train_1st} ---")

        # Create configuration for this run
        config = ExperimentConfig(
            n_train=args.n_train_0th,  # Will be overridden by kwargs
            n_test=args.n_test,
            n_epochs=args.n_epochs,
            eval_every=args.eval_every,
            device=args.device,
            seed=args.seed,
            model_type="mlp",
            target_type=args.target,
            hidden_dims=[args.hidden_dim],
            n_layers=[args.n_layers],
            use_derivatives=(n_train_1st > 0),
            deriv_alpha=args.deriv_alpha,
            deriv_beta=args.deriv_beta,
            save_dir=args.save_dir,
        )

        # Create target with current 1st order point count
        target = create_target(
            config,
            n_train_0th=args.n_train_0th,
            n_train_1st=n_train_1st,
            k=args.k,
        )

        # Create model
        model_config = ModelConfig(
            hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device
        )

        # Train and evaluate using our new infrastructure
        print(f"Training model...")
        run_results = run_mlp_experiment(
            target=target,
            config=model_config,
            n_epochs=args.n_epochs,
            eval_every=args.eval_every,
            exp_dir=os.path.join(args.save_dir, f"n1st_{n_train_1st}"),
            optimizer_name=args.optimizer,
            weight_evals=[compute_svds],
        )

        # Extract results
        rmse_train = run_results["rmse_train"]
        l_inf_train = run_results["l_inf_train"]
        rmse_test = run_results["rmse_test"]
        l_inf_test = run_results["l_inf_test"]
        u_pred_train = run_results["u_pred_train"]
        u_pred_test = run_results["u_pred_test"]
        model = run_results["model"]

        # Compute L2RE
        l2re_test = l2_relative_error(u_pred_test, target.test_values)

        # Compute weight norms
        weight_norms = compute_weight_norms(model)
        svd_dict = compute_svds(model)

        # Store results
        result = {
            "n_train_1st": n_train_1st,
            "rmse_train": rmse_train,
            "l_inf_train": l_inf_train,
            "rmse_test": rmse_test,
            "l_inf_test": l_inf_test,
            "l2re_test": l2re_test,
            "weight_norms": weight_norms,
            "singular_values": svd_dict,
        }
        results.append(result)

        print(f"Results: RMSE_test={rmse_test:.2e}, L2RE_test={l2re_test:.2e}")

        # Save weight norms plot for this run
        plot_weight_norms(
            weight_norms,
            os.path.join(args.save_dir, f"weight_norms_n1st_{n_train_1st}.png"),
        )
        # NEW: plot SVDs per layer for this run
        json_log_path = os.path.join(
            args.save_dir, f"n1st_{n_train_1st}", 
            f"mlp_hdim{args.hidden_dim}_layers{args.n_layers}.json"
        )
        
        plot_svd_norms(
            svd_dict,
            save_dir=args.save_dir,
            run_tag=f"n1st_{n_train_1st}",
            use_logy=True,
            json_file_path=json_log_path,
            every_k_eval=args.svd_plot_every,
        )

        # Save custom solution plot for this run
        title_suffix = f" (n_1st={n_train_1st})"
        plot_sine_target_solution(
            target,
            model,
            os.path.join(args.save_dir, f"solution_n1st_{n_train_1st}.png"),
            title_suffix,
        )

        # Plot training curves (loss and eval metrics) from JSON log
        json_log_path = os.path.join(
            args.save_dir, f"n1st_{n_train_1st}", 
            f"mlp_hdim{args.hidden_dim}_layers{args.n_layers}",
            f"mlp_hdim{args.hidden_dim}_layers{args.n_layers}.json"
        )

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f"model_n1st_{n_train_1st}.pt"),
        )

    # Save all results
    with open(os.path.join(args.save_dir, "deriv_sweep_results.json"), "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for r in results:
            json_r = r.copy()
            json_r["weight_norms"] = {k: float(v) for k, v in r["weight_norms"].items()}
            # Convert singular values numpy arrays to lists
            json_r["singular_values"] = {k: v.tolist() for k, v in r["singular_values"].items()}
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    # Create derivative sweep plot
    plot_deriv_sweep_results(
        results, os.path.join(args.save_dir, "deriv_sweep_l2re.png")
    )

    # Create summary table
    print("\n" + "=" * 80)
    print("DERIVATIVE SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'n_train_1st':<12} {'RMSE_test':<12} {'L2RE_test':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r['n_train_1st']:<12} {r['rmse_test']:<12.2e} {r['l2re_test']:<12.2e}")

    print(f"\nResults saved to: {args.save_dir}")
    print("Files created:")
    print(f"  - deriv_sweep_results.json: All numerical results")
    print(f"  - deriv_sweep_l2re.png: L2RE vs n_train_1st plot")
    print(f"  - weight_norms_n1st_*.png: Weight norm plots for each run")
    print(f"  - solution_n1st_*.png: Custom solution plots for each run")
    print(f"  - training_curves_n1st_*.png: Loss and L2RE training curves for each run")
    print(f"  - svd_*_n1st_*.png: Singular value plots for each layer and run")
    print(f"  - model_n1st_*.pt: Trained model states")
    print(f"  - n1st_*/mlp_hdim*_layers*.json: Training logs with SVD data for each run")


if __name__ == "__main__":
    main()
