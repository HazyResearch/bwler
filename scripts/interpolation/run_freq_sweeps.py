#!/usr/bin/env python3
"""
Comprehensive frequency analysis during training for understanding model learning dynamics.

Implements:
1. Function Approximation Decomposition - tracks how model builds up target function
4. Frequency-Specific Learning Rates - analyzes convergence per Fourier mode
5. Weight Specialization Analysis - correlates weights with frequency content
8. Information Flow Analysis - tracks gradients in frequency domain

This script follows the same structure as run_mlp_k_sweep.py but adds comprehensive
frequency analysis during training.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import sys
import re
from scipy.interpolate import interp1d

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.experiments.interpolation.config import ExperimentConfig, ModelConfig
from src.experiments.interpolation.targets import create_target
from src.experiments.interpolation.models import create_mlp, run_mlp_experiment
from src.utils.metrics import l2_relative_error, l2_error, max_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MLP wavenumber sweep with comprehensive frequency analysis"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="sine",
        choices=["sine", "two_harmonic"],
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
    parser.add_argument("--n_epochs", type=int, default=1500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument(
        "--custom_eval_schedule",
        type=str,
        default=None,
        help="Custom evaluation schedule as 'freq1:end1,freq2:end2,freq3:end3'. Example: '25:1000,100:1500,1000:500000'"
    )
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
        choices=["adam", "sgd", "ssbroyden"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1.0,2.0,4.0,8.0,16.0,32.0,64.0",
        help="Comma-separated list of wavenumbers to sweep over",
    )
    parser.add_argument(
        "--svd_plot_every",
        type=int,
        default=10,
        help="Plot every k-th evaluation in SVD evolution plots to reduce clutter (higher values = fewer curves)",
    )
    # Embedding arguments
    parser.add_argument(
        "--embedding",
        type=str,
        default="none",
        choices=["none", "theta", "cheb", "bary"],
        help="Embedding to use: 'none', 'theta', 'cheb', or 'bary'"
    )
    parser.add_argument(
        "--embedding_M",
        type=int,
        default=None,
        help="Feature count for 'cheb' or degree for 'bary' embedding"
    )
    parser.add_argument(
        "--num_bands",
        type=int,
        default=1,
        help="Number of frequency bands: 1=sin(kx), 2=sin(x)+sin(kx), 3=sin(x)+sin(2kx)+sin(kx), etc."
    )
    return parser.parse_args()


def parse_custom_eval_schedule(schedule_str, n_epochs):
    """
    Parse custom evaluation schedule string and return list of evaluation epochs.

    Args:
        schedule_str: String like '25:1000,100:1500,1000:500000'
        n_epochs: Total number of epochs

    Returns:
        List of epoch numbers when to evaluate
    """
    if not schedule_str:
        return None

    eval_epochs = []
    current_epoch = 0

    segments = schedule_str.split(',')
    for segment in segments:
        freq_str, end_str = segment.split(':')
        freq = int(freq_str)
        end_epoch = min(int(end_str), n_epochs)  # Don't go beyond n_epochs

        # Add evaluation points for this segment
        while current_epoch + freq <= end_epoch:
            current_epoch += freq
            eval_epochs.append(current_epoch)

    # Always include the final epoch if not already included
    if eval_epochs[-1] != n_epochs:
        eval_epochs.append(n_epochs)

    return eval_epochs


def filter_json_logs_to_schedule(json_file_path, target_epochs):
    """
    Filter JSON training logs to only include evaluations at target epochs.

    Args:
        json_file_path: Path to the JSON log file
        target_epochs: List of epoch numbers to keep
    """
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load JSON file {json_file_path}: {e}")
        return

    target_epochs_set = set(target_epochs)

    # Filter each logged metric
    for key in data:
        if 'iter' in data[key] and 'value' in data[key]:
            original_iters = data[key]['iter']
            original_values = data[key]['value']

            # Find indices of epochs we want to keep
            keep_indices = [i for i, epoch in enumerate(original_iters)
                           if epoch in target_epochs_set or epoch == -1]  # Always keep epoch -1 (initial)

            # Filter the data
            data[key]['iter'] = [original_iters[i] for i in keep_indices]
            data[key]['value'] = [original_values[i] for i in keep_indices]

    # Write back the filtered data
    try:
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Filtered JSON logs to {len(target_epochs)} custom evaluation points")
    except Exception as e:
        print(f"Warning: Could not write filtered JSON file {json_file_path}: {e}")


# Frequency analysis functions as weight evaluators
def function_decomposition_analyzer(model, u_eval=None, u_true=None, **kwargs):
    """
    Analysis 1: Function Approximation Decomposition
    Track how model builds up sine function over training
    """
    if u_eval is None or u_true is None:
        return {}

    # Remove duplicate point at boundary for periodic functions
    u_eval_clean = u_eval[:-1] if len(u_eval) > 1 else u_eval
    u_true_clean = u_true[:-1] if len(u_true) > 1 else u_true

    # Compute residual (what's left to learn) and learned content
    residual = u_true_clean - u_eval_clean  # Error in function values
    learned = u_eval_clean  # What model currently predicts
    target = u_true_clean   # True function

    # FFT analysis - use rfft for real signals to avoid redundant negative frequencies
    residual_fft = torch.fft.rfft(residual)
    learned_fft = torch.fft.rfft(learned)
    target_fft = torch.fft.rfft(target)

    return {
        'residual_fft_magnitude': torch.abs(residual_fft).detach().cpu(),
        'learned_fft_magnitude': torch.abs(learned_fft).detach().cpu(),
        'target_fft_magnitude': torch.abs(target_fft).detach().cpu(),
    }


def frequency_learning_rates_analyzer(model, u_eval=None, u_true=None, **kwargs):
    """
    Analysis 4: Frequency-Specific Learning Rates
    Track convergence rate of each Fourier coefficient
    """
    if u_eval is None or u_true is None:
        return {}

    # Remove duplicate point at boundary for periodic functions
    model_vals = u_eval[:-1] if len(u_eval) > 1 else u_eval

    # Project model output onto Fourier basis using rfft
    model_fft = torch.fft.rfft(model_vals)

    # Get dominant frequency components (top 10)
    magnitudes = torch.abs(model_fft)
    top_freq_indices = torch.topk(magnitudes, min(10, len(magnitudes))).indices

    result = {}
    for i, freq_idx in enumerate(top_freq_indices):
        result[f'freq_{freq_idx.item()}_magnitude'] = magnitudes[freq_idx].detach().cpu()
        result[f'freq_{freq_idx.item()}_phase'] = torch.angle(model_fft[freq_idx]).detach().cpu()

    return result


def weight_specialization_analyzer(model, **kwargs):
    """
    Analysis 5: Weight Specialization Analysis
    Correlate weight changes with frequency content
    """
    results = {}

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # 2D FFT of weight matrix
            weight_fft = torch.fft.fft2(param.detach())

            # Spectral properties
            results[f'{name}_spectral_norm'] = torch.linalg.matrix_norm(param.detach(), ord=2).cpu()
            results[f'{name}_frobenius_norm'] = torch.linalg.matrix_norm(param.detach(), ord='fro').cpu()
            results[f'{name}_fft_magnitude_mean'] = torch.abs(weight_fft).mean().cpu()
            results[f'{name}_fft_magnitude_max'] = torch.abs(weight_fft).max().cpu()

            # Effective rank
            u, s, v = torch.linalg.svd(param.detach())
            threshold = 0.01 * s[0]  # 1% of largest singular value
            effective_rank = (s > threshold).sum().cpu()
            results[f'{name}_effective_rank'] = effective_rank

    return results


def compute_svds(model, **kwargs):
    """Compute singular values of weight matrices in the MLP."""
    svd_dict = {}

    for name, param in model.named_parameters():
        if "weight" in name:
            # Compute SVD
            weight_matrix = param.data.cpu().numpy()
            u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
            svd_dict[name] = s  # Store singular values

    return svd_dict


# Plotting functions for frequency analysis
def plot_frequency_evolution_timeline(json_file_path, save_path, every_k_eval=1, k_value=None, num_bands=1):
    """Plot learning timeline showing which frequencies are learned when."""
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training data from {json_file_path}: {e}")
        return

    # Look for function decomposition data
    residual_key = "weight_function_decomposition_analyzer_residual_fft_magnitude"
    learned_key = "weight_function_decomposition_analyzer_learned_fft_magnitude"

    if residual_key not in training_data or learned_key not in training_data:
        print(f"Warning: No function decomposition data found")
        return

    # Filter to only show up to epoch 2000, but at every evaluation (ignore every_k_eval for epoch filtering)
    all_iterations = training_data[residual_key]['iter']
    all_residual_fft = np.array(training_data[residual_key]['value'])
    all_learned_fft = np.array(training_data[learned_key]['value'])

    print(f"Total evaluations in data: {len(all_iterations)}")
    print(f"Epoch range in data: {min(all_iterations) if all_iterations else 'N/A'} to {max(all_iterations) if all_iterations else 'N/A'}")

    # Find indices for epochs <= 2000
    epoch_mask = np.array(all_iterations) <= 2000
    n_epochs_under_2000 = np.sum(epoch_mask)

    print(f"Evaluations with epochs <= 2000: {n_epochs_under_2000}")
    if n_epochs_under_2000 > 0:
        epochs_under_2000 = [all_iterations[i] for i in range(len(all_iterations)) if epoch_mask[i]]
        print(f"Epochs <= 2000: {epochs_under_2000[:10]}{'...' if len(epochs_under_2000) > 10 else ''}")

    # Apply epoch filter first, then every_k_eval sampling
    filtered_iterations = [all_iterations[i] for i in range(len(all_iterations)) if epoch_mask[i]]
    filtered_residual = all_residual_fft[epoch_mask]
    filtered_learned = all_learned_fft[epoch_mask]

    print(f"After epoch filtering: {len(filtered_iterations)} evaluations")
    print(f"every_k_eval parameter: {every_k_eval} (ignored - showing every evaluation)")

    # Use every evaluation up to epoch 2000 (ignore every_k_eval parameter)
    iterations = filtered_iterations
    residual_fft = filtered_residual
    learned_fft = filtered_learned

    print(f"Final: {len(iterations)} evaluations for plotting (every evaluation up to epoch 2000)")

    # Calculate frequency range - using rfft so we have positive frequencies only
    n_freqs = residual_fft.shape[1]  # rfft output length
    n_points = (n_freqs - 1) * 2  # Original signal length before rfft
    freqs = np.fft.rfftfreq(n_points, d=2.0/n_points)  # Frequency axis for rfft

    if k_value is not None and num_bands is not None:
        # For harmonics: highest frequency is num_bands * k / (2π) + buffer
        max_freq = (num_bands * k_value) / (2 * np.pi) + 4
    elif k_value is not None:
        # Single frequency case
        max_freq = k_value / (2 * np.pi) + 5
    else:
        max_freq = 10  # Default fallback

    # Find indices for frequencies 0 to k/(2*pi) + 5
    freq_mask = (freqs >= 0) & (freqs <= max_freq)
    freq_indices = np.where(freq_mask)[0]
    freqs_subset = freqs[freq_indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Use evenly spaced evaluation indices instead of actual epoch numbers
    # This gives equal spacing between evaluations regardless of epoch intervals
    eval_indices = np.arange(len(iterations))  # 0, 1, 2, 3, ... for each evaluation

    print(f"Plotting {len(iterations)} evaluations with even spacing")
    print(f"Epoch range: {iterations[0]} to {iterations[-1]}")

    # Use imshow with evenly spaced evaluation indices
    # Plot residual evolution (what's left to learn)
    im1 = ax1.imshow(residual_fft[:, freq_indices].T,
                    aspect='auto', origin='lower',
                    extent=[eval_indices[0], eval_indices[-1], freqs_subset[0], freqs_subset[-1]],
                    norm=LogNorm(), cmap='viridis')
    # Plot learned content evolution
    im2 = ax2.imshow(learned_fft[:, freq_indices].T,
                    aspect='auto', origin='lower',
                    extent=[eval_indices[0], eval_indices[-1], freqs_subset[0], freqs_subset[-1]],
                    norm=LogNorm(), cmap='plasma')

    # Create custom x-axis labels showing actual epochs at key evaluation points
    n_ticks = min(8, len(iterations))  # Show up to 8 epoch labels
    tick_indices = np.linspace(0, len(iterations)-1, n_ticks, dtype=int)
    tick_positions = eval_indices[tick_indices]
    tick_labels = [str(iterations[i]) for i in tick_indices]

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    # Set custom y-axis (frequency) ticks - granularity 0.05, labels every 5
    freq_min = freqs_subset[0]
    freq_max = freqs_subset[-1]

    # Show tick labels every 5.0 (0, 5, 10, 15, ...)
    labeled_ticks = np.arange(0, freq_max + 5.0, 5.0)
    labeled_ticks = labeled_ticks[labeled_ticks <= freq_max]

    # Minor ticks every 0.05 for granularity (no labels)
    minor_ticks = np.arange(0, freq_max + 0.05, 0.05)
    minor_ticks = minor_ticks[minor_ticks <= freq_max]

    ax1.set_yticks(labeled_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    ax2.set_yticks(labeled_ticks)
    ax2.set_yticks(minor_ticks, minor=True)

    ax1.set_ylabel('Frequency')
    freq_range_str = f"0 to {max_freq:.1f}" if k_value is not None else "0-10"
    ax1.set_title(f'Residual FFT Magnitude (What\'s left to learn) - Frequencies {freq_range_str}')
    plt.colorbar(im1, ax=ax1)

    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Training Epoch (Evenly Spaced Evaluations)')
    ax2.set_title(f'Learned FFT Magnitude (What model has captured) - Frequencies {freq_range_str}')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Frequency evolution timeline saved to: {save_path}")


def plot_target_vs_prediction(target, model, k_value, save_path):
    """Plot target function vs model prediction for visual comparison."""
    # Generate high-resolution test points
    x_test = torch.linspace(-1, 1, 1000, device=target.device, dtype=target.dtype)

    # Get target values
    target_values = target.get_function([x_test]).detach().cpu().numpy()

    # Get model predictions
    with torch.no_grad():
        pred_values = model([x_test]).detach().cpu().numpy()

    x_test_np = x_test.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))

    # Plot target and prediction
    plt.subplot(2, 1, 1)
    plt.plot(x_test_np, target_values, 'b-', linewidth=2, label='Target', alpha=0.8)
    plt.plot(x_test_np, pred_values, 'r--', linewidth=2, label='Prediction', alpha=0.8)
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.title(f'Target vs Prediction (k={k_value})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot error
    plt.subplot(2, 1, 2)
    error = pred_values - target_values
    plt.plot(x_test_np, error, 'g-', linewidth=2, label='Error (Pred - Target)')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title(f'Prediction Error (k={k_value})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Target vs prediction plot saved to: {save_path}")


def plot_weight_specialization_evolution(json_file_path, save_path, every_k_eval=1):
    """Plot weight specialization metrics evolution."""
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training data from {json_file_path}: {e}")
        return

    # Find all weight specialization keys
    specialization_keys = [k for k in training_data.keys()
                          if k.startswith("weight_weight_specialization_analyzer")]

    if not specialization_keys:
        print(f"Warning: No weight specialization data found")
        return

    plt.figure(figsize=(15, 10))

    # Create subplots for different metrics
    n_cols = 2
    n_rows = (len(specialization_keys) + 1) // n_cols

    for i, key in enumerate(specialization_keys):
        iterations = training_data[key]['iter'][::every_k_eval]
        values = training_data[key]['value'][::every_k_eval]

        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(iterations, values, linewidth=2)
        plt.xlabel('Training Epoch')
        plt.ylabel('Value')
        plt.title(key.split('_')[-1])  # Extract metric name
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weight specialization evolution saved to: {save_path}")


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
        run_tag: optional string to include in filename (e.g., 'k_2.0')
        use_logy: whether to use log scale for y-axis
        json_file_path: optional path to JSON file with training history
        every_k_eval: only plot every k-th evaluation from training history to reduce clutter
    """
    # Define cleaning function at the top of the function
    def clean_for_log(data, min_val=1e-12):
        cleaned = np.array(data, dtype=float)
        # Replace inf, -inf, nan, and values <= 0 with min_val
        mask = ~np.isfinite(cleaned) | (cleaned <= 0)
        if np.any(mask):
            cleaned[mask] = min_val
        return cleaned

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

                    # Clean SVD values for log plotting if needed
                    if use_logy:
                        svd_sorted = clean_for_log(svd_sorted)

                    alpha = 0.3 + 0.7 * (i / max(1, len(filtered_iterations) - 1))  # Fade in over time

                    if iteration == -1:
                        label = "Initial (epoch -1)"
                        color = 'gray'
                        linewidth = 2
                    else:
                        label = f"Epoch {iteration}"
                        color = plt.cm.viridis(i / max(1, len(filtered_iterations) - 1))
                        linewidth = 1.5

                    plt.plot(np.arange(1, len(svd_sorted) + 1), svd_sorted,
                            color=color, alpha=alpha, linewidth=linewidth,
                            marker='o' if len(svd_sorted) <= 10 else None,
                            markersize=3, label=label)

        # Plot final state (from svd_dict) with emphasis
        plt.plot(np.arange(1, len(s_sorted) + 1), s_sorted,
                color='red', linewidth=3, marker='o', markersize=4,
                label="Final state", alpha=0.9)

        if use_logy:
            # Clean the final SVD values
            s_sorted = clean_for_log(s_sorted)
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


def plot_k_sweep_results(results, save_path):
    """Plot L2RE vs k for the wavenumber sweep."""
    plt.figure(figsize=(10, 6))

    k_list = [r["k"] for r in results]
    l2re_values = [r["l2re_test"] for r in results]

    # Plot with log scale for both x-axis and y-axis
    plt.loglog(k_list, l2re_values, "o-", linewidth=2, markersize=8)

    # Customize plot
    plt.xlabel("Wavenumber (k)")
    plt.ylabel("L2 Relative Error")
    plt.title("L2RE vs Wavenumber")
    plt.grid(True, alpha=0.3)

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


def plot_data_periodogram_comparison(target, save_path, title_suffix=""):
    """
    Compare the periodogram of training data vs evaluation data to verify
    that training data is representative of the target function's frequency content.
    This is done before any training to check data quality.
    """
    plt.figure(figsize=(15, 10))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Interpolate both training and test data onto a common high-resolution grid
    # Use a grid that's dense enough to capture the frequency content
    x_dense = torch.linspace(-1, 1, 2048, device=target.test_points.device, dtype=target.test_points.dtype)

    # Get true function values on dense grid
    true_values_dense = target.get_function([x_dense])

    # Interpolate training data onto dense grid (simple linear interpolation)
    train_x = target.train_points_0th.cpu().numpy()
    train_y = target.train_values_0th.cpu().numpy()

    # Sort training points for interpolation
    sort_idx = np.argsort(train_x)
    train_x_sorted = train_x[sort_idx]
    train_y_sorted = train_y[sort_idx]

    # Interpolate training data to dense grid
    train_interp = np.interp(x_dense.cpu().numpy(), train_x_sorted, train_y_sorted)
    train_interp_tensor = torch.tensor(train_interp, device=x_dense.device, dtype=x_dense.dtype)

    # Compute periodograms using the same method as in training
    def compute_psd_from_signal(signal):
        """Compute PSD using the same windowing as in training."""
        # Drop duplicate point at the seam and apply Hann window
        signal_windowed = signal[:-1]  # Remove duplicate
        N = signal_windowed.shape[-1]
        dx = 2.0 / N

        # Create Hann window
        n = torch.arange(N, device=signal.device, dtype=signal.dtype)
        w = 0.5 - 0.5 * torch.cos(2 * torch.pi * n / N)

        # Apply window
        xw = signal_windowed * w

        # Compute FFT
        X = torch.fft.rfft(xw, dim=-1)

        # Scale for energy conservation
        Wpow = (w**2).sum()
        Sxx = (X.abs()**2) * (dx / Wpow)

        return Sxx.cpu().numpy()

    # Compute PSDs
    true_psd = compute_psd_from_signal(true_values_dense)
    train_psd = compute_psd_from_signal(train_interp_tensor)

    # Create frequency axis
    N = len(true_values_dense) - 1  # After dropping duplicate
    freqs = np.fft.rfftfreq(N, d=2.0/N)  # Frequency bins

    # Plot 1: True function
    axes[0, 0].plot(x_dense.cpu().numpy(), true_values_dense.cpu().numpy(), 'b-', linewidth=2, label='True Function')
    axes[0, 0].scatter(train_x, train_y, c='red', s=20, alpha=0.7, label=f'Training Points (n={len(train_x)})')
    axes[0, 0].set_title(f'Function Comparison{title_suffix}')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u(x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Interpolated training data
    axes[0, 1].plot(x_dense.cpu().numpy(), true_values_dense.cpu().numpy(), 'b-', linewidth=2, label='True Function')
    axes[0, 1].plot(x_dense.cpu().numpy(), train_interp, 'r--', linewidth=2, alpha=0.8, label='Interpolated Training')
    axes[0, 1].set_title(f'Training Data Interpolation{title_suffix}')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u(x)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: PSD comparison (linear scale)
    axes[1, 0].semilogy(freqs, true_psd, 'b-', linewidth=2, label='True Function PSD')
    axes[1, 0].semilogy(freqs, train_psd, 'r--', linewidth=2, alpha=0.8, label='Training Data PSD')
    axes[1, 0].set_title(f'Periodogram Comparison (Log Scale){title_suffix}')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('PSD Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: PSD ratio to show discrepancies
    # Avoid division by zero
    psd_ratio = np.divide(train_psd, true_psd, out=np.ones_like(train_psd), where=true_psd!=0)
    axes[1, 1].semilogy(freqs[1:], psd_ratio[1:], 'g-', linewidth=2)  # Skip DC component
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Match')
    axes[1, 1].set_title(f'PSD Ratio (Training/True){title_suffix}')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('Ratio (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add some statistics as text
    mse_interp = np.mean((true_values_dense.cpu().numpy() - train_interp)**2)
    psd_mse = np.mean((true_psd - train_psd)**2)

    fig.suptitle(f'Data Quality Check{title_suffix}\n'
                f'Interpolation MSE: {mse_interp:.2e}, PSD MSE: {psd_mse:.2e}',
                fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Data periodogram comparison saved to: {save_path}")
    print(f"  Interpolation MSE: {mse_interp:.2e}")
    print(f"  PSD MSE: {psd_mse:.2e}")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse k values from command line
    k_list = [float(k.strip()) for k in args.k_values.split(',')]

    # Validate embedding requirements
    if args.embedding in ["cheb", "bary"] and args.embedding_M is None:
        raise ValueError(f"For '{args.embedding}' embedding you must specify --embedding_M > 0.")

    # Parse custom evaluation schedule if provided
    custom_eval_epochs = None
    effective_eval_every = args.eval_every

    if args.custom_eval_schedule:
        custom_eval_epochs = parse_custom_eval_schedule(args.custom_eval_schedule, args.n_epochs)
        print(f"Using custom evaluation schedule: {len(custom_eval_epochs)} evaluations")
        print(f"Eval epochs: {custom_eval_epochs[:10]}{'...' if len(custom_eval_epochs) > 10 else ''}")

        # Auto-adjust svd_plot_every for reasonable number of curves (target ~20-30 curves max)
        if args.svd_plot_every == 10:  # Only auto-adjust if using default value
            target_curves = 25
            auto_every = max(1, len(custom_eval_epochs) // target_curves)
            if auto_every > args.svd_plot_every:
                print(f"Auto-adjusting --svd_plot_every from {args.svd_plot_every} to {auto_every} for reasonable plotting")
                args.svd_plot_every = auto_every

        print(f"Will show every {args.svd_plot_every}th evaluation in SVD plots (~{len(custom_eval_epochs) // args.svd_plot_every} curves)")

        # Custom evaluation schedule will be used directly in training

    # Store results
    results = []

    print(
        f"Running frequency analysis sweep for MLP: {args.n_layers} layers, {args.hidden_dim} hidden dim"
    )
    print(f"Target: {args.target}, Device: {args.device}, Optimizer: {args.optimizer}")
    print(f"Embedding: {args.embedding}" + (f" (M={args.embedding_M})" if args.embedding_M else ""))
    print(f"Fixed 0th order points: {args.n_train_0th}, 1st order points: {args.n_train_1st}")
    print(f"Sweeping over wavenumbers: {k_list}")
    print(f"Training for {args.n_epochs} epochs with {'custom' if args.custom_eval_schedule else 'fixed'} evaluation schedule")

    for k in k_list:
        print(f"\n--- Testing k = {k} with frequency analysis ---")

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
            use_derivatives=(args.n_train_1st > 0),
            deriv_alpha=args.deriv_alpha,
            deriv_beta=args.deriv_beta,
            save_dir=args.save_dir,
        )

        # Create target with current wavenumber
        target = create_target(
            config,
            n_train_0th=args.n_train_0th,
            n_train_1st=args.n_train_1st,
            k=k,
            num_bands=args.num_bands,
        )

        # Create model config with embedding parameters
        model_config = ModelConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            device=args.device,
            embedding=args.embedding,
            embedding_M=args.embedding_M,
        )

        # SANITY CHECK: Compare periodogram of training data vs evaluation data
        print(f"Performing data quality check...")
        plot_data_periodogram_comparison(
            target,
            os.path.join(args.save_dir, f"data_quality_check_k_{k}.png"),
            title_suffix=f" (k={k})"
        )

        # Train and evaluate using frequency analyzers + SVD
        print(f"Training model with comprehensive frequency analysis...")

        # Use custom evaluation schedule if provided, otherwise use regular eval_every
        if args.custom_eval_schedule:
            print(f"Training with custom evaluation schedule: {len(custom_eval_epochs)} evaluation points")
            run_results = run_mlp_experiment(
                target=target,
                config=model_config,
                n_epochs=args.n_epochs,
                eval_every=args.eval_every,  # This parameter is ignored when custom_eval_epochs is provided
                exp_dir=os.path.join(args.save_dir, f"k_{k}"),
                optimizer_name=args.optimizer,
                weight_evals=[
                    function_decomposition_analyzer,
                    frequency_learning_rates_analyzer,
                    weight_specialization_analyzer,
                    compute_svds,
                ],
                custom_eval_epochs=custom_eval_epochs,
            )
        else:
            print(f"Training with fixed evaluation every {args.eval_every} epochs")
            run_results = run_mlp_experiment(
                target=target,
                config=model_config,
                n_epochs=args.n_epochs,
                eval_every=args.eval_every,
                exp_dir=os.path.join(args.save_dir, f"k_{k}"),
                optimizer_name=args.optimizer,
                weight_evals=[
                    function_decomposition_analyzer,
                    frequency_learning_rates_analyzer,
                    weight_specialization_analyzer,
                    compute_svds,
                ],
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

        # Compute SVDs for final plotting
        svd_dict = compute_svds(model)

        # Store results
        result = {
            "k": k,
            "rmse_train": rmse_train,
            "l_inf_train": l_inf_train,
            "rmse_test": rmse_test,
            "l_inf_test": l_inf_test,
            "l2re_test": l2re_test,
            "singular_values": svd_dict,
        }
        results.append(result)

        print(f"Results: RMSE_test={rmse_test:.2e}, L2RE_test={l2re_test:.2e}")

        # Generate frequency analysis plots
        json_log_path = os.path.join(
            args.save_dir, f"k_{k}",
            f"mlp_hdim{args.hidden_dim}_layers{args.n_layers}.json"
        )

        # Plot function decomposition timeline
        plot_frequency_evolution_timeline(
            json_log_path,
            os.path.join(args.save_dir, f"freq_evolution_timeline_k_{k}.png"),
            every_k_eval=args.svd_plot_every,
            k_value=k,
            num_bands=args.num_bands,
        )

        # Plot weight specialization evolution
        plot_weight_specialization_evolution(
            json_log_path,
            os.path.join(args.save_dir, f"weight_specialization_k_{k}.png"),
            every_k_eval=args.svd_plot_every,
        )

        # Plot target vs prediction comparison
        plot_target_vs_prediction(
            target,
            model,
            k,
            os.path.join(args.save_dir, f"target_vs_prediction_k_{k}.png"),
        )

        # Plot SVD evolution for each weight matrix
        plot_svd_norms(
            svd_dict,
            save_dir=args.save_dir,
            run_tag=f"k_{k}",
            use_logy=True,
            json_file_path=json_log_path,
            every_k_eval=args.svd_plot_every,
        )

        # Plot solution comparison (same as size k sweep)
        plot_sine_target_solution(
            target,
            model,
            os.path.join(args.save_dir, f"solution_k_{k}.png"),
            title_suffix=f" (k={k})"
        )

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f"model_k_{k}.pt"),
        )

    # Save all results
    with open(os.path.join(args.save_dir, "freq_analysis_results.json"), "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for r in results:
            json_r = r.copy()
            # Convert singular values numpy arrays to lists
            json_r["singular_values"] = {k: v.tolist() for k, v in r["singular_values"].items()}
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    # Create wavenumber sweep plot
    plot_k_sweep_results(
        results, os.path.join(args.save_dir, "k_sweep_l2re.png")
    )

    # Create summary table
    print("\n" + "=" * 80)
    print("FREQUENCY ANALYSIS SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'k':<12} {'RMSE_test':<12} {'L2RE_test':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r['k']:<12} {r['rmse_test']:<12.2e} {r['l2re_test']:<12.2e}")

    print(f"\nResults saved to: {args.save_dir}")
    print("Files created:")
    print(f"  - freq_analysis_results.json: All numerical results")
    print(f"  - k_sweep_l2re.png: L2RE vs k plot")
    print(f"  - data_quality_check_k_*.png: Pre-training data quality analysis")
    print(f"  - freq_evolution_timeline_k_*.png: Function decomposition timelines")
    print(f"  - weight_specialization_k_*.png: Weight specialization evolution")
    print(f"  - target_vs_prediction_k_*.png: Target vs prediction comparisons")
    print(f"  - svd_*_k_*.png: Singular value evolution plots for each layer and run")
    print(f"  - solution_k_*.png: Solution comparison plots for each run")
    print(f"  - model_k_*.pt: Trained model states")
    print(f"  - k_*/mlp_hdim*_layers*.json: Training logs with frequency data")

    print("\nFrequency analysis completed successfully!")
    print("This analysis provides insights into:")
    print("  1. Function Approximation Decomposition - how model builds up target")
    print("  4. Frequency-Specific Learning Rates - convergence per Fourier mode")
    print("  5. Weight Specialization Analysis - weight correlation with frequencies")


if __name__ == "__main__":
    main()
