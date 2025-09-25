# Plotting functions for frequency analysis
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
    epoch_mask = np.array(all_iterations) <= 7000
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

if __name__ == "__main__":
    main()
