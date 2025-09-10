#!/usr/bin/env python3
"""
Script to run wavenumber sweep for a fixed MLP configuration using the new SineTarget.
Sweeps over different wavenumbers (k) for the sine target and analyzes weight norms.
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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.experiments.interpolation.config import ExperimentConfig, ModelConfig
from src.experiments.interpolation.targets import create_target
from src.experiments.interpolation.models import create_mlp, run_mlp_experiment
from src.utils.metrics import l2_relative_error, l2_error, max_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MLP wavenumber sweep for fixed configuration"
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
        choices=["adam", "sgd", "ssbroyden"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1.0,2.0,4.0,8.0",
        help="Comma-separated list of wavenumbers to sweep over (e.g., '1.0,2.0,4.0,8.0')",
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

def plot_psd_heatmap(json_file_path, save_path, every_k_eval=1):
    """
    Create a heatmap showing the evolution of the PSD (periodogram) over training.
    
    Args:
        json_file_path: path to JSON file with training history
        save_path: where to save the heatmap PNG
        every_k_eval: only plot every k-th evaluation to reduce data density
    """
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found at {json_file_path}")
        return
        
    try:
        with open(json_file_path, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training data from {json_file_path}: {e}")
        return
    
    # Look for periodogram data from dyadic gradients
    psd_key = "weight_dyadic_psd_gradients_Sxx"
    if psd_key not in training_data:
        print(f"Warning: No periodogram data found in {json_file_path}")
        return
    
    iterations = training_data[psd_key]['iter'][::every_k_eval]
    psd_history = training_data[psd_key]['value'][::every_k_eval]  # List of PSD arrays
    
    if len(psd_history) == 0:
        print(f"Warning: No PSD data available")
        return
    
    # Convert to numpy array: (n_iterations, n_frequencies)
    psd_array = np.array(psd_history)
    n_iters, n_freqs = psd_array.shape
    
    # Create frequency axis (assuming normalized frequencies from FFT)
    frequencies = np.arange(n_freqs)
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with log scale for PSD values
    im = plt.imshow(psd_array, 
                    aspect='auto', 
                    origin='lower',
                    cmap='viridis',
                    norm=LogNorm(vmin=psd_array[psd_array > 0].min(), 
                                   vmax=psd_array.max()),
                    extent=[-0.5, n_freqs-0.5, iterations[0], iterations[-1]])
    
    plt.colorbar(im, label='PSD Magnitude (log scale)')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Training Iteration')
    plt.title('PSD Evolution During Training')
    
    # Add some frequency tick labels
    if n_freqs > 10:
        freq_ticks = np.linspace(0, n_freqs-1, min(10, n_freqs), dtype=int)
        plt.xticks(freq_ticks)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PSD heatmap saved to: {save_path}")

def plot_gradient_norm_heatmap(json_file_path, save_path, every_k_eval=1):
    """
    Create a heatmap showing the evolution of gradient norms across frequency probes over training.
    
    Args:
        json_file_path: path to JSON file with training history
        save_path: where to save the heatmap PNG
        every_k_eval: only plot every k-th evaluation to reduce data density
    """
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found at {json_file_path}")
        return
        
    try:
        with open(json_file_path, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training data from {json_file_path}: {e}")
        return
    
    # Look for dyadic gradient data
    grad_key = "weight_dyadic_psd_gradients_grad_norms"
    centers_key = "weight_dyadic_psd_gradients_centers"
    
    if grad_key not in training_data or centers_key not in training_data:
        print(f"Warning: No dyadic gradient data found in {json_file_path}")
        return
    
    iterations = training_data[grad_key]['iter'][::every_k_eval]
    grad_history = training_data[grad_key]['value'][::every_k_eval]  # List of gradient norm arrays
    
    # Get the frequency centers (should be consistent across iterations)
    centers_history = training_data[centers_key]['value']
    if len(centers_history) > 0:
        centers = centers_history[0]  # Use first iteration's centers
    else:
        print(f"Warning: No frequency centers data available")
        return
    
    if len(grad_history) == 0:
        print(f"Warning: No gradient data available")
        return
    
    # Convert to numpy array: (n_iterations, n_probes)
    grad_array = np.array(grad_history)
    n_iters, n_probes = grad_array.shape
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with log scale for gradient norms
    im = plt.imshow(grad_array, 
                    aspect='auto', 
                    origin='lower',
                    cmap='viridis',
                    norm=LogNorm(vmin=grad_array[grad_array > 0].min(), 
                                   vmax=grad_array.max()),
                    extent=[-0.5, n_probes-0.5, iterations[0], iterations[-1]])
    
    plt.colorbar(im, label='Gradient Norm (log scale)')
    plt.xlabel('Frequency Probe')
    plt.ylabel('Training Iteration')
    plt.title('Gradient Norms Across Frequency Probes During Training')
    
    # Add frequency center labels on x-axis - align with imshow extent
    probe_labels = [f'{int(c)}' if c == int(c) else f'{c:.1f}' for c in centers]
    tick_positions = np.arange(n_probes)  # 0, 1, 2, ..., n_probes-1
    plt.xticks(tick_positions, probe_labels, rotation=45)
    
    # Add secondary x-axis label explaining the centers
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xlabel('Frequency Centers (DC, 1, 2, 4, 8, ...)')
    ax2.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient norm heatmap saved to: {save_path}")

def plot_dyadic_energy_heatmap(json_file_path, save_path, every_k_eval=1):
    """
    Create a heatmap showing the evolution of dyadic probe energies over training.
    
    Args:
        json_file_path: path to JSON file with training history
        save_path: where to save the heatmap PNG
        every_k_eval: only plot every k-th evaluation to reduce data density
    """
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found at {json_file_path}")
        return
        
    try:
        with open(json_file_path, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training data from {json_file_path}: {e}")
        return
    
    # Look for dyadic energy data
    energy_key = "weight_dyadic_psd_gradients_energies"
    centers_key = "weight_dyadic_psd_gradients_centers"
    
    if energy_key not in training_data or centers_key not in training_data:
        print(f"Warning: No dyadic energy data found in {json_file_path}")
        return
    
    iterations = training_data[energy_key]['iter'][::every_k_eval]
    energy_history = training_data[energy_key]['value'][::every_k_eval]  # List of energy arrays
    
    # Get the frequency centers
    centers_history = training_data[centers_key]['value']
    if len(centers_history) > 0:
        centers = centers_history[0]  # Use first iteration's centers
    else:
        print(f"Warning: No frequency centers data available")
        return
    
    if len(energy_history) == 0:
        print(f"Warning: No energy data available")
        return
    
    # Convert to numpy array: (n_iterations, n_probes)
    energy_array = np.array(energy_history)
    n_iters, n_probes = energy_array.shape
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with log scale for energies
    im = plt.imshow(energy_array, 
                    aspect='auto', 
                    origin='lower',
                    cmap='viridis',
                    norm=LogNorm(vmin=energy_array[energy_array > 0].min(), 
                                   vmax=energy_array.max()),
                    extent=[-0.5, n_probes-0.5, iterations[0], iterations[-1]])
    
    plt.colorbar(im, label='Probe Energy (log scale)')
    plt.xlabel('Frequency Probe')
    plt.ylabel('Training Iteration')
    plt.title('Dyadic Probe Energies During Training')
    
    # Add frequency center labels on x-axis - align with imshow extent
    probe_labels = [f'{int(c)}' if c == int(c) else f'{c:.1f}' for c in centers]
    tick_positions = np.arange(n_probes)  # 0, 1, 2, ..., n_probes-1
    plt.xticks(tick_positions, probe_labels, rotation=45)
    
    # Add secondary x-axis label
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xlabel('Frequency Centers (DC, 1, 2, 4, 8, ...)')
    ax2.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dyadic energy heatmap saved to: {save_path}")

def hann_window(N, device = None, dtype=torch.float64):
    n = torch.arange(N, device = device, dtype=dtype)
    # periodic Hann matched to FFT (no duplicate at the seam)
    return 0.5 - 0.5*torch.cos(2*torch.pi*n/N)

def compute_periodogram(model, u_eval = None, u_true = None, numpy = False, **kwargs):
    """Compute periodogram of the residual."""
    # Work entirely in PyTorch for consistency
    residual = (u_eval - u_true)[:-1]  # drop duplicate point at the seam, keep as tensor
    N = residual.shape[-1]
    dx = 2.0 / N
    
    # Create Hann window as torch tensor
    n = torch.arange(N, device=residual.device, dtype=residual.dtype)
    w = 0.5 - 0.5 * torch.cos(2 * torch.pi * n / N)
    
    # Apply window
    xw = residual * w
    
    # Compute FFT
    X = torch.fft.rfft(xw, dim=-1)
    
    # Scale for energy conservation: see Parseval with window; include dx to integrate over x
    Wpow = (w**2).sum()
    Sxx = (X.abs()**2) * (dx / Wpow)  # power spectral density ~ |FFT|^2 normalized
    
    # Return as dictionary like compute_svds to match logging system expectations
    if numpy:
        return Sxx.cpu().numpy()
    return Sxx

def _dyadic_centers(F: int, start_bin: int = 1):
    """Centers at 1,2,4,8,... < F (start_bin is clamped to >=1)."""
    c = []
    v = max(1, int(start_bin))
    while v < F:
        c.append(v)
        v <<= 1
    if len(c) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(c, dtype=torch.long)

def _gaussian_bins(F: int, centers: torch.Tensor, sigma_bins: float,
                   device, dtype, normalize: bool = True):
    """
    Gaussian kernels in *bin index* space.
    sigma_bins ~ 1–2 is a good start with Hann.
    Returns W: (J, F); optionally row-normalized (mean over support).
    """
    if centers.numel() == 0:
        return torch.empty(0, F, device=device, dtype=dtype)
    m = torch.arange(F, device=device, dtype=dtype)[None, :]      # (1,F)
    c = centers.to(device=device, dtype=dtype)[:, None]           # (J,1)
    W = torch.exp(-0.5 * ((m - c) / float(sigma_bins))**2)        # (J,F)
    if normalize:
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
    return W

def _gaussian_octave(F: int, centers: torch.Tensor, sigma_oct: float,
                     device, dtype, normalize: bool = True):
    """
    Constant-Q style: Gaussian in log2(bin_index) (ignoring DC).
    sigma_oct ~ 1/3–1/2 octave is typical.
    """
    if centers.numel() == 0:
        return torch.empty(0, F, device=device, dtype=dtype)
    m = torch.arange(F, device=device, dtype=dtype)
    logm = torch.log2(torch.clamp(m, min=1)).to(dtype)[None, :]   # (1,F)
    logc = torch.log2(torch.clamp(centers.to(device=device), min=1)).to(dtype)[:, None]  # (J,1)
    W = torch.exp(-0.5 * ((logm - logc) / float(sigma_oct))**2)   # (J,F)
    # (optionally zero DC row entries, but log clamp to 1 already suppresses it)
    if normalize:
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
    return W

def dyadic_psd_gradients(
    model,
    u_eval: torch.Tensor,
    u_true: torch.Tensor,
    *,
    kernel_type: str = "bins",     # "bins" or "octave"
    sigma_bins: float = 1.5,       # ~1–2 bins catches Hann main-lobe
    sigma_oct: float = 0.35,       # ~1/3 octave for constant-Q
    start_bin: int = 1,            # dyadic centers start (1→ 1,2,4,8,...)
    include_dc: bool = True,       # add DC as its own probe at bin 0
    reduction: str = "mean",       # "mean" (row-normalized kernels) or "sum"
):
    """
    Build dyadic probes at powers of 2 and compute one gradient per probe for the PSD (periodogram):
      E_j = <W_j, Sxx>, where Sxx is your unweighted periodogram.
    NOTE: run without no_grad/inference_mode; graph must be alive.

    Returns:
      dict with:
        - 'energies': per-probe energies (detached, shape (P,))
        - 'grad_norms': list[float] length P (L2 over all params)
        - 'centers': list of probe centers (0 if include_dc else starts at 1,2,4,...)
        - 'Sxx': detached PSD (for your normal logging)
        - 'P': number of probes
    """

    params = list(model.parameters())

    # 1) PSD with your normalization (keeps graph)
    Sxx = compute_periodogram(model, u_eval=u_eval, u_true=u_true, numpy=False)  # (..., F)
    F = Sxx.shape[-1]
    device, dtype = Sxx.device, Sxx.dtype

    # 2) Build kernels at dyadic centers
    centers = _dyadic_centers(F, start_bin=start_bin)
    normalize = (reduction == "mean")
    if kernel_type == "octave":
        W = _gaussian_octave(F, centers, sigma_oct, device, dtype, normalize=normalize)
    else:
        W = _gaussian_bins(F, centers, sigma_bins, device, dtype, normalize=normalize)

    W_rows = []
    probe_centers = []

    if include_dc:
        dc = torch.zeros(1, F, device=device, dtype=dtype); dc[0, 0] = 1.0
        # for "sum" vs "mean" the DC row is already a single-bin kernel
        W_rows.append(dc)
        probe_centers.append(0)

    if W.numel() > 0:
        W_rows.append(W)
        probe_centers.extend(centers.tolist())

    if not W_rows:
        return {"energies": torch.empty(0), "grad_norms": [], "centers": [], "Sxx": Sxx.detach(), "P": 0}

    W_all = torch.cat(W_rows, dim=0)  # (P, F)
    P = W_all.shape[0]

    # If reduction == "sum", convert row-normalized kernels into sum kernels:
    if reduction == "sum":
        # Undo normalization by scaling with effective support size (approx by L1 norm inverse).
        # If you prefer exact sum of Sxx over bins, build boxcar bands instead.
        # Here we just use the unnormalized Gaussians: rebuild them quickly.
        if kernel_type == "octave":
            W_all = _gaussian_octave(F, centers, sigma_oct, device, dtype, normalize=False)
            if include_dc:
                dc = torch.zeros(1, F, device=device, dtype=dtype); dc[0, 0] = 1.0
                W_all = torch.cat([dc, W_all], dim=0)
        else:
            W_all = _gaussian_bins(F, centers, sigma_bins, device, dtype, normalize=False)
            if include_dc:
                dc = torch.zeros(1, F, device=device, dtype=dtype); dc[0, 0] = 1.0
                W_all = torch.cat([dc, W_all], dim=0)

    # 3) Probe energies (for logging) – mean over any leading batch dims
    E = (Sxx * W_all).sum(dim=-1)              # (..., P)
    
    if E.ndim > 1:
        E_log = E.mean(dim=tuple(range(E.ndim - 1))).detach().cpu()  # (P,)
    else:
        E_log = E.detach().cpu()                                      # (P,)

    # 4) One VJP per probe
    grad_norms = []
    for p in range(P):
        e_p = E[..., p].sum()  # scalar
        retain = (p < P - 1)
        grads = torch.autograd.grad(e_p, params, retain_graph=retain, allow_unused=True)
        g2 = 0.0
        for g in grads:
            if g is not None:
                g2 = g2 + (g**2).sum()
        grad_norms.append(float(torch.sqrt(g2).detach().cpu()))

    return {
        "energies": E_log,
        "grad_norms": grad_norms,
        "centers": probe_centers,   # [0, 1, 2, 4, 8, ...] if include_dc else [1,2,4,...]
        "kernel_type": kernel_type,
        "sigma_bins": sigma_bins,
        "sigma_oct": sigma_oct,
        "reduction": reduction,
        "Sxx": Sxx.detach(),
        "P": P,
    }


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


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse k values from command line
    k_list = [float(k.strip()) for k in args.k_values.split(',')]

    # Store results
    results = []

    print(
        f"Running wavenumber sweep for MLP: {args.n_layers} layers, {args.hidden_dim} hidden dim"
    )
    print(f"Target: {args.target}, Device: {args.device}, Optimizer: {args.optimizer}")
    print(f"Fixed 0th order points: {args.n_train_0th}, 1st order points: {args.n_train_1st}")
    print(f"Sweeping over wavenumbers: {k_list}")

    for k in k_list:
        print(f"\n--- Testing k = {k} ---")

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
            exp_dir=os.path.join(args.save_dir, f"k_{k}"),
            optimizer_name=args.optimizer,
            weight_evals=[compute_svds, dyadic_psd_gradients],
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
            "k": k,
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
            os.path.join(args.save_dir, f"weight_norms_k_{k}.png"),
        )
        
        # Plot SVDs per layer for this run
        json_log_path = os.path.join(
            args.save_dir, f"k_{k}", 
            f"mlp_hdim{args.hidden_dim}_layers{args.n_layers}.json"
        )
        
        plot_svd_norms(
            svd_dict,
            save_dir=args.save_dir,
            run_tag=f"k_{k}",
            use_logy=True,
            json_file_path=json_log_path,
            every_k_eval=args.svd_plot_every,
        )

        # Generate heatmaps for PSD and gradient analysis
        plot_psd_heatmap(
            json_log_path,
            os.path.join(args.save_dir, f"psd_heatmap_k_{k}.png"),
            every_k_eval=args.svd_plot_every,
        )
        
        plot_gradient_norm_heatmap(
            json_log_path,
            os.path.join(args.save_dir, f"gradient_heatmap_k_{k}.png"),
            every_k_eval=args.svd_plot_every,
        )
        
        plot_dyadic_energy_heatmap(
            json_log_path,
            os.path.join(args.save_dir, f"energy_heatmap_k_{k}.png"),
            every_k_eval=args.svd_plot_every,
        )

        # Save custom solution plot for this run
        title_suffix = f" (k={k})"
        plot_sine_target_solution(
            target,
            model,
            os.path.join(args.save_dir, f"solution_k_{k}.png"),
            title_suffix,
        )

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f"model_k_{k}.pt"),
        )

    # Save all results
    with open(os.path.join(args.save_dir, "k_sweep_results.json"), "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for r in results:
            json_r = r.copy()
            json_r["weight_norms"] = {k: float(v) for k, v in r["weight_norms"].items()}
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
    print("WAVENUMBER SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'k':<12} {'RMSE_test':<12} {'L2RE_test':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r['k']:<12} {r['rmse_test']:<12.2e} {r['l2re_test']:<12.2e}")

    print(f"\nResults saved to: {args.save_dir}")
    print("Files created:")
    print(f"  - k_sweep_results.json: All numerical results")
    print(f"  - k_sweep_l2re.png: L2RE vs k plot")
    print(f"  - weight_norms_k_*.png: Weight norm plots for each run")
    print(f"  - solution_k_*.png: Custom solution plots for each run")
    print(f"  - svd_*_k_*.png: Singular value plots for each layer and run")
    print(f"  - psd_heatmap_k_*.png: PSD evolution heatmaps for each run")
    print(f"  - gradient_heatmap_k_*.png: Gradient norm heatmaps for each run")
    print(f"  - energy_heatmap_k_*.png: Dyadic energy heatmaps for each run")
    print(f"  - model_k_*.pt: Trained model states")
    print(f"  - k_*/mlp_hdim*_layers*.json: Training logs with SVD data for each run")


if __name__ == "__main__":
    main()
