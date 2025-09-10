#!/usr/bin/env python3
"""
Script to run all MLP derivative sweeps directly on GPU without SLURM.
Supports running all Adam sweeps or all SSBroyden sweeps via command line arguments.

Usage:
    # Run all Adam sweeps
    python scripts/interpolation/run_all_mlp_sweeps.py --optimizer adam

    # Run all SSBroyden sweeps
    python scripts/interpolation/run_all_mlp_sweeps.py --optimizer ssbroyden

    # Use CPU instead of GPU
    python scripts/interpolation/run_all_mlp_sweeps.py --optimizer adam --device cpu

    # Change save directory
    python scripts/interpolation/run_all_mlp_sweeps.py --optimizer adam --save_root /path/to/results

    # Dry run (print commands without executing)
    python scripts/interpolation/run_all_mlp_sweeps.py --optimizer adam --dry_run

Configuration:
    - MLP layers: [2, 3]
    - Hidden dimensions: [32, 64, 128]
    - Derivative points: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    - Adam: 10,000 epochs, log every 1,000 epochs
    - SSBroyden: 1,000 epochs, log every 100 epochs
"""
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all MLP derivative sweeps directly on GPU"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["adam", "ssbroyden"],
        help="Optimizer to use for all sweeps",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/pscratch/sd/j/jwl50/bwler/plots/interpolation",
        help="Root directory to save results",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without executing"
    )
    return parser.parse_args()


def get_sweep_configs():
    """Get the MLP configurations and derivative points from the bash script."""
    # MLP configurations to test (from the bash script)
    layers_list = [2, 3]
    hdim_list = [32, 64, 128]

    # Derivative points to sweep (exponential spacing)
    deriv_points_list = [0]

    return layers_list, hdim_list, deriv_points_list


def get_optimizer_params(optimizer):
    """Get optimizer-specific training parameters."""
    if optimizer.lower() == "ssbroyden":
        n_epochs = 10000  # SSBroyden: 10000 epochs for full training
        eval_every = 1000  # SSBroyden: log every 1000 epochs
    else:  # adam
        n_epochs = 20000  # Adam: 20000 epochs for full training
        eval_every = 1000  # Adam: log every 1000 epochs

    return n_epochs, eval_every


def run_single_sweep(layers, hdim, optimizer, device, save_root, dry_run=False):
    """Run a single MLP derivative sweep."""
    # Get optimizer-specific parameters
    n_epochs, eval_every = get_optimizer_params(optimizer)

    # Configuration
    target = "sine"
    seed = 0
    n_train_0th = 100  # points for value loss
    n_train_1st = 100  # points for derivative loss (will be swept over)
    n_test = 1000
    deriv_alpha = 1.0
    deriv_beta = 1.0

    # Create save directory
    config_name = f"layers{layers}_hdim{hdim}"
    save_dir = f"{save_root}/mlp_size_sweep_{optimizer.lower()}/{config_name}"

    # Build command
    cmd = [
        "python",
        "scripts/interpolation/run_mlp_deriv_sweep.py",
        "--target",
        target,
        "--device",
        device,
        "--seed",
        str(seed),
        "--n_train_0th",
        str(n_train_0th),
        "--n_train_1st",
        str(n_train_1st),
        "--n_test",
        str(n_test),
        "--n_epochs",
        str(n_epochs),
        "--eval_every",
        str(eval_every),
        "--save_dir",
        save_dir,
        "--n_layers",
        str(layers),
        "--hidden_dim",
        str(hdim),
        "--deriv_alpha",
        str(deriv_alpha),
        "--deriv_beta",
        str(deriv_beta),
        "--optimizer",
        optimizer.lower(),
    ]

    print(f"\n{'='*80}")
    print(f"Running MLP sweep: {config_name} with {optimizer}")
    print(f"Save directory: {save_dir}")
    print(f"Epochs: {n_epochs}, Log every: {eval_every}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")

    if dry_run:
        print("DRY RUN - Command would be executed")
        return

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        print(
            f"\n✅ Completed {config_name} with {optimizer} in {end_time - start_time:.1f} seconds"
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed {config_name} with {optimizer}: {e}")
        return False

    return True


def main():
    args = parse_args()

    # Get configurations
    layers_list, hdim_list, deriv_points_list = get_sweep_configs()

    print(f"Running all {args.optimizer.upper()} sweeps directly on {args.device}")
    print(
        f"MLP configurations: {len(layers_list)} layers × {len(hdim_list)} hidden dims = {len(layers_list) * len(hdim_list)} total"
    )
    print(f"Derivative points to sweep: {deriv_points_list}")
    print(f"Save root: {args.save_root}")

    if args.dry_run:
        print("\nDRY RUN MODE - No commands will be executed")

    # Track results
    total_configs = len(layers_list) * len(hdim_list)
    successful_runs = 0
    failed_runs = 0

    # Run all sweeps
    for layers in layers_list:
        for hdim in hdim_list:
            success = run_single_sweep(
                layers=layers,
                hdim=hdim,
                optimizer=args.optimizer,
                device=args.device,
                save_root=args.save_root,
                dry_run=args.dry_run,
            )

            if success is None:  # dry run
                continue
            elif success:
                successful_runs += 1
            else:
                failed_runs += 1

    # Summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Total configurations: {total_configs}")
    if not args.dry_run:
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {failed_runs}")
        print(f"Success rate: {successful_runs/total_configs*100:.1f}%")

    print(
        f"\nResults saved to: {args.save_root}/mlp_size_sweep_{args.optimizer.lower()}/"
    )
    print("Each configuration creates its own subdirectory with:")
    print("  - deriv_sweep_results.json: All numerical results")
    print("  - deriv_sweep_l2re.png: L2RE vs n_train_1st plot")
    print("  - weight_norms_n1st_*.png: Weight norm plots for each run")
    print("  - solution_n1st_*.png: Custom solution plots for each run")
    print("  - model_n1st_*.pt: Trained model states")


if __name__ == "__main__":
    main()
