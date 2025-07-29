#!/usr/bin/env python3
"""
Sweep Manager for PDE Experiments
Generates job commands for different methods and problems.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path

@dataclass
class ProblemConfig:
    """Configuration for a specific PDE problem."""
    name: str
    base_args: Dict[str, Any]
    model_params: Dict[str, Dict[str, Any]]  # method -> params

@dataclass
class JobCommand:
    """Represents a single job command."""
    problem: str
    method: str
    model: str
    command: str
    job_name: str
    save_dir: str

class SweepManager:
    def __init__(self, base_dir: str = "scripts/rebuttal"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "generated_jobs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Define problem configurations
        self.problems = self._define_problems()
        
    def _define_problems(self) -> Dict[str, ProblemConfig]:
        """Define configurations for each problem."""
        return {
            "convection": ProblemConfig(
                name="convection",
                base_args={
                    "c": 40,
                    "n_epochs": 20000,
                    "eval_every": 50,
                    "sample_type": "standard",
                },
                model_params={
                    "mlp_ssbroyden": {
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                    },
                    "bwler_hat": {
                        "n_t": 81,
                        "n_x": 80,
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                        "mlpinterp_random_collocation": False,  # Default to fixed BWLer nodes
                    },
                    "explicit_bwler": {
                        "n_t": 81,
                        "n_x": 80,
                    },
                }
            ),
            "reaction": ProblemConfig(
                name="reaction",
                base_args={
                    "rho": 5,
                    "n_epochs": 20000,
                    "eval_every": 50,
                    "sample_type": "standard",
                },
                model_params={
                    "mlp_ssbroyden": {
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                    },
                    "bwler_hat": {
                        "n_t": 41,
                        "n_x": 41,
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                        "mlpinterp_random_collocation": False,  # Default to fixed BWLer nodes
                    },
                    "explicit_bwler": {
                        "n_t": 81,
                        "n_x": 81,
                    },
                }
            ),
            "wave": ProblemConfig(
                name="wave",
                base_args={
                    "c": 2,
                    "beta": 5,
                    "n_epochs": 20000,
                    "eval_every": 50,
                    "sample_type": "standard",
                },
                model_params={
                    "mlp_ssbroyden": {
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                    },
                    "bwler_hat": {
                        "n_t": 41,
                        "n_x": 41,
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                        "mlpinterp_random_collocation": False,  # Default to fixed BWLer nodes
                    },
                    "explicit_bwler": {
                        "n_t": 41,
                        "n_x": 41,
                    },
                }
            ),
            "burgers": ProblemConfig(
                name="burgers",
                base_args={
                    "n_epochs": 1000000,
                    "eval_every": 50,
                    "sample_type": "standard",
                },
                model_params={
                    "mlp_ssbroyden": {
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                    },
                    "bwler_hat": {
                        "n_t": 321,
                        "n_x": 321,
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                        "mlpinterp_random_collocation": False,  # Default to fixed BWLer nodes
                    },
                    "explicit_bwler": {
                        "n_t": 161,
                        "n_x": 161,
                    },
                }
            ),
            "allen_cahn": ProblemConfig(
                name="allen_cahn",
                base_args={
                    "n_epochs": 1000000,
                    "eval_every": 50,
                    "sample_type": "standard",
                },
                model_params={
                    "mlp_ssbroyden": {
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                    },
                    "bwler_hat": {
                        "n_t": 321,
                        "n_x": 321,
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                        "mlpinterp_random_collocation": False,  # Default to fixed BWLer nodes
                    },
                    "explicit_bwler": {
                        "n_t": 161,
                        "n_x": 161,
                    },
                }
            ),
            "poisson": ProblemConfig(
                name="poisson",
                base_args={
                    "n_epochs": 1000000,
                    "eval_every": 50,
                },
                model_params={
                    "mlp_ssbroyden": {
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                    },
                    "bwler_hat": {
                        "n_x": 51,
                        "n_y": 51,
                        "n_layers": 3,
                        "hidden_dim": 32,
                        "activation": "tanh",
                        "mlpinterp_random_collocation": False,  # Default to fixed BWLer nodes
                    },
                    "explicit_bwler": {
                        "n_x": 51,
                        "n_y": 51,
                    },
                }
            ),
        }
    
    def _build_command(self, problem: str, method: str, model: str, 
                      base_args: Dict, model_params: Dict) -> JobCommand:
        """Build a job command for a specific configuration."""
        
        # Map method to actual optimizer and model type
        method_mapping = {
            "mlp_ssbroyden": ("ssbroyden", "mlp"),
            "bwler_hat": ("ssbroyden", "mlpinterp"),
            "explicit_bwler": ("ssbroyden", "polynomial"),
        }
        
        optimizer, model_type = method_mapping[method]
        
        # Determine the module path based on problem type
        if problem in ["burgers", "poisson", "allen_cahn"]:
            module_path = f"src.experiments.pdes.benchmarks.{problem}"
        else:
            module_path = f"src.experiments.pdes.simple.{problem}"
        
        # Build command arguments
        cmd_parts = [
            "CUDA_VISIBLE_DEVICES=0",
            f"python -m {module_path}",
        ]
        
        # Add base arguments
        for key, value in base_args.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key} {value}")
        
        # Add model-specific arguments
        for key, value in model_params.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key} {value}")
        
        # Add method and model
        cmd_parts.extend([f"--method {optimizer}", f"--model {model_type}"])
        
        command = " ".join(cmd_parts)
        
        # Generate job name and save directory
        job_name = f"{problem}_{method}"
        save_dir = f"plots/pdes/rebuttal/{problem}/{method}"
        
        return JobCommand(
            problem=problem,
            method=method,
            model=model_type,
            command=command,
            job_name=job_name,
            save_dir=save_dir
        )
    
    def generate_all_jobs(self) -> List[JobCommand]:
        """Generate all job commands."""
        jobs = []
        
        for problem_name, problem_config in self.problems.items():
            for method_name, model_params in problem_config.model_params.items():
                job = self._build_command(
                    problem_name, method_name, method_name,
                    problem_config.base_args, model_params
                )
                jobs.append(job)
        
        return jobs
    
    def generate_slurm_scripts(self):
        """Generate individual SLURM scripts for each job."""
        jobs = self.generate_all_jobs()
        
        # Create a master submit script
        master_script = self.output_dir / "submit_all.sh"
        
        with open(master_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Master submit script for all PDE experiments\n\n")
            
            for job in jobs:
                # Create individual job script
                job_script = self.output_dir / f"{job.job_name}.sh"
                
                # Determine runtime based on problem difficulty
                easy_problems = ["convection", "reaction", "wave"]
                if job.problem in easy_problems:
                    runtime = "1:00:00"
                else:
                    runtime = "10:00:00"
                
                with open(job_script, 'w') as job_f:
                    job_f.write(f"""#!/bin/bash -l
#SBATCH --time={runtime}
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m1266
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o slurm_{job.job_name}.out
#SBATCH --job-name {job.job_name}

conda activate /pscratch/sd/j/jwl50/bwler/.pyenv

{job.command}
""")
                
                # Add to master script with full path
                f.write(f"sbatch {job_script}\n")
        
        # Make scripts executable
        os.chmod(master_script, 0o755)
        for job in jobs:
            job_script = self.output_dir / f"{job.job_name}.sh"
            os.chmod(job_script, 0o755)
        
        print(f"Generated {len(jobs)} job scripts in {self.output_dir}")
        print(f"Master submit script: {master_script}")
        
        return jobs
    
    def generate_job_list(self) -> str:
        """Generate a simple list of all jobs for manual submission."""
        jobs = self.generate_all_jobs()
        
        output = []
        for job in jobs:
            output.append(f"# {job.job_name}")
            output.append(f"# {job.save_dir}")
            output.append(job.command)
            output.append("")
        
        return "\n".join(output)

def main():
    """Generate all job scripts."""
    manager = SweepManager()
    
    # Generate SLURM scripts
    jobs = manager.generate_slurm_scripts()
    
    # Also save job list for reference
    job_list_file = manager.output_dir / "job_list.txt"
    with open(job_list_file, 'w') as f:
        f.write(manager.generate_job_list())
    
    print(f"\nJob list saved to: {job_list_file}")
    print(f"\nTotal jobs: {len(jobs)}")
    print("\nTo submit all jobs:")
    print(f"cd {manager.output_dir}")
    print("bash submit_all.sh")

if __name__ == "__main__":
    main() 