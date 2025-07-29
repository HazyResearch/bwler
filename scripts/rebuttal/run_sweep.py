#!/usr/bin/env python3
"""
Main script to run the complete PDE sweep.
This script:
1. Generates all job scripts
2. Provides instructions for submission
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
    else:
        print("✗ Failed")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def main():
    """Run the complete PDE sweep setup."""
    print("🚀 Setting up PDE sweep...")
    
    # Step 1: Generate all job scripts
    print("\n📋 Step 1: Generating job scripts")
    if not run_command("python scripts/rebuttal/sweep_manager.py", 
                      "Generating sweep manager"):
        print("Error: Failed to generate job scripts")
        sys.exit(1)
    
    # Step 2: Show summary
    print("\n📊 Summary:")
    print("Generated job scripts for:")
    print("  Problems: convection, reaction, wave, burgers, allen_cahn, poisson")
    print("  Methods: mlp (ssbroyden), bwler-hat, explicit bwler")
    print("  Total: 18 jobs (6 problems × 3 methods)")
    print("  Runtime: 1 hour (easy problems), 10 hours (hard problems)")
    
    # Step 3: Show next steps
    print("\n🎯 Next steps:")
    print("1. Review generated scripts in scripts/rebuttal/generated_jobs/")
    print("2. Submit all jobs:")
    print("   cd scripts/rebuttal/generated_jobs")
    print("   bash submit_all.sh")
    print("3. Monitor jobs:")
    print("   squeue -u $USER")
    print("4. Check results in plots/pdes/rebuttal/")

if __name__ == "__main__":
    main() 