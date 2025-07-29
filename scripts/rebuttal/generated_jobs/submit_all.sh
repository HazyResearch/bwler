#!/bin/bash
# Master submit script for all PDE experiments

sbatch scripts/rebuttal/generated_jobs/convection_mlp_ssbroyden.sh
sbatch scripts/rebuttal/generated_jobs/convection_bwler_hat.sh
sbatch scripts/rebuttal/generated_jobs/convection_explicit_bwler.sh
sbatch scripts/rebuttal/generated_jobs/reaction_mlp_ssbroyden.sh
sbatch scripts/rebuttal/generated_jobs/reaction_bwler_hat.sh
sbatch scripts/rebuttal/generated_jobs/reaction_explicit_bwler.sh
sbatch scripts/rebuttal/generated_jobs/wave_mlp_ssbroyden.sh
sbatch scripts/rebuttal/generated_jobs/wave_bwler_hat.sh
sbatch scripts/rebuttal/generated_jobs/wave_explicit_bwler.sh
sbatch scripts/rebuttal/generated_jobs/burgers_mlp_ssbroyden.sh
sbatch scripts/rebuttal/generated_jobs/burgers_bwler_hat.sh
sbatch scripts/rebuttal/generated_jobs/burgers_explicit_bwler.sh
sbatch scripts/rebuttal/generated_jobs/allen_cahn_mlp_ssbroyden.sh
sbatch scripts/rebuttal/generated_jobs/allen_cahn_bwler_hat.sh
sbatch scripts/rebuttal/generated_jobs/allen_cahn_explicit_bwler.sh
sbatch scripts/rebuttal/generated_jobs/poisson_mlp_ssbroyden.sh
sbatch scripts/rebuttal/generated_jobs/poisson_bwler_hat.sh
sbatch scripts/rebuttal/generated_jobs/poisson_explicit_bwler.sh
