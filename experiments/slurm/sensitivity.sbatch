#!/bin/bash
# MIP-gap / time-limit sweep of the MO-ILP (run_sensitivity.py), one array task per workload.
#   sbatch experiments/slurm/sensitivity.sbatch          (from the repository root)
# Each task runs 3 time limits x 6 gaps sequentially: at most (100+300+1000)*6 s of solver
# time plus model building, hence the 8 h limit. Completed runs are skipped on resubmission.
#SBATCH --job-name=moo-ilp-sens
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail

# --- adapt to the cluster -------------------------------------------------------------
# module load julia gurobi python
# export GUROBI_HOME=/path/to/gurobi1100/linux64
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"   # WLS or token-server license
# export JULIA_DEPOT_PATH=$HOME/.julia                              # precompiled on the login node
# source .venv/bin/activate
# ---------------------------------------------------------------------------------------

export GKSwstype=100                            # GR/Plots without a display
export JULIA_NUM_THREADS=1                      # parallelism is in Gurobi (Threads below)

USECASES=(ai iiot smartcity vr)
SIZES=(50 100)
UC=${USECASES[$((SLURM_ARRAY_TASK_ID / 2))]}
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "task $SLURM_ARRAY_TASK_ID: $UC$SIZE on $(hostname), $SLURM_CPUS_PER_TASK cpus"
python experiments/run_sensitivity.py --usecases "$UC" --sizes "$SIZE" --threads "$SLURM_CPUS_PER_TASK"
