#!/bin/bash
# MIP-gap / time-limit sweep of the MO-ILP (run_sensitivity.py), one array task per workload.
#   sbatch experiments/slurm/sensitivity.sbatch          (from the repository root)
#   sbatch --array=0 experiments/slurm/sensitivity.sbatch   (ai50 only, as a first test)
# Each task runs 3 time limits x 6 gaps sequentially: at most (100+300+1000)*6 s of solver
# time plus model building, hence the 8 h limit. Completed runs are skipped on resubmission.
#SBATCH --job-name=moo-ilp-sens
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out
#SBATCH --partition=cpu_amd_zen4

set -euo pipefail

# --- adapt to the cluster -------------------------------------------------------------
# module load julia
# export GUROBI_HOME=/path/to/gurobi1200/linux64
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}" # WLS or token-server license
# export JULIA_DEPOT_PATH=$HOME/.julia                              # precompiled on the login node
source .venv/bin/activate
# ---------------------------------------------------------------------------------------

# fail here rather than in each of the 18 runs (their errors only end up in output.txt)
[ -f experiments/run_sensitivity.py ] || { echo "submit from the repository root"; exit 1; }
command -v julia >/dev/null || { echo "julia not in PATH (module load julia?)"; exit 1; }
[ -f "$GRB_LICENSE_FILE" ] || { echo "Gurobi license not found: $GRB_LICENSE_FILE"; exit 1; }

export JULIA_NUM_THREADS=1 # parallelism is in Gurobi (Threads below)

USECASES=(ai iiot smartcity vr)
SIZES=(50 100)
UC=${USECASES[$((SLURM_ARRAY_TASK_ID / 2))]}
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "task $SLURM_ARRAY_TASK_ID: $UC$SIZE on $(hostname), $SLURM_CPUS_PER_TASK cpus"
python -u experiments/run_sensitivity.py --usecases "$UC" --sizes "$SIZE" --threads "$SLURM_CPUS_PER_TASK"
