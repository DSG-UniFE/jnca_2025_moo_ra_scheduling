#!/bin/bash
# Large-scale evaluation of the MOEAs (run_moea.py large: 250, 500 and 1000 requests) with the
# fixed decoder (MOEA_2026), one array task per use case.
# Each task has 54 runs (3 sizes x 6 algorithms x 3 seeds) shared by SLURM_CPUS_PER_TASK workers.
#   sbatch experiments/slurm/large.sbatch                (from the repository root)
#   sbatch --array=0 experiments/slurm/large.sbatch      (ai only, as a first test)
#   SEEDS=10 sbatch experiments/slurm/large.sbatch       (more seeds; finished runs are skipped)
# Estimate with 8 workers: about 2 h per task (6 to 7 h with 10 seeds), longest single run
# about 1 h (1000-request MSPSO). Workers do not start a run that is not expected to finish within
# the time limit, so a task that runs out of time loses no half-done run: resubmit to finish it.
#SBATCH --job-name=moea-large
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out
#SBATCH --partition=cpu_amd_zen4

set -euo pipefail

# --- adapt to the cluster -------------------------------------------------------------
# module load python
source .venv/bin/activate # needs jmetalpy==1.7.0
# ---------------------------------------------------------------------------------------

[ -f experiments/run_moea.py ] || { echo "submit from the repository root"; exit 1; }
[ -f services/types_large/vr1000.csv ] || { echo "missing services/types_large/: run experiments/gen_requests.py"; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 # one core per worker
export MPLBACKEND=Agg

USECASES=(ai iiot smartcity vr)
UC=${USECASES[$SLURM_ARRAY_TASK_ID]}
export MOEA_DIR=MOEA_2026
SEEDS=${SEEDS:-3}

# minutes left in the allocation, minus a margin for writing the last results
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L) # [D-]HH:MM:SS or MM:SS
days=0; [[ $left == *-* ]] && { days=${left%%-*}; left=${left#*-}; }
IFS=: read -r -a p <<< "$left"
while [ ${#p[@]} -lt 3 ]; do p=(0 "${p[@]}"); done
BUDGET=$(( days * 1440 + 10#${p[0]} * 60 + 10#${p[1]} - 10 ))
[ "$BUDGET" -gt 0 ] || { echo "time limit too short"; exit 1; }

echo "task $SLURM_ARRAY_TASK_ID: $UC $MOEA_DIR seeds=$SEEDS on $(hostname), $SLURM_CPUS_PER_TASK cpus, budget ${BUDGET} min"
python -u experiments/run_moea.py large --usecases "$UC" --seeds "$SEEDS" \
  --workers "$SLURM_CPUS_PER_TASK" --processes 1 --time-budget "$BUDGET"
