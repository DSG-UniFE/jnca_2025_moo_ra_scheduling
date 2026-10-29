#!/bin/bash
# Paper evaluation of the MOEAs (run_moea.py paper), one array task per use case and
# objective set (180 runs each: 3 sizes x 6 algorithms x 10 seeds), run by
# SLURM_CPUS_PER_TASK workers that share the grid. Completed runs are skipped on resubmission.
#   sbatch experiments/slurm/moea_paper.sbatch           (from the repository root)
#SBATCH --job-name=moea-paper
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/slurm/logs/%x_%A_%a.out
#SBATCH --partition=cpu_amd_zen4_2x48

set -euo pipefail

# --- adapt to the cluster -------------------------------------------------------------
# module load python
source .venv/bin/activate # needs jmetalpy==1.7.0
# ---------------------------------------------------------------------------------------

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 # one core per worker
export MPLBACKEND=Agg

USECASES=(ai iiot smartcity vr)
OSETS=(f1f2f3 f1f2 f1f3 f2f3)
UC=${USECASES[$((SLURM_ARRAY_TASK_ID / 4))]}
OSET=${OSETS[$((SLURM_ARRAY_TASK_ID % 4))]}

echo "task $SLURM_ARRAY_TASK_ID: $UC $OSET on $(hostname), $SLURM_CPUS_PER_TASK cpus"
python experiments/run_moea.py paper --usecases "$UC" --objective-sets "$OSET" \
  --workers "$SLURM_CPUS_PER_TASK" --processes 1
