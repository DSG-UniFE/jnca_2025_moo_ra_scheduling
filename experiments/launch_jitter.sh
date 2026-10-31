#!/bin/sh
# Latency jitter analysis for the improved MOEA (MOEA_2026). Nothing runs until you execute this script.
#
#   sh experiments/launch_jitter.sh            # run the analysis
#   DRY=1 sh experiments/launch_jitter.sh      # only list what would be evaluated
#   SIGMA="0.1 0.2" REAL=500 sh experiments/launch_jitter.sh
#   OLD=1 sh experiments/launch_jitter.sh      # also the original decoder (ablation fronts and the paper's fronts)
#
# MOEA_2026: the 3-objective INFIT fronts of the paper rerun (experiments/results/MOEA_2026/paper),
#            3 seeds, 12 workloads (small, 50 and 100 requests)
# Results: experiments/results/MOEA_2026/reports/jitter_paper.csv and jitter_paper_summary.csv
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/.venvs/jnca-moea/bin/python}"
SIGMA="${SIGMA:-0.1 0.2 0.4}"
REAL="${REAL:-1000}"
extra=""
[ -n "$DRY" ] && extra="--dry-run"

run() {  # run <version> <args...>
  v=$1; shift
  MOEA_DIR=$v "$PY" experiments/jitter_eval.py --sigma $SIGMA --realizations "$REAL" $extra "$@"
}

if [ -n "$DRY" ]; then
  run MOEA_2026 --source json --exp paper
  if [ -n "$OLD" ]; then
    run MOEA --source json --exp ablation
    run MOEA --source paper
  fi
else
  if [ -n "$OLD" ]; then
    ( run MOEA --source json --exp ablation > /tmp/jitter_MOEA_ablation.log 2>&1
      run MOEA --source paper > /tmp/jitter_MOEA_paper.log 2>&1 ) &
  fi
  run MOEA_2026 --source json --exp paper > /tmp/jitter_MOEA_2026.log 2>&1
  wait
  echo "done; logs in /tmp/jitter_*.log"
fi
