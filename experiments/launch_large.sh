#!/bin/sh
# Start the large-scale evaluation (250, 500 and 1000 requests, four use cases) for both code
# versions. Nothing runs until you execute this script.
#
#   sh experiments/launch_large.sh                 # 3 seeds, 7 workers per version
#   SEEDS=1 WORKERS=4 sh experiments/launch_large.sh
#   BUDGET=55 sh experiments/launch_large.sh       # stop starting new runs after 55 minutes
#   VERSIONS="MOEA_2026" sh experiments/launch_large.sh
#   DRY=1 sh experiments/launch_large.sh           # only count the runs, start nothing
#
# Runs are ordered seed first and small workloads first, so stopping early leaves complete
# seeds. Every run is saved on its own and skipped on restart: run the same command again to
# resume. Progress: MOEA_DIR=<version> python experiments/run_moea.py large --seeds 3 --status
# Stop everything: pkill -f run_moea.py
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/.venvs/jnca-moea/bin/python}"
SEEDS="${SEEDS:-3}"
WORKERS="${WORKERS:-7}"
VERSIONS="${VERSIONS:-MOEA_2026 MOEA}"

# request files of the large workloads (deterministic, seed 0); skip if they exist
if [ ! -f services/types_large/vr1000.csv ] || [ ! -f services/types_large/ai250.csv ]; then
  "$PY" experiments/gen_requests.py --sizes 250 500 1000
fi

for v in $VERSIONS; do
  if [ -n "$DRY" ]; then
    MOEA_DIR=$v "$PY" experiments/run_moea.py large --seeds "$SEEDS" --dry-run | tail -1
    continue
  fi
  extra=""
  [ -n "$BUDGET" ] && extra="--time-budget $BUDGET"
  MOEA_DIR=$v nohup caffeinate -i "$PY" experiments/run_moea.py large --seeds "$SEEDS" --workers "$WORKERS" $extra \
    > "/tmp/launch_large_$v.log" 2>&1 &
  echo "started $v with $WORKERS workers (log /tmp/launch_large_$v.log)"
done
