#!/bin/sh
# Large-scale evaluation (250, 500 and 1000 requests, four use cases) with the improved MOEA (MOEA_2026).
# Nothing runs until you execute this script. It can be stopped and restarted at any time:
# every run is saved on its own, finished runs are skipped, and a killed run only loses itself.
#
#   sh experiments/launch_large.sh                        # 3 seeds, 12 workers, no deadline
#   UNTIL=14:30 sh experiments/launch_large.sh            # do not start a run that cannot finish by 14:30
#   BUDGET=90 sh experiments/launch_large.sh              # the same, 90 minutes from now
#   ESTIMATE=1 sh experiments/launch_large.sh             # print the estimated time left and exit
#   STATUS=1 sh experiments/launch_large.sh               # print the progress and exit
#   DRY=1 UNTIL=14:30 sh experiments/launch_large.sh      # list what would start / be skipped
#   VERSIONS="MOEA_2026 MOEA" WORKERS=6 sh experiments/launch_large.sh   # add the original decoder
#   SEEDS=1 sh experiments/launch_large.sh
#   USECASES="iiot" sh experiments/launch_large.sh        # only some use cases (ai iiot smartcity vr)
#
# With UNTIL or BUDGET, a worker skips runs whose estimated duration (times SAFETY, default 1.5)
# does not fit before the deadline and keeps picking shorter ones, so you can leave without
# losing a half-finished run. Stop at once with: sh experiments/stop_runs.sh
# Runs go seed first and small workloads first, so a partial result is a set of complete seeds.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/.venvs/jnca-moea/bin/python}"
SEEDS="${SEEDS:-3}"
WORKERS="${WORKERS:-12}"
VERSIONS="${VERSIONS:-MOEA_2026}"

# request files of the large workloads (deterministic, seed 0); skip if they exist
if [ ! -f services/types_large/vr1000.csv ] || [ ! -f services/types_large/ai250.csv ]; then
  "$PY" experiments/gen_requests.py --sizes 250 500 1000
fi

# the request files must not contain latency bounds that no plan can satisfy
"$PY" experiments/gen_requests.py --check > /dev/null || {
  echo "Impossible latency bounds in the request files. Regenerate them: python experiments/gen_requests.py"; exit 1; }

extra=""
[ -n "$UNTIL" ] && extra="$extra --until $UNTIL"
[ -n "$BUDGET" ] && extra="$extra --time-budget $BUDGET"
[ -n "$SAFETY" ] && extra="$extra --safety $SAFETY"
[ -n "$USECASES" ] && extra="$extra --usecases $USECASES"

for v in $VERSIONS; do
  if [ -n "$STATUS" ]; then
    MOEA_DIR=$v "$PY" experiments/run_moea.py large --seeds "$SEEDS" --status | head -5
  elif [ -n "$ESTIMATE" ]; then
    MOEA_DIR=$v "$PY" experiments/run_moea.py large --seeds "$SEEDS" --estimate --workers "$WORKERS" | head -3
  elif [ -n "$DRY" ]; then
    MOEA_DIR=$v "$PY" experiments/run_moea.py large --seeds "$SEEDS" --dry-run $extra | tail -3
  else
    MOEA_DIR=$v nohup caffeinate -i "$PY" experiments/run_moea.py large --seeds "$SEEDS" --workers "$WORKERS" $extra \
      > "/tmp/launch_large_$v.log" 2>&1 &
    echo "started $v with $WORKERS workers (log /tmp/launch_large_$v.log)"
  fi
done
