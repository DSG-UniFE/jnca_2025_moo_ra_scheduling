#!/bin/sh
# Stop all experiment workers at once, remove their lock files and show what is finished.
# A run that is in progress is lost and starts again on the next launch; finished runs are kept.
#   sh experiments/stop_runs.sh                 # stop and print the progress of the large-scale runs
#   PACK=1 sh experiments/stop_runs.sh          # also pack the results to ~/Desktop/jnca_results
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/.venvs/jnca-moea/bin/python}"
pkill -f run_moea.py
sleep 3
find experiments/results \( -name "*.lock" -o -name "*.tmp" \) -delete 2>/dev/null
echo "workers left: $(pgrep -f run_moea.py | wc -l)"
for v in ${VERSIONS:-MOEA_2026}; do
  MOEA_DIR=$v "$PY" experiments/run_moea.py large --seeds "${SEEDS:-3}" --status | head -4
done
if [ -n "$PACK" ]; then
  for v in ${VERSIONS:-MOEA_2026}; do MOEA_DIR=$v "$PY" experiments/run_moea.py report large > /dev/null 2>&1 || true; done
  "$PY" experiments/pack_results.py --out "$HOME/Desktop/jnca_results" --exp large | tail -3
  "$PY" experiments/pack_results.py --out "$HOME/Desktop/jnca_results" --exp large --full | tail -3
fi
