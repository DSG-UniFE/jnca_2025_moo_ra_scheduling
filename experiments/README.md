# Script Experiments 

Setup (jMetalPy 1.7.0 is the version tested; 1.5.x and 1.8+ lack the integer operators
that `MOEA/solve.py` imports from `jmetal.operator`):

    pip install "jmetalpy==1.7.0" seaborn pandas numpy scipy dask

Run everything from the repository root.

| item | Command |
|---|---|
| Rerun of the paper evaluation (fixed decoder, `MOEA_2026/`) | `python experiments/run_moea.py paper` |
| Large scale | `python experiments/gen_requests.py` then `python experiments/run_moea.py large` |
| INFIT ablation | `python experiments/run_moea.py ablation` |
| Latency jitter | `python experiments/jitter_eval.py --source paper` (existing fronts, no new runs) |
| Gap and time-limit sweep | `python experiments/run_sensitivity.py` (needs Julia and Gurobi) |
| Metrics tables | `python experiments/run_moea.py report large` / `report ablation` |

Useful options: `--sizes`, `--usecases`, `--algos`, `--seeds N` (default 10),
`--evals` (default 50000), `--processes` (default 8), `--dry-run`. Every run is saved
to its own JSON file and skipped on restart, so jobs can be interrupted and resumed.

## Code versions

The runner uses `MOEA_2026/` (fixed decoder, see `MOEA_2026/README.md`) by default. `MOEA_DIR=MOEA`
runs the code of the submitted paper (three objectives only). The paper evaluation covers the
small, 50 and 100-request workloads with the 3-objective and the three 2-objective problems;
`paper --objective-sets f1f2f3 f1f2 f1f3 f2f3` selects them.

## Ablation variants

* `infit`: full INFIT (the paper's setting).
* `penalty`: both repair loops off, constraint violations visible to jMetal (negative values).
* `norepair`: repair off with violations stored as positive numbers, as the original code does.
  jMetal reads `constraint < 0` as a violation, so the algorithms ignore them here. Use it only
  as a diagnostic, not as the "penalty only" variant.

## Changes to existing code

`MOEA/moo_ra_4.py` got two optional constructor arguments, `repair=True` and
`penalize=False`, plus a guard on the debug print. The defaults reproduce the original
behavior exactly. `MO-ILP_Gpu_added.jl` is not modified; `run_sensitivity.py` patches a copy.

## Metrics

HV uses the reference point `1.1 * max` of the pooled solutions, IGD uses the pooled
solutions of all algorithms, variants and seeds of a workload as reference, and sparsity
follows `MOEA/plot_and_compare.py`. `--feasible-only` drops infeasible solutions first.
The Dominance-based Gap is added when `results/usecase/<workload>/objectives/objectives_gap_0.0.txt`
exists (it does for the 50 and 100-request workloads).

## Jitter model

Each (origin, data center) latency is multiplied by `exp(N(-s^2/2, s^2))` (mean one), drawn
independently per realization and shared by all solutions. `--sigma` sets `s`. The ILP fronts
are not covered: their assignments are in the Gurobi solution files, not in a variable vector.
