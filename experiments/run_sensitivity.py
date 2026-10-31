"""MIP-gap and time-limit sensitivity of the MO-ILP .

MO-ILP_Gpu_added.jl hard-codes the use case, the gap list and the 1000 s time
limit, and writes into results/usecase/. To leave that file and the paper's
results untouched, this script writes a patched copy of the Julia script to
experiments/results/sensitivity/<case>/ and runs it from there with every output
path redirected. Each substitution is checked to match exactly once. Julia runs
inside the run directory, so Gurobi's SolFiles of parallel runs do not collide.

    python experiments/run_sensitivity.py --usecases vr --sizes 50 100 \
        --gaps 0.0 0.05 0.1 0.25 0.5 0.75 --time-limits 100 300 1000 --dry-run
    python experiments/run_sensitivity.py            # defaults above, all use cases

Needs Julia with JuMP, Gurobi (licensed), MultiObjectiveAlgorithms, DataFrames, CSV and
BenchmarkTools. The figures of the original script are skipped (a stub replaces Plots, which
does not load on Julia 1.13 on macOS); pass --plots to keep them. --no-bridges creates the model
with add_bridges = false to lower the peak memory of the 100-request workloads; the model passed
to Gurobi is the same (same Model fingerprint in output.txt). --direct uses direct_model, which
does not keep JuMP's copy of the model and saves more memory, but passes the rows to Gurobi in
another order (different fingerprint), so time-limited runs can differ from the default ones: its
results go to experiments/results/sensitivity_direct/ and do not overwrite the others. --out
writes the results to another directory (e.g. runs made on another machine).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JL = os.path.join(ROOT, "MO-ILP_Gpu_added.jl")
OUT = os.path.join(ROOT, "experiments", "results", "sensitivity")
OUT_DIRECT = OUT + "_direct"


PLOTS_STUB = '''module Plots  # stand-in for the plotting package: the sweep only needs the numbers
export scatter3d, scatter3d!, savefig
scatter3d(args...; kwargs...) = nothing
scatter3d!(args...; kwargs...) = nothing
savefig(args...; kwargs...) = nothing
end
using .Plots'''


def patch(src, usecase, gap, tlimit, results_dir, plots=False, threads=0, no_bridges=False, direct=False):
    def sub(pattern, repl, text):
        new, n = re.subn(pattern, lambda m: repl, text, flags=re.M)
        if n != 1:
            sys.exit(f"patch failed: {pattern!r} matched {n} times (was MO-ILP_Gpu_added.jl edited?)")
        return new

    s = src
    if not plots:
        s = sub(r"^using Plots$", PLOTS_STUB, s)
    if direct:
        s = sub(r"model = Model\(Gurobi\.Optimizer\)", "model = direct_model(Gurobi.Optimizer())", s)
    elif no_bridges:
        s = sub(r"model = Model\(Gurobi\.Optimizer\)", "model = Model(Gurobi.Optimizer; add_bridges = false)", s)
    s = sub(r"^mip_gaps = \[0\.0\].*$", f"mip_gaps = [{gap}]", s)
    s = sub(r"TimeLimitSec\(\), 1000\)", f"TimeLimitSec(), {tlimit})", s)
    s = sub(r'^\s*#set_optimizer_attribute\(model, "Threads", 4\)\s*$',
            f'\tset_optimizer_attribute(model, "Threads", {threads})', s)
    s = sub(r'^usecase = "[a-z0-9]+".*$', f'usecase = "{usecase}"', s)
    s = sub(r'^results_path = .*$', f'results_path = "{results_dir}/"', s)
    s = sub(r'^latency_file = .*$', f'latency_file = "{ROOT}/AWS/latency.csv"', s)
    s = sub(r'^instance_file = .*$', f'instance_file = "{ROOT}/AWS/pricing.csv"', s)
    s = sub(r'^service_file = .*$', f'service_file = "{ROOT}/services/types/{usecase}.csv"', s)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usecases", nargs="+", default=["ai", "iiot", "smartcity", "vr"])
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    p.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75])
    p.add_argument("--time-limits", type=int, nargs="+", default=[100, 300, 1000])
    p.add_argument("--julia", default="julia")
    p.add_argument("--plots", action="store_true", help="keep the figures of the original script (needs a working Plots)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)),
                   help="Gurobi Threads (0 = Gurobi default); defaults to SLURM_CPUS_PER_TASK when set")
    p.add_argument("--no-bridges", action="store_true", help="create the model with add_bridges = false (less memory)")
    p.add_argument("--direct", action="store_true",
                   help="create the model with direct_model (least memory, other row order); results in sensitivity_direct/")
    p.add_argument("--out", help="results directory (default experiments/results/sensitivity, or sensitivity_direct with --direct)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    out = os.path.abspath(a.out) if a.out else OUT_DIRECT if a.direct else OUT
    src = open(JL).read()
    for uc in a.usecases:
        for size in a.sizes:
            case = f"{uc}{size}"
            for tl in a.time_limits:
                for gap in a.gaps:
                    d = os.path.join(out, case, f"tl{tl}_gap{gap}")
                    if os.path.exists(os.path.join(d, f"results_usecase_{case}.csv")):
                        print("skip (done):", d)
                        continue
                    print("run:", case, "time limit", tl, "gap", gap)
                    if a.dry_run:
                        patch(src, case, gap, tl, d, a.plots, a.threads, a.no_bridges, a.direct)  # still validates the substitutions
                        continue
                    os.makedirs(d, exist_ok=True)
                    script = os.path.join(d, "run.jl")
                    with open(script, "w") as f:
                        f.write(patch(src, case, gap, tl, d, a.plots, a.threads, a.no_bridges, a.direct))
                    with open(os.path.join(d, "output.txt"), "w") as log:
                        subprocess.run([a.julia, script], cwd=d, stdout=log, stderr=subprocess.STDOUT, check=False)


if __name__ == "__main__":
    main()
