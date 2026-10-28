"""MIP-gap and time-limit sensitivity of the MO-ILP .

MO-ILP_Gpu_added.jl hard-codes the use case, the gap list and the 1000 s time
limit, and writes into results/usecase/. To leave that file and the paper's
results untouched, this script writes a patched copy of the Julia script to
experiments/results/sensitivity/<case>/ and runs it from there with every output
path redirected. Each substitution is checked to match exactly once.

    python experiments/run_sensitivity.py --usecases vr --sizes 50 100 \
        --gaps 0.0 0.05 0.1 0.25 0.5 0.75 --time-limits 100 300 1000 --dry-run
    python experiments/run_sensitivity.py            # defaults above, all use cases

Needs Julia with JuMP, Gurobi (licensed), MultiObjectiveAlgorithms, DataFrames, CSV,
Plots and BenchmarkTools, as for the original script.
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


def patch(src, usecase, gap, tlimit, results_dir):
    def sub(pattern, repl, text):
        new, n = re.subn(pattern, lambda m: repl, text, flags=re.M)
        if n != 1:
            sys.exit(f"patch failed: {pattern!r} matched {n} times (was MO-ILP_Gpu_added.jl edited?)")
        return new

    s = src
    s = sub(r"^mip_gaps = \[0\.0\].*$", f"mip_gaps = [{gap}]", s)
    s = sub(r"TimeLimitSec\(\), 1000\)", f"TimeLimitSec(), {tlimit})", s)
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
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    src = open(JL).read()
    for uc in a.usecases:
        for size in a.sizes:
            case = f"{uc}{size}"
            for tl in a.time_limits:
                for gap in a.gaps:
                    d = os.path.join(OUT, case, f"tl{tl}_gap{gap}")
                    if os.path.exists(os.path.join(d, f"results_usecase_{case}.csv")):
                        print("skip (done):", d)
                        continue
                    print("run:", case, "time limit", tl, "gap", gap)
                    if a.dry_run:
                        patch(src, case, gap, tl, d)  # still validates the substitutions
                        continue
                    os.makedirs(d, exist_ok=True)
                    script = os.path.join(d, "run.jl")
                    with open(script, "w") as f:
                        f.write(patch(src, case, gap, tl, d))
                    with open(os.path.join(d, "output.txt"), "w") as log:
                        subprocess.run([a.julia, script], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)


if __name__ == "__main__":
    main()
