"""Run the MOEA experiments requested in the response letter.

    large     R3-5  large-scale evaluation (metaheuristics only, 3 objectives)
    ablation  R6-5  INFIT ablation (full INFIT vs. INFIT without repair)
    report          HV, IGD, sparsity, feasibility, time (and DG) from saved runs

Every run is written to its own JSON file and skipped if it already exists, so
an interrupted job can be restarted with the same command.

Examples (from the repository root):
    python experiments/gen_requests.py --sizes 250 500 1000
    python experiments/run_moea.py large --sizes 250 --usecases vr --seeds 1 --evals 2000 --dry-run
    python experiments/run_moea.py large
    python experiments/run_moea.py ablation
    python experiments/run_moea.py report large
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import freeze_support

import numpy as np

import algos
from algos import ALGORITHMS, ROOT, MOEA_DIR

USECASES = ["ai", "iiot", "smartcity", "vr"]
OUT = os.path.join(ROOT, "experiments", "results")
# Paper services files (50/100 requests) and the large files made by gen_requests.py
SERVICES = {"paper": os.path.join(ROOT, "services", "types"),
            "large": os.path.join(ROOT, "services", "types_large")}
# variant name -> MooRa4 kwargs.
#   infit    full INFIT (paper setting)
#   norepair repair loops off, violations stored as positive numbers as in the original
#            code, which jMetal does not treat as violations (the algorithms ignore them)
#   penalty  repair loops off, violations visible to jMetal (constrained dominance)
VARIANTS = {"infit": {"repair": True}, "norepair": {"repair": False},
            "penalty": {"repair": False, "penalize": True}}


def service_file(usecase, size):
    for d in (SERVICES["paper"], SERVICES["large"]):
        f = os.path.join(d, f"{usecase}{size}.csv")
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"no request file for {usecase}{size}; run gen_requests.py")


def run_one(problem_cls, sfile, kwargs, alg, seed, evals, processes, out_json):
    algos.seed_everything(seed)
    problem = problem_cls(sfile, **kwargs)
    algorithm = algos.build(alg, problem, evals=evals, processes=processes)
    t0 = time.time()
    algorithm.run()
    elapsed = time.time() - t0
    result = algorithm.result()
    from jmetal.util.solution import get_non_dominated_solutions
    front = get_non_dominated_solutions(result)
    feasible = lambda s: all(c >= 0 for c in s.constraints) and sum(abs(c) for c in s.constraints) == 0
    rec = {
        "algorithm": alg, "seed": seed, "evals": evals, "repair": kwargs.get("repair", True),
        "service_file": os.path.basename(sfile), "num_requests": problem.num_requests,
        "num_replicas": problem.num_replicas, "time_s": elapsed,
        "final_population_size": len(result),
        "feasible_share_population": float(np.mean([feasible(s) for s in result])) if result else 0.0,
        "feasible_share_front": float(np.mean([feasible(s) for s in front])) if front else 0.0,
        "objectives": [list(map(float, s.objectives)) for s in front],
        "constraints": [list(map(float, s.constraints)) for s in front],
        "variables": [list(map(int, s.variables)) for s in front],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rec, f)
    return rec


def run_grid(exp, variants, sizes, usecases, algs, seeds, evals, processes, dry):
    from moo_ra_4 import MooRa4
    n = 0
    for variant in variants:
        for uc in usecases:
            for size in sizes:
                for alg in algs:
                    for seed in range(seeds):
                        out = os.path.join(OUT, exp, variant, f"{uc}{size}", f"seed{seed}", f"{alg}.json")
                        n += 1
                        if os.path.exists(out):
                            print("skip (done):", out)
                            continue
                        if dry:
                            print("would run:", exp, variant, uc, size, alg, "seed", seed)
                            continue
                        print(f"[{n}] {exp} {variant} {uc}{size} {alg} seed={seed}", flush=True)
                        rec = run_one(MooRa4, service_file(uc, size), VARIANTS[variant], alg, seed, evals, processes, out)
                        print(f"    time={rec['time_s']:.1f}s front={len(rec['objectives'])} "
                              f"feasible(pop)={rec['feasible_share_population']:.2f}", flush=True)
    print("configurations:", n)


# ---------------------------------------------------------------- report
def load_runs(exp):
    runs = []
    for f in glob.glob(os.path.join(OUT, exp, "*", "*", "seed*", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        parts = f.split(os.sep)
        r["variant"], r["workload"] = parts[-4], parts[-3]
        runs.append(r)
    return runs


def sparsity(objs):
    if len(objs) <= 1:
        return 0.0
    objs = np.array(objs)
    s = 0.0
    for j in range(objs.shape[1]):
        o = objs[np.argsort(objs[:, j])]
        s += float(np.sum((o[:-1, j] - o[1:, j]) ** 2))
    return s / (len(objs) - 1)


def report(exp, feasible_only, ilp_dir):
    import pandas as pd
    from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance
    sys.path.insert(0, MOEA_DIR)
    from common import dominance_based_gap

    runs = load_runs(exp)
    if not runs:
        sys.exit(f"no runs found under {os.path.join(OUT, exp)}")
    rows = []
    for wl in sorted({r["workload"] for r in runs}):
        group = [r for r in runs if r["workload"] == wl]

        def objs_of(r):
            o = r["objectives"]
            if feasible_only:
                o = [x for x, c in zip(o, r["constraints"]) if sum(abs(v) for v in c) == 0]
            return o

        # Reference set: ensemble of every solution of every algorithm/variant/seed
        # for this workload (same choice as MOEA/plot_and_compare.py).
        pool = np.array([x for r in group for x in objs_of(r)])
        if len(pool) == 0:
            continue
        hv = HyperVolume((pool.max(axis=0) * 1.1).tolist())
        igd = InvertedGenerationalDistance(pool)
        ilp = None
        uc = wl.rstrip("0123456789")
        size = wl[len(uc):]
        ilp_file = os.path.join(ilp_dir, f"{uc}{size}", "objectives", "objectives_gap_0.0.txt") if ilp_dir else None
        if ilp_file and os.path.exists(ilp_file):
            ilp = np.loadtxt(ilp_file, ndmin=2)
        for r in group:
            o = np.array(objs_of(r))
            row = {"workload": wl, "variant": r["variant"], "algorithm": r["algorithm"], "seed": r["seed"],
                   "time_s": r["time_s"], "front_size": len(o),
                   "feasible_pop": r["feasible_share_population"], "feasible_front": r["feasible_share_front"],
                   "hv": hv.compute(o) if len(o) else 0.0,
                   "igd": igd.compute(o) if len(o) else np.nan,
                   "sparsity": sparsity(o)}
            if ilp is not None and len(o):
                g = dominance_based_gap(o, ilp)
                row["dg_sum"] = float(sum(g)) if g is not None else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    raw = os.path.join(OUT, "reports", f"{exp}_per_run.csv")
    df.to_csv(raw, index=False)
    metrics = [c for c in ["time_s", "front_size", "feasible_pop", "feasible_front", "hv", "igd", "sparsity", "dg_sum"] if c in df]
    summ = df.groupby(["workload", "variant", "algorithm"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    out = os.path.join(OUT, "reports", f"{exp}_summary.csv")
    summ.to_csv(out)
    print(summ.round(4).to_string())
    print("\nwrote", raw, "and", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, sizes, variants in (("large", [250, 500, 1000], ["infit"]), ("ablation", [50, 100], ["infit", "penalty"])):
        s = sub.add_parser(name)
        s.add_argument("--sizes", type=int, nargs="+", default=sizes)
        s.add_argument("--usecases", nargs="+", default=USECASES, choices=USECASES)
        s.add_argument("--algos", nargs="+", default=ALGORITHMS, choices=ALGORITHMS)
        s.add_argument("--variants", nargs="+", default=variants, choices=list(VARIANTS))
        s.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 per configuration")
        s.add_argument("--evals", type=int, default=50_000)
        s.add_argument("--processes", type=int, default=8, help="1 disables multiprocessing")
        s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("exp", choices=["large", "ablation"])
    r.add_argument("--feasible-only", action="store_true", help="drop infeasible solutions before computing metrics")
    r.add_argument("--ilp-dir", default=os.path.join(ROOT, "results", "usecase"),
                   help="folder with <usecase>/objectives/objectives_gap_0.0.txt, used for the Dominance-based Gap")
    a = p.parse_args()

    os.chdir(MOEA_DIR)  # MooRa4 loads ./cnsm_data/*.csv relative to MOEA/
    if a.cmd == "report":
        report(a.exp, a.feasible_only, a.ilp_dir)
    else:
        run_grid(a.cmd, a.variants, a.sizes, a.usecases, a.algos, a.seeds, a.evals, a.processes, a.dry_run)


if __name__ == "__main__":
    freeze_support()
    main()
