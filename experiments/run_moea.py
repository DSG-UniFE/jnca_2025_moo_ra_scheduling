"""Run the MOEA experiments with the fixed decoder in MOEA_2026/.

    paper     rerun of the paper evaluation (small, 50 and 100 requests; 3-objective and the
              three 2-objective problems) with the fixed code
    large     large-scale evaluation (metaheuristics only, 3 objectives)
    ablation  INFIT ablation (full INFIT vs. INFIT without repair)
    report    HV, IGD, sparsity, feasibility, time (and DG) from saved runs

The code version is chosen with the environment variable MOEA_DIR (default MOEA_2026;
MOEA_DIR=MOEA runs the code of the paper, three objectives only).

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
from algos import ALGORITHMS, ROOT, MOEA_DIR, MOEA_NAME

USECASES = ["ai", "iiot", "smartcity", "vr"]
OUT = algos.RESULTS
# Paper services files (50/100 requests) and the large files made by gen_requests.py
SERVICES = {"paper": os.path.join(ROOT, "services", "types"),
            "large": os.path.join(ROOT, "services", "types_large")}
# variant name -> MooRa4 kwargs.
#   infit    full INFIT (paper setting)
#   norepair repair loops off, violations stored as positive numbers as in the original
#            code, which jMetal does not treat as violations (the algorithms ignore them)
#   penalty  repair loops off, violations visible to jMetal (constrained dominance)
VARIANTS = {"infit": {"repair": True}, "norepair": {"repair": False, "penalize": False},
            "penalty": {"repair": False, "penalize": True}}
# objective sets of the paper: indexes into (f1, f2, f3)
OBJECTIVE_SETS = {"f1f2f3": (0, 1, 2), "f1f2": (0, 1), "f1f3": (0, 2), "f2f3": (1, 2)}


def service_file(usecase, size):
    for d in (SERVICES["paper"], SERVICES["large"]):
        f = os.path.join(d, f"{usecase}{size}.csv")  # size "" is the small workload (e.g. ai.csv)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"no request file for {usecase}{size}; run gen_requests.py")


def run_one(problem_cls, sfile, kwargs, alg, seed, evals, processes, out_json, objectives=(0, 1, 2)):
    algos.seed_everything(seed)
    kwargs = dict(kwargs)
    if MOEA_NAME == "MOEA":  # the original class has no objective selection
        if tuple(objectives) != (0, 1, 2):
            raise SystemExit("the original MOEA folder only supports the three-objective problem")
    else:
        kwargs["objectives"] = tuple(objectives)
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
        "service_file": os.path.basename(sfile), "objectives": list(objectives), "moea": MOEA_NAME, "num_requests": problem.num_requests,
        "num_replicas": problem.num_replicas, "time_s": elapsed,
        "final_population_size": len(result),
        "feasible_share_population": float(np.mean([feasible(s) for s in result])) if result else 0.0,
        "feasible_share_front": float(np.mean([feasible(s) for s in front])) if front else 0.0,
        "objectives": [list(map(float, s.objectives)) for s in front],
        "constraints": [list(map(float, s.constraints)) for s in front],
        "variables": [list(map(int, s.variables)) for s in front],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out_json)  # a killed job never leaves a half-written result
    return rec


def build_tasks(exp, variants, sizes, usecases, algs, seeds, objective_sets):
    """All runs in priority order: seed first, so that stopping early leaves complete seeds,
    then small workloads before large ones."""
    tasks = []
    for seed in range(seeds):
        for size in sizes:
            size = "" if size == "small" else size
            for uc in usecases:
                for oset in objective_sets:
                    for variant in variants:
                        for alg in algs:
                            wl = f"{uc}{size}" if exp != "paper" else f"{uc}{size or 'small'}_{oset}"
                            out = os.path.join(OUT, exp, variant, wl, f"seed{seed}", f"{alg}.json")
                            tasks.append(dict(exp=exp, variant=variant, wl=wl, uc=uc, size=size, alg=alg,
                                              seed=seed, oset=oset, out=out))
    return tasks


def is_done(task, evals):
    """A result counts only if it was produced with the requested number of evaluations."""
    if not os.path.exists(task["out"]):
        return False
    try:
        with open(task["out"]) as fh:
            return json.load(fh).get("evals") == evals
    except (OSError, ValueError):
        return False


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim(task, stale_min):
    """Claim a run with an exclusive lock file so that several workers can share the grid.
    A lock whose process is gone, or that is older than stale_min minutes, is taken over."""
    lock = task["out"] + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                pid = int(open(lock).read().strip() or 0)
                age = (time.time() - os.path.getmtime(lock)) / 60
                if (pid and not pid_alive(pid)) or age > stale_min:
                    os.remove(lock)
                    continue
            except (OSError, ValueError):
                pass
            return None
    return None


def run_grid(exp, variants, sizes, usecases, algs, seeds, evals, processes, dry,
             objective_sets=("f1f2f3",), time_budget=None, stale_min=180):
    from moo_ra_4 import MooRa4
    tasks = build_tasks(exp, variants, sizes, usecases, algs, seeds, objective_sets)
    start = time.time()
    done = ran = 0
    for n, t in enumerate(tasks, 1):
        if is_done(t, evals):
            done += 1
            continue
        if dry:
            print("would run:", exp, t["variant"], t["wl"], t["alg"], "seed", t["seed"])
            continue
        if time_budget is not None and (time.time() - start) / 60 >= time_budget:
            print(f"time budget of {time_budget} min reached: no new runs started", flush=True)
            break
        lock = claim(t, stale_min)
        if lock is None:
            continue  # another worker has it
        try:
            print(f"[{n}/{len(tasks)}] {MOEA_NAME} {exp} {t['variant']} {t['wl']} {t['alg']} seed={t['seed']}", flush=True)
            rec = run_one(MooRa4, service_file(t["uc"], t["size"]), VARIANTS[t["variant"]], t["alg"], t["seed"],
                          evals, processes, t["out"], OBJECTIVE_SETS[t["oset"]])
            ran += 1
            print(f"    time={rec['time_s']:.1f}s front={len(rec['objectives'])} "
                  f"feasible(pop)={rec['feasible_share_population']:.2f}", flush=True)
        finally:
            if os.path.exists(lock):
                os.remove(lock)
    print(f"configurations: {len(tasks)}  already done: {done}  run by this worker: {ran}")


def status(exp, variants, sizes, usecases, algs, seeds, evals, objective_sets):
    tasks = build_tasks(exp, variants, sizes, usecases, algs, seeds, objective_sets)
    done = sum(is_done(t, evals) for t in tasks)
    def alive(t):
        try:
            return pid_alive(int(open(t["out"] + ".lock").read().strip()))
        except (OSError, ValueError):
            return False
    running = [t for t in tasks if os.path.exists(t["out"] + ".lock") and alive(t)]
    print(f"{MOEA_NAME} {exp}: {done}/{len(tasks)} runs done, {len(running)} running")
    for seed in range(seeds):
        ts = [t for t in tasks if t["seed"] == seed]
        print(f"  seed {seed}: {sum(is_done(t, evals) for t in ts)}/{len(ts)}")
    for t in running:
        print("  running:", t["variant"], t["wl"], t["alg"], "seed", t["seed"])


def launch_workers(n, argv):
    """Start n worker processes with the same arguments (minus --workers) and wait for them."""
    import subprocess
    logdir = os.path.join(OUT, "logs")
    os.makedirs(logdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    args = []
    skip = False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == "--workers":
            skip = True
            continue
        if x.startswith("--workers="):
            continue
        args.append(x)
    procs = []
    for i in range(n):
        log = open(os.path.join(logdir, f"{stamp}_{exp_name(argv)}_w{i}.log"), "w")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args,
                                      stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()))
        time.sleep(1)  # stagger the first claims
    print(f"started {n} workers of {MOEA_NAME}; logs in {logdir}", flush=True)
    for p in procs:
        p.wait()


def exp_name(argv):
    return next((x for x in argv if x in ("paper", "large", "ablation")), "run")


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
        base = wl.split("_")[0]
        uc = base.rstrip("0123456789")
        size = base[len(uc):]
        three_obj = wl.endswith("_f1f2f3") or "_" not in wl
        ilp_file = os.path.join(ilp_dir, f"{uc}{size}", "objectives", "objectives_gap_0.0.txt") if ilp_dir and three_obj else None
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
    for name, sizes, variants in (("large", [250, 500, 1000], ["infit"]), ("ablation", [50, 100], ["infit", "penalty"]),
                                  ("paper", ["small", 50, 100], ["infit"])):
        s = sub.add_parser(name)
        s.add_argument("--sizes", nargs="+", default=sizes, help="workload sizes; 'small' is the file without a size suffix")
        s.add_argument("--usecases", nargs="+", default=USECASES, choices=USECASES)
        s.add_argument("--algos", nargs="+", default=ALGORITHMS, choices=ALGORITHMS)
        s.add_argument("--variants", nargs="+", default=variants, choices=list(VARIANTS))
        s.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 per configuration")
        s.add_argument("--evals", type=int, default=50_000)
        s.add_argument("--processes", type=int, default=1, help="processes of the jMetal evaluator inside one run (use --workers to parallelize runs)")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--time-budget", type=float, default=None,
                       help="minutes after which this worker stops starting new runs (running ones finish)")
        s.add_argument("--workers", type=int, default=1, help="parallel worker processes sharing the grid")
        s.add_argument("--stale-min", type=float, default=180, help="take over a lock older than this")
        s.add_argument("--status", action="store_true", help="print the progress of the grid and exit")
        if name == "paper":
            s.add_argument("--objective-sets", nargs="+", default=list(OBJECTIVE_SETS), choices=list(OBJECTIVE_SETS))
    r = sub.add_parser("report")
    r.add_argument("exp", choices=["large", "ablation", "paper"])
    r.add_argument("--feasible-only", action="store_true", help="drop infeasible solutions before computing metrics")
    r.add_argument("--ilp-dir", default=os.path.join(ROOT, "results", "usecase"),
                   help="folder with <usecase>/objectives/objectives_gap_0.0.txt, used for the Dominance-based Gap")
    a = p.parse_args()

    os.chdir(MOEA_DIR)  # MooRa4 loads ./cnsm_data/*.csv relative to MOEA/
    if a.cmd == "report":
        report(a.exp, a.feasible_only, a.ilp_dir)
    else:
        sizes = [x if x == "small" else int(x) for x in a.sizes]
        osets = getattr(a, "objective_sets", ["f1f2f3"])
        if a.status:
            status(a.cmd, a.variants, sizes, a.usecases, a.algos, a.seeds, a.evals, osets)
        elif a.workers > 1:
            launch_workers(a.workers, sys.argv[1:])
        else:
            run_grid(a.cmd, a.variants, sizes, a.usecases, a.algos, a.seeds, a.evals, a.processes, a.dry_run,
                     osets, a.time_budget, a.stale_min)


if __name__ == "__main__":
    freeze_support()
    main()
