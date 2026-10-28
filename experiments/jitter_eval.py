"""Robustness of Pareto-optimal plans under latency jitter .

Re-evaluates existing fronts under sampled latency realizations. It needs no new
optimization runs. The mean latency of every (origin region, data center) pair is
the AWS value used for planning; each realization multiplies it by a
mean-preserving lognormal factor, exp(N(-s^2/2, s^2)), drawn independently per
pair. All solutions share the same realizations (common random numbers).

For each solution it reports, over R realizations:
  viol_request_rate  share of requests whose realized latency exceeds their bound
  viol_plan_rate     share of realizations in which at least one request violates
  f1_degradation_pct mean realized f1 vs. the nominal f1 of the same plan (in %)

Sources of fronts
  --source json   runs written by run_moea.py (experiments/results/<exp>/...)
  --source paper  fronts of the paper (MOEA/results/<workload>/*.FUN.* and *.VAR.*)

The ILP fronts are not included: their assignments are in the Gurobi solution
files, not in a variable vector. Add a loader to include the ILP row.

    python experiments/jitter_eval.py --source paper --sigma 0.1 0.2 0.4
    python experiments/jitter_eval.py --source json --exp large
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import algos
from algos import ROOT, MOEA_DIR

OUT = algos.RESULTS
ALG_FROM_FILE = {"Random Search": "RandomSearch"}


def load_paper(workloads):
    for wdir in sorted(glob.glob(os.path.join(MOEA_DIR, "results", "*"))):
        wl = os.path.basename(wdir)
        if workloads and wl not in workloads:
            continue
        for fun in glob.glob(os.path.join(wdir, "*.FUN.*")):
            alg = os.path.basename(fun).split(".FUN.")[0]
            var = fun.replace(".FUN.", ".VAR.")
            if not os.path.exists(var):
                continue
            f = np.atleast_2d(np.loadtxt(fun))
            v = np.atleast_2d(np.loadtxt(var)).astype(int)
            yield {"workload": wl, "variant": "paper", "algorithm": ALG_FROM_FILE.get(alg, alg),
                   "seed": 0, "objectives": f.tolist(), "variables": v.tolist(),
                   "service_file": f"{wl}.csv"}


def load_json(exp, workloads):
    for f in glob.glob(os.path.join(OUT, exp, "*", "*", "seed*", "*.json")):
        parts = f.split(os.sep)
        if workloads and parts[-3] not in workloads:
            continue
        with open(f) as fh:
            r = json.load(fh)
        r["variant"], r["workload"] = parts[-4], parts[-3]
        yield r


def find_service_file(name):
    for d in ("services/types", "services/types_large"):
        p = os.path.join(ROOT, d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def evaluate(problem, variables, noise):
    """noise: (R, n_loc, n_dc) multiplicative factors. Returns per-solution metrics."""
    locs = sorted({problem.requests_location[i] for i in range(problem.num_requests)})
    loc_idx = {l: k for k, l in enumerate(locs)}
    base = np.array([[problem.latency_lookup[(l, dc)] for dc in problem.datacenters] for l in locs])
    req_loc = np.array([loc_idx[problem.requests_location[i]] for i in range(problem.num_requests)])
    bound = np.array([problem.requests_latency[i] for i in range(problem.num_requests)], dtype=float)
    mapping = np.array(problem.mapping)
    R = noise.shape[0]
    realized_lat = base[None, :, :] * noise  # (R, n_loc, n_dc)

    out = []
    for var in variables:
        dcs = np.array([problem.decode(v)[0] for v in var[problem.num_requests:]], dtype=int)
        # replica -> (request, dc); latency of a request = max over its replicas
        nominal = np.zeros(problem.num_requests)
        real = np.zeros((R, problem.num_requests))
        for rep, (req, dc) in enumerate(zip(mapping, dcs)):
            nominal[req] = max(nominal[req], base[req_loc[req], dc])
            real[:, req] = np.maximum(real[:, req], realized_lat[:, req_loc[req], dc])
        viol = real > bound[None, :]
        f1_nom = nominal.mean()
        f1_real = real.mean(axis=1)
        out.append({
            "f1_nominal": f1_nom,
            "nominal_violations": int((nominal > bound).sum()),
            "viol_request_rate": float(viol.mean()),
            "viol_plan_rate": float(viol.any(axis=1).mean()),
            "f1_degradation_pct": float((f1_real.mean() - f1_nom) / f1_nom * 100) if f1_nom > 0 else np.nan,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["paper", "json"], default="paper")
    p.add_argument("--exp", default="large", help="experiment folder when --source json")
    p.add_argument("--workloads", nargs="*", help="e.g. vr50 ai100 (default: all)")
    p.add_argument("--sigma", type=float, nargs="+", default=[0.1, 0.2, 0.4],
                   help="log-scale std of the latency jitter")
    p.add_argument("--realizations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.chdir(MOEA_DIR)
    from moo_ra_4 import MooRa4
    runs = list(load_paper(a.workloads) if a.source == "paper" else load_json(a.exp, a.workloads))
    if not runs:
        raise SystemExit("no fronts found")

    rows = []
    cache = {}
    for r in runs:
        sf = find_service_file(r["service_file"])
        if sf not in cache:
            cache[sf] = MooRa4(sf)
        problem = cache[sf]
        n_loc = len({problem.requests_location[i] for i in range(problem.num_requests)})
        for sigma in a.sigma:
            rng = np.random.default_rng(a.seed)
            noise = np.exp(rng.normal(-sigma**2 / 2, sigma, size=(a.realizations, n_loc, problem.num_datacenters)))
            res = evaluate(problem, r["variables"], noise)
            df = pd.DataFrame(res)
            rows.append({"workload": r["workload"], "variant": r["variant"], "algorithm": r["algorithm"],
                         "seed": r["seed"], "sigma": sigma, "front_size": len(df),
                         **df.mean().to_dict()})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    path = os.path.join(OUT, "reports", f"jitter_{a.source}.csv")
    out.to_csv(path, index=False)
    summ = out.groupby(["workload", "algorithm", "sigma"])[["viol_request_rate", "viol_plan_rate", "f1_degradation_pct"]].mean()
    print(summ.round(4).to_string())
    print("\nwrote", path)


if __name__ == "__main__":
    main()
