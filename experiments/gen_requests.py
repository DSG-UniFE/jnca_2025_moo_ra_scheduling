"""Generate the large-scale request files for the four use cases.

Mirrors request_generator.py (same distributions and ranges) but takes the sizes
and a seed from the command line and writes to services/types_large/, so the
files used in the paper (services/types/) are never touched.

A request whose latency bound is below the lowest latency reachable from its region of origin
cannot be satisfied by any plan (for example a 5 ms bound from us-east-1, whose lowest latency is
5.15 ms): the MO-ILP is infeasible and every MOEA run ends with a residual violation. Such a bound
is replaced by a value drawn uniformly from the feasible part of the range of its use case. The
replacement uses its own random stream, so every other value of every request stays unchanged.

    python experiments/gen_requests.py --sizes 250 500 1000 --seed 0
    python experiments/gen_requests.py --check          # report impossible bounds in all request files
"""
import argparse
import glob
import math
import os
import random

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same ranges as request_generator.py (order: ai, iiot, smartcity, vr)
MAX_REPLICAS = 6
VCPU = [(1, 6), (1, 4), (1, 3), (2, 4)]
VRAM = [(2, 8), (2, 6), (1, 3), (2, 4)]
GPU = [(0, 16), (0, 8), (0, 4), (0, 12)]
DURATION = (10, 50)
LATENCY = [(20, 40), (5, 15), (200, 500), (10, 120)]
DCS = ["eu-south-1", "eu-west-2", "us-east-1", "us-west-1", "ap-northeast-3", "ap-south-1"]
USECASES = ["ai", "iiot", "smartcity", "vr"]
NAMES = [
    ("preprocessing", "inference", "postprocessing"),
    ("datacollection", "preprocessing", "anomaly"),
    ("aggregation", "analysis"),
    ("capturing", "preprocessing", "decoding", "rendering"),
]


def min_reachable():
    """Lowest latency (ms) from each origin region to any data center."""
    lat = pd.read_csv(os.path.join(ROOT, "AWS", "latency.csv")).set_index("DataCenter")
    return lat.min(axis=1)


def generate(idx, n, rng, repair_rng=None, min_reach=None):
    rows = []
    for _ in range(n):
        rows.append({
            "usecase": USECASES[idx],
            "name": rng.choice(NAMES[idx]),
            "cpu": rng.randint(*VCPU[idx]),
            "gpu": rng.randint(*GPU[idx]),
            "ram": rng.randint(*VRAM[idx]),
            "duration": rng.randint(*DURATION),
            "location": rng.choice(DCS),
            "replicas": rng.randint(1, MAX_REPLICAS),
            "latency": rng.randint(*LATENCY[idx]),
        })
        if min_reach is not None:
            row = rows[-1]
            floor = math.ceil(min_reach[row["location"]])
            if row["latency"] < floor:  # impossible bound: draw a feasible one from its own stream
                lo, hi = max(LATENCY[idx][0], floor), LATENCY[idx][1]
                if lo > hi:
                    raise ValueError(f"no feasible latency bound for {row['usecase']} from {row['location']}")
                row["latency"] = repair_rng.randint(lo, hi)
    return pd.DataFrame(rows)


def check(paths):
    """Print the requests with an impossible latency bound; return their number."""
    mn = min_reachable()
    total = 0
    for p in paths:
        d = pd.read_csv(p)
        bad = d[d.latency < d.location.map(mn)]
        total += len(bad)
        if len(bad):
            print(f"{os.path.relpath(p, ROOT)}: {len(bad)} of {len(d)} requests have a bound below the lowest reachable latency")
    print(f"checked {len(paths)} files: {total} impossible requests")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000])
    p.add_argument("--usecases", nargs="+", default=USECASES, choices=USECASES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "services", "types_large"))
    p.add_argument("--check", action="store_true",
                   help="only check services/types and services/types_large for impossible latency bounds; exit 1 if any")
    a = p.parse_args()
    if a.check:
        files = sorted(glob.glob(os.path.join(ROOT, "services", "types", "*.csv"))
                       + glob.glob(os.path.join(ROOT, "services", "types_large", "*.csv")))
        raise SystemExit(1 if check(files) else 0)
    os.makedirs(a.out, exist_ok=True)
    mn = min_reachable()
    for uc in a.usecases:
        idx = USECASES.index(uc)
        for n in a.sizes:
            rng = random.Random(f"{a.seed}-{uc}-{n}")
            path = os.path.join(a.out, f"{uc}{n}.csv")
            repair_rng = random.Random(f"{a.seed}-{uc}-{n}-bound-repair")
            generate(idx, n, rng, repair_rng, mn).to_csv(path, index=False)
            print("wrote", path)


if __name__ == "__main__":
    main()
