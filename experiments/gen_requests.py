"""Generate the large-scale request files for the four use cases.

Mirrors request_generator.py (same distributions and ranges) but takes the sizes
and a seed from the command line and writes to services/types_large/, so the
files used in the paper (services/types/) are never touched.

    python experiments/gen_requests.py --sizes 250 500 1000 --seed 0
"""
import argparse
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


def generate(idx, n, rng):
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
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000])
    p.add_argument("--usecases", nargs="+", default=USECASES, choices=USECASES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "services", "types_large"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for uc in a.usecases:
        idx = USECASES.index(uc)
        for n in a.sizes:
            rng = random.Random(f"{a.seed}-{uc}-{n}")
            path = os.path.join(a.out, f"{uc}{n}.csv")
            generate(idx, n, rng).to_csv(path, index=False)
            print("wrote", path)


if __name__ == "__main__":
    main()
