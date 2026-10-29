"""Time a fixed piece of the real workload, to compare machines (laptop vs cluster node).

Evaluates 600 random plans of the 100-request VR workload with the decoder, and runs SPEA2 for
2,000 evaluations. Both are single-process and take well under a minute. Run it on each machine,
alone (no other jobs), and compare the two printed times:

    MOEA_DIR=MOEA_2026 python experiments/benchmark_node.py
    MOEA_DIR=MOEA      python experiments/benchmark_node.py
"""
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algos
from algos import ROOT, MOEA_DIR, MOEA_NAME


def cpu_model():
    for cmd in (["sysctl", "-n", "machdep.cpu.brand_string"], ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "unknown"


def main():
    os.chdir(MOEA_DIR)
    from moo_ra_4 import MooRa4
    problem = MooRa4(os.path.join(ROOT, "services", "types", "vr100.csv"))
    rng = random.Random(0)
    sols = []
    for _ in range(600):
        s = problem.create_solution()
        s.variables = [rng.randint(lo, hi) for lo, hi in zip(problem.lower_bound, problem.upper_bound)]
        sols.append(s)
    t0 = time.perf_counter()
    for s in sols:
        problem.evaluate(s)
    decode_s = time.perf_counter() - t0

    algos.seed_everything(0)
    alg = algos.build("SPEA2", MooRa4(os.path.join(ROOT, "services", "types", "vr100.csv")), evals=2000, processes=1)
    t0 = time.perf_counter()
    alg.run()
    spea_s = time.perf_counter() - t0
    print("\n=== benchmark ===")
    print(f"code version     : {MOEA_NAME}")
    print(f"cpu              : {cpu_model()}")
    print(f"logical cpus     : {os.cpu_count()}")
    print(f"python / numpy   : {platform.python_version()} / {np.__version__}")
    print(f"decoder, 600 plans, 100 requests: {decode_s:.2f} s")
    print(f"SPEA2, 2,000 evaluations         : {spea_s:.2f} s")
    print("Compare both times between machines; a run of 50,000 evaluations takes about 25x the SPEA2 time.")


if __name__ == "__main__":
    main()
