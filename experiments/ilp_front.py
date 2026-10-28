"""Load the MO-ILP front (3 objectives, MIP gap 0.0) of a workload from the paper's results.

Reads results/usecase/<workload>/objectives/objectives_gap_0.0.txt when it exists, otherwise the
Julia output results/usecase/<workload>/f1_f2_f3/results_usecase_<workload>[_mac].csv.
The old results tree is only read, never written.
"""
import ast
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ILP = os.path.join(ROOT, "results", "usecase")


def ilp_front(workload, gap="0.0"):
    txt = os.path.join(ILP, workload, "objectives", f"objectives_gap_{gap}.txt")
    if os.path.exists(txt):
        return np.loadtxt(txt, ndmin=2)
    for name in (f"results_usecase_{workload}_mac.csv", f"results_usecase_{workload}.csv"):
        path = os.path.join(ILP, workload, "f1_f2_f3", name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if float(row["MIP_Gap"]) == float(gap):
                    try:
                        cols = [ast.literal_eval(row[k]) for k in ("Latency", "Cost", "Unavailability")]
                    except Exception:
                        continue
                    pts = np.array(cols, dtype=float).T
                    return pts if pts.ndim == 2 and len(pts) else None
    return None
