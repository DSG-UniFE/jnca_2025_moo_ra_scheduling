"""Compare the code versions (MOEA vs MOEA_2026) and the ablation variants on the same seeds.

HV and IGD are only comparable inside one version, because the two decoders compute the
objectives differently (f1 before or after the repair, Reserved billing, instance sharing).
The MO-ILP front is the common reference: for every run we take the solution closest to the
ILP reference point (smallest summed percentage gap) and report its gap per objective.

    python experiments/compare_versions.py --seeds 2
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from ilp_front import ilp_front

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
ILP = os.path.join(ROOT, "results", "usecase")


def load(version, nseeds, exp="ablation"):
    rows = []
    for f in glob.glob(os.path.join(RES, version, exp, "*", "*", "seed*", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r["seed"] >= nseeds:
            continue
        parts = f.split(os.sep)
        r["variant"], r["workload"] = parts[-4], parts[-3]
        rows.append(r)
    return rows


def ilp_reference(workload):
    pts = ilp_front(workload)
    if pts is None:
        return None
    ideal = pts.min(axis=0)
    return pts[np.argmin(np.linalg.norm(pts - ideal, axis=1))]


def best_gap(objs, ref):
    o = np.array(objs)
    g = (o - ref) / np.maximum(1e-6, np.minimum(o, ref)) * 100
    k = np.argmin(g.sum(axis=1))
    return g[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    rows = []
    for v in ("MOEA", "MOEA_2026"):
        for r in load(v, a.seeds):
            ref = ilp_reference(r["workload"])
            row = dict(version=v, variant=r["variant"], workload=r["workload"], algorithm=r["algorithm"], seed=r["seed"],
                       time_s=r["time_s"], front=len(r["objectives"]), feasible=r["feasible_share_population"])
            if ref is not None and r["objectives"]:
                g = best_gap(r["objectives"], ref)
                row.update(gap_f1=g[0], gap_f2=g[1], gap_f3=g[2])
            rows.append(row)
    d = pd.DataFrame(rows)
    print(f"{len(d)} runs, seeds 0..{a.seeds - 1}: ", d.groupby(["version", "variant"]).size().to_dict())

    print("\n== Feasibility, front size, time (mean) ==")
    print(d.groupby(["version", "variant", "algorithm"])[["feasible", "front", "time_s"]].mean().round(2).to_string())

    inf = d[d.variant == "infit"].dropna(subset=["gap_f1"])
    print("\n== Gap to the MO-ILP reference point, INFIT runs: median over workloads and seeds (%) ==")
    print("   positive = worse than the ILP point, negative = better")
    print(inf.groupby(["version", "algorithm"])[["gap_f1", "gap_f2", "gap_f3"]].median().round(1).to_string())
    print("\n   all algorithms together:")
    print(inf.groupby("version")[["gap_f1", "gap_f2", "gap_f3"]].median().round(1).to_string())
    print("\n   by size (all algorithms, median f1 / f2 gap):")
    inf = inf.assign(size=inf.workload.str.extract(r"(\d+)$")[0])
    print(inf.groupby(["version", "size"])[["gap_f1", "gap_f2"]].median().round(1).to_string())

    pen = d[d.variant == "penalty"].dropna(subset=["gap_f1"])
    print("\n== Penalty-only variant: gap to the ILP point of its (infeasible) solutions, median % ==")
    print(pen.groupby("version")[["gap_f1", "gap_f2", "gap_f3"]].median().round(1).to_string())
    d.to_csv(os.path.join(RES, "compare_versions_per_run.csv"), index=False)


if __name__ == "__main__":
    main()
