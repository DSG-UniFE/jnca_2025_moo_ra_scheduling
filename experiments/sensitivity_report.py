"""MIP-gap and time-limit sensitivity of the MO-ILP, compared with both MOEA versions.

The ILP runs do not depend on the MOEA version. What depends on it is the comparison: for every
(workload, time limit, MIP gap) the ILP front is compared with the fronts of the original decoder
(MOEA, ablation `infit` runs) and of the fixed decoder (MOEA_2026, paper-rerun runs), seeds 0..2.

ILP sources
  paper   results/usecase/<workload>/f1_f2_f3/results_usecase_<workload>_mac.csv (time limit 1000 s;
          all six gaps at 50 requests, only gap 0.0 at 100 requests)
  sweep   experiments/results/sensitivity/<workload>/tl<T>_gap<G>/results_usecase_<workload>.csv
          (written by run_sensitivity.py)

Per row: ILP execution time, front size, whether the time limit was hit, and for each MOEA version
the best-cost ratio to the ILP, the Dominance-based-Gap-style gaps of the solution closest to the
ILP reference point, and the hypervolume of the ILP front relative to the MOEA fronts.

    python experiments/sensitivity_report.py
"""
import ast
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
WORKLOADS = [f"{u}{n}" for u in ("ai", "iiot", "smartcity", "vr") for n in (50, 100)]
VERSION_SOURCE = {"MOEA": ("MOEA", "ablation", "{wl}"), "MOEA_2026": ("MOEA_2026", "paper", "{wl}_f1f2f3")}


def read_ilp_csv(path, tl, source):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                pts = np.array([ast.literal_eval(row[k]) for k in ("Latency", "Cost", "Unavailability")], dtype=float).T
            except Exception:
                continue
            if pts.ndim != 2 or not len(pts):
                continue
            out.append(dict(tl=tl, gap=float(row["MIP_Gap"]), time=float(row["Execution_Time"]), front=pts, source=source))
    return out


SENS_DIR = os.path.join(RES, "sensitivity")  # overridden by --sens-dir
MOEA_ROOT = {"MOEA": os.path.join(RES, "MOEA"), "MOEA_2026": os.path.join(RES, "MOEA_2026")}  # overridden by --moea-2026-dir


def ilp_runs(wl):
    runs = read_ilp_csv(os.path.join(ROOT, "results", "usecase", wl, "f1_f2_f3", f"results_usecase_{wl}_mac.csv"), 1000, "paper")
    for d in sorted(glob.glob(os.path.join(SENS_DIR, wl, "tl*_gap*"))):
        tl = int(os.path.basename(d).split("_")[0][2:])
        src = "paper (copied)" if os.path.exists(os.path.join(d, "SOURCE.txt")) else "sweep"
        runs += read_ilp_csv(os.path.join(d, f"results_usecase_{wl}.csv"), tl, src)
    # a setting may exist in both sources: keep the sweep (newer) one
    best = {}
    for r in runs:
        best[(r["tl"], r["gap"])] = r
    return list(best.values())


def moea_fronts(version, wl):
    v, exp, pat = VERSION_SOURCE[version]
    fronts = []
    for f in glob.glob(os.path.join(MOEA_ROOT[version], exp, "infit", pat.format(wl=wl), "seed*", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r["objectives"]:
            fronts.append((r["algorithm"], np.array(r["objectives"])))
    return fronts


def gaps(front, ref):
    g = (front - ref) / np.maximum(1e-6, np.minimum(front, ref)) * 100
    return g[np.argmin(g.sum(axis=1))]


def hv(points, lo, hi):
    """Hypervolume after min-max normalizing every objective over the pooled points, reference point 1.1.
    Without the normalization the interruption objective, whose values are tiny, dominates the volume."""
    from jmetal.core.quality_indicator import HyperVolume
    span = np.maximum(hi - lo, 1e-12)
    return HyperVolume([1.1, 1.1, 1.1]).compute((points - lo) / span)


def main():
    global SENS_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens-dir", help="folder with <workload>/tl<T>_gap<G>/ runs (default experiments/results/sensitivity)")
    ap.add_argument("--moea-2026-dir", help="folder with paper/infit/... runs of MOEA_2026 (default experiments/results/MOEA_2026)")
    ap.add_argument("--versions", nargs="+", default=list(VERSION_SOURCE), choices=list(VERSION_SOURCE))
    ap.add_argument("--out", help="folder for sensitivity_report.csv (default next to the sweep)")
    a = ap.parse_args()
    if a.sens_dir:
        SENS_DIR = os.path.abspath(a.sens_dir)
    if a.moea_2026_dir:
        MOEA_ROOT["MOEA_2026"] = os.path.abspath(a.moea_2026_dir)
    rows = []
    for wl in WORKLOADS:
        runs = ilp_runs(wl)
        if not runs:
            continue
        for version in a.versions:
            fronts = moea_fronts(version, wl)
            if not fronts:
                continue
            pool = np.vstack([r["front"] for r in runs] + [f for _, f in fronts])
            lo, hi = pool.min(axis=0), pool.max(axis=0)
            moea_hv = float(np.median([hv(f, lo, hi) for _, f in fronts]))
            for r in runs:
                front = r["front"]
                ideal = front.min(axis=0)
                ref = front[np.argmin(np.linalg.norm(front - ideal, axis=1))]
                g = np.array([gaps(f, ref) for _, f in fronts])
                ratio = np.array([f[:, 1].min() / front[:, 1].min() for _, f in fronts])
                rows.append(dict(workload=wl, size=int(wl.lstrip("abcdefghijklmnopqrstuvwxyz")), version=version, tl=r["tl"], gap=r["gap"],
                                 source=r["source"], ilp_time_s=r["time"], ilp_points=len(front),
                                 timed_out=r["time"] >= 0.97 * r["tl"], moea_fronts=len(fronts),
                                 gap_f1=float(np.median(g[:, 0])), gap_f2=float(np.median(g[:, 1])),
                                 cost_ratio=float(np.median(ratio)), share_cheaper=float((ratio < 1).mean()),
                                 hv_ilp_over_moea=hv(front, lo, hi) / moea_hv if moea_hv > 0 else np.nan))
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no ILP and MOEA data found")
    outdir = os.path.abspath(a.out) if a.out else SENS_DIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "sensitivity_report.csv")
    d.to_csv(path, index=False)

    ilp = d[d.version == a.versions[0]].drop_duplicates(["workload", "tl", "gap"])
    print(f"{ilp.workload.nunique()} workloads, {len(ilp)} ILP settings (time limit x gap x workload)\n")
    print("== MO-ILP: median over workloads, by time limit and MIP gap ==")
    print(ilp.groupby(["tl", "gap"]).agg(runs=("workload", "size"), time_s=("ilp_time_s", "median"), points=("ilp_points", "median"),
                                          timed_out=("timed_out", "mean")).round(2).to_string())
    print("\n== Each MOEA version against the ILP front of the same setting (median over workloads) ==")
    print("   cost_ratio = best MOEA cost / best ILP cost (1 = equal); share_cheaper = runs whose cheapest plan beats the ILP's;")
    print("   hv_ilp_over_moea = normalized hypervolume of the ILP front relative to the median MOEA front (>1 = ILP front is larger)")
    cols = ["cost_ratio", "share_cheaper", "gap_f1", "gap_f2", "hv_ilp_over_moea"]
    print(d.groupby(["version", "size", "tl", "gap"])[cols].median().round(2).to_string())
    print("\nwrote", path)


if __name__ == "__main__":
    main()
