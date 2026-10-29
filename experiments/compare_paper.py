"""Compare the paper rerun (MOEA_2026) with the metrics stored for the published paper.

1. Dominance-based Gap of the 3-objective workloads: paper value against the rerun (median over seeds).
2. Agreement of the algorithm ranking per workload for hypervolume, IGD, sparsity (all four
   objective sets). The absolute values are not comparable: the paper pools the MO-ILP solutions
   into the reference set and the rerun pools only the MOEA runs, and the decoders differ.

    python experiments/compare_paper.py [--version MOEA_2026]
"""
import argparse
import json
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilp_front import ilp_front

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALG = {"Random Search": "RandomSearch", "Random": "RandomSearch"}
SETS = {"f1f2f3": "", "f1f2": "_f1f2", "f1f3": "_f1f3", "f2f3": "_f2f3"}
METRICS = {"hv": ("hypervolumes", "hypervolume", "Hypervolume"), "igd": ("igds", "igd", "IGD"),
           "sparsity": ("sparsities", "sparsity", "Sparsity")}


def paper_metric(metric, oset, wl):
    folder, stem, label = METRICS[metric]
    path = os.path.join(ROOT, folder + SETS[oset], f"{stem}_{wl}.txt")
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(rf"(.+?) {label}: ([\d.eE+-]+)$", line.strip())
        if m and "ILP" not in m.group(1) and "Gap" not in m.group(1):
            out[ALG.get(m.group(1).strip(), m.group(1).strip())] = float(m.group(2))
    return out


def paper_dg(wl):
    out = {}
    path = os.path.join(ROOT, "dominance_gaps", f"dominance_gaps_{wl}.txt")
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"(.+?) Dominance Gap(?: 0\.00)?: (\[.*\]|None)", line.strip())
            if m and m.group(2) != "None":
                out[ALG.get(m.group(1).strip(), m.group(1).strip())] = json.loads(m.group(2))
    return out


def gap(objs, ref):
    o = np.array(objs)
    g = (o - ref) / np.maximum(1e-6, np.minimum(o, ref)) * 100
    return g[np.argmin(g.sum(axis=1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="MOEA_2026")
    a = ap.parse_args()
    res = os.path.join(ROOT, "experiments", "results", a.version)
    runs = pd.read_csv(os.path.join(res, "reports", "paper_per_run.csv"))

    # ---- 1. Dominance-based Gap, 3 objectives
    rows = []
    for uc in ("ai", "iiot", "smartcity", "vr"):
        for size in ("", "50", "100"):
            wl = uc + size
            ilp = ilp_front(wl)
            if ilp is None:
                continue
            ideal = ilp.min(axis=0)
            ref = ilp[np.argmin(np.linalg.norm(ilp - ideal, axis=1))]
            pdg = paper_dg(wl)
            for alg in ("MOCell", "NSGAII", "NSGAIII", "MSPSO", "SPEA2", "RandomSearch"):
                gs = [gap(json.load(open(f))["objectives"], ref)
                      for f in glob.glob(os.path.join(res, "paper", "infit", f"{uc}{size or 'small'}_f1f2f3", "seed*", f"{alg}.json"))]
                if alg in pdg and gs:
                    gs = np.array(gs)
                    rows.append(dict(wl=wl, size=size or "small", alg=alg, paper_f1=pdg[alg][0], rerun_f1=np.median(gs[:, 0]),
                                     paper_f2=pdg[alg][1], rerun_f2=np.median(gs[:, 1])))
    d = pd.DataFrame(rows)
    print(f"== 1. Dominance-based Gap (3 objectives): {len(d)} workload x algorithm pairs ==")
    print("median over pairs (%), by size:")
    print(d.groupby("size")[["paper_f1", "rerun_f1", "paper_f2", "rerun_f2"]].median().round(0).to_string())
    print("\nmedian over pairs (%), by algorithm:")
    print(d.groupby("alg")[["paper_f1", "rerun_f1", "paper_f2", "rerun_f2"]].median().round(0).to_string())
    better = (d.rerun_f2 < d.paper_f2).mean()
    print(f"\nshare of pairs where the rerun cost gap is smaller than the paper's: {better:.2f}")
    print("pairs where the rerun cost gap is larger by >25%:",
          d[d.rerun_f2 > 1.25 * d.paper_f2][["wl", "alg", "paper_f2", "rerun_f2"]].round(0).to_dict("records"))

    # ---- 2. Ranking agreement
    out = []
    for oset in SETS:
        for uc in ("ai", "iiot", "smartcity", "vr"):
            for size in ("", "50", "100"):
                wl = uc + size
                rr = runs[runs.workload == f"{uc}{size or 'small'}_{oset}"]
                if rr.empty:
                    continue
                for metric, better_high in (("hv", True), ("igd", False), ("sparsity", False)):
                    pv = paper_metric(metric, oset, wl)
                    rv = rr.groupby("algorithm")[metric].median().to_dict()
                    algs = [x for x in pv if x in rv]
                    if len(algs) < 4:
                        continue
                    p = pd.Series({x: pv[x] for x in algs})
                    r = pd.Series({x: rv[x] for x in algs})
                    rho = p.rank().corr(r.rank())
                    best_p = p.idxmax() if better_high else p.idxmin()
                    best_r = r.idxmax() if better_high else r.idxmin()
                    out.append(dict(oset=oset, wl=wl, metric=metric, rho=rho, same_best=best_p == best_r, n=len(algs)))
    o = pd.DataFrame(out)
    print("\n== 2. Agreement of the algorithm ranking with the paper (per workload) ==")
    print("Spearman correlation of the ranks (median) and share of workloads with the same best algorithm:")
    t = o.groupby(["metric", "oset"]).agg(median_rho=("rho", "median"), same_best=("same_best", "mean"), workloads=("wl", "size")).round(2)
    print(t.to_string())
    print("\nall objective sets:")
    print(o.groupby("metric").agg(median_rho=("rho", "median"), same_best=("same_best", "mean"), workloads=("wl", "size")).round(2).to_string())
    d.to_csv(os.path.join(res, "reports", "compare_paper_dg.csv"), index=False)
    o.to_csv(os.path.join(res, "reports", "compare_paper_ranks.csv"), index=False)


if __name__ == "__main__":
    main()
