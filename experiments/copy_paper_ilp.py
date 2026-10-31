"""Copy the ILP runs of the paper into the layout of the MIP-gap / time-limit sweep.

The paper's ILP runs (results/usecase/<workload>/f1_f2_f3/results_usecase_<workload>_mac.csv) used a
time limit of 1000 s and several gaps. Copying them to experiments/results/sensitivity/<workload>/
tl1000_gap<G>/ makes run_sensitivity.py skip those settings. Each copy gets a SOURCE.txt: the paper
runs used Gurobi 12.0.0, the new sweep runs use the installed version, so the two sources can be told apart.

    python experiments/copy_paper_ilp.py [--dry-run]
"""
import argparse
import ast
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENS = os.path.join(ROOT, "experiments", "results", "sensitivity")
WORKLOADS = [f"{u}{n}" for u in ("ai", "iiot", "smartcity", "vr") for n in (50, 100)]
TIME_LIMIT = 1000  # seconds, as in the paper's Julia script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    copied = skipped = 0
    for wl in WORKLOADS:
        path = os.path.join(ROOT, "results", "usecase", wl, "f1_f2_f3", f"results_usecase_{wl}_mac.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            for row in reader:
                try:
                    ok = all(len(ast.literal_eval(row[k])) > 0 for k in ("Latency", "Cost", "Unavailability"))
                except Exception:
                    ok = False
                if not ok:
                    skipped += 1
                    continue
                d = os.path.join(SENS, wl, f"tl{TIME_LIMIT}_gap{float(row['MIP_Gap'])}")
                out = os.path.join(d, f"results_usecase_{wl}.csv")
                if os.path.exists(out):
                    continue
                copied += 1
                if a.dry_run:
                    print("would copy:", wl, "gap", row["MIP_Gap"])
                    continue
                os.makedirs(d, exist_ok=True)
                with open(out, "w", newline="") as o:
                    w = csv.DictWriter(o, fieldnames=header)
                    w.writeheader()
                    w.writerow(row)
                with open(os.path.join(d, "SOURCE.txt"), "w") as o:
                    o.write(f"Copied from the paper's ILP runs ({os.path.relpath(path, ROOT)}).\n"
                            f"Time limit {TIME_LIMIT} s, Gurobi 12.0.0. Not produced by run_sensitivity.py.\n")
    print(f"copied {copied} settings, skipped {skipped} rows without a front")


if __name__ == "__main__":
    main()
