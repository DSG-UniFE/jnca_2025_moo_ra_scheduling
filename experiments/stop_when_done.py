"""Stop the running workers once the given seeds of a grid are complete, or at a deadline.

Polls the status of one code version, then kills the workers (a run in progress is lost and
starts again on the next launch), removes their lock files, refreshes the reports of all
versions and packs the results.

    python experiments/stop_when_done.py --version MOEA_2026 --seeds 2 --pack-out ~/Desktop/jnca_results
    python experiments/stop_when_done.py --exp paper --seeds 1 --deadline-min 27
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUN = os.path.join(ROOT, "experiments", "run_moea.py")


def run(version, *args):
    env = dict(os.environ, MOEA_DIR=version)
    return subprocess.run([PY, RUN, *args], cwd=ROOT, env=env, capture_output=True, text=True).stdout


def seeds_done(version, n, exp="ablation"):
    extra = ["--sizes", "50", "100"] if exp == "ablation" else []
    out = run(version, exp, *extra, "--seeds", "3", "--status")
    done = {int(s): (int(a), int(b)) for s, a, b in re.findall(r"seed (\d+): (\d+)/(\d+)", out)}
    return all(done.get(s, (0, 1))[0] == done.get(s, (0, 1))[1] for s in range(n)) and len(done) > 0, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="MOEA_2026")
    p.add_argument("--seeds", type=int, default=2, help="stop when seeds 0..N-1 are complete")
    p.add_argument("--pack-out", default=os.path.expanduser("~/Desktop/jnca_results"))
    p.add_argument("--poll", type=int, default=20)
    p.add_argument("--exp", default="ablation", choices=["ablation", "paper", "large"])
    p.add_argument("--deadline-min", type=float, default=None, help="stop at this many minutes from now even if the seeds are not complete")
    a = p.parse_args()
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    start = time.time()
    while True:
        ok, out = seeds_done(a.version, a.seeds, a.exp)
        if ok:
            log(f"seeds 0..{a.seeds - 1} of {a.version} {a.exp} complete: stopping workers")
            break
        if a.deadline_min is not None and (time.time() - start) / 60 >= a.deadline_min:
            log(f"deadline of {a.deadline_min} min reached: stopping workers\n{out}")
            break
        time.sleep(a.poll)
    subprocess.run(["pkill", "-f", "run_moea.py"])
    time.sleep(3)
    for lock in glob.glob(os.path.join(ROOT, "experiments", "results", "*", "*", "*", "*", "seed*", "*.lock")):
        os.remove(lock)
    for tmp in glob.glob(os.path.join(ROOT, "experiments", "results", "*", "*", "*", "*", "seed*", "*.tmp")):
        os.remove(tmp)
    for v in ("MOEA", "MOEA_2026"):
        for exp in dict.fromkeys(("ablation", a.exp)):  # the ablation and the experiment being stopped
            if not glob.glob(os.path.join(ROOT, "experiments", "results", v, exp, "*", "*", "seed*", "*.json")):
                continue
            log(f"report {exp} {v}")
            run(v, "report", exp)
            run(v, "report", exp, "--feasible-only")
    log("packing")
    for extra in ([], ["--full"]):
        print(subprocess.run([PY, os.path.join(ROOT, "experiments", "pack_results.py"), "--out", a.pack_out, *extra],
                             cwd=ROOT, capture_output=True, text=True).stdout, flush=True)
    log("done")


if __name__ == "__main__":
    main()
