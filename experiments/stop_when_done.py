"""Stop the running workers once the given seeds of an ablation grid are complete.

Polls the status of one code version, then kills the workers (a run in progress is lost and
starts again on the next launch), removes their lock files, refreshes the reports of all
versions and packs the results.

    python experiments/stop_when_done.py --version MOEA_2026 --seeds 2 --pack-out ~/Desktop/jnca_results
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


def seeds_done(version, n):
    out = run(version, "ablation", "--sizes", "50", "100", "--seeds", "3", "--status")
    done = {int(s): (int(a), int(b)) for s, a, b in re.findall(r"seed (\d+): (\d+)/(\d+)", out)}
    return all(done.get(s, (0, 1))[0] == done.get(s, (0, 1))[1] for s in range(n)) and len(done) > 0, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="MOEA_2026")
    p.add_argument("--seeds", type=int, default=2, help="stop when seeds 0..N-1 are complete")
    p.add_argument("--pack-out", default=os.path.expanduser("~/Desktop/jnca_results"))
    p.add_argument("--poll", type=int, default=20)
    a = p.parse_args()
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    while True:
        ok, out = seeds_done(a.version, a.seeds)
        if ok:
            break
        time.sleep(a.poll)
    log(f"seeds 0..{a.seeds - 1} of {a.version} complete: stopping workers")
    subprocess.run(["pkill", "-f", "run_moea.py"])
    time.sleep(3)
    for lock in glob.glob(os.path.join(ROOT, "experiments", "results", "*", "*", "*", "*", "seed*", "*.lock")):
        os.remove(lock)
    for tmp in glob.glob(os.path.join(ROOT, "experiments", "results", "*", "*", "*", "*", "seed*", "*.tmp")):
        os.remove(tmp)
    for v in ("MOEA", "MOEA_2026"):
        log(f"report {v}")
        run(v, "report", "ablation")
        run(v, "report", "ablation", "--feasible-only")
    log("packing")
    for extra in ([], ["--full"]):
        print(subprocess.run([PY, os.path.join(ROOT, "experiments", "pack_results.py"), "--out", a.pack_out, *extra],
                             cwd=ROOT, capture_output=True, text=True).stdout, flush=True)
    log("done")


if __name__ == "__main__":
    main()
