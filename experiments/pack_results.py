"""Pack the experiment results into one .tar.gz for sharing outside the public repository.

The archive unpacks into experiments/results/, so extracting it in the repository root restores
the folder layout expected by `run_moea.py report` and `jitter_eval.py`:

    tar -xzf jnca_results_slim_YYYYMMDD.tar.gz -C <repository root>

It contains, for every code version found (MOEA, MOEA_2026):
  reports/*.csv   the metric tables (run `run_moea.py report <experiment>` first to refresh them)
  <experiment>/   every run as a JSON file; without the variable vectors unless --full
  logs/           the worker logs
and experiments/results/MANIFEST.txt with the date, the code commit and the run counts.
A .sha256 file is written next to the archive.

    python experiments/pack_results.py --out ~/Desktop            # slim: tables, figures, DG
    python experiments/pack_results.py --out ~/Desktop --full     # adds the variable vectors (jitter analysis)
"""
import argparse
import datetime
import glob
import hashlib
import io
import json
import os
import platform
import subprocess
import tarfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")


def git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def add_bytes(tar, arcname, data, mtime):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(data))


def manifest(versions, runs, full):
    try:
        from importlib.metadata import version
        jm = version("jmetalpy")
    except Exception:
        jm = "unknown"
    lines = [
        "JNCA revision: experiment results",
        f"created: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"code commit: {git('rev-parse', 'HEAD')}" + ("  (uncommitted changes present)" if git("status", "--porcelain", "--untracked-files=no") else ""),
        f"python {platform.python_version()}, jMetalPy {jm}",
        f"variable vectors included: {'yes' if full else 'no (slim; the jitter analysis needs --full)'}",
        "",
        "Code versions: MOEA = decoder of the submitted manuscript; MOEA_2026 = fixed decoder (see MOEA_2026/README.md).",
        "",
        "Runs per version, experiment and variant:",
    ]
    for v in versions:
        c = Counter()
        evals = set()
        seeds = set()
        for f, rec in runs[v]:
            parts = f.split(os.sep)
            c[(parts[-5], parts[-4])] += 1
            evals.add(rec.get("evals"))
            seeds.add(rec.get("seed"))
        lines.append(f"  {v}: evaluations {sorted(e for e in evals if e)}, seeds {sorted(s for s in seeds if s is not None)}")
        for (exp, var), n in sorted(c.items()):
            lines.append(f"    {exp}/{var}: {n} runs")
    lines += ["", "Reproduce a table:", "  MOEA_DIR=<version> python experiments/run_moea.py report ablation",
              "Reproduce a run: python experiments/run_moea.py ablation ... (see experiments/README.md)"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.dirname(ROOT), help="folder for the archive (outside the repository)")
    p.add_argument("--full", action="store_true", help="keep the variable vectors of every solution")
    a = p.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    if out.startswith(ROOT + os.sep):
        raise SystemExit("--out must be outside the repository")
    os.makedirs(out, exist_ok=True)

    versions = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RESULTS, "*"))) if os.path.isdir(d)]
    runs = {}
    for v in versions:
        files = sorted(glob.glob(os.path.join(RESULTS, v, "*", "*", "*", "seed*", "*.json")))
        runs[v] = []
        for f in files:
            with open(f) as fh:
                runs[v].append((f, json.load(fh)))
    if not any(runs.values()):
        raise SystemExit("no runs found")

    stamp = datetime.date.today().strftime("%Y%m%d")
    name = f"jnca_results_{'full' if a.full else 'slim'}_{stamp}.tar.gz"
    path = os.path.join(out, name)
    now = datetime.datetime.now().timestamp()
    with tarfile.open(path, "w:gz") as tar:
        add_bytes(tar, "experiments/results/MANIFEST.txt", manifest(versions, runs, a.full).encode(), now)
        for v in versions:
            for f, rec in runs[v]:
                arc = os.path.join("experiments", "results", os.path.relpath(f, RESULTS))
                if a.full:
                    tar.add(f, arcname=arc)
                else:
                    rec = dict(rec)
                    rec.pop("variables", None)
                    add_bytes(tar, arc, json.dumps(rec).encode(), os.path.getmtime(f))
            for sub in ("reports", "logs"):
                d = os.path.join(RESULTS, v, sub)
                if os.path.isdir(d):
                    tar.add(d, arcname=os.path.join("experiments", "results", v, sub))
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as fh:
        fh.write(f"{digest}  {name}\n")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)\nwrote {path}.sha256")
    for v in versions:
        print(f"  {v}: {len(runs[v])} runs")


if __name__ == "__main__":
    main()
