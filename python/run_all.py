"""Reproduce every experiment, then analyse and plot.

    python run_all.py --tier pilot --jobs 20     # quick pass
    python run_all.py --tier full  --jobs 20     # the paper-grade run

Each stage is skipped if its results already exist, so this can be re-run after
an interruption. Use --force to recompute, or --only to run one stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STAGES = ("main", "ablation", "stress", "multivariate", "realdata")


def sh(args: list[str], label: str) -> bool:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    r = subprocess.run([PY, "-u", *args], cwd=HERE)
    if r.returncode != 0:
        print(f"!! stage failed: {label} (exit {r.returncode})", file=sys.stderr)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="pilot", choices=("smoke", "pilot", "full"))
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help=f"comma-separated subset of {STAGES}")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    stages = [s for s in (args.only.split(",") if args.only else STAGES) if s]
    force = ["--force"] if args.force else []
    ok = True

    for suite in [s for s in stages if s != "realdata"]:
        name = f"{suite}_{args.tier}"
        ok &= sh(["experiments/run_benchmark.py", "--suite", suite, "--tier", args.tier,
                  "--jobs", str(args.jobs), "--out", args.out, "--name", name, *force],
                 f"RUN  {suite} ({args.tier})")

    if "realdata" in stages:
        ok &= sh(["experiments/run_realdata.py", "--splits", str(args.splits),
                  "--out", f"{args.out}/realdata"],
                 "RUN  real-data application")

    for suite in [s for s in stages if s != "realdata"]:
        run_dir = Path(args.out) / f"{suite}_{args.tier}"
        if not (HERE / run_dir).exists():
            continue
        ref = "akde_ridge" if suite == "ablation" else "akde"
        ok &= sh(["experiments/analyze.py", "--run", str(run_dir),
                  "--reference", ref, "--boot", str(args.boot)],
                 f"ANALYSE  {suite}")
        if not args.skip_figures:
            ok &= sh(["experiments/make_figures.py", "--run", str(run_dir),
                      "--reference", ref],
                     f"FIGURES  {suite}")

    print("\nall stages complete" if ok else "\ncompleted with failures (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
