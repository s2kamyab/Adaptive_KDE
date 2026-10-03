"""Smoke test: fit every registered estimator once and report accuracy + cost.

Run this before launching a long experiment.  It catches configuration errors
in seconds and prints the per-fit timings that determine whether a given run
tier is affordable.

    python experiments/smoke.py [--target f9] [--n 500]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akde import metrics, registry
from akde.densities import get_target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="f9")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-loc", type=int, default=150)
    ap.add_argument("--n-scale", type=int, default=10)
    ap.add_argument("--methods", default="")
    args = ap.parse_args()

    target = get_target(args.target)
    rng = np.random.default_rng(1)
    X = target.sample(args.n, rng)

    names = args.methods.split(",") if args.methods else (registry.ESSENTIAL + registry.ABLATIONS)
    names = [n for n in dict.fromkeys(names) if n]

    print(f"target={args.target} ({target.label})  N={args.n}  "
          f"dictionary={args.n_loc}x{args.n_scale}={args.n_loc * args.n_scale}\n")
    hdr = f"{'method':22s} {'ISE':>11s} {'supp':>6s} {'mass':>8s} {'valid':>6s} {'tune s':>8s} {'fit s':>8s}"
    print(hdr)
    print("-" * len(hdr))

    failures = 0
    rows = []
    for name in names:
        try:
            est = registry.build(name, n_loc=args.n_loc, n_scale=args.n_scale)
            t0 = perf_counter()
            fit = est.fit(X, np.random.default_rng(7))
            wall = perf_counter() - t0
            ise = metrics.ise(fit, target)
            d = fit.diagnostics
            rows.append((name, ise, wall))
            print(f"{name:22s} {ise:11.3e} {fit.n_active():6d} "
                  f"{fit.total_mass():8.4f} {str(fit.is_valid_density()):>6s} "
                  f"{d.get('tune_time_s', 0):8.2f} {d.get('fit_time_s', 0):8.2f}")
        except Exception as exc:
            failures += 1
            print(f"{name:22s} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)

    if rows:
        total = sum(r[2] for r in rows)
        print(f"\ntotal wall time for one replicate: {total:.1f} s")
        best = min(rows, key=lambda r: r[1])
        print(f"lowest ISE on this single draw: {best[0]} ({best[1]:.3e})")
        print("(one draw proves nothing -- use the bootstrap over replicates)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
