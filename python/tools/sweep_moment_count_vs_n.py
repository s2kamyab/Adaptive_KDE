"""Does the AKDE error plateau lift when the moment count grows with N?

Motivation
----------
In the main run the two estimators that match a *fixed finite* feature set
plateau in N, while every criterion aligned with L2 converges at roughly the
textbook rate (median log-log slopes, 15 densities x 30 replicates):

    akde            -0.115      finite-moment, Omega-weighted, m = 300 fixed
    song            -0.120      finite-moment, kernel embedding
    spades_simplex  -0.872      L2 / integrated-square, SAME dictionary
    kde_lscv        -0.586      L2-tuned KDE

Since SPADES shares the dictionary and converges, the plateau is not the
representation. The remaining suspect is the manuscript's own caveat in S2.3:
"A finite collection of features need not distinguish arbitrary
distributions." With m held at 300 while N grows, the moment criterion is
satisfied to within sampling error long before the density is pinned down, so
the residual bias stops falling.

This script tests the implied prescription: m must grow with N. If the slope
in N steepens as m grows, the plateau is a fixed-m artefact and the estimator
is repairable; if it does not, the limitation is in the criterion itself.

Usage
-----
    python tools/sweep_moment_count_vs_n.py --jobs 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from time import perf_counter  # noqa: E402

from akde import metrics, registry  # noqa: E402
from akde.densities import get_target  # noqa: E402
from akde.runner import name_key  # noqa: E402

#: m = MEAS_N_SCALE * meas_n_loc
MEAS_N_SCALE = 3
M_GRID = (99, 300, 600, 1200)


def one(target_name: str, N: int, m: int, rep: int, seed: int) -> dict:
    target = get_target(target_name)
    rng_data = np.random.default_rng(
        np.random.SeedSequence([seed, name_key(target_name), N, rep, 0]))
    X = target.sample(N, rng_data)
    rec = {"target": target_name, "N": N, "m": m, "rep": rep}
    try:
        est = registry.build("akde", meas_n_loc=m // MEAS_N_SCALE,
                             meas_n_scale=MEAS_N_SCALE)
        t0 = perf_counter()
        fit = est.fit(X, np.random.default_rng(
            np.random.SeedSequence([seed, name_key(target_name), N, rep, 1])))
        rec["wall_s"] = perf_counter() - t0
        rec["ise"] = metrics.ise(fit, target)
        rec["n_active"] = fit.n_active(tol=1e-8)
        for k in ("cov_rank_numeric", "cov_effective_rank", "omega_cond", "rho",
                  "lam_ratio", "n_measurements", "fallback_used"):
            rec[k] = fit.diagnostics.get(k, np.nan)
        rec["error"] = ""
    except Exception as exc:
        rec["ise"] = np.nan
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="f2,f5,f9")
    ap.add_argument("--sizes", default="500,5000")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", default="results/moment_count")
    args = ap.parse_args()

    targets = args.targets.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(t, N, m, r) for t in targets for N in sizes
            for m in M_GRID for r in range(args.reps)]
    print(f"{len(jobs)} fits: targets={targets} sizes={sizes} "
          f"m={list(M_GRID)} reps={args.reps} jobs={args.jobs}", flush=True)

    rows = Parallel(n_jobs=args.jobs, verbose=5, backend="loky",
                    inner_max_num_threads=1)(
        delayed(one)(t, N, m, r, args.seed) for t, N, m, r in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)

    ok = df[df.error == ""]
    print("\n=== median ISE by (N, m), pooled over targets ===")
    piv = ok.pivot_table(index="m", columns="N", values="ise", aggfunc="median")
    print(piv.to_string(float_format=lambda x: f"{x:.4g}"))

    print("\n=== log-log slope of median ISE in N, per m ===")
    print("(more negative = converging; ~0 = plateau)")
    for m in piv.index:
        y = piv.loc[m].values.astype(float)
        x = np.log(piv.columns.values.astype(float))
        if np.all(np.isfinite(y)) and np.all(y > 0):
            print(f"  m={m:5d}  slope {np.polyfit(x, np.log(y), 1)[0]:+.3f}")

    print("\n=== per target ===")
    print(ok.pivot_table(index=["target", "m"], columns="N", values="ise",
                         aggfunc="median").to_string(
                             float_format=lambda x: f"{x:.4g}"))
    print("\n=== conditioning and cost ===")
    print(ok.pivot_table(index="m", columns="N",
                         values=["cov_rank_numeric", "omega_cond", "wall_s",
                                 "n_active"],
                         aggfunc="median").to_string(
                             float_format=lambda x: f"{x:.4g}"))
    print(f"\nwritten to {out/'raw.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
