"""Development reference fits for Figure A: adapted SPADES and Sheather-Jones.

No compatible cached fits exist: the V2 rows were produced on a different seed
namespace, a cross-validated dictionary size and a 5-fold tuning budget, so
their dependency hashes do not match this stage (see
protocol_followup_locked.yaml -> provenance.reuse_rule).  These two references
are therefore run on the 40 development datasets: 2 x 40 = 80 fits, logged
separately from the Stage-A budget, under the same locked tuning policy as the
confirmation comparators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from akde import metrics                                     # noqa: E402
from akde.densities import get_target                        # noqa: E402
from followup.akde_followup.comparators import build_comparator  # noqa: E402
from followup.akde_followup.manifest import write_manifest   # noqa: E402
from followup.akde_followup.seeds import data_rng, fit_rng   # noqa: E402

OUT = ROOT / "followup" / "results" / "dev_references"
TARGETS = ("f0", "f2", "f5", "f9")
SIZES = (500, 5000)
REFS = ("spades_simplex", "kde_sj")


def _one(target, N, reps):
    tgt = get_target(target)
    rows = []
    fitdir = OUT / "fits"
    fitdir.mkdir(parents=True, exist_ok=True)
    for rep in range(reps):
        X = tgt.sample(N, data_rng("development", target, N, rep))
        for ri, key in enumerate(REFS):
            est = build_comparator(key)
            t0 = perf_counter()
            fit = est.fit(X, fit_rng("figure_dev", target, N, rep, ri))
            wall = perf_counter() - t0
            ise = float(metrics.ise(fit, tgt))
            rows.append({"stage": "dev_references", "namespace": "development",
                         "target": target, "N": N, "rep": rep, "variant": key,
                         "family": "reference", "wall_time_s": wall, "ise": ise,
                         "n_active": int(fit.n_active(tol=1e-8)),
                         "is_valid_density": bool(fit.is_valid_density()),
                         "total_mass": float(fit.total_mass()), "error": ""})
            np.savez_compressed(fitdir / ("%s_N%d_r%d_%s.npz" % (target, N, rep, key)),
                                weights=fit.weights, means=fit.means,
                                variances=fit.variances, ise=ise)
        print("  %s N=%-5d rep=%d done" % (target, N, rep), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / "all_rows.csv"
    if final.exists() and not a.force:
        print("cached:", final)
        return 0

    from joblib import Parallel, delayed
    cells = [(t, n) for t in TARGETS for n in SIZES]
    print("running %d reference fits (%d cells x %d reps x %d methods)"
          % (len(cells) * a.reps * len(REFS), len(cells), a.reps, len(REFS)), flush=True)
    res = Parallel(n_jobs=a.jobs, backend="loky", verbose=5,
                   inner_max_num_threads=1)(
        delayed(_one)(t, n, a.reps) for t, n in cells)
    df = pd.DataFrame([r for chunk in res for r in chunk])
    df.to_csv(final, index=False)
    write_manifest(OUT / "manifest.json", stage="dev_references",
                   config={"targets": list(TARGETS), "sizes": list(SIZES),
                           "reps": a.reps, "methods": list(REFS),
                           "policy": "locked follow-up tuning (3 folds, pinned 60x5 dict)"},
                   outputs={"rows": final})
    print("\n%d rows" % len(df))
    print(df.pivot_table(index="variant", columns="N", values="ise",
                         aggfunc="median").to_string(float_format=lambda v: "%.5g" % v))
    print("wrote", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
