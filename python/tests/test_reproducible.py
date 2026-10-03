"""Verify a replicate is bit-reproducible under single-threaded BLAS.

The moment design matrix is numerically singular (condition numbers ~1e17), so
a last-bit difference in a matrix product can change which atoms are selected.
With multi-threaded BLAS, summation order depends on thread scheduling and the
same replicate can return a materially different ISE. Pinning BLAS to one
thread removes that source of variation.

This test must be run as a script (it sets thread environment variables before
importing numpy):

    python tests/test_reproducible.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from akde.runner import RunConfig, run_one_replicate  # noqa: E402

ESTIMATORS = ("akde", "spades_simplex", "song", "kde_sj", "abramson", "thin_kt")


def main() -> int:
    cfg = RunConfig(name="repro", targets=("f9",), sample_sizes=(200,), n_reps=1,
                    estimators=ESTIMATORS, compute_iae=False, compute_hellinger=False)

    a = {r["estimator"]: r["ise"] for r in run_one_replicate(cfg, "f9", 200, 0)}
    b = {r["estimator"]: r["ise"] for r in run_one_replicate(cfg, "f9", 200, 0)}

    failed = 0
    for est in ESTIMATORS:
        va, vb = a.get(est, np.nan), b.get(est, np.nan)
        same = (np.isnan(va) and np.isnan(vb)) or va == vb
        print(f"  {est:16s} {va:.10e}  {vb:.10e}  {'OK' if same else 'DIFFERS'}")
        if not same:
            failed += 1

    # A different replicate index must give a genuinely different dataset,
    # otherwise the replication is not independent.
    c = {r["estimator"]: r["ise"] for r in run_one_replicate(cfg, "f9", 200, 1)}
    distinct = sum(1 for est in ESTIMATORS if c.get(est) != a.get(est))
    print(f"\n  rep 0 vs rep 1 differ for {distinct}/{len(ESTIMATORS)} estimators "
          f"(should be all of them)")
    if distinct != len(ESTIMATORS):
        failed += 1

    print(f"\n{'PASS' if not failed else 'FAIL'}: repeated identical replicates "
          f"{'agree exactly' if not failed else 'disagree'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
