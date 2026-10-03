"""Reproduce saved V2 rows from their recorded seeds, and audit the V2 seed keys.

Two things this establishes:
  * the V2 seed scheme really does give unique, paired keys over the executed
    grid (crc32-based, no salted ``hash()``);
  * a row in ``results/main_full/*.csv`` can be regenerated from its
    ``(seed, target, N, rep)`` alone, i.e. the saved results are provenance for
    the code as it now stands.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akde import metrics                                    # noqa: E402
from akde.densities import get_target                       # noqa: E402
from akde.registry import build                             # noqa: E402
from akde.runner import _child_seed, name_key               # noqa: E402

OUT = ROOT / "followup" / "results" / "audit_tables"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 1. seed-key audit over the grid that was actually executed -------------
keys, states = [], []
for t in ["f%d" % i for i in range(15)]:
    for n in [50, 100, 200, 500, 1000, 5000]:
        for r in range(30):
            for s in [0, 1] + [100 + i for i in range(10)]:
                k = (20260915, name_key(t), n, r, s)
                keys.append(k)
                states.append(int(np.random.default_rng(
                    np.random.SeedSequence(list(k))).integers(0, 2 ** 62)))
print("V2 seed keys: %d, unique keys %d, unique first draws %d"
      % (len(keys), len(set(keys)), len(set(states))), flush=True)

# pairing: the data stream must not depend on which estimator is being fitted
a = get_target("f9").sample(5, _child_seed(20260915, "f9", 500, 3, 0))
b = get_target("f9").sample(5, _child_seed(20260915, "f9", 500, 3, 0))
print("pairing across methods (identical data stream):", np.array_equal(a, b), flush=True)

# ---- 2. end-to-end reproduction of saved rows ------------------------------
ESTS = [("kde_sj", 4), ("spades", 1), ("song", 3), ("akde", 0)]
rows = []
for tname, N, reps in [("f9", 500, [0, 7]), ("f5", 5000, [0])]:
    df = pd.read_csv(ROOT / "results" / "main_full" / ("%s_N%d.csv" % (tname, N)),
                     low_memory=False)
    tgt = get_target(tname)
    for rep in reps:
        X = tgt.sample(N, _child_seed(20260915, tname, N, rep, 0))
        for est_name, ei in ESTS:
            est = build(est_name, n_loc=150, n_scale=10)
            est.tuning.n_folds = 5
            fit = est.fit(X, _child_seed(20260915, tname, N, rep, 100 + ei))
            got = float(metrics.ise(fit, tgt))
            saved = float(df[(df.rep == rep) & (df.estimator == est_name)].ise.iloc[0])
            rel = abs(got - saved) / max(saved, 1e-300)
            rows.append({"target": tname, "N": N, "rep": rep, "estimator": est_name,
                         "ise_recomputed": got, "ise_saved": saved, "rel_err": rel})
            print("  %s N=%d rep=%d %-8s recomputed %.10g  saved %.10g  rel %.2e"
                  % (tname, N, rep, est_name, got, saved, rel), flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT / "v2_reproduction_check.csv", index=False)
print("\nmax relative error over %d reproduced rows: %.3e"
      % (len(out), out.rel_err.max()), flush=True)
print("wrote", OUT / "v2_reproduction_check.csv")
