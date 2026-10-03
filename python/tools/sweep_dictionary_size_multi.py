"""Check the dictionary-size effect across several densities and draws.

One draw on one density is not enough to set a default. This repeats the
comparison over three densities of different character and three seeds.
"""
import os
import sys
from pathlib import Path
from time import perf_counter

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from akde import metrics, registry
from akde.densities import get_target

SIZES = ((60, 5), (100, 6), (150, 10))
print("{:>5s} {:>6s} {:>6s} {:>26s} {:>8s}".format("tgt", "N", "M", "median ISE (3 draws)", "time/fit"))
print("-" * 58)
for tname in ("f5", "f9", "f13"):
    target = get_target(tname)
    for N in (500, 5000):
        for (nl, ns) in SIZES:
            ises, times = [], []
            for seed in (1, 2, 3):
                X = target.sample(N, np.random.default_rng(seed))
                est = registry.build("akde", n_loc=nl, n_scale=ns)
                t0 = perf_counter()
                fit = est.fit(X, np.random.default_rng(7))
                times.append(perf_counter() - t0)
                ises.append(metrics.ise(fit, target))
            print("{:>5s} {:6d} {:6d} {:26.4e} {:7.1f}s".format(
                tname, N, nl * ns, float(np.median(ises)), float(np.median(times))))
            sys.stdout.flush()
    print()

