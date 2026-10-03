"""Cost and accuracy against dictionary size.

The manuscript used pq = 1500 for its Figure 1 and pq = 600 for Figure 2, so
both are faithful settings. Coordinate descent costs O(M^2) per sweep, so this
choice dominates the runtime of the whole experiment.
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

print("{:>6s} {:>6s} {:>7s} {:>8s} {:>12s} {:>6s}".format(
    "N", "n_loc", "n_scale", "M", "ISE", "time"))
print("-" * 52)
for N in (200, 1000, 5000):
    target = get_target("f9")
    X = target.sample(N, np.random.default_rng(1))
    for (nl, ns) in ((100, 6), (150, 10)):
        est = registry.build("akde", n_loc=nl, n_scale=ns)
        t0 = perf_counter()
        fit = est.fit(X, np.random.default_rng(7))
        dt = perf_counter() - t0
        print("{:6d} {:6d} {:7d} {:8d} {:12.4e} {:5.1f}s".format(
            N, nl, ns, nl * ns, metrics.ise(fit, target), dt))
        sys.stdout.flush()

