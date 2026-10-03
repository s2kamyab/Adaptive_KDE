import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import registry, metrics, solvers
from akde.densities import get_target

target = get_target("f9")
X = target.sample(200, np.random.default_rng(1))

for name in ["kde_sj", "kde_lscv", "kde_botev", "abramson", "gmm_em",
             "thin_kt", "spades", "spades_simplex", "song", "akde"]:
    est = registry.build(name, n_loc=150, n_scale=10)
    t0 = perf_counter()
    try:
        fit = est.fit(X, np.random.default_rng(7))
        dt = perf_counter() - t0
        print("{:18s} {:8.2f}s  ISE={:.3e}  supp={}".format(
            name, dt, metrics.ise(fit, target), fit.n_active()))
    except Exception as e:
        print("{:18s} FAILED {}: {}".format(name, type(e).__name__, e))
    sys.stdout.flush()

