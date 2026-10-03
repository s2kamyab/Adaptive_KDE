"""How much does the estimator improve as the measurement bank grows?

The moment objective constrains the density only through m linear functionals.
With m far below the dictionary size the fit is underdetermined in a way the
L1 penalty does not fix, so m is a first-class design parameter, not a detail.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import metrics, registry
from akde.densities import get_target
from akde.estimators.akde import AKDE, AKDEConfig
from akde.registry import shared_dictionary

target_name = sys.argv[1] if len(sys.argv) > 1 else "f9"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
target = get_target(target_name)
X = target.sample(N, np.random.default_rng(1))
print(f"target={target_name} ({target.label})  N={N}")

for name in ("kde_sj", "kde_lscv", "abramson", "spades_simplex"):
    f = registry.build(name, n_loc=150, n_scale=10).fit(X, np.random.default_rng(7))
    print(f"  reference {name:16s} ISE={metrics.ise(f, target):.4e}")
print()

hdr = "{:>10s} {:>9s} {:>6s} {:>7s} {:>12s} {:>6s} {:>9s}"
print(hdr.format("weighting", "meas", "m", "rank", "best ISE", "supp", "lam"))
print("-" * 70)

for weighting in ("identity", "diagonal", "ridge", "trunc_pinv"):
    for (nl, ns) in ((40, 3), (100, 3), (200, 3), (150, 5), (300, 5)):
        cfg = AKDEConfig(
            dictionary=shared_dictionary(150, 10),
            weighting=weighting, meas_mode="independent",
            meas_n_loc=nl, meas_n_scale=ns,
            meas_scale_lo_mult=0.25, meas_scale_hi_mult=2.0,
            rho_grid=(1e-6, 1e-4, 1e-2, 1e-1, 1.0),
        )
        est = AKDE(cfg)
        cands = est.candidates(X)
        try:
            fits = est.fit_path(X, cands, np.random.default_rng(7))
            ises = np.array([metrics.ise(f, target) for f in fits])
            i = int(np.argmin(ises))
            d = fits[i].diagnostics
            print(hdr.format(weighting, f"{nl}x{ns}", str(d["n_measurements"]),
                             str(d.get("cov_rank_numeric")), f"{ises[i]:.4e}",
                             str(d["n_selected"]), f"{cands[i]['lam_ratio']:.1e}"))
        except Exception as e:
            print(f"  {weighting} {nl}x{ns} FAILED {type(e).__name__}: {e}")
    print()

