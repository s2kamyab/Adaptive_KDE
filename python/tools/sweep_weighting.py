"""Separate three possible causes of a poor fit:

  1. the dictionary cannot represent the target (approximation error),
  2. the weighting matrix is wrong,
  3. the measurement bank is badly designed.

(1) is answered by an ORACLE diagnostic: the best simplex combination of the
dictionary atoms measured directly against the true density.  That is a lower
bound no data-driven fit can beat, and it uses the truth, so it is a
diagnostic only and never part of the comparison.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import gaussops, metrics, registry, solvers
from akde.densities import get_target
from akde.dictionary import build_dictionary
from akde.estimators.akde import AKDE, AKDEConfig
from akde.registry import shared_dictionary
from akde.model import mixture_density


def oracle_dictionary_ise(X, target, dict_cfg, rng):
    """Best achievable ISE over the simplex of dictionary atoms (uses the truth)."""
    atoms = build_dictionary(X, dict_cfg, rng=rng)
    G = gaussops.gram_matrix(atoms.means, atoms.variances)
    # c_i = int b_i f0, closed form against the true mixture
    c = gaussops.overlap_diag(atoms.means, atoms.variances,
                              target.means, target.variances) @ target.weights
    res = solvers.simplex_qp(2.0 * G, -2.0 * c)
    w = res.x
    keep = w > 0
    fit = mixture_density(w[keep], atoms.means[keep], atoms.variances[keep])
    return metrics.ise(fit, target), int(keep.sum()), atoms.size


target_name = sys.argv[1] if len(sys.argv) > 1 else "f9"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
target = get_target(target_name)
X = target.sample(N, np.random.default_rng(1))

print(f"target={target_name} ({target.label})  N={N}\n")

cfg = shared_dictionary(150, 10)
ise_o, supp_o, size_o = oracle_dictionary_ise(X, target, cfg, np.random.default_rng(7))
print(f"ORACLE dictionary bound (150x10 = {size_o} atoms): "
      f"ISE={ise_o:.4e} with {supp_o} atoms")
print("  -> no data-driven fit on this dictionary can do better than this.\n")

print("reference estimators:")
for name in ("kde_sj", "kde_lscv", "abramson"):
    est = registry.build(name)
    f = est.fit(X, np.random.default_rng(7))
    print(f"  {name:14s} ISE={metrics.ise(f, target):.4e}")

print("\nbest-on-path ISE by weighting scheme (oracle-selected lambda, so an "
      "upper bound on what CV can deliver):")
rho_grid = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)
for weighting in ("identity", "diagonal", "ridge", "trunc_pinv"):
    for meas_mode in ("matched", "independent"):
        c = AKDEConfig(dictionary=shared_dictionary(150, 10), weighting=weighting,
                       meas_mode=meas_mode, rho_grid=rho_grid)
        est = AKDE(c)
        cands = est.candidates(X)
        fits = est.fit_path(X, cands, np.random.default_rng(7))
        ises = np.array([metrics.ise(f, target) for f in fits])
        i = int(np.argmin(ises))
        print(f"  {weighting:11s} meas={meas_mode:12s} best ISE={ises[i]:.4e}  "
              f"(rho={cands[i]['rho']:.0e}, lam={cands[i]['lam_ratio']:.1e}, "
              f"supp={fits[i].diagnostics['n_selected']}, "
              f"rank={fits[i].diagnostics.get('cov_rank_numeric')}/"
              f"{fits[i].diagnostics.get('n_measurements')})")

