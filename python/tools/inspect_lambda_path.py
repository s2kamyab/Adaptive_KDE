"""Inspect the AKDE lambda path directly: support size, J-score and true ISE."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import metrics, registry
from akde.densities import get_target

target = get_target("f9")
X = target.sample(200, np.random.default_rng(1))

est = registry.build("akde", n_loc=60, n_scale=6)
cands = est.candidates(X)
print("n candidates:", len(cands))
fits = est.fit_path(X, cands, np.random.default_rng(7))

print("\n{:>10s} {:>10s} {:>6s} {:>7s} {:>12s} {:>12s} {:>8s}".format(
    "rho", "lam_ratio", "supp", "fallbk", "J(train)", "ISE", "mass"))
for c, f in zip(cands, fits):
    j = metrics.j_score(f, X)
    print("{:10.1e} {:10.2e} {:6d} {:>7s} {:12.5f} {:12.4e} {:8.4f}".format(
        c["rho"], c["lam_ratio"], f.diagnostics["n_selected"],
        str(f.diagnostics["fallback_used"]), j, metrics.ise(f, target), f.total_mass()))

print("\ncovariance diagnostics for the last fit:")
for k in ("cov_eig_max", "cov_eig_min", "cov_rank_numeric", "cov_effective_rank", "omega_cond"):
    print("   {:22s} {}".format(k, fits[-1].diagnostics.get(k)))
print("   n_measurements        ", fits[-1].diagnostics.get("n_measurements"))
print("   n_dictionary          ", fits[-1].diagnostics.get("n_dictionary"))

