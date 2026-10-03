"""Decisive test: fit the estimator to the EXACT moment vector.

y_exact_j = int k_j(x) f0(x) dx is available in closed form (both Gaussian
mixtures).  Replacing the empirical y by y_exact removes all sampling noise
while leaving the optimisation, the dictionary and the measurement design
untouched.

  * tiny ISE with y_exact  -> the machinery is correct and the limitation is
    statistical: the estimator is chasing noise in the moment vector.
  * large ISE with y_exact -> the moment conditions do not identify the
    density at this (m, M), independently of any sampling noise.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import gaussops, metrics, solvers
from akde.densities import get_target
from akde.dictionary import build_dictionary, build_measurements
from akde.model import mixture_density
from akde.registry import shared_dictionary

target_name = sys.argv[1] if len(sys.argv) > 1 else "f9"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
target = get_target(target_name)
rng = np.random.default_rng(1)
X = target.sample(N, rng)

dcfg = shared_dictionary(150, 10)
atoms = build_dictionary(X, dcfg, rng=np.random.default_rng(7))
print(f"target={target_name}  N={N}  dictionary={atoms.size}")

for (nl, ns) in ((40, 3), (150, 5), (300, 5)):
    meas = build_measurements(X, "independent", n_loc=nl, n_scale=ns,
                              scale_lo_mult=0.25, scale_hi_mult=2.0,
                              rng=np.random.default_rng(7))
    D = gaussops.design_matrix(meas.means, meas.variances, atoms.means, atoms.variances)

    # empirical moment vector
    diff = X[:, None, :] - meas.means[None, :, :]
    v = meas.variances[None, :, :]
    K = np.exp(-0.5 * np.sum(diff * diff / v + np.log(v) + gaussops.LOG_2PI, axis=2))
    y_emp = K.mean(axis=0)

    # exact moment vector: int k_j f0, closed form
    y_exact = gaussops.overlap_diag(meas.means, meas.variances,
                                    target.means, target.variances) @ target.weights

    rel = np.linalg.norm(y_emp - y_exact) / np.linalg.norm(y_exact)
    print(f"\n m={meas.size:5d}  relative moment noise ||y_emp - y_exact|| / ||y_exact|| = {rel:.4f}")

    for label, y in (("empirical", y_emp), ("EXACT", y_exact)):
        H = D.T @ D
        H = 0.5 * (H + H.T)
        g = D.T @ y
        lam_max = solvers.lambda_max_nonneg(g)
        lams = np.geomspace(0.7, 1e-6, 20) * lam_max
        best = (np.inf, None, None)
        for c, res in zip(np.geomspace(0.7, 1e-6, 20),
                          solvers.nonneg_lasso_path(H, g, lams)):
            u = res.x
            if u.max() <= 0:
                continue
            S = np.flatnonzero(u > 1e-3 * u.max())
            DS = D[:, S]
            r2 = solvers.simplex_qp(DS.T @ DS, -(DS.T @ y))
            w = np.zeros(u.size)
            w[S] = r2.x
            keep = w > 0
            fit = mixture_density(w[keep], atoms.means[keep], atoms.variances[keep])
            e = metrics.ise(fit, target)
            if e < best[0]:
                best = (e, c, int(S.size))
        print(f"   {label:10s} best ISE over lambda path = {best[0]:.4e} "
              f"(lam_ratio={best[1]:.1e}, support={best[2]})")

