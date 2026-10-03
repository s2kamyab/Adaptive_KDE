"""Targeted verification of the AKDE implementation before changing anything.

Every check prints PASS/FAIL and the numbers behind it.  These are the checks
named in the follow-up brief; the 16 checks already in tests/test_core.py are
run separately and are not duplicated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akde import gaussops, solvers                                   # noqa: E402
from akde.bandwidths import silverman                                # noqa: E402
from akde.densities import get_target                                # noqa: E402
from akde.dictionary import (DictionaryConfig, build_dictionary,     # noqa: E402
                             build_measurements)
from akde.estimators.akde import AKDE, AKDEConfig                    # noqa: E402
from akde.estimators.song import SongConfig                          # noqa: E402
from akde.metrics import integrate_1d                                # noqa: E402
from akde.registry import shared_dictionary                          # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print("[%s] %s   %s" % (tag, name, detail))


def gpdf(t, mu, var):
    return np.exp(-0.5 * ((t - mu) ** 2 / var + np.log(var) + gaussops.LOG_2PI))


rng = np.random.default_rng(20260927)
tgt = get_target("f9")            # MW10 claw
X = tgt.sample(500, np.random.default_rng(1))

dcfg = shared_dictionary(20, 3)
atoms = build_dictionary(X, dcfg)
meas = build_measurements(X, "independent", n_loc=12, n_scale=3,
                          scale_lo_mult=0.25, scale_hi_mult=2.0)

# ---------------------------------------------------------------- 1. D, G vs quadrature
D = gaussops.design_matrix(meas.means, meas.variances, atoms.means, atoms.variances)
LO, HI = float(X.min()) - 15, float(X.max()) + 15
err = 0.0
for j in [0, 5, 17, meas.size - 1]:
    for i in [0, 3, 29, atoms.size - 1]:
        num = integrate_1d(
            lambda t, j=j, i=i: gpdf(t, meas.means[j, 0], meas.variances[j, 0])
            * gpdf(t, atoms.means[i, 0], atoms.variances[i, 0]),
            LO, HI, tol=1e-13)
        err = max(err, abs(num - D[j, i]) / max(abs(D[j, i]), 1e-12))
check("D_ji = int k_j b_i matches quadrature", err < 1e-7, "max rel err %.3e" % err)

G = gaussops.gram_matrix(atoms.means, atoms.variances)
gerr = 0.0
for j, i in [(0, 0), (2, 11), (30, 31), (atoms.size - 1, atoms.size - 1)]:
    num = integrate_1d(
        lambda t, j=j, i=i: gpdf(t, atoms.means[j, 0], atoms.variances[j, 0])
        * gpdf(t, atoms.means[i, 0], atoms.variances[i, 0]),
        LO, HI, tol=1e-13)
    gerr = max(gerr, abs(num - G[j, i]) / max(abs(G[j, i]), 1e-12))
check("G_jk = int b_j b_k matches quadrature", gerr < 1e-7, "max rel err %.3e" % gerr)

# ---------------------------------------------------------------- 2. analytic ISE vs quadrature
w = rng.random(atoms.size)
w /= w.sum()
ana = gaussops.ise_mixtures(w, atoms.means, atoms.variances,
                            tgt.weights, tgt.means, tgt.variances)
num = integrate_1d(lambda t: (gaussops.mixture_pdf_diag(t, w, atoms.means, atoms.variances)
                              - tgt.pdf(t)) ** 2, -20, 20, tol=1e-13)
check("analytic ISE matches quadrature", abs(ana - num) / ana < 1e-8,
      "%.10g vs %.10g" % (ana, num))

# ---------------------------------------------------------------- 3. sd vs variance, unit mass
f5 = get_target("f5")
check("MW6 stores variance (sd^2), not sd", np.allclose(f5.variances, (2 / 3) ** 2),
      "variances %s" % np.unique(f5.variances))
check("MW10 narrow components var = 0.01", np.isclose(np.min(tgt.variances), 0.01),
      "min var %g" % np.min(tgt.variances))
worst = 0.0
for name in ["f%d" % i for i in range(15)]:
    t = get_target(name)
    m = integrate_1d(lambda z: t.pdf(z), *t.plot_range(pad=12), tol=1e-12)
    worst = max(worst, abs(m - 1))
check("all 15 MW densities integrate to 1", worst < 1e-9, "max |mass-1| = %.3e" % worst)

# ---------------------------------------------------------------- 4. covariance of the sample mean
def eval_kernels(Xa, mb):
    diff = Xa[:, None, :] - mb.means[None, :, :]
    v = mb.variances[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff / v + np.log(v) + gaussops.LOG_2PI, axis=2))


K = eval_kernels(X, meas)
N = X.shape[0]
y = K.mean(axis=0)
Kc = K - y
C_code = (Kc.T @ Kc) / (N * (N - 1.0))
C_ref = np.cov(K, rowvar=False, ddof=1) / N
check("C_hat == Cov(k(X))/N (covariance of the sample mean)",
      np.allclose(C_code, C_ref, rtol=1e-10, atol=1e-18),
      "max abs diff %.3e" % np.abs(C_code - C_ref).max())

# ---------------------------------------------------------------- 5. Stage 1 objective / KKT
apply_omega, covd = AKDE._omega_apply(C_code, "ridge", 1e-3, 1e-6)
H1 = D.T @ apply_omega(D)
H1 = 0.5 * (H1 + H1.T)
g = D.T @ apply_omega(y)
lam = 0.1 * solvers.lambda_max_nonneg(g)
res = solvers.nonneg_lasso_path(H1, g, np.array([lam]))[0]
u = res.x
lhs = 0.5 * float((D @ u - y) @ apply_omega(D @ u - y)) + lam * u.sum()
rhs = res.objective + 0.5 * float(y @ apply_omega(y))
check("stage-1 objective == 0.5(Du-y)'Om(Du-y)+lam*sum(u)",
      abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-9, "%.10g vs %.10g" % (lhs, rhs))
check("stage-1 has no unit-sum constraint", abs(u.sum() - 1.0) > 1e-6,
      "sum(u) = %.6g" % u.sum())
grad = H1 @ u - g + lam
kkt = max(np.abs(grad[u > 0]).max(initial=0.0), max(0.0, -grad[u <= 0].min(initial=0.0)))
refscale = abs(g).max() + np.mean(np.diag(H1))
# This one is EXPECTED to fail on V2's solver and is recorded, not asserted:
# cyclic coordinate descent stalls on a numerically singular Hessian and the
# 500-sweep cap binds.  See audit_findings.md §3.1 and
# followup/scripts/compare_stage1_solvers.py.
print("[NOTE] stage-1 relative KKT on this instance: %.3e (converged=%s, "
      "iters=%d) — V2's solver does not reach its own 1e-8 tolerance; this is "
      "a recorded defect, not a script failure."
      % (kkt / refscale, res.converged, res.iterations))
check("stage-1 KKT residual is at least as good as V2 reports for it",
      kkt / refscale < 1e-3, "rel KKT %.3e" % (kkt / refscale))

# ---------------------------------------------------------------- 6. Stage 2
S = np.flatnonzero(u > 1e-3 * u.max())
DS = D[:, S]
H2 = DS.T @ apply_omega(DS)
H2 = 0.5 * (H2 + H2.T)
c2 = -DS.T @ apply_omega(y)
r2 = solvers.simplex_qp(H2, c2, x0=u[S] / u[S].sum())
a = r2.x
check("stage-2 feasible (simplex)", abs(a.sum() - 1) < 1e-10 and a.min() >= -1e-12,
      "sum %.12g  min %.3e" % (a.sum(), a.min()))


def loss(v):
    return 0.5 * float((DS @ v - y) @ apply_omega(DS @ v - y))


norm_u = u[S] / u[S].sum()
check("stage-2 loss <= normalised stage-1 coefficients on same support",
      loss(a) <= loss(norm_u) + 1e-12 * max(1.0, abs(loss(norm_u))),
      "refit %.10g vs normalised %.10g" % (loss(a), loss(norm_u)))

# ---------------------------------------------------------------- 7. threshold indexing
umax = u.max()
S_ref = np.array([i for i in range(u.size) if u[i] > 1e-3 * umax])
check("support threshold indexing correct", np.array_equal(S, S_ref), "|S|=%d" % S.size)

# ---------------------------------------------------------------- 8. Q1 dictionary identity
a300 = build_dictionary(X, shared_dictionary(60, 5))
b300 = build_dictionary(X, DictionaryConfig(n_loc=60, n_scale=5, location_rule="grid",
                                            scale_lo_mult=0.05, scale_hi_mult=4.0))
check("dictionary build deterministic for a given (n_loc,n_scale)",
      np.array_equal(a300.means, b300.means) and np.array_equal(a300.variances, b300.variances))
a1500 = build_dictionary(X, shared_dictionary(150, 10))
shared_loc = int(np.isin(np.round(a300.means[:, 0], 12),
                         np.round(a1500.means[:, 0], 12)).sum())
shared_sd = int(np.isin(np.round(np.unique(a300.variances), 12),
                        np.round(np.unique(a1500.variances), 12)).sum())
check("60x5 dictionary is NOT a sub-bank of the 150x10 one",
      shared_loc < a300.size,
      "shared locations %d/%d, shared scales %d/5" % (shared_loc, a300.size, shared_sd))

# ---------------------------------------------------------------- 9. Q2 Song fidelity
scfg = SongConfig(dictionary=shared_dictionary(8, 2))
sa = build_dictionary(X, scfg.dictionary)
h = silverman(X[:, 0])
s2 = (1.0 * h) ** 2
Qc = gaussops.overlap_diag(sa.means, sa.variances + s2, sa.means, sa.variances)
qc = gaussops.overlap_diag(sa.means, sa.variances + s2, X, np.zeros_like(X) + 1e-300).mean(axis=1)


def kern(a_, b_):
    return gpdf(a_ - b_, 0.0, s2)


# Reference by 2-D Gauss-Hermite, which is exact for Gaussian weights.  (A
# nested adaptive-quadrature reference was tried first and agreed only to
# ~1e-6; refining it drove the discrepancy to zero, so the residual was in the
# reference, not in the code.  480 nodes is where this converges.)
nodes, wts = np.polynomial.hermite_e.hermegauss(480)
wts = wts / np.sqrt(2.0 * np.pi)
qerr = 0.0
for i, j in [(0, 0), (0, 5), (3, 9), (sa.size - 1, sa.size - 2), (2, 7)]:
    xi = sa.means[i, 0] + np.sqrt(sa.variances[i, 0]) * nodes
    xj = sa.means[j, 0] + np.sqrt(sa.variances[j, 0]) * nodes
    ref = float(wts @ kern(xi[:, None], xj[None, :]) @ wts)
    qerr = max(qerr, abs(ref - Qc[i, j]) / max(abs(Qc[i, j]), 1e-300))
check("Song Q_ij = E_{b_i,b_j}[kappa]  (exact RKHS criterion, not a feature approximation)",
      qerr < 1e-10, "max rel err vs 2-D Gauss-Hermite (480 nodes) %.3e" % qerr)

lerr = 0.0
for i in [0, 4, sa.size - 1]:
    numl = integrate_1d(
        lambda t, i=i: gpdf(t, sa.means[i, 0], sa.variances[i, 0])
        * np.mean(kern(t[:, None], X[None, :, 0]), axis=1), -25, 25, tol=1e-11)
    lerr = max(lerr, abs(numl - qc[i]) / max(abs(qc[i]), 1e-15))
check("Song l_i = N^-1 sum_n E_{b_i}[kappa(x_n,.)]", lerr < 1e-6, "max rel err %.3e" % lerr)

# ---------------------------------------------------------------- 10. Q8 measurement operator vs N
print("\n-- measurement bank geometry as a function of N (m held at 300) --")
for n in [50, 200, 500, 1000, 5000]:
    Xn = tgt.sample(n, np.random.default_rng(7))
    mb = build_measurements(Xn, "independent", n_loc=100, n_scale=3,
                            scale_lo_mult=0.25, scale_hi_mult=2.0)
    sds = np.sqrt(np.unique(mb.variances))
    print("   N=%5d  m=%d  centres [%+.3f,%+.3f]  sd %s"
          % (n, mb.size, mb.means.min(), mb.means.max(),
             np.array2string(sds, precision=4)))
m1 = build_measurements(tgt.sample(500, np.random.default_rng(11)), "independent",
                        n_loc=100, n_scale=3, scale_lo_mult=0.25, scale_hi_mult=2.0)
m2 = build_measurements(tgt.sample(500, np.random.default_rng(12)), "independent",
                        n_loc=100, n_scale=3, scale_lo_mult=0.25, scale_hi_mult=2.0)
check("measurement operator is NOT fixed: it moves with the training sample",
      not np.allclose(m1.means, m2.means),
      "max centre shift %.4f, sd ratio %.4f"
      % (np.abs(m1.means - m2.means).max(),
         float(np.sqrt(m1.variances.max() / m2.variances.max()))))

# ---------------------------------------------------------------- 11. rank measured pre-ridge
_, info = AKDE._omega_apply(C_code, "ridge", 1e-1, 1e-6)
eig_raw = np.linalg.eigvalsh(C_code)
rank_raw = int(np.sum(eig_raw > max(eig_raw[-1], 0.0) * 1e-12))
reg = C_code + 1e-1 * np.mean(np.diag(C_code)) * np.eye(C_code.shape[0])
eig_reg = np.linalg.eigvalsh(reg)
rank_reg = int(np.sum(eig_reg > eig_reg[-1] * 1e-12))
check("cov_rank_numeric is the PRE-ridge numerical rank",
      info["cov_rank_numeric"] == rank_raw and rank_reg == C_code.shape[0],
      "reported %d, pre-ridge %d, post-ridge %d (m=%d)"
      % (info["cov_rank_numeric"], rank_raw, rank_reg, C_code.shape[0]))

print("\n" + ("ALL CHECKS PASSED" if not FAILS else "FAILURES: %s" % FAILS))
sys.exit(1 if FAILS else 0)
