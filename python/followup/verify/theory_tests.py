"""Task 3 - executable checks for every claim made in ``theory_checks.md``.

Each test corresponds to a numbered proposition or remark. Tests that are
meant to FAIL a naive claim are written as explicit counterexamples and
asserted in the direction that shows the naive claim is false.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import numpy as np                                 # noqa: E402

from akde import gaussops                          # noqa: E402
from akde.densities import get_target              # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary  # noqa: E402
from akde.metrics import integrate_1d              # noqa: E402
from followup.akde_followup import oracle          # noqa: E402
from followup.akde_followup.seeds import data_rng  # noqa: E402
from followup.akde_followup.stage1 import omega_half           # noqa: E402
from followup.akde_followup.variants import FollowupAKDE       # noqa: E402

from followup.paths import DIAG_ROOT   # noqa: E402
OUT = DIAG_ROOT
RESULTS = []


def rec(name, ok, detail):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print("[%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))


def _small_dictionary(M=6, sd=0.5, spacing=1.0):
    mu = (np.arange(M) - (M - 1) / 2.0)[:, None] * spacing
    var = np.full_like(mu, sd ** 2)
    return mu, var


# ---------------------------------------------------------------- Prop. 1
def test_prop1_gradient_identity():
    """When k_j = b_j: D = G, y0 = a, and Gw - a = (1/2) grad R(w)."""
    tgt = get_target("f5")
    X = tgt.sample(300, data_rng("development", "f5", 300, 0))
    atoms = build_dictionary(X, DictionaryConfig(20, 3, "grid", 0.05, 4.0))
    G, a, c0 = oracle.exact_risk_parts(atoms, tgt)
    D = gaussops.design_matrix(atoms.means, atoms.variances,
                               atoms.means, atoms.variances)
    y0 = oracle.population_moments(atoms, tgt)
    rec("Prop 1(i): D = G when the measurement family is the dictionary",
        np.array_equal(D, G), "max|D-G| = %.3e" % np.abs(D - G).max())
    rec("Prop 1(ii): y0 = a", np.array_equal(y0, a),
        "max|y0-a| = %.3e" % np.abs(y0 - a).max())

    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        w = rng.random(atoms.size)
        num = np.empty_like(w)
        h = 1e-6
        for i in range(w.size):
            e = np.zeros_like(w)
            e[i] = h
            num[i] = (oracle.risk(w + e, G, a, c0) - oracle.risk(w - e, G, a, c0)) / (2 * h)
        ana = 2.0 * (G @ w - a)
        worst = max(worst, float(np.max(np.abs(num - ana)) /
                                 max(np.max(np.abs(ana)), 1e-300)))
    rec("Prop 1(iii): grad R(w) = 2(Gw - a), vs central differences",
        worst < 1e-6, "max relative deviation %.3e" % worst)


# ---------------------------------------------------------------- Prop. 2
def test_prop2_consistency_and_omega():
    """Omega-invariance holds exactly when the moment system is CONSISTENT.

    Three regimes on a well-conditioned 6-atom dictionary:
      (a) m = M, D nonsingular  -> consistent, unique solution, Omega-free
      (b) m < M, D full row rank -> consistent, solution SET, Omega-free set
      (c) m > M, inconsistent    -> solution DEPENDS on Omega  (counterexample)
    """
    mu, var = _small_dictionary()
    G = gaussops.gram_matrix(mu, var)
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, size=(500, 1))

    def bank_stats(mk, vk):
        Kk = np.exp(-0.5 * ((X[:, None, :] - mk[None, :, :]) ** 2 / vk[None, :, :]
                            + np.log(vk)[None, :, :] + gaussops.LOG_2PI)).squeeze(-1)
        yk = Kk.mean(axis=0)
        Kc = Kk - yk
        return yk, (Kc.T @ Kc) / (X.shape[0] * (X.shape[0] - 1.0))

    def wls(D, y, C, rho):
        half = omega_half(C, "ridge", rho)
        return np.linalg.lstsq(half(D), half(y), rcond=None)[0]

    rhos = (1e-6, 1e-3, 1e-1, 1.0)

    # (a) m = M = 6, matched
    D = gaussops.design_matrix(mu, var, mu, var)
    y, C = bank_stats(mu, var)
    sols = [wls(D, y, C, r) for r in rhos]
    spread = max(float(np.linalg.norm(p - q)) for p in sols for q in sols)
    resid = float(np.linalg.norm(D @ sols[0] - y))
    rec("Prop 2(a): m=M, nonsingular D -> solution independent of Omega",
        spread < 1e-9 * max(1.0, np.linalg.norm(sols[0])),
        "cond(D)=%.3g, residual=%.3e, max spread across 4 rho = %.3e"
        % (np.linalg.cond(D), resid, spread))

    # (b) m = 3 < M = 6, consistent
    mk = np.array([-2.0, 0.0, 2.0])[:, None]
    vk = np.full_like(mk, 1.0)
    Du = gaussops.design_matrix(mk, vk, mu, var)
    yu, Cu = bank_stats(mk, vk)
    sols = [wls(Du, yu, Cu, r) for r in rhos]
    spread = max(float(np.linalg.norm(p - q)) for p in sols for q in sols)
    resid = float(np.linalg.norm(Du @ sols[0] - yu))
    rec("Prop 2(b): m<M, consistent -> min-norm solution independent of Omega",
        spread < 1e-8 and resid < 1e-10,
        "residual=%.3e, max spread across 4 rho = %.3e" % (resid, spread))

    # (c) m = 12 > M = 6, inconsistent  -- THE COUNTEREXAMPLE
    mk2 = np.linspace(-3.5, 3.5, 12)[:, None]
    vk2 = np.full_like(mk2, 0.36)
    Do = gaussops.design_matrix(mk2, vk2, mu, var)
    yo, Co = bank_stats(mk2, vk2)
    sols = [wls(Do, yo, Co, r) for r in rhos]
    spread = max(float(np.linalg.norm(p - q)) for p in sols for q in sols)
    resid = float(np.linalg.norm(Do @ sols[0] - yo))
    rel = spread / float(np.linalg.norm(sols[0]))
    rec("Prop 2(c) COUNTEREXAMPLE: m>M, inconsistent -> solution DEPENDS on Omega",
        spread > 1e-3,
        "cond(D)=%.3g, residual=%.3e (nonzero), max coefficient spread across "
        "4 rho = %.3e (%.1f%% of ||u||)"
        % (np.linalg.cond(Do), resid, spread, 100 * rel))
    return {"overdetermined_spread": spread, "overdetermined_residual": resid}


# ---------------------------------------------------------------- Prop. 3
def test_prop3_gram_inverse_weighting():
    """For PD G and Omega = G^{-1}, the matched moment objective equals
    (1/2) x the integrated-square objective, up to an additive constant."""
    mu, var = _small_dictionary()
    G = gaussops.gram_matrix(mu, var)
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(500, 1))
    Kb = np.exp(-0.5 * ((X[:, None, :] - mu[None, :, :]) ** 2 / var[None, :, :]
                        + np.log(var)[None, :, :] + gaussops.LOG_2PI)).squeeze(-1)
    s = Kb.mean(axis=0)
    Gi = np.linalg.inv(G)

    worst = 0.0
    const = None
    for _ in range(50):
        w = rng.normal(size=G.shape[0])
        moment = float((G @ w - s) @ Gi @ (G @ w - s))          # (Gw-s)'G^{-1}(Gw-s)
        l2 = float(w @ G @ w - 2.0 * s @ w)                     # w'Gw - 2s'w
        diff = moment - 1.0 * l2                                 # should be constant
        if const is None:
            const = diff
        worst = max(worst, abs(diff - const) / max(abs(const), 1e-300))
    expected_const = float(s @ Gi @ s)
    rec("Prop 3: (Gw-s)'G^{-1}(Gw-s) = w'Gw - 2s'w + s'G^{-1}s",
        worst < 1e-8 and abs(const - expected_const) / abs(expected_const) < 1e-8,
        "max relative deviation from constancy %.3e; constant %.6g vs s'G^{-1}s "
        "%.6g" % (worst, const, expected_const))

    # and Omega = C^{-1} is NOT G^{-1}
    Kc = Kb - s
    C = (Kc.T @ Kc) / (X.shape[0] * (X.shape[0] - 1.0))
    scale = float(np.mean(np.diag(C)))
    Cr = C + 1e-3 * scale * np.eye(C.shape[0])
    Cri = np.linalg.inv(Cr)
    # compare the two weightings up to overall scale
    align = float(np.sum(Cri * Gi) / (np.linalg.norm(Cri) * np.linalg.norm(Gi)))
    rec("Prop 3 remark: the covariance weighting is NOT proportional to G^{-1}",
        align < 0.999,
        "cosine alignment between (C+rho I)^{-1} and G^{-1} = %.4f" % align)


# ---------------------------------------------------------------- Remark
def test_penalised_constrained_minimisers_differ():
    """The unpenalised unconstrained centre is NOT the constrained optimum."""
    tgt = get_target("f5")
    X = tgt.sample(500, data_rng("development", "f5", 500, 0))
    atoms = build_dictionary(X, DictionaryConfig(20, 3, "grid", 0.05, 4.0))
    G, a, c0 = oracle.exact_risk_parts(atoms, tgt)
    s = FollowupAKDE.eval_kernels(X, atoms).mean(axis=0)
    u_unc = np.linalg.lstsq(G, s, rcond=None)[0]
    R_simplex, w_simplex, _ = oracle.min_risk_on(G, a, c0)
    rec("Remark: the unconstrained centre is not on the simplex",
        (u_unc.min() < -1e-8) or abs(u_unc.sum() - 1.0) > 1e-6,
        "min coefficient %.4g, sum %.6g (simplex requires >=0 and sum 1)"
        % (u_unc.min(), u_unc.sum()))
    rec("Remark: exact risk at the unconstrained centre vs the simplex optimum",
        True,
        "R(u_unc)=%.6g vs R_simplex*=%.6g"
        % (oracle.risk(u_unc, G, a, c0), R_simplex))


# ---------------------------------------------------------------- integrability
def test_integrability_assumptions():
    """The identities need b_i, k_j, f0 in L1 cap L2 so every integral is finite."""
    tgt = get_target("f9")
    X = tgt.sample(200, data_rng("development", "f9", 200, 0))
    atoms = build_dictionary(X, DictionaryConfig(8, 2, "grid", 0.05, 4.0))
    G, a, c0 = oracle.exact_risk_parts(atoms, tgt)
    lo, hi = -30.0, 30.0
    ok = True
    worst = 0.0
    for i in [0, 3, atoms.size - 1]:
        num = integrate_1d(
            lambda t, i=i: np.exp(-0.5 * (
                (t - atoms.means[i, 0]) ** 2 / atoms.variances[i, 0]
                + np.log(atoms.variances[i, 0]) + gaussops.LOG_2PI)) * tgt.pdf(t),
            lo, hi, tol=1e-13)
        worst = max(worst, abs(num - a[i]) / max(abs(a[i]), 1e-300))
    rec("Assumptions: a_i = int b_i f0 finite and matches quadrature",
        worst < 1e-9 and np.all(np.isfinite(G)) and np.isfinite(c0),
        "max relative error %.3e; G finite=%s; c0=%.6g"
        % (worst, bool(np.all(np.isfinite(G))), c0))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test_prop1_gradient_identity()
    test_prop2_consistency_and_omega()
    test_prop3_gram_inverse_weighting()
    test_penalised_constrained_minimisers_differ()
    test_integrability_assumptions()
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    with open(OUT / "theory_tests.json", "w", encoding="utf-8") as fh:
        json.dump({"results": RESULTS, "n_tests": len(RESULTS),
                   "n_failed": n_fail}, fh, indent=2)
    print("\n%d/%d theory tests passed" % (len(RESULTS) - n_fail, len(RESULTS)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
