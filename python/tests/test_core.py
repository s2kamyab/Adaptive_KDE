"""Correctness checks for the numerical core.

These verify the pieces every reported number depends on: the closed-form
Gaussian overlap against quadrature, the Marron-Wand parameters against their
defining properties, the analytic ISE against numerical integration, the
solvers against their KKT conditions, and the bandwidth selectors against
independently known behaviour.

Run with:  python -m pytest tests -q      (or: python tests/test_core.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akde import densities, gaussops, metrics, solvers, bandwidths  # noqa: E402
from akde.model import mixture_density  # noqa: E402


def _quad(fn, lo, hi, n=200001):
    x = np.linspace(lo, hi, n)
    return np.trapezoid(fn(x), x)


# --------------------------------------------------------------------------
# Gaussian overlap identity (manuscript Proposition 2)
# --------------------------------------------------------------------------

def test_overlap_matches_quadrature():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu, nu = rng.normal(0, 2, 2)
        s2, t2 = rng.uniform(0.01, 3, 2)
        analytic = float(gaussops.overlap_diag([[mu]], [[s2]], [[nu]], [[t2]])[0, 0])

        def prod(x):
            a = np.exp(-0.5 * (x - mu) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
            b = np.exp(-0.5 * (x - nu) ** 2 / t2) / np.sqrt(2 * np.pi * t2)
            return a * b

        lo = min(mu, nu) - 12 * np.sqrt(max(s2, t2))
        hi = max(mu, nu) + 12 * np.sqrt(max(s2, t2))
        assert np.isclose(analytic, _quad(prod, lo, hi), rtol=1e-8, atol=1e-12)
    print("PASS  overlap identity == quadrature")


def test_overlap_multivariate():
    """Covariances add in d dimensions too (the Remark's multivariate extension)."""
    rng = np.random.default_rng(1)
    d = 3
    mu, nu = rng.normal(0, 1, d), rng.normal(0, 1, d)
    s2, t2 = rng.uniform(0.2, 2, d), rng.uniform(0.2, 2, d)
    analytic = float(gaussops.overlap_diag([mu], [s2], [nu], [t2])[0, 0])
    # against the full-covariance path
    full = float(gaussops.overlap_full([mu], [np.diag(s2)], [nu], [np.diag(t2)])[0, 0])
    assert np.isclose(analytic, full, rtol=1e-12)
    print("PASS  multivariate overlap: diagonal path == full-covariance path")


# --------------------------------------------------------------------------
# Marron-Wand suite
# --------------------------------------------------------------------------

def test_marron_wand_integrates_to_one():
    for spec in densities.marron_wand_suite():
        lo, hi = spec.plot_range(pad=12)
        total = _quad(lambda x: spec.pdf(x[:, None]), lo, hi, n=400001)
        assert np.isclose(total, 1.0, atol=2e-4), f"{spec.name}: mass {total}"
    print("PASS  all 15 Marron-Wand densities integrate to 1")


def test_marron_wand_known_moments():
    """Spot-check against the published definitions.

    MW1 is N(0,1). MW6 (bimodal) is symmetric about 0 with variance
    4/9 + 1 = 1.444. MW7 (separated bimodal) has variance 1/4 + 9/4 = 2.5.
    """
    suite = densities.marron_wand_suite()
    for name, idx, mean, var in [("f0", 0, 0.0, 1.0), ("f5", 5, 0.0, 4 / 9 + 1.0),
                                 ("f6", 6, 0.0, 0.25 + 2.25)]:
        s = suite[idx]
        w, m, v = s.weights, s.means[:, 0], s.variances[:, 0]
        got_mean = float(w @ m)
        got_var = float(w @ (v + m ** 2) - got_mean ** 2)
        assert np.isclose(got_mean, mean, atol=1e-12), f"{name} mean {got_mean}"
        assert np.isclose(got_var, var, atol=1e-12), f"{name} var {got_var}"
    print("PASS  Marron-Wand moments match the published definitions")


def test_self_overlap_matches_quadrature():
    for spec in densities.marron_wand_suite()[:6]:
        lo, hi = spec.plot_range(pad=12)
        num = _quad(lambda x: spec.pdf(x[:, None]) ** 2, lo, hi, n=400001)
        ana = spec.self_overlap()
        assert np.isclose(num, ana, rtol=2e-4), f"{spec.name}: {num} vs {ana}"
    print("PASS  analytic int f^2 == quadrature")


# --------------------------------------------------------------------------
# ISE
# --------------------------------------------------------------------------

def test_analytic_ise_matches_quadrature():
    rng = np.random.default_rng(3)
    target = densities.marron_wand_suite()[8]        # trimodal
    X = target.sample(200, rng)
    h = bandwidths.silverman(X[:, 0])
    fit = mixture_density(np.full(200, 1 / 200), X, np.full((200, 1), h ** 2))

    ana = metrics.ise(fit, target)
    lo, hi = -12.0, 12.0
    num = _quad(lambda x: (fit.pdf(x[:, None]) - target.pdf(x[:, None])) ** 2, lo, hi, n=400001)
    assert np.isclose(ana, num, rtol=1e-4), f"{ana} vs {num}"
    print(f"PASS  analytic ISE == quadrature  ({ana:.6g} vs {num:.6g})")


def test_ise_zero_for_identical():
    target = densities.marron_wand_suite()[1]
    fit = mixture_density(target.weights, target.means, target.variances)
    assert metrics.ise(fit, target) < 1e-18
    print("PASS  ISE of a density against itself is 0")


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def test_nonneg_lasso_kkt():
    rng = np.random.default_rng(4)
    n = 60
    A = rng.normal(size=(80, n))
    H = A.T @ A
    g = A.T @ rng.normal(size=80)
    lam_max = solvers.lambda_max_nonneg(g)
    lams = np.geomspace(lam_max, lam_max * 1e-3, 10)
    for res in solvers.nonneg_lasso_path(H, g, lams):
        assert np.all(res.x >= 0), "non-negativity violated"
        # kkt_residual is reported relative to the problem scale
        assert res.kkt_residual < 1e-7, f"KKT {res.kkt_residual}"
    # at lam_max the solution must be exactly zero
    first = solvers.nonneg_lasso_path(H, g, np.array([lam_max * (1 + 1e-9)]))[0]
    assert np.all(first.x == 0), "solution not zero at lambda_max"
    print("PASS  non-negative lasso satisfies KKT; collapses to 0 at lambda_max")


def test_signed_lasso_kkt():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(70, 50))
    H = A.T @ A
    g = A.T @ rng.normal(size=70)
    lam_max = solvers.lambda_max_signed(g)
    for res in solvers.signed_lasso_path(H, g, np.geomspace(lam_max, lam_max * 1e-3, 8)):
        assert res.kkt_residual < 1e-7, f"KKT {res.kkt_residual}"
    print("PASS  signed lasso satisfies KKT")


def test_simplex_qp_against_bruteforce():
    """Small random simplex QPs against a dense projected-gradient reference."""
    rng = np.random.default_rng(6)
    for _ in range(15):
        k = rng.integers(2, 9)
        A = rng.normal(size=(k + 3, k))
        H = A.T @ A
        c = rng.normal(size=k)
        res = solvers.simplex_qp(H, c)
        assert np.all(res.x >= -1e-12), "negative weight"
        assert np.isclose(res.x.sum(), 1.0, atol=1e-9), f"sum {res.x.sum()}"

        # brute-force reference: long projected gradient descent
        x = np.full(k, 1 / k)
        L = np.linalg.eigvalsh(H)[-1]
        for _ in range(200000):
            x = solvers.project_simplex(x - (H @ x + c) / L)
        ref = 0.5 * x @ H @ x + c @ x
        assert res.objective <= ref + 1e-9, f"{res.objective} > {ref}"
    print("PASS  simplex QP matches / beats projected-gradient reference")


def test_simplex_qp_large():
    """Frank-Wolfe path on a dictionary-sized problem."""
    rng = np.random.default_rng(7)
    k = 900
    A = rng.normal(size=(120, k))
    H = A.T @ A + 1e-8 * np.eye(k)
    c = -np.abs(rng.normal(size=k))
    res = solvers.simplex_qp(H, c)
    assert np.isclose(res.x.sum(), 1.0, atol=1e-9)
    assert np.all(res.x >= -1e-12)
    assert res.kkt_residual < 1e-6, f"KKT {res.kkt_residual}"
    print(f"PASS  large simplex QP (k={k}) solved, support={int((res.x > 1e-10).sum())}")


def test_project_simplex():
    rng = np.random.default_rng(8)
    for _ in range(50):
        v = rng.normal(size=rng.integers(2, 20))
        p = solvers.project_simplex(v)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p >= 0)
    print("PASS  simplex projection")


# --------------------------------------------------------------------------
# Bandwidth selectors
# --------------------------------------------------------------------------

def test_bandwidth_selectors_sane():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(500)
    # For N(0,1) with n=500 the AMISE-optimal Gaussian-kernel bandwidth is
    # (4/3)^(1/5) n^(-1/5) ~ 0.306.
    opt = (4.0 / 3.0) ** 0.2 * 500 ** -0.2
    for rule in ("silverman", "sj", "sj_dpi", "botev_isj"):
        h = bandwidths.select_bandwidth(x, rule)
        assert 0.4 * opt < h < 2.5 * opt, f"{rule}: h={h:.4f} vs optimal {opt:.4f}"
        print(f"      {rule:10s} h={h:.4f}  (AMISE-optimal {opt:.4f})")

    # LSCV is deliberately held to a much looser band.  Its criterion is
    # extremely flat near the optimum and routinely has a second, lower local
    # minimum well below it, so it undersmooths badly -- a documented property
    # (Hall & Marron 1991), not an implementation defect.  On this sample the
    # global minimum is at h ~ 0.108 with J = -0.27234 against J = -0.27029 at
    # the sensible h ~ 0.28; R's bw.ucv selects the same point.  The
    # closed-form score is separately verified against the leave-one-out
    # definition in test_lscv_score_matches_definition.
    h = bandwidths.select_bandwidth(x, "lscv")
    assert 0.2 * opt < h < 2.5 * opt, f"lscv: h={h:.4f} vs optimal {opt:.4f}"
    print(f"      {'lscv':10s} h={h:.4f}  (AMISE-optimal {opt:.4f}; known to undersmooth)")
    print("PASS  bandwidth selectors behave as documented for Gaussian data")


def test_lscv_undersmoothing_is_real():
    """Confirm the LSCV minimum is a true global minimum, not a search failure."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(500)
    h_sel = bandwidths.select_bandwidth(x, "lscv")
    grid = np.geomspace(0.01, 1.5, 300)
    scores = np.array([bandwidths.lscv_score(x, h) for h in grid])
    assert bandwidths.lscv_score(x, h_sel) <= scores.min() + 1e-9, "search missed the minimum"
    print(f"PASS  LSCV selection attains the global minimum of its own criterion "
          f"(h={h_sel:.4f})")


def test_lscv_score_matches_definition():
    """The closed-form LSCV must equal its own definition computed directly."""
    rng = np.random.default_rng(10)
    x = rng.standard_normal(60)
    h = 0.4
    closed = bandwidths.lscv_score(x, h)

    n = x.size
    fit = mixture_density(np.full(n, 1 / n), x[:, None], np.full((n, 1), h ** 2))
    int_f2 = fit.diagnostics and None  # noqa
    int_f2 = gaussops.self_overlap_quad(np.full(n, 1 / n), x[:, None], np.full((n, 1), h ** 2))
    loo = 0.0
    for i in range(n):
        others = np.delete(x, i)
        loo += np.mean(np.exp(-0.5 * ((x[i] - others) / h) ** 2) / (h * np.sqrt(2 * np.pi)))
    direct = int_f2 - 2.0 * loo / n
    assert np.isclose(closed, direct, rtol=1e-10), f"{closed} vs {direct}"
    print("PASS  closed-form LSCV == leave-one-out definition")


# --------------------------------------------------------------------------
# J-score
# --------------------------------------------------------------------------

def test_j_score_tracks_ise():
    """J-score differs from ISE by the constant int f0^2, in expectation.

    This is why it can tune hyperparameters without seeing the truth.
    """
    rng = np.random.default_rng(11)
    target = densities.marron_wand_suite()[5]
    X = target.sample(400, rng)
    val = target.sample(40000, rng)
    const = target.self_overlap()
    for h in (0.1, 0.2, 0.35, 0.6, 1.0):
        n = X.shape[0]
        fit = mixture_density(np.full(n, 1 / n), X, np.full((n, 1), h ** 2))
        j = metrics.j_score(fit, val)
        ise = metrics.ise(fit, target)
        assert abs((j + const) - ise) < 0.02, f"h={h}: J+c={j + const:.4f} ISE={ise:.4f}"
    print("PASS  J-score + int f0^2 tracks ISE")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
