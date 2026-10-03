"""Accuracy and validity measures.

Primary accuracy is **integrated squared error**, not a grid MSE.  Where both
the truth and the estimate are Gaussian mixtures -- which covers almost the
whole comparison set, since a KDE is itself a Gaussian mixture with one
component per retained point -- ISE is computed exactly from pairwise
Gaussian overlaps.  Only estimators that are not mixtures fall back to
quadrature, and those calls record the integration domain and the achieved
convergence, as the protocol requires.

Also provided: integrated absolute error, Hellinger distance, held-out
negative log likelihood (guarded so it is only ever reported for validated
densities), the held-out integrated-square (``J``) score used for all
hyperparameter tuning, and mass/negativity diagnostics.
"""

from __future__ import annotations

import numpy as np

from . import gaussops
from .densities import MixtureSpec
from .model import FittedDensity

__all__ = [
    "IntegrationReport",
    "integrate_1d",
    "ise",
    "iae",
    "hellinger",
    "j_score",
    "held_out_nll",
    "negative_mass",
    "l1_mass_error",
]


class IntegrationReport(dict):
    """Diagnostics from an adaptive quadrature call."""


def _gauss_legendre_panels(fn, lo, hi, n_panels, n_nodes=20):
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    edges = np.linspace(lo, hi, n_panels + 1)
    a, b = edges[:-1], edges[1:]
    mid = 0.5 * (a + b)[:, None]
    half = 0.5 * (b - a)[:, None]
    pts = (mid + half * nodes[None, :]).ravel()
    vals = np.asarray(fn(pts), dtype=float).ravel()
    vals = vals.reshape(n_panels, n_nodes)
    return float(np.sum(half.ravel() * (vals @ weights)))


def integrate_1d(fn, lo: float, hi: float, tol: float = 1e-10,
                 max_panels: int = 1 << 16, report: IntegrationReport | None = None) -> float:
    """Composite Gauss-Legendre with panel doubling until convergence.

    Returns the integral; if ``report`` is supplied it is filled with the
    domain, final panel count and the last relative change, so that a run can
    later be audited for under-resolved integration rather than trusting a
    fixed grid.
    """
    n_panels = 64
    prev = _gauss_legendre_panels(fn, lo, hi, n_panels)
    rel = np.inf
    while n_panels < max_panels:
        n_panels *= 2
        cur = _gauss_legendre_panels(fn, lo, hi, n_panels)
        denom = max(abs(cur), 1e-300)
        rel = abs(cur - prev) / denom
        prev = cur
        if rel < tol:
            break
    if report is not None:
        report.update(domain=(lo, hi), panels=n_panels, rel_change=rel, converged=rel < tol)
    return prev


def _integration_domain(target: MixtureSpec, fit: FittedDensity, pad: float = 6.0) -> tuple[float, float]:
    lo, hi = target.plot_range(pad=pad)
    if fit.is_mixture:
        sd = np.sqrt(fit.variances.max(axis=0))
        lo = min(lo, float((fit.means - pad * sd).min()))
        hi = max(hi, float((fit.means + pad * sd).max()))
    return lo, hi


def ise(fit: FittedDensity, target: MixtureSpec, report: IntegrationReport | None = None) -> float:
    """Integrated squared error ``int (fhat - f0)^2``.

    Exact when the estimate is a Gaussian mixture, which is the case for
    every estimator in the essential comparison set.
    """
    if fit.is_mixture:
        return gaussops.ise_mixtures(
            fit.weights, fit.means, fit.variances,
            target.weights, target.means, target.variances,
        )
    if target.dim != 1:
        raise NotImplementedError("numeric ISE is implemented for d = 1 only")
    lo, hi = _integration_domain(target, fit)
    fn = lambda t: (fit.pdf(t) - target.pdf(t)) ** 2
    return float(integrate_1d(fn, lo, hi, report=report))


# Absolute-error and Hellinger distances have no closed form, so they need
# quadrature -- and the cost of a quadrature node is proportional to the number
# of mixture components.  A KDE at N = 5000 has 5000 components, so a converged
# integration evaluates billions of Gaussian terms.  ISE, the primary measure,
# is unaffected because it is analytic.  These secondary measures are therefore
# skipped above a component budget and the skip is recorded, rather than being
# silently computed on a grid too coarse to trust.
IAE_MAX_COMPONENTS = 1200
_IAE_MAX_PANELS = 8192


def _too_large(fit: FittedDensity) -> bool:
    return bool(fit.is_mixture and fit.weights.size > IAE_MAX_COMPONENTS)


def iae(fit: FittedDensity, target: MixtureSpec, report: IntegrationReport | None = None) -> float:
    """Integrated absolute error ``int |fhat - f0|`` (quadrature; d = 1).

    Returns NaN when the estimate has more components than
    :data:`IAE_MAX_COMPONENTS`; see the note above.
    """
    if target.dim != 1:
        raise NotImplementedError("IAE is implemented for d = 1 only")
    if _too_large(fit):
        return float("nan")
    lo, hi = _integration_domain(target, fit)
    fn = lambda t: np.abs(fit.pdf(t) - target.pdf(t))
    return float(integrate_1d(fn, lo, hi, tol=1e-8, max_panels=_IAE_MAX_PANELS,
                              report=report))


def hellinger(fit: FittedDensity, target: MixtureSpec,
              report: IntegrationReport | None = None) -> float:
    """Hellinger distance; defined only for non-negative estimates.

    Returns NaN for estimates carrying negative mass rather than silently
    clipping them, so a signed estimator is visibly absent from this column
    instead of appearing to score well on it.  Also NaN above the component
    budget, as for :func:`iae`.
    """
    if target.dim != 1:
        raise NotImplementedError("Hellinger is implemented for d = 1 only")
    if fit.is_mixture and np.any(fit.weights < -1e-14):
        return float("nan")
    if _too_large(fit):
        return float("nan")
    lo, hi = _integration_domain(target, fit)

    def fn(t):
        p = np.maximum(fit.pdf(t), 0.0)
        q = target.pdf(t)
        return (np.sqrt(p) - np.sqrt(q)) ** 2

    val = integrate_1d(fn, lo, hi, tol=1e-8, max_panels=_IAE_MAX_PANELS, report=report)
    return float(np.sqrt(max(val, 0.0) / 2.0))


def j_score(fit: FittedDensity, x_val: np.ndarray,
            domain: tuple[float, float] | None = None) -> float:
    """Held-out integrated-square score.

        J(f) = int f(x)^2 dx - (2 / n_val) sum_j f(x_j^val)

    This is the tuning criterion throughout.  It is valid for signed
    estimates, needs no knowledge of the true density, and is (up to the
    constant ``int f0^2``) an unbiased estimate of the ISE -- which is why it
    can select hyperparameters without leaking the truth into the comparison.
    """
    x_val = np.asarray(x_val, dtype=float)
    if x_val.ndim == 1:
        x_val = x_val[:, None]
    if fit.is_mixture:
        int_f2 = gaussops.self_overlap_quad(fit.weights, fit.means, fit.variances)
    else:
        if domain is None:
            raise ValueError("a domain is required for the J-score of a non-mixture estimate")
        int_f2 = integrate_1d(lambda t: fit.pdf(t) ** 2, domain[0], domain[1])
    mean_f = float(np.mean(fit.pdf(x_val)))
    return float(int_f2 - 2.0 * mean_f)


def held_out_nll(fit: FittedDensity, x_test: np.ndarray, floor: float = 1e-300,
                 require_valid: bool = True) -> tuple[float, int]:
    """Mean held-out negative log likelihood, plus the number of floored points.

    Returns ``(nan, 0)`` unless the estimate is a validated probability
    density: a likelihood score computed for an unnormalised or signed
    estimate is not comparable with one computed for a density, and the
    protocol asks for this measure to be restricted accordingly.
    """
    if require_valid and not fit.is_valid_density():
        return float("nan"), 0
    x_test = np.asarray(x_test, dtype=float)
    if x_test.ndim == 1:
        x_test = x_test[:, None]
    p = fit.pdf(x_test)
    n_floored = int(np.sum(p < floor))
    p = np.maximum(p, floor)
    return float(-np.mean(np.log(p))), n_floored


def negative_mass(fit: FittedDensity, target: MixtureSpec) -> float:
    """``int max(0, -fhat)``: how much probability mass is negative.

    Zero by construction for the two-stage estimator and for every
    simplex-constrained baseline; strictly positive for signed SPADES.
    """
    if fit.is_mixture and np.all(fit.weights >= -1e-14):
        return 0.0
    if target.dim != 1:
        return float("nan")
    lo, hi = _integration_domain(target, fit)
    return float(integrate_1d(lambda t: np.maximum(-fit.pdf(t), 0.0), lo, hi, tol=1e-9))


def l1_mass_error(fit: FittedDensity) -> float:
    """``|int fhat - 1|``, the normalisation error."""
    return float(abs(fit.total_mass() - 1.0))
