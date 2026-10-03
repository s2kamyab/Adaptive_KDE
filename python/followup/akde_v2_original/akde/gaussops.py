"""Closed-form Gaussian overlap algebra.

Everything in this experimental framework that can be written as a Gaussian
mixture is scored *analytically* rather than on a grid.  The single identity
behind all of it is Proposition 2 of the manuscript,

    int N(x; mu, S) N(x; nu, T) dx = N(mu; nu, S + T),

i.e. the covariances add.  This gives, in closed form,

  * the moment design matrix  D_ji = int k_j(x) b_i(x) dx,
  * the SPADES Gram matrix    G_jk = int b_j(x) b_k(x) dx,
  * the integrated squared error between any two Gaussian mixtures,
  * the ``int f^2`` term of the held-out integrated-square (J) score.

Avoiding quadrature matters here: several Marron-Wand densities have
components with sigma = 0.01, so a grid fine enough to resolve them is
expensive and a grid that is not fine enough silently biases every reported
error.  See ``metrics.py`` for the numeric fallback used by the (few)
estimators that are not Gaussian mixtures.

Shape conventions
-----------------
means       (K, d)
variances   (K, d)      diagonal covariance, one row per component
covs        (K, d, d)   full covariance, only for estimators that need it

The d == 1 case is special-cased throughout because it is the workhorse: the
main benchmark runs 15 univariate densities and some mixtures carry one
component per observation (N up to 5000).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LOG_2PI",
    "overlap_diag",
    "quad_overlap_diag",
    "overlap_full",
    "quad_overlap_full",
    "design_matrix",
    "gram_matrix",
    "mixture_pdf_diag",
    "mixture_pdf_full",
    "ise_mixtures",
    "self_overlap_quad",
]

LOG_2PI = float(np.log(2.0 * np.pi))

# Materialise at most this many float64 entries at once when accumulating
# quadratic forms.  32e6 entries ~ 256 MB, comfortable on a 32 GB machine
# while still letting several worker processes run side by side.
_BLOCK_ENTRIES = 8_000_000


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return a


def overlap_diag(
    means1: np.ndarray,
    vars1: np.ndarray,
    means2: np.ndarray,
    vars2: np.ndarray,
) -> np.ndarray:
    """Pairwise Gaussian overlaps for diagonal covariances.

    Returns the (K1, K2) matrix with entries ``N(mu_i; nu_j, V_i + W_j)``.

    Computed in log space and then exponentiated.  Underflow to exactly zero
    is the correct answer for well-separated narrow components (e.g. the claw
    densities) and is left to happen rather than being clipped.
    """
    m1, v1 = _as_2d(means1), _as_2d(vars1)
    m2, v2 = _as_2d(means2), _as_2d(vars2)
    if m1.shape[1] != m2.shape[1]:
        raise ValueError(f"dimension mismatch: {m1.shape[1]} vs {m2.shape[1]}")

    # s[i, j, k] = v1[i, k] + v2[j, k]
    s = v1[:, None, :] + v2[None, :, :]
    diff = m1[:, None, :] - m2[None, :, :]
    logp = -0.5 * np.sum(diff * diff / s + np.log(s) + LOG_2PI, axis=2)
    return np.exp(logp)


def quad_overlap_diag(
    w1: np.ndarray,
    means1: np.ndarray,
    vars1: np.ndarray,
    w2: np.ndarray,
    means2: np.ndarray,
    vars2: np.ndarray,
) -> float:
    """``w1^T S w2`` where ``S`` is the pairwise overlap matrix.

    Accumulated in row blocks so that a mixture with one component per
    observation (K = 5000 gives a 25M-entry matrix) never has its full
    overlap matrix materialised.
    """
    w1 = np.asarray(w1, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()
    m1, v1 = _as_2d(means1), _as_2d(vars1)
    m2, v2 = _as_2d(means2), _as_2d(vars2)

    k2 = m2.shape[0]
    rows = max(1, int(_BLOCK_ENTRIES // max(k2, 1)))
    total = 0.0
    for start in range(0, m1.shape[0], rows):
        stop = min(start + rows, m1.shape[0])
        block = overlap_diag(m1[start:stop], v1[start:stop], m2, v2)
        total += float(w1[start:stop] @ (block @ w2))
    return total


def _chol_logdet_solve(cov: np.ndarray, diff: np.ndarray) -> tuple[float, np.ndarray]:
    L = np.linalg.cholesky(cov)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    sol = np.linalg.solve(L, diff.T)
    return logdet, sol


def overlap_full(
    means1: np.ndarray,
    covs1: np.ndarray,
    means2: np.ndarray,
    covs2: np.ndarray,
) -> np.ndarray:
    """Pairwise overlaps for full covariances.

    One Cholesky per (i, j) pair, so this is only used where at least one
    mixture has few components (the EM baseline, and multivariate targets).
    Diagonal-covariance mixtures should go through :func:`overlap_diag`.
    """
    m1, m2 = _as_2d(means1), _as_2d(means2)
    c1 = np.asarray(covs1, dtype=float)
    c2 = np.asarray(covs2, dtype=float)
    d = m1.shape[1]
    out = np.empty((m1.shape[0], m2.shape[0]), dtype=float)
    for i in range(m1.shape[0]):
        for j in range(m2.shape[0]):
            cov = c1[i] + c2[j]
            diff = (m1[i] - m2[j])[None, :]
            logdet, sol = _chol_logdet_solve(cov, diff)
            maha = float(np.sum(sol * sol))
            out[i, j] = np.exp(-0.5 * (maha + logdet + d * LOG_2PI))
    return out


def quad_overlap_full(w1, means1, covs1, w2, means2, covs2) -> float:
    w1 = np.asarray(w1, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()
    return float(w1 @ overlap_full(means1, covs1, means2, covs2) @ w2)


def design_matrix(
    kernel_means: np.ndarray,
    kernel_vars: np.ndarray,
    basis_means: np.ndarray,
    basis_vars: np.ndarray,
) -> np.ndarray:
    """Moment design matrix ``D_ji = int k_j(x) b_i(x) dx``.

    This is manuscript equation (eq:D-closed-form): ``m * M`` closed-form
    Gaussian evaluations in place of ``m * M`` quadratures.  Rows index the
    ``m`` measurement kernels, columns the ``M`` dictionary atoms.
    """
    return overlap_diag(kernel_means, kernel_vars, basis_means, basis_vars)


def gram_matrix(basis_means: np.ndarray, basis_vars: np.ndarray) -> np.ndarray:
    """SPADES Gram matrix ``G_jk = int b_j b_k``.

    The same identity with the measurement family set equal to the basis
    family, which is exactly the structural difference the manuscript draws
    between the two objectives.
    """
    return overlap_diag(basis_means, basis_vars, basis_means, basis_vars)


def mixture_pdf_diag(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Evaluate a diagonal-covariance Gaussian mixture at ``x`` (n, d)."""
    x = _as_2d(x)
    m, v = _as_2d(means), _as_2d(variances)
    w = np.asarray(weights, dtype=float).ravel()

    k = m.shape[0]
    rows = max(1, int(_BLOCK_ENTRIES // max(k, 1)))
    out = np.empty(x.shape[0], dtype=float)
    for start in range(0, x.shape[0], rows):
        stop = min(start + rows, x.shape[0])
        diff = x[start:stop, None, :] - m[None, :, :]
        logp = -0.5 * np.sum(diff * diff / v[None, :, :] + np.log(v)[None, :, :] + LOG_2PI, axis=2)
        out[start:stop] = np.exp(logp) @ w
    return out


def mixture_pdf_full(x: np.ndarray, weights, means, covs) -> np.ndarray:
    x = _as_2d(x)
    m = _as_2d(means)
    covs = np.asarray(covs, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    d = m.shape[1]
    out = np.zeros(x.shape[0], dtype=float)
    for j in range(m.shape[0]):
        L = np.linalg.cholesky(covs[j])
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        sol = np.linalg.solve(L, (x - m[j]).T)
        maha = np.sum(sol * sol, axis=0)
        out += w[j] * np.exp(-0.5 * (maha + logdet + d * LOG_2PI))
    return out


def self_overlap_quad(weights, means, variances) -> float:
    """``int f(x)^2 dx`` for a diagonal-covariance Gaussian mixture."""
    return quad_overlap_diag(weights, means, variances, weights, means, variances)


def ise_mixtures(
    w1, means1, vars1,
    w2, means2, vars2,
) -> float:
    """Exact integrated squared error between two diagonal Gaussian mixtures.

    ``int (f - g)^2 = int f^2 - 2 int f g + int g^2``, every term closed form.
    Clipped at zero: the three terms are computed separately, so cancellation
    can return a tiny negative value when the two mixtures are near-identical.
    """
    ff = quad_overlap_diag(w1, means1, vars1, w1, means1, vars1)
    fg = quad_overlap_diag(w1, means1, vars1, w2, means2, vars2)
    gg = quad_overlap_diag(w2, means2, vars2, w2, means2, vars2)
    return max(0.0, ff - 2.0 * fg + gg)
