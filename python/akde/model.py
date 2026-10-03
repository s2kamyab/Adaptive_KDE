"""Container for a fitted density estimate.

Every estimator returns a :class:`FittedDensity`.  Two representations are
supported:

``gmm_diag``
    A Gaussian mixture with diagonal component covariances.  Weights may be
    negative and need not sum to one -- the original SPADES objective returns
    exactly such a signed, unnormalised object, and reporting it that way
    (rather than dividing by the coefficient sum) is deliberate: the
    recommendations warn against silently repairing an estimate before
    scoring it.  The mass diagnostics below quantify the difference.

``callable``
    Anything else; scored by quadrature.

Keeping the mixture form wherever possible is what makes integrated squared
error analytic for almost the whole comparison set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import gaussops

__all__ = ["FittedDensity", "mixture_density", "callable_density"]


@dataclass
class FittedDensity:
    kind: str
    weights: np.ndarray | None = None
    means: np.ndarray | None = None
    variances: np.ndarray | None = None
    pdf_fn: Callable[[np.ndarray], np.ndarray] | None = None
    support: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_mixture(self) -> bool:
        return self.kind == "gmm_diag"

    @property
    def dim(self) -> int:
        if self.is_mixture:
            return int(self.means.shape[1])
        return int(self.diagnostics.get("dim", 1))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if self.is_mixture:
            return gaussops.mixture_pdf_diag(x, self.weights, self.means, self.variances)
        assert self.pdf_fn is not None
        return np.asarray(self.pdf_fn(x), dtype=float).ravel()

    # -- representation-size and validity diagnostics -----------------------

    def n_active(self, tol: float = 1e-8, relative: bool = True) -> int:
        """Number of non-negligible coefficients.

        A count of non-zero coefficients is meaningless without a stated
        tolerance, so the tolerance is explicit and, by default, relative to
        the largest coefficient.
        """
        if not self.is_mixture:
            return int(self.diagnostics.get("n_active", -1))
        w = np.abs(self.weights)
        if w.size == 0:
            return 0
        thresh = tol * w.max() if relative else tol
        return int(np.sum(w > thresh))

    def total_mass(self) -> float:
        """``int fhat``. Exactly ``sum(w)`` for normalised atoms."""
        if self.is_mixture:
            return float(np.sum(self.weights))
        return float(self.diagnostics.get("total_mass", np.nan))

    def negative_weight_mass(self) -> float:
        """Total mass carried by negative coefficients (0 for valid densities)."""
        if self.is_mixture:
            return float(np.sum(np.minimum(self.weights, 0.0)))
        return float(self.diagnostics.get("negative_mass", np.nan))

    def is_valid_density(self, mass_tol: float = 1e-6) -> bool:
        """True when the estimate is provably a probability density.

        For a mixture of normalised non-negative atoms this is decidable
        exactly from the coefficients; no pointwise check is needed.
        """
        if not self.is_mixture:
            return bool(self.diagnostics.get("is_valid_density", False))
        return bool(np.all(self.weights >= -1e-14) and abs(self.total_mass() - 1.0) <= mass_tol)


def mixture_density(weights, means, variances, support=None, **diagnostics) -> FittedDensity:
    w = np.asarray(weights, dtype=float).ravel()
    m = np.asarray(means, dtype=float)
    v = np.asarray(variances, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    if v.ndim == 1:
        v = v[:, None]
    if v.shape != m.shape:
        if v.shape[0] == m.shape[0] and v.shape[1] == 1:
            v = np.repeat(v, m.shape[1], axis=1)
        else:
            raise ValueError(f"variance shape {v.shape} incompatible with means {m.shape}")
    return FittedDensity("gmm_diag", w, m, v, None, support, dict(diagnostics))


def callable_density(fn, dim: int = 1, **diagnostics) -> FittedDensity:
    diagnostics.setdefault("dim", dim)
    return FittedDensity("callable", pdf_fn=fn, diagnostics=dict(diagnostics))
