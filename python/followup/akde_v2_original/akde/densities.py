"""Test densities.

The 15 univariate targets used by the existing MATLAB code (``marronwand.m``,
labelled ``f0``..``f14``) were checked component-by-component against Marron &
Wand (1992), *Exact mean integrated squared error*, Ann. Statist. 20(2),
712-736, Table 1.  They are that suite exactly, in the published order, so
``f0`` is MW density #1 (Gaussian) and ``f14`` is MW density #15 (discrete
comb).  ``MARRON_WAND_MAPPING`` records the correspondence; the manuscript
should cite Marron & Wand and reproduce it.

Densities are stored as diagonal-covariance Gaussian mixtures, which makes the
true ``int f^2`` and every ISE against a Gaussian-mixture estimate analytic.

Also provided:
  * ``multivariate_targets`` - full-rank 2D and 5D mixtures with unequal
    scales, overlapping modes and skewness, as the evaluation protocol asks
    for.
  * ``stress_targets`` - outlier contamination and a rare narrow mode, kept
    as a separate stress test rather than averaged into the main table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import gaussops

__all__ = [
    "MixtureSpec",
    "marron_wand_suite",
    "MARRON_WAND_MAPPING",
    "MARRON_WAND_NAMES",
    "multivariate_targets",
    "stress_targets",
    "get_target",
    "list_targets",
]


@dataclass(frozen=True)
class MixtureSpec:
    """A Gaussian mixture with diagonal component covariances.

    Parameters
    ----------
    weights : (K,) mixing proportions, summing to one.
    means : (K, d)
    variances : (K, d) diagonal covariance entries (variances, not std devs).
    """

    name: str
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float).ravel()
        m = np.asarray(self.means, dtype=float)
        v = np.asarray(self.variances, dtype=float)
        if m.ndim == 1:
            m = m[:, None]
        if v.ndim == 1:
            v = v[:, None]
        if not (m.shape == v.shape and w.shape[0] == m.shape[0]):
            raise ValueError(f"{self.name}: inconsistent mixture shapes")
        if not np.isclose(w.sum(), 1.0, atol=1e-12):
            raise ValueError(f"{self.name}: weights sum to {w.sum()!r}, not 1")
        if np.any(v <= 0):
            raise ValueError(f"{self.name}: non-positive variance")
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "variances", v)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` observations, shape (n, d)."""
        idx = rng.choice(self.n_components, size=n, p=self.weights)
        return self.means[idx] + rng.standard_normal((n, self.dim)) * np.sqrt(self.variances[idx])

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return gaussops.mixture_pdf_diag(x, self.weights, self.means, self.variances)

    def self_overlap(self) -> float:
        """``int f^2``, used by the ISE decomposition and the J-score."""
        return gaussops.self_overlap_quad(self.weights, self.means, self.variances)

    def plot_range(self, pad: float = 4.0) -> tuple[float, float]:
        sd = np.sqrt(self.variances.max(axis=0))
        lo = float((self.means - pad * sd).min())
        hi = float((self.means + pad * sd).max())
        return lo, hi


def _mw(means: Sequence[float], stds: Sequence[float], priors: Sequence[float]) -> tuple:
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    priors = np.asarray(priors, dtype=float)
    return priors, means[:, None], (stds ** 2)[:, None]


MARRON_WAND_NAMES = [
    "Gaussian",
    "Skewed unimodal",
    "Strongly skewed",
    "Kurtotic unimodal",
    "Outlier",
    "Bimodal",
    "Separated bimodal",
    "Skewed bimodal",
    "Trimodal",
    "Claw",
    "Double claw",
    "Asymmetric claw",
    "Asymmetric double claw",
    "Smooth comb",
    "Discrete comb",
]

#: ``f{i}`` in the existing MATLAB code / manuscript figures  ->  Marron-Wand
#: density number (1-based, as printed in Marron & Wand 1992, Table 1).
MARRON_WAND_MAPPING = {f"f{i}": (i + 1, MARRON_WAND_NAMES[i]) for i in range(15)}


def marron_wand_suite() -> list[MixtureSpec]:
    """The 15 Marron-Wand normal mixture densities, in published order."""
    specs: list[tuple] = []

    # #1 Gaussian
    specs.append(_mw([0.0], [1.0], [1.0]))

    # #2 Skewed unimodal
    specs.append(_mw([0.0, 0.5, 13 / 12], [1.0, 2 / 3, 5 / 9], np.array([1, 1, 3]) / 5))

    # #3 Strongly skewed
    l = np.arange(8)
    specs.append(_mw(3 * ((2 / 3) ** l - 1), (2 / 3) ** l, np.ones(8) / 8))

    # #4 Kurtotic unimodal
    specs.append(_mw([0.0, 0.0], [1.0, 0.1], np.array([2, 1]) / 3))

    # #5 Outlier
    specs.append(_mw([0.0, 0.0], [1.0, 0.1], np.array([1, 9]) / 10))

    # #6 Bimodal
    specs.append(_mw([-1.0, 1.0], [2 / 3, 2 / 3], [0.5, 0.5]))

    # #7 Separated bimodal
    specs.append(_mw([-1.5, 1.5], [0.5, 0.5], [0.5, 0.5]))

    # #8 Skewed bimodal
    specs.append(_mw([0.0, 1.5], [1.0, 1 / 3], np.array([3, 1]) / 4))

    # #9 Trimodal
    specs.append(_mw([-1.2, 1.2, 0.0], [0.6, 0.6, 0.25], np.array([9, 9, 2]) / 20))

    # #10 Claw
    l = np.arange(5)
    specs.append(
        _mw(
            np.concatenate([[0.0], l / 2 - 1]),
            np.concatenate([[1.0], np.ones(5) / 10]),
            np.concatenate([[0.5], np.ones(5) / 10]),
        )
    )

    # #11 Double claw
    l = np.arange(7)
    specs.append(
        _mw(
            np.concatenate([[-1.0, 1.0], (l - 3) / 2]),
            np.concatenate([[2 / 3, 2 / 3], np.ones(7) / 100]),
            np.concatenate([[49 / 100, 49 / 100], np.ones(7) / 350]),
        )
    )

    # #12 Asymmetric claw
    l = np.arange(-2, 3)
    specs.append(
        _mw(
            np.concatenate([[0.0], l + 0.5]),
            np.concatenate([[1.0], (2.0 ** (-l)) / 10]),
            np.concatenate([[0.5], (2.0 ** (1 - l)) / 31]),
        )
    )

    # #13 Asymmetric double claw
    l = np.arange(1, 4)
    specs.append(
        _mw(
            np.concatenate([[-1.0, 1.0], -l / 2, l / 2]),
            np.concatenate([[2 / 3, 2 / 3], np.ones(3) / 100, np.ones(3) * 7 / 100]),
            np.concatenate([[46 / 100, 46 / 100], np.ones(3) / 300, np.ones(3) * 7 / 300]),
        )
    )

    # #14 Smooth comb
    l = np.arange(6)
    specs.append(
        _mw(
            (65 - 96 * (0.5 ** l)) / 21,
            (32 / 63) / (2.0 ** l),
            (2.0 ** (5 - l)) / 63,
        )
    )

    # #15 Discrete comb
    l1, l2 = np.arange(3), np.arange(8, 11)
    specs.append(
        _mw(
            np.concatenate([(12 * l1 - 15) / 7, 2 * l2 / 7]),
            np.concatenate([np.full(3, 2 / 7), np.full(3, 1 / 21)]),
            np.concatenate([np.full(3, 2 / 7), np.full(3, 1 / 21)]),
        )
    )

    out = []
    for i, (w, m, v) in enumerate(specs):
        out.append(
            MixtureSpec(
                name=f"f{i}",
                weights=w,
                means=m,
                variances=v,
                label=f"MW{i + 1} {MARRON_WAND_NAMES[i]}",
                meta={"suite": "marron_wand", "mw_number": i + 1},
            )
        )
    return out


def multivariate_targets() -> list[MixtureSpec]:
    """Full-rank 2D and 5D targets with unequal scales, overlap and skewness.

    Components are kept axis-aligned (diagonal covariance) so that ISE stays
    analytic through the same overlap identity; anisotropy is produced by
    giving each component different per-axis variances and by offsetting modes
    along different axes, rather than by rotating a shared covariance.
    """
    out: list[MixtureSpec] = []

    # --- 2D: three overlapping modes, very unequal per-axis scales ---
    w = np.array([0.45, 0.35, 0.20])
    m = np.array([[0.0, 0.0], [1.6, 0.8], [-1.2, 1.9]])
    v = np.array([[1.00, 0.09], [0.16, 0.64], [0.25, 0.04]])
    out.append(
        MixtureSpec("m2_overlap", w, m, v, "2D overlapping, unequal scales",
                    meta={"suite": "multivariate", "dim": 2})
    )

    # --- 2D: skewed, built as a geometric cascade in the first coordinate ---
    k = 6
    l = np.arange(k)
    w = (2.0 ** (k - 1 - l))
    w = w / w.sum()
    m = np.stack([3 * ((2 / 3) ** l - 1), 0.5 * l - 1.0], axis=1)
    v = np.stack([(2 / 3) ** (2 * l), np.full(k, 0.3)], axis=1)
    out.append(
        MixtureSpec("m2_skew", w, m, v, "2D skewed cascade",
                    meta={"suite": "multivariate", "dim": 2})
    )

    # --- 5D: four full-rank modes, per-axis variances spanning two decades ---
    w = np.array([0.35, 0.30, 0.20, 0.15])
    m = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.5, -1.0, 0.5, 0.0, 1.0],
        [-1.0, 1.5, 0.0, 1.0, -0.5],
        [0.5, 0.5, -1.5, -1.0, 0.0],
    ])
    v = np.array([
        [1.00, 0.25, 0.64, 0.09, 0.49],
        [0.16, 1.00, 0.09, 0.36, 0.25],
        [0.49, 0.09, 1.00, 0.25, 0.16],
        [0.09, 0.36, 0.25, 1.00, 0.64],
    ])
    out.append(
        MixtureSpec("m5_overlap", w, m, v, "5D overlapping, unequal scales",
                    meta={"suite": "multivariate", "dim": 5})
    )
    return out


def stress_targets() -> list[MixtureSpec]:
    """Outlier contamination and rare narrow modes, reported separately."""
    out: list[MixtureSpec] = []

    # Heavy, far contamination: 3% of mass at 8 sigma with a wide scale.
    out.append(
        MixtureSpec(
            "s_outlier",
            np.array([0.97, 0.03]),
            np.array([[0.0], [8.0]]),
            np.array([[1.0], [9.0]]),
            "Far heavy contamination (3%)",
            meta={"suite": "stress"},
        )
    )

    # A single very rare, very narrow mode on a broad background.
    out.append(
        MixtureSpec(
            "s_rare_spike",
            np.array([0.99, 0.01]),
            np.array([[0.0], [2.0]]),
            np.array([[1.0], [0.02 ** 2]]),
            "Rare narrow mode (1%, sd 0.02)",
            meta={"suite": "stress"},
        )
    )

    # Two narrow spikes of unequal mass riding on a wide base.
    out.append(
        MixtureSpec(
            "s_twin_spike",
            np.array([0.90, 0.07, 0.03]),
            np.array([[0.0], [-1.5], [1.5]]),
            np.array([[1.0], [0.05 ** 2], [0.03 ** 2]]),
            "Twin narrow spikes on wide base",
            meta={"suite": "stress"},
        )
    )
    return out


_REGISTRY: dict[str, MixtureSpec] | None = None


def _registry() -> dict[str, MixtureSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        for spec in marron_wand_suite() + multivariate_targets() + stress_targets():
            _REGISTRY[spec.name] = spec
    return _REGISTRY


def get_target(name: str) -> MixtureSpec:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown target {name!r}; available: {sorted(reg)}")
    return reg[name]


def list_targets(suite: str | None = None) -> list[str]:
    reg = _registry()
    if suite is None:
        return list(reg)
    return [k for k, v in reg.items() if v.meta.get("suite") == suite]
