"""Multiscale dictionaries and measurement-kernel banks.

The manuscript draws a structural distinction that SPADES cannot make: the
*basis* functions ``b_i`` that the density is built from and the *measurement*
functions ``k_j`` that produce the moment vector need not be the same family.
Both are built here, through the same code path but with independent
settings, so that the "matched versus independently designed" ablation the
recommendations ask for is a configuration change rather than a rewrite.

All dictionaries are anchored to a data-driven reference bandwidth (Silverman)
rather than to the raw data range.  This matters: the old fixed grid
``c * (max - min) / M`` ties the finest available scale to the sample range,
so a single outlier coarsens the whole dictionary -- which is one plausible
mechanism behind the anomalous historical ``f5`` (Marron-Wand "outlier")
result that the recommendations single out for investigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bandwidths import silverman

__all__ = ["Atoms", "build_dictionary", "build_measurements", "DictionaryConfig"]


@dataclass
class Atoms:
    """A bank of diagonal Gaussian atoms."""

    means: np.ndarray      # (M, d)
    variances: np.ndarray  # (M, d)
    n_loc: int
    n_scale: int
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.means.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])


@dataclass
class DictionaryConfig:
    """Dictionary geometry.  All of these are tuned or ablated, never fixed by fiat."""

    n_loc: int = 150
    n_scale: int = 10
    location_rule: str = "grid"      # 'grid' | 'quantile' | 'data'
    scale_lo_mult: float = 0.05      # finest scale, as a multiple of the reference bandwidth
    scale_hi_mult: float = 4.0       # coarsest scale
    pad_mult: float = 1.0            # extend locations past the data by pad_mult * coarsest scale


def _reference_bandwidth(X: np.ndarray) -> np.ndarray:
    """Per-axis Silverman bandwidth, used only to set the dictionary scale grid."""
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    d = X.shape[1]
    out = np.empty(d)
    for k in range(d):
        h = silverman(X[:, k])
        out[k] = h if np.isfinite(h) and h > 0 else 1.0
    return out


def _locations(X: np.ndarray, n_loc: int, rule: str, pad: np.ndarray,
               rng: np.random.Generator | None) -> np.ndarray:
    n, d = X.shape
    if rule == "data":
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(n_loc, n), replace=False)
        return X[idx]

    if d == 1:
        if rule == "quantile":
            qs = np.linspace(0.0, 1.0, n_loc)
            loc = np.quantile(X[:, 0], qs)
            # de-duplicate: heavy ties collapse quantiles onto each other
            loc = np.unique(loc)
            if loc.size < n_loc:
                extra = np.linspace(X.min() - pad[0], X.max() + pad[0], n_loc - loc.size + 2)[1:-1]
                loc = np.unique(np.concatenate([loc, extra]))
            return loc[:, None]
        return np.linspace(X.min() - pad[0], X.max() + pad[0], n_loc)[:, None]

    # d > 1: a full product grid is exponential in d, so locations come from a
    # subsample of the observations, which also puts atoms where the mass is.
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n_loc, n), replace=False)
    return X[idx]


def build_dictionary(
    X: np.ndarray,
    config: DictionaryConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Build the ``M = p * q`` location-scale dictionary of manuscript §3.3.

    Scales are geometrically spaced between ``scale_lo_mult`` and
    ``scale_hi_mult`` times the reference bandwidth, so the dictionary spans
    roughly two decades of resolution by default.
    """
    config = config or DictionaryConfig()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1:
        X = X.T
    d = X.shape[1]

    h_ref = _reference_bandwidth(X)
    mults = np.geomspace(config.scale_lo_mult, config.scale_hi_mult, config.n_scale)
    pad = config.pad_mult * config.scale_hi_mult * h_ref

    loc = _locations(X, config.n_loc, config.location_rule, pad, rng)
    p = loc.shape[0]

    means = np.tile(loc, (config.n_scale, 1))
    variances = np.empty((p * config.n_scale, d))
    for li, mult in enumerate(mults):
        sd = mult * h_ref
        variances[li * p:(li + 1) * p, :] = (sd ** 2)[None, :]

    return Atoms(
        means=means,
        variances=variances,
        n_loc=p,
        n_scale=config.n_scale,
        scales=mults,
        meta={
            "h_ref": h_ref,
            "location_rule": config.location_rule,
            "scale_lo_mult": config.scale_lo_mult,
            "scale_hi_mult": config.scale_hi_mult,
        },
    )


def build_measurements(
    X: np.ndarray,
    mode: str = "matched",
    dictionary: Atoms | None = None,
    n_loc: int = 40,
    n_scale: int = 3,
    scale_lo_mult: float = 0.5,
    scale_hi_mult: float = 2.0,
    location_rule: str = "grid",
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Build the measurement-kernel bank ``{k_j}``.

    ``mode='matched'``
        Reuse the dictionary's own geometry, subsampled to ``n_loc`` locations
        and ``n_scale`` scales.  This reproduces the configuration used in the
        historical experiments, where the two families coincide.

    ``mode='independent'``
        Build a separate, deliberately coarser bank.  This is the decoupling
        the manuscript flags as available but unexplored; the moment vector
        then measures the density at a resolution chosen for conditioning
        rather than for representation.

    Coarser measurement kernels are usually better conditioned: narrow kernels
    on a fine grid give a near-singular covariance because neighbouring
    readings are almost the same statistic.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1:
        X = X.T

    if mode == "matched":
        if dictionary is None:
            raise ValueError("mode='matched' requires the dictionary")
        p, q = dictionary.n_loc, dictionary.n_scale
        loc_idx = np.unique(np.linspace(0, p - 1, min(n_loc, p)).astype(int))
        scale_idx = np.unique(np.linspace(0, q - 1, min(n_scale, q)).astype(int))
        rows = np.concatenate([li * p + loc_idx for li in scale_idx])
        return Atoms(
            means=dictionary.means[rows],
            variances=dictionary.variances[rows],
            n_loc=loc_idx.size,
            n_scale=scale_idx.size,
            scales=dictionary.scales[scale_idx],
            meta={"mode": "matched"},
        )

    if mode == "independent":
        cfg = DictionaryConfig(
            n_loc=n_loc,
            n_scale=n_scale,
            location_rule=location_rule,
            scale_lo_mult=scale_lo_mult,
            scale_hi_mult=scale_hi_mult,
        )
        atoms = build_dictionary(X, cfg, rng=rng)
        atoms.meta["mode"] = "independent"
        return atoms

    raise ValueError(f"unknown measurement mode {mode!r}")
