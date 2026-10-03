"""Conventional and adaptive kernel density estimators.

These are the references that decide whether the proposed method is worth
anything: if a well-tuned ordinary KDE matches it, the extra machinery is not
earning its keep.  They are therefore implemented with the real selectors
(see ``bandwidths.py``), not rules of thumb.

``FixedKDE``
    Gaussian KDE at a bandwidth from a named selector: Sheather-Jones
    solve-the-equation, least-squares CV, Botev's improved Sheather-Jones, or
    Silverman as a weak control.  Each is a Gaussian mixture with one
    component per observation, so its ISE is exact.

``AbramsonKDE``
    Abramson's square-root law: ``h_i ∝ h_0 f_pilot(x_i)^{-1/2}``, normalised
    by the geometric mean of the pilot so ``h_0`` keeps its scale.  Because
    the bandwidth varies by *sample point* and each component is a normalised
    kernel, the estimate integrates to one exactly -- unlike the balloon
    variant that varies bandwidth by evaluation point.  The density floor and
    the bandwidth caps are explicit parameters and are reported, since an
    unreported floor is doing unacknowledged work on the outlier densities.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .. import bandwidths
from ..model import FittedDensity, mixture_density
from .base import Estimator, TuningConfig

__all__ = ["FixedKDE", "AbramsonKDE"]


_SELECTOR_NOTES = {
    "sj": "Sheather-Jones solve-the-equation plug-in (as in R bw.SJ(method='ste')).",
    "sj_dpi": "Sheather-Jones direct plug-in (as in R bw.SJ(method='dpi')).",
    "lscv": "Least-squares cross-validation, exact closed-form leave-one-out risk.",
    "botev_isj": (
        "Improved Sheather-Jones bandwidth of Botev, Grotowski & Kroese (2010), "
        "used in a fixed-bandwidth Gaussian KDE. This is what the authors' released "
        "kde.m computes; it is NOT the fully adaptive linear-diffusion estimator of "
        "their Section 5, which is not implemented here."
    ),
    "silverman": "Normal reference rule (weak control).",
    "scott": "Scott's rule (weak control).",
}


class FixedKDE(Estimator):
    """Gaussian KDE with a data-driven global bandwidth."""

    def __init__(self, rule: str = "sj", tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning)
        self.rule = rule
        self.name = f"KDE[{rule}]"
        self.provenance = _SELECTOR_NOTES.get(rule, "")

    def candidates(self, X: np.ndarray) -> list[dict]:
        # The selector *is* the tuning; there is nothing left to cross-validate.
        return [{"rule": self.rule}]

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        d = X.shape[1]
        t0 = perf_counter()
        # Per-axis bandwidth; for d == 1 this is the scalar selector.
        h = np.array([bandwidths.select_bandwidth(X[:, k], self.rule) for k in range(d)])
        elapsed = perf_counter() - t0

        n = X.shape[0]
        w = np.full(n, 1.0 / n)
        var = np.tile((h ** 2)[None, :], (n, 1))
        return [
            mixture_density(
                w, X, var,
                n_selected=n,
                n_dictionary=n,
                bandwidth=h.tolist(),
                selector=self.rule,
                setup_time_s=elapsed,
                stage1_time_s=0.0,
                stage2_time_s=0.0,
            )
        ]


@dataclass
class AbramsonConfig:
    pilot_rule: str = "sj"
    #: multipliers on the pilot-derived global bandwidth h0, cross-validated
    h0_mults: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.2)
    #: pilot density floor, as a fraction of the pilot's maximum
    density_floor: float = 1e-3
    #: caps on the local bandwidth factor lambda_i
    lam_min: float = 0.2
    lam_max: float = 5.0


class AbramsonKDE(Estimator):
    """Abramson (1982) square-root-law sample-point adaptive KDE."""

    name = "Abramson"
    provenance = (
        "Sample-point adaptive KDE, h_i = h0 * lambda_i with "
        "lambda_i = (f_pilot(x_i) / geometric-mean)^(-1/2), pilot floored at "
        "1e-3 of its maximum and lambda clipped to [0.2, 5]. Integrates to one exactly."
    )

    def __init__(self, config: AbramsonConfig | None = None,
                 tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning)
        self.config = config or AbramsonConfig()

    def candidates(self, X: np.ndarray) -> list[dict]:
        return [{"h0_mult": m} for m in self.config.h0_mults]

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        cfg = self.config
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape

        t0 = perf_counter()
        h_pilot = np.array([bandwidths.select_bandwidth(X[:, k], cfg.pilot_rule) for k in range(d)])

        # Pilot density at the sample points, accumulated in row blocks.
        # The result is only a length-n vector, so there is no reason to
        # materialise the n x n x d difference array -- at n = 5000 that alone
        # is 200 MB per intermediate, and three of them live at once.
        v = (h_pilot ** 2)[None, None, :]
        log_norm = float(np.sum(np.log(h_pilot))) + 0.5 * d * np.log(2 * np.pi)
        pilot = np.empty(n, dtype=float)
        block = max(1, int(4_000_000 // max(n, 1)))
        for start in range(0, n, block):
            stop = min(start + block, n)
            diff = X[start:stop, None, :] - X[None, :, :]
            quad = np.sum(diff * diff / v, axis=2)
            pilot[start:stop] = np.exp(-0.5 * quad - log_norm).mean(axis=1)

        floor = cfg.density_floor * float(pilot.max()) if pilot.max() > 0 else 1e-300
        pilot = np.maximum(pilot, floor)

        log_g = float(np.mean(np.log(pilot)))
        lam = np.exp(-0.5 * (np.log(pilot) - log_g))
        lam = np.clip(lam, cfg.lam_min, cfg.lam_max)
        setup = perf_counter() - t0

        w = np.full(n, 1.0 / n)
        out = []
        for c in cands:
            h_i = (c["h0_mult"] * h_pilot)[None, :] * lam[:, None]
            out.append(
                mixture_density(
                    w, X, h_i ** 2,
                    n_selected=n,
                    n_dictionary=n,
                    h0=(c["h0_mult"] * h_pilot).tolist(),
                    lam_min_used=float(lam.min()),
                    lam_max_used=float(lam.max()),
                    lam_clipped=int(np.sum((lam <= cfg.lam_min) | (lam >= cfg.lam_max))),
                    density_floor=cfg.density_floor,
                    pilot_rule=cfg.pilot_rule,
                    setup_time_s=setup,
                    stage1_time_s=0.0,
                    stage2_time_s=0.0,
                )
            )
        return out
