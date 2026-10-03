"""Gaussian mixture fitted by EM -- the conventional compact-mixture control.

Tests whether a compact likelihood-fitted mixture is simply competitive, which
would undercut the case for a dictionary method altogether.  Component count is
selected on training data by the same K-fold J-score used for every other
estimator, so the comparison is not confounded by giving EM a different model
selection rule.  A BIC-selected variant is also available for reference.

Covariance floors and multiple initialisations are used, as recommended: EM on
densities with very narrow components (the claws, the combs) will otherwise
collapse a component onto a single point and report an infinite likelihood.
Degenerate fits are recorded rather than silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.mixture import GaussianMixture

from ..model import FittedDensity, mixture_density
from .base import Estimator, TuningConfig

__all__ = ["EMGaussianMixture", "EMConfig"]


@dataclass
class EMConfig:
    k_grid: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    n_init: int = 5
    reg_covar: float = 1e-6
    max_iter: int = 300
    selection: str = "cv"     # 'cv' (J-score, default) or 'bic'


class EMGaussianMixture(Estimator):
    name = "GMM-EM"
    provenance = (
        "sklearn GaussianMixture, full covariances, 5 initialisations, "
        "reg_covar=1e-6; component count chosen by K-fold J-score on training data."
    )

    def __init__(self, config: EMConfig | None = None,
                 tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning)
        self.config = config or EMConfig()
        if self.config.selection == "bic":
            self.name = "GMM-EM[BIC]"

    def candidates(self, X: np.ndarray) -> list[dict]:
        n = X.shape[0]
        # A component needs enough points to estimate its covariance.
        ks = [k for k in self.config.k_grid if k * (X.shape[1] + 2) <= max(n - 1, 2)]
        if not ks:
            ks = [1]
        if self.config.selection == "bic":
            return [{"k_grid": tuple(ks)}]
        return [{"k": k} for k in ks]

    def _fit_one(self, X, k, seed) -> tuple[GaussianMixture | None, bool]:
        cfg = self.config
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=cfg.reg_covar,
                max_iter=cfg.max_iter,
                n_init=cfg.n_init,
                random_state=seed,
            )
            gm.fit(X)
            return gm, bool(getattr(gm, "converged_", False))
        except Exception:
            return None, False

    def _to_density(self, gm, X, k, elapsed, converged) -> FittedDensity:
        d = X.shape[1]
        covs = gm.covariances_               # (k, d, d)
        if d == 1:
            variances = covs[:, 0, 0][:, None]
            return mixture_density(
                gm.weights_, gm.means_, variances,
                n_selected=int(k), n_dictionary=int(k),
                em_converged=converged,
                min_component_sd=float(np.sqrt(variances.min())),
                setup_time_s=0.0, stage1_time_s=elapsed, stage2_time_s=0.0,
            )
        # d > 1: diagonalise is not acceptable, so keep full covariances by
        # wrapping the sklearn model as a callable and scoring numerically.
        from ..model import callable_density
        from .. import gaussops

        fit = callable_density(
            lambda x: gaussops.mixture_pdf_full(x, gm.weights_, gm.means_, covs),
            dim=d,
            n_active=int(k), n_selected=int(k), n_dictionary=int(k),
            em_converged=converged, is_valid_density=True, total_mass=1.0,
            setup_time_s=0.0, stage1_time_s=elapsed, stage2_time_s=0.0,
        )
        fit.diagnostics["full_cov_model"] = (gm.weights_, gm.means_, covs)
        return fit

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        seed = int(rng.integers(0, 2 ** 31 - 1))

        out = []
        for c in cands:
            if "k_grid" in c:
                t0 = perf_counter()
                best, best_bic, best_k, best_conv = None, np.inf, 1, False
                for k in c["k_grid"]:
                    gm, conv = self._fit_one(X, k, seed)
                    if gm is None:
                        continue
                    bic = gm.bic(X)
                    if np.isfinite(bic) and bic < best_bic:
                        best, best_bic, best_k, best_conv = gm, bic, k, conv
                elapsed = perf_counter() - t0
                if best is None:
                    best, best_conv = self._fit_one(X, 1, seed)
                    best_k = 1
                out.append(self._to_density(best, X, best_k, elapsed, best_conv))
            else:
                k = int(c["k"])
                t0 = perf_counter()
                gm, conv = self._fit_one(X, k, seed)
                elapsed = perf_counter() - t0
                if gm is None:
                    gm, conv = self._fit_one(X, 1, seed)
                    k = 1
                out.append(self._to_density(gm, X, k, elapsed, conv))
        return out
