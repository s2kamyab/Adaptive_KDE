"""Kernel-embedding mixture fitting (Song, Zhang, Smola, Gretton & Scholkopf, ICML 2008).

The closest predecessor to the proposed estimator, and the one that decides
whether covariance weighting adds anything beyond convex kernel moment
matching.  Their fixed-prototype formulation minimises the squared maximum
mean discrepancy between the empirical kernel mean embedding and the embedding
of a mixture over fixed prototypes, subject to simplex constraints:

    min_{w in simplex}  w^T Q w - 2 q^T w + nu ||w||^2,
    Q_ij = E_{x ~ b_i, x' ~ b_j} kappa(x, x'),
    q_i  = (1/N) sum_n E_{x ~ b_i} kappa(x, x_n).

With a Gaussian RBF kernel ``kappa(x, x') = N(x - x'; 0, s^2 I)`` and Gaussian
prototypes, both expectations are closed form by the same overlap identity:

    Q_ij = N(mu_i; mu_j, s^2 + sigma_i^2 + sigma_j^2),
    q_i  = (1/N) sum_n N(mu_i; x_n, s^2 + sigma_i^2).

Two things are worth stating plainly, because they are what the comparison
turns on:

* This is already a **convex QP over the probability simplex** -- so it is
  structurally much closer to the proposed method than the phrase "method of
  moments" suggests, and it produces a valid density without needing a
  two-stage construction.
* It has **no sparsity mechanism**.  Its representation size is the whole
  dictionary, which is precisely why the accuracy-versus-support-size plot is
  the informative comparison rather than accuracy alone.

Running it on the proposed method's dictionary is a common-dictionary
adaptation; it is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .. import gaussops, solvers
from ..bandwidths import silverman
from ..dictionary import DictionaryConfig, build_dictionary
from ..model import FittedDensity, mixture_density
from .base import Estimator, TuningConfig

__all__ = ["SongKernelMoment", "SongConfig"]


@dataclass
class SongConfig:
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    #: RBF kernel widths, as multiples of the Silverman reference bandwidth
    kernel_mults: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    #: L2 regularisation on the weights (their ``nu``), relative to trace(Q)/M
    nu_grid: tuple[float, ...] = (0.0, 1e-6, 1e-4)
    weight_tol: float = 1e-8


class SongKernelMoment(Estimator):
    name = "Song-KME"
    provenance = (
        "Song et al. (ICML 2008) fixed-prototype kernel moment matching: convex QP "
        "over the simplex, Gaussian RBF kernel, closed-form Q and q. Run on the "
        "shared multiscale dictionary (common-dictionary adaptation). No sparsity term."
    )

    def __init__(self, config: SongConfig | None = None,
                 tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning)
        self.config = config or SongConfig()

    def candidates(self, X: np.ndarray) -> list[dict]:
        return [{"kernel_mult": k, "nu": nu}
                for k in self.config.kernel_mults
                for nu in self.config.nu_grid]

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        cfg = self.config
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        d = X.shape[1]

        t0 = perf_counter()
        atoms = build_dictionary(X, cfg.dictionary, rng=rng)
        h_ref = np.array([silverman(X[:, k]) for k in range(d)])
        h_ref = np.where(np.isfinite(h_ref) & (h_ref > 0), h_ref, 1.0)
        setup = perf_counter() - t0

        out: list[FittedDensity] = []
        cache: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
        for c in cands:
            km = float(c["kernel_mult"])
            if km not in cache:
                t1 = perf_counter()
                s2 = (km * h_ref) ** 2            # RBF kernel variance, per axis
                # Q_ij = N(mu_i; mu_j, s^2 + sigma_i^2 + sigma_j^2)
                Q = gaussops.overlap_diag(
                    atoms.means, atoms.variances + s2[None, :],
                    atoms.means, atoms.variances,
                )
                Q = 0.5 * (Q + Q.T)
                # q_i = mean_n N(mu_i; x_n, s^2 + sigma_i^2)
                q = gaussops.overlap_diag(
                    atoms.means, atoms.variances + s2[None, :],
                    X, np.zeros_like(X) + 1e-300,
                ).mean(axis=1)
                cache[km] = (Q, q, perf_counter() - t1)
            Q, q, build_time = cache[km]

            M = Q.shape[0]
            nu_rel = float(c["nu"])
            nu = nu_rel * float(np.trace(Q)) / M
            H = 2.0 * (Q + nu * np.eye(M))
            cvec = -2.0 * q

            t2 = perf_counter()
            res = solvers.simplex_qp(H, cvec)
            solve_time = perf_counter() - t2

            w = res.x
            keep = w > cfg.weight_tol
            if not keep.any():
                keep = np.zeros(M, dtype=bool)
                keep[int(np.argmax(w))] = True
            wk = w[keep]
            wk = wk / wk.sum()
            out.append(
                mixture_density(
                    wk, atoms.means[keep], atoms.variances[keep],
                    support=np.flatnonzero(keep),
                    n_selected=int(keep.sum()),
                    n_dictionary=M,
                    kernel_mult=km,
                    nu=nu_rel,
                    stage2_kkt=res.kkt_residual,
                    setup_time_s=setup + build_time,
                    stage1_time_s=0.0,
                    stage2_time_s=solve_time,
                )
            )
        return out
