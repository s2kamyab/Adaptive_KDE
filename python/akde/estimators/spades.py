"""SPADES (Bunea, Tsybakov, Wegkamp & Barbu, Ann. Statist. 2010).

Two estimators are provided and they are *not* interchangeable:

``SPADES``
    The published objective, exactly as the manuscript states it in
    eq:spades-raw --

        min_w  w^T G w - 2 s^T w + gamma ||w||_1,
        G_jk = int b_j b_k,   s_i = (1/N) sum_n b_i(x_n),

    with **no** non-negativity or unit-sum constraint.  Its solution is a
    signed, unnormalised function.  It is scored that way.  The historical
    MATLAB divided the coefficients by their sum before scoring, which turns a
    signed object into something that integrates to one without making it a
    density, and flatters or penalises it unpredictably depending on the sign
    of the sum.

``SPADESSimplex``
    The density-valid comparison variant defined in the manuscript's
    "A density-valid comparison variant" paragraph: non-negative penalised
    selection on the same integrated-square loss, same relative-threshold
    support rule, then a simplex refit.  This isolates the *fitting loss*
    (Gram versus covariance-weighted moments) from the *density constraint*,
    which the plain comparison confounds.  It is an adaptation for controlled
    comparison, not something the original paper proposed.

Both share the proposed estimator's dictionary so the comparison is about the
objective rather than the representation, and both tune ``gamma``
independently of the proposed method's ``lambda``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .. import gaussops, solvers
from ..dictionary import DictionaryConfig, build_dictionary
from ..model import FittedDensity, mixture_density
from .base import Estimator, TuningConfig

__all__ = ["SPADES", "SPADESSimplex", "SPADESConfig"]


@dataclass
class SPADESConfig:
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    n_lambda: int = 20
    #: See AKDEConfig: the top of the path stops strictly below the penalty at
    #: which the solution collapses to zero.
    lam_max_ratio: float = 0.7
    lam_min_ratio: float = 1e-4
    tau: float = 1e-3
    max_support: int = 400


class _SPADESBase(Estimator):
    def __init__(self, config: SPADESConfig | None = None,
                 tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning)
        self.config = config or SPADESConfig()

    def candidates(self, X: np.ndarray) -> list[dict]:
        cfg = self.config
        return [{"lam_ratio": r}
                for r in np.geomspace(cfg.lam_max_ratio, cfg.lam_min_ratio, cfg.n_lambda)]

    def _setup(self, X, rng):
        atoms = build_dictionary(X, self.config.dictionary, rng=rng)
        # G in closed form via the same Gaussian overlap identity: the
        # manuscript notes G_jk = D_jk when the measurement family equals the
        # basis family.
        G = gaussops.gram_matrix(atoms.means, atoms.variances)
        s = self._basis_means(X, atoms)
        return atoms, G, s

    @staticmethod
    def _basis_means(X, atoms) -> np.ndarray:
        """``s_i = (1/N) sum_n b_i(x_n)``."""
        diff = X[:, None, :] - atoms.means[None, :, :]
        v = atoms.variances[None, :, :]
        logp = -0.5 * np.sum(diff * diff / v + np.log(v) + gaussops.LOG_2PI, axis=2)
        return np.exp(logp).mean(axis=0)


class SPADES(_SPADESBase):
    """Original signed SPADES.  Produces a signed, unnormalised estimate."""

    name = "SPADES"
    provenance = (
        "Original eq:spades-raw objective: signed, no simplex constraint, "
        "scored without post-hoc normalisation."
    )

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        t0 = perf_counter()
        atoms, G, s = self._setup(X, rng)
        setup_time = perf_counter() - t0

        # Convert w'Gw - 2s'w + gamma||w||_1 to the 1/2 x'Hx - g'x + lam||x||_1
        # convention: H = 2G, g = 2s, lam = gamma.
        H = 2.0 * G
        g = 2.0 * s
        lam_max = solvers.lambda_max_signed(g)
        lams = np.array([c["lam_ratio"] for c in cands], dtype=float) * lam_max

        t1 = perf_counter()
        path = solvers.signed_lasso_path(H, g, lams)
        per = (perf_counter() - t1) / max(len(cands), 1)

        out = []
        for c, res in zip(cands, path):
            w = res.x
            keep = w != 0
            out.append(
                mixture_density(
                    w[keep], atoms.means[keep], atoms.variances[keep],
                    support=np.flatnonzero(keep),
                    n_selected=int(keep.sum()),
                    n_dictionary=int(w.size),
                    stage1_kkt=res.kkt_residual,
                    stage1_converged=res.converged,
                    setup_time_s=setup_time,
                    stage1_time_s=per,
                    stage2_time_s=0.0,
                    lam_ratio=c["lam_ratio"],
                    signed=True,
                )
            )
        return out


class SPADESSimplex(_SPADESBase):
    """Non-negative SPADES-style selection plus simplex refit (manuscript variant)."""

    name = "SPADES-simplex"
    provenance = (
        "Density-valid adaptation described in the manuscript: non-negative "
        "L1-penalised selection on the SPADES integrated-square loss, then a "
        "simplex refit. Not proposed by the original paper."
    )

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        cfg = self.config
        t0 = perf_counter()
        atoms, G, s = self._setup(X, rng)
        setup_time = perf_counter() - t0

        H = 2.0 * G
        g = 2.0 * s
        lam_max = solvers.lambda_max_nonneg(g)
        lams = np.array([c["lam_ratio"] for c in cands], dtype=float) * lam_max

        t1 = perf_counter()
        path = solvers.nonneg_lasso_path(H, g, lams)
        per = (perf_counter() - t1) / max(len(cands), 1)

        out = []
        for c, res in zip(cands, path):
            u = res.x
            umax = float(u.max()) if u.size else 0.0
            fallback = False
            capped = False
            if umax > 0:
                S = np.flatnonzero(u > cfg.tau * umax)
            else:
                S = np.array([], dtype=int)
            if S.size > cfg.max_support:
                S = np.sort(S[np.argsort(-u[S])[: cfg.max_support]])
                capped = True
            if S.size == 0:
                # same explicit fallback rule: singleton minimising G_ii - 2 s_i
                fallback = True
                S = np.array([int(np.argmin(np.diag(G) - 2.0 * s))])

            t2 = perf_counter()
            H2 = 2.0 * G[np.ix_(S, S)]
            c2 = -2.0 * s[S]
            start = u[S] / u[S].sum() if u[S].sum() > 0 else None
            r2 = solvers.simplex_qp(H2, c2, x0=start)
            stage2 = perf_counter() - t2

            w = np.zeros(u.size)
            w[S] = r2.x
            keep = w > 0
            out.append(
                mixture_density(
                    w[keep], atoms.means[keep], atoms.variances[keep], support=S,
                    n_selected=int(S.size),
                    n_dictionary=int(w.size),
                    fallback_used=fallback,
                    support_capped=capped,
                    stage1_kkt=res.kkt_residual,
                    stage1_converged=res.converged,
                    stage2_kkt=r2.kkt_residual,
                    setup_time_s=setup_time,
                    stage1_time_s=per,
                    stage2_time_s=stage2,
                    lam_ratio=c["lam_ratio"],
                )
            )
        return out
