"""The proposed estimator: non-negative penalised selection, then simplex refit.

Implements manuscript §3.2-§3.3 as written, with none of the shortcuts the
review flagged in the historical MATLAB code:

* **The covariance is estimated from the training sample only.**  The old
  ``demo.m`` drew 50 *fresh* datasets from the true simulation density to
  build ``Sig``.  That is privileged information no practitioner has, and it
  is the single change most likely to move the published numbers.  Here
  ``C_hat`` is eq:covariance-regularized, computed from the same ``x_n`` the
  estimator is given, with the plug-in nature of that estimate disclosed
  rather than hidden.

* **The estimate is a probability density by construction**, not by dividing
  the coefficient vector by its sum afterwards.  Stage 1 solves a
  non-negative penalised problem to pick a support; stage 2 refits on that
  support over the simplex.  The old code's ``w = w / sum(w)`` applied to a
  signed solution is exactly the silent repair the recommendations rule out.

* **The penalty is anchored to the scale of the problem it penalises.**
  ``lam`` is parameterised as a fraction of the value at which the solution
  collapses to zero, so switching between identity, diagonal, ridge and
  truncated-pseudoinverse weighting -- whose scales differ by factors of N --
  does not silently change the amount of regularisation.

* **Conditioning is measured, not assumed.**  Every fit records the condition
  number and effective rank of the regularised covariance, the selected and
  final support sizes, whether the zero-solution fallback fired, and the KKT
  residual of each stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .. import gaussops, solvers
from ..dictionary import Atoms, DictionaryConfig, build_dictionary, build_measurements
from ..model import FittedDensity, mixture_density
from .base import Estimator, TuningConfig

__all__ = ["AKDE", "AKDEConfig", "WEIGHTINGS"]

#: Weighting schemes compared in the core ablation.
WEIGHTINGS = ("identity", "diagonal", "ridge", "trunc_pinv")


@dataclass
class AKDEConfig:
    """Default settings, and the evidence behind them.

    A measurement-bank sweep on MW10 (claw, N=200) with the dictionary held
    fixed at 150x10 gave, for the best lambda on each path:

        m     identity   diagonal
        120   1.98e-1    1.81e-1
        300   1.52e-1    1.09e-1
        600   1.48e-1    1.21e-1
        1500  1.47e-1    1.10e-1

    so accuracy improves up to roughly m = 300 and then flattens.  ``m = 300``
    is therefore the default: larger banks cost more (the covariance is m x m)
    without buying accuracy.

    An ``independent`` measurement bank beat a ``matched`` one consistently
    (e.g. diagonal weighting, 6.4e-2 versus 1.9e-1 on the same draw).  The
    manuscript permits this explicitly -- the measurement and basis families
    need not coincide -- and notes it as an unexplored direction; the sweep
    says it is worth taking.  ``matched`` is kept as a registered ablation so
    the historical configuration stays reproducible.
    """

    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    #: ``(n_loc, n_scale)`` options for the dictionary, selected by the same
    #: K-fold J-score as every other hyperparameter.  Set to ``None`` to pin
    #: the dictionary to ``dictionary`` instead (used by the size ablations).
    #:
    #: Dictionary size is validated rather than fixed because it turned out to
    #: be one of the most consequential settings, and not monotonically: an
    #: over-complete dictionary packs atoms whose moment signatures nearly
    #: coincide, which pushes the condition number of the design matrix
    #: towards 1e17 and leaves the moment fit under-determined in L2.  On MW10
    #: at N=5000, M=600 reached ISE 1.7e-2 where M=1500 reached 2.2e-1.  The
    #: evaluation protocol asks for dictionary settings to be chosen on
    #: training data, which is what this does.
    dict_sizes: tuple[tuple[int, int], ...] | None = ((60, 5), (100, 6), (150, 10))
    meas_mode: str = "independent"    # 'matched' | 'independent'
    meas_n_loc: int = 100
    meas_n_scale: int = 3
    meas_scale_lo_mult: float = 0.25
    meas_scale_hi_mult: float = 2.0

    weighting: str = "ridge"
    #: Relative to the mean diagonal of C_hat.  The grid reaches 1.0 because
    #: C_hat is badly rank deficient (rank <= min(m, N-1), and numerically far
    #: below even that), so heavy shrinkage towards a scaled identity is a
    #: setting cross-validation must be allowed to choose.
    rho_grid: tuple[float, ...] = (1e-6, 1e-3, 1e-1, 1.0)
    trunc_eta: float = 1e-6                            # eigenvalue cut for trunc_pinv

    n_lambda: int = 10
    #: Path runs from ``lam_max_ratio * lam_max`` down to ``lam_min_ratio * lam_max``.
    #: The top of the path stops strictly below 1: ``lam_max`` is *defined* as the
    #: smallest penalty at which the solution collapses to zero, so a candidate at
    #: ratio 1.0 always returns the degenerate singleton fallback and can only win
    #: a cross-validation comparison by noise.
    lam_max_ratio: float = 0.7
    #: Not taken below ~3e-4: past that the active set grows towards the whole
    #: dictionary, coordinate descent slows sharply, and the extra candidates
    #: were not selected by cross-validation in the pilot sweeps.
    lam_min_ratio: float = 3e-4
    tau: float = 1e-3                 # relative support threshold, eq:selected-support
    max_support: int = 400            # runtime guard; triggers are recorded

    single_stage: bool = False        # ablation: skip the simplex refit
    no_sparsity: bool = False         # ablation: lam = 0, refit on the full dictionary


class AKDE(Estimator):
    """Two-stage covariance-weighted moment estimator over a multiscale dictionary."""

    def __init__(self, config: AKDEConfig | None = None, tuning: TuningConfig | None = None,
                 name: str | None = None) -> None:
        super().__init__(tuning)
        self.config = config or AKDEConfig()
        self.name = name or f"AKDE[{self.config.weighting}]"
        self.provenance = (
            "Two-stage: non-negative L1-penalised support selection, then simplex refit. "
            "Covariance from training data only (plug-in; measurement bank is data-anchored)."
        )

    # -- candidates ----------------------------------------------------

    def candidates(self, X: np.ndarray) -> list[dict]:
        cfg = self.config
        if cfg.no_sparsity:
            lam_ratios = [0.0]
        else:
            lam_ratios = np.geomspace(cfg.lam_max_ratio, cfg.lam_min_ratio,
                                      cfg.n_lambda).tolist()
        rhos = (0.0,) if cfg.weighting in ("identity",) else cfg.rho_grid
        if cfg.dict_sizes:
            dicts = list(cfg.dict_sizes)
        else:
            dicts = [(cfg.dictionary.n_loc, cfg.dictionary.n_scale)]
        return [{"dict": d, "rho": r, "lam_ratio": l}
                for d in dicts for r in rhos for l in lam_ratios]

    def _dict_config(self, spec: tuple[int, int]) -> DictionaryConfig:
        base = self.config.dictionary
        return DictionaryConfig(
            n_loc=spec[0], n_scale=spec[1],
            location_rule=base.location_rule,
            scale_lo_mult=base.scale_lo_mult,
            scale_hi_mult=base.scale_hi_mult,
            pad_mult=base.pad_mult,
        )

    # -- weighting -----------------------------------------------------

    @staticmethod
    def _omega_apply(C: np.ndarray, weighting: str, rho_rel: float, eta: float):
        """Return ``apply(B) = Omega @ B`` plus conditioning diagnostics.

        ``Omega`` is never formed explicitly when it is an inverse; linear
        systems are solved instead, as the manuscript specifies.
        """
        m = C.shape[0]
        eigs = np.linalg.eigvalsh(C)
        scale = float(np.mean(np.diag(C)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        diag_info = {
            "cov_eig_max": float(eigs[-1]),
            "cov_eig_min": float(eigs[0]),
            "cov_rank_numeric": int(np.sum(eigs > max(eigs[-1], 0.0) * 1e-12)),
        }
        pos = eigs[eigs > 0]
        if pos.size:
            p = pos / pos.sum()
            diag_info["cov_effective_rank"] = float(np.exp(-np.sum(p * np.log(p))))
        else:
            diag_info["cov_effective_rank"] = 0.0

        if weighting == "identity":
            diag_info["omega_cond"] = 1.0
            return (lambda B: B), diag_info

        if weighting == "diagonal":
            d = np.diag(C) + rho_rel * scale
            d = np.where(d > 0, d, scale)
            inv = 1.0 / d
            diag_info["omega_cond"] = float(inv.max() / inv.min())
            return (lambda B: inv[:, None] * B if B.ndim == 2 else inv * B), diag_info

        if weighting == "trunc_pinv":
            vals, vecs = np.linalg.eigh(C)
            cut = max(vals[-1], 0.0) * eta
            keep = vals > cut
            diag_info["omega_cond"] = (
                float(vals[keep].max() / vals[keep].min()) if keep.any() else np.inf
            )
            diag_info["trunc_kept"] = int(keep.sum())
            if not keep.any():
                return (lambda B: B), diag_info
            V = vecs[:, keep]
            inv_vals = 1.0 / vals[keep]

            def apply(B):
                return V @ (inv_vals[:, None] * (V.T @ B)) if B.ndim == 2 else V @ (inv_vals * (V.T @ B))

            return apply, diag_info

        # 'ridge': Omega = (C_hat + rho I)^{-1}, eq:covariance-regularized
        reg = C + rho_rel * scale * np.eye(m)
        eig_reg = eigs + rho_rel * scale
        diag_info["omega_cond"] = float(eig_reg[-1] / max(eig_reg[0], 1e-300))
        try:
            chol = np.linalg.cholesky(reg)
        except np.linalg.LinAlgError:
            reg = reg + (abs(eig_reg[0]) + 1e-10 * scale) * np.eye(m)
            chol = np.linalg.cholesky(reg)

        def apply(B):
            return np.linalg.solve(chol.T, np.linalg.solve(chol, B))

        return apply, diag_info

    # -- fitting -------------------------------------------------------

    def fit_path(self, X: np.ndarray, cands: list[dict],
                 rng: np.random.Generator) -> list[FittedDensity]:
        cfg = self.config
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        N = X.shape[0]

        # Group candidates first by dictionary spec (the dictionary, design
        # matrix and covariance are rebuilt per spec) and then by rho (the
        # weighted Hessian is built once per rho, and the whole lambda path is
        # solved from it with warm starts).
        out: list[FittedDensity | None] = [None] * len(cands)
        by_dict: dict[tuple[int, int], list[int]] = {}
        for i, c in enumerate(cands):
            spec = tuple(c.get("dict", (cfg.dictionary.n_loc, cfg.dictionary.n_scale)))
            by_dict.setdefault(spec, []).append(i)

        for spec, dict_idxs in by_dict.items():
            t_setup = perf_counter()
            atoms: Atoms = build_dictionary(X, self._dict_config(spec), rng=rng)
            if cfg.meas_mode == "matched":
                meas = build_measurements(X, "matched", dictionary=atoms,
                                          n_loc=cfg.meas_n_loc, n_scale=cfg.meas_n_scale)
            else:
                meas = build_measurements(X, "independent", n_loc=cfg.meas_n_loc,
                                          n_scale=cfg.meas_n_scale,
                                          scale_lo_mult=cfg.meas_scale_lo_mult,
                                          scale_hi_mult=cfg.meas_scale_hi_mult, rng=rng)

            # Moment design matrix in closed form (Proposition 2) -- no quadrature.
            D = gaussops.design_matrix(meas.means, meas.variances,
                                       atoms.means, atoms.variances)

            # Empirical measurements and their covariance, from training data only.
            K = self._eval_kernels(X, meas)
            y = K.mean(axis=0)
            Kc = K - y
            C_hat = (Kc.T @ Kc) / (N * (N - 1.0)) if N > 1 else np.eye(meas.size)
            setup_time = perf_counter() - t_setup

            by_rho: dict[float, list[int]] = {}
            for i in dict_idxs:
                by_rho.setdefault(float(cands[i]["rho"]), []).append(i)

            for rho, idxs in by_rho.items():
                apply_omega, cov_diag = self._omega_apply(
                    C_hat, cfg.weighting, rho, cfg.trunc_eta)
                OmD = apply_omega(D)
                H1 = D.T @ OmD
                H1 = 0.5 * (H1 + H1.T)
                g = D.T @ apply_omega(y)

                lam_max = solvers.lambda_max_nonneg(g)
                ratios = np.array([cands[i]["lam_ratio"] for i in idxs], dtype=float)
                lams = ratios * lam_max
                if lam_max <= 0:
                    lams = np.zeros_like(lams)

                t_stage1 = perf_counter()
                path = solvers.nonneg_lasso_path(H1, g, lams)
                stage1_time = (perf_counter() - t_stage1) / max(len(idxs), 1)

                for slot, res in zip(idxs, path):
                    out[slot] = self._finalise(
                        res, H1, g, D, y, apply_omega, atoms, cov_diag,
                        setup_time / max(len(dict_idxs), 1), stage1_time, rho,
                        cands[slot]["lam_ratio"], N, meas.size,
                    )

        return [o for o in out]

    @staticmethod
    def _eval_kernels(X: np.ndarray, meas: Atoms) -> np.ndarray:
        """``K[n, j] = k_j(x_n)`` for the Gaussian measurement bank."""
        diff = X[:, None, :] - meas.means[None, :, :]
        v = meas.variances[None, :, :]
        logp = -0.5 * np.sum(diff * diff / v + np.log(v) + gaussops.LOG_2PI, axis=2)
        return np.exp(logp)

    def _finalise(self, res, H1, g, D, y, apply_omega, atoms, cov_diag,
                  setup_time, stage1_time, rho, lam_ratio, N, m) -> FittedDensity:
        cfg = self.config
        u = res.x
        M = u.size

        # --- support selection, eq:selected-support / eq:empty-support ---
        fallback = False
        umax = float(u.max()) if u.size else 0.0
        if umax > 0:
            S = np.flatnonzero(u > cfg.tau * umax)
        else:
            S = np.array([], dtype=int)

        capped = False
        if S.size > cfg.max_support:
            # keep the largest coefficients; recorded, never silent
            S = S[np.argsort(-u[S])[: cfg.max_support]]
            S.sort()
            capped = True

        if S.size == 0:
            # explicit fallback: the single atom minimising the loss
            fallback = True
            loss_i = 0.5 * np.diag(H1) - g
            S = np.array([int(np.argmin(loss_i))])

        n_selected = int(S.size)

        if cfg.single_stage:
            # ablation: normalise stage 1 instead of refitting (this is what the
            # historical code effectively did, and is reported as such)
            w = np.zeros(M)
            s = u[S].sum()
            w[S] = u[S] / s if s > 0 else 1.0 / S.size
            kkt2 = float("nan")
            stage2_time = 0.0
        else:
            t2 = perf_counter()
            DS = D[:, S]
            H2 = DS.T @ apply_omega(DS)
            H2 = 0.5 * (H2 + H2.T)
            c2 = -DS.T @ apply_omega(y)
            start = u[S] / u[S].sum() if u[S].sum() > 0 else None
            r2 = solvers.simplex_qp(H2, c2, x0=start)
            stage2_time = perf_counter() - t2
            w = np.zeros(M)
            w[S] = r2.x
            kkt2 = r2.kkt_residual

        keep = w > 0
        fit = mixture_density(
            w[keep], atoms.means[keep], atoms.variances[keep], support=S,
            n_selected=n_selected,
            n_dictionary=M,
            n_measurements=m,
            fallback_used=fallback,
            support_capped=capped,
            stage1_kkt=res.kkt_residual,
            stage1_converged=res.converged,
            stage2_kkt=kkt2,
            setup_time_s=setup_time,
            stage1_time_s=stage1_time,
            stage2_time_s=stage2_time,
            rho=rho,
            lam_ratio=lam_ratio,
            weighting=cfg.weighting,
            **cov_diag,
        )
        return fit
