"""Configurable follow-up estimator: one code path, every Stage-A/B variant.

The V2 estimator (`akde.estimators.akde.AKDE`) is left untouched.  This module
re-expresses the same pipeline with each design choice as an explicit switch,
so a variant is a configuration rather than a fork:

    dictionary   pinned to one literal atom set per cell (never CV-selected)
    measurement  count, scale range and location rule, all separately settable
    weighting    identity | diagonal | ridge
    selection    'l1' (stage-1 non-negative lasso) | 'none' (full dictionary)
    refit        'moment' (Omega-weighted, as V2) | 'l2' (empirical integrated
                 square) | 'normalise' (V2's single-stage ablation)

Setting ``preset='v2'`` reproduces the V2 estimator bit-for-bit on a pinned
dictionary; `followup/scripts/check_v2_identity.py` is the test that enforces
this and must keep passing after any edit here.

Naming discipline required by the brief: a configuration with
``selection='l1'`` and ``refit='l2'`` is a **hybrid**, not moment fitting, and
is labelled ``hybrid_*`` everywhere it appears.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import perf_counter

import numpy as np

_PY = Path(__file__).resolve().parents[2]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from akde import gaussops, solvers                                    # noqa: E402
from akde.dictionary import Atoms, DictionaryConfig, build_dictionary  # noqa: E402
from akde.estimators.base import Estimator, TuningConfig              # noqa: E402
from akde.model import FittedDensity, mixture_density                 # noqa: E402

__all__ = ["FollowupConfig", "FollowupAKDE", "MeasurementSpec", "build_measurement_bank",
           "STAGE_A", "STAGE_B", "make_variant"]


# ---------------------------------------------------------------- measurement banks

@dataclass(frozen=True)
class MeasurementSpec:
    """A measurement bank, described by its geometry rather than by a mode name.

    ``n_loc x n_scale`` Gaussian kernels.  Locations are an equally spaced grid
    over the padded sample range; scales are ``geomspace(lo, hi)`` times the
    per-axis Silverman reference bandwidth of the *training* sample.

    Nesting: two specs with the same ``(lo, hi)`` and location rule nest when
    ``(n_loc_big - 1) % (n_loc_small - 1) == 0`` and likewise for scales, so
    ``100 -> 199`` locations and ``3 -> 5`` scales are nested refinements.
    ``nests_in`` checks this so the brief's "separate adding features from
    changing their scale range" requirement can be audited, not asserted.
    """

    n_loc: int = 100
    n_scale: int = 3
    lo_mult: float = 0.25
    hi_mult: float = 2.0
    source: str = "independent"     # 'independent' | 'dictionary'
    label: str = ""

    @property
    def m(self) -> int:
        return self.n_loc * self.n_scale

    def nests_in(self, other: "MeasurementSpec") -> bool:
        if self.source != other.source:
            return False
        if (self.lo_mult, self.hi_mult) != (other.lo_mult, other.hi_mult):
            return False
        for a, b in ((self.n_loc, other.n_loc), (self.n_scale, other.n_scale)):
            if b < a:
                return False
            if a > 1 and (b - 1) % (a - 1) != 0:
                return False
            if a == 1 and b != 1:
                return False
        return True


def build_measurement_bank(X: np.ndarray, spec: MeasurementSpec,
                           dictionary: Atoms | None = None) -> Atoms:
    if spec.source == "dictionary":
        if dictionary is None:
            raise ValueError("source='dictionary' needs the dictionary")
        p, q = dictionary.n_loc, dictionary.n_scale
        loc_idx = np.unique(np.linspace(0, p - 1, min(spec.n_loc, p)).astype(int))
        sc_idx = np.unique(np.linspace(0, q - 1, min(spec.n_scale, q)).astype(int))
        rows = np.concatenate([li * p + loc_idx for li in sc_idx])
        return Atoms(means=dictionary.means[rows], variances=dictionary.variances[rows],
                     n_loc=loc_idx.size, n_scale=sc_idx.size,
                     scales=dictionary.scales[sc_idx],
                     meta={"source": "dictionary", "label": spec.label})
    cfg = DictionaryConfig(n_loc=spec.n_loc, n_scale=spec.n_scale,
                           location_rule="grid",
                           scale_lo_mult=spec.lo_mult, scale_hi_mult=spec.hi_mult)
    a = build_dictionary(X, cfg)
    a.meta["source"] = "independent"
    a.meta["label"] = spec.label
    return a


# ---------------------------------------------------------------- configuration

@dataclass
class FollowupConfig:
    #: Pinned dictionary geometry. Never cross-validated in this study: the
    #: mechanism experiments require one literal atom set per cell.
    dict_n_loc: int = 60
    dict_n_scale: int = 5
    dict_lo_mult: float = 0.05
    dict_hi_mult: float = 4.0

    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)

    weighting: str = "ridge"                   # identity | diagonal | ridge
    rho_grid: tuple[float, ...] = (1e-6, 1e-3, 1e-1, 1.0)

    selection: str = "l1"                      # 'l1' | 'none'
    refit: str = "moment"                      # 'moment' | 'l2' | 'normalise'

    n_lambda: int = 6
    lam_max_ratio: float = 0.7
    lam_min_ratio: float = 3e-4
    tau: float = 1e-3
    max_support: int = 400

    #: Stage-1 solver.  'cd_v2' is V2's coordinate descent with its 500-sweep
    #: cap (the V2 estimator, exactly).  'fista' solves the same objective in
    #: residual space and actually reaches the KKT tolerance; it is a
    #: *different answer*, not a free speed-up, and is reported as such.
    stage1_solver: str = "cd_v2"        # 'cd_v2' | 'fista'
    fista_kkt_tol: float = 1e-8
    fista_max_iter: int = 20000
    #: Diagnostic-only early stop for 'cd_v2'.  Changes the answer; never used
    #: for a reported estimator.  Default False.
    stagnation_stop: bool = False
    stagnation_patience: int = 3
    max_cd_iter: int = 500

    label: str = "followup"
    family: str = "moment"        # 'moment' | 'hybrid' | 'l2' — used for reporting
    #: When true the measurement bank is chosen by ``n_aware_bank(N)`` instead
    #: of ``measurement``.  The rule is frozen in this module; the runner calls
    #: ``resolve_config`` once per cell and records the resulting bank.
    n_aware_measurement: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        d["measurement"] = asdict(self.measurement)
        return d


# ---------------------------------------------------------------- the estimator

class FollowupAKDE(Estimator):
    """AKDE with every design choice exposed; V2 is one setting of it."""

    def __init__(self, config: FollowupConfig | None = None,
                 tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning or TuningConfig(n_folds=3))
        self.config = config or FollowupConfig()
        self.name = self.config.label
        self.provenance = (
            f"followup[{self.config.family}] dict={self.config.dict_n_loc}x"
            f"{self.config.dict_n_scale} m={self.config.measurement.m} "
            f"({self.config.measurement.source}, scales "
            f"{self.config.measurement.lo_mult}-{self.config.measurement.hi_mult}h) "
            f"W={self.config.weighting} sel={self.config.selection} "
            f"refit={self.config.refit}"
        )

    # -- candidates ---------------------------------------------------

    def candidates(self, X: np.ndarray) -> list[dict]:
        cfg = self.config
        rhos = (0.0,) if cfg.weighting == "identity" else cfg.rho_grid
        if cfg.selection == "none":
            lams = [0.0]
        else:
            lams = np.geomspace(cfg.lam_max_ratio, cfg.lam_min_ratio,
                                cfg.n_lambda).tolist()
        # With refit='l2' and no selection, Omega plays no part at all, so the
        # rho grid would enumerate identical fits; collapse it and say so.
        if cfg.selection == "none" and cfg.refit == "l2":
            rhos = (rhos[0],)
        return [{"rho": r, "lam_ratio": l} for r in rhos for l in lams]

    # -- weighting ----------------------------------------------------

    @staticmethod
    def omega_apply(C: np.ndarray, weighting: str, rho_rel: float):
        """``apply(B) = Omega B`` plus conditioning diagnostics of the RAW Ĉ."""
        m = C.shape[0]
        eigs = np.linalg.eigvalsh(C)
        scale = float(np.mean(np.diag(C)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        pos = eigs[eigs > 0]
        # Effective rank = exp(entropy of the normalised positive spectrum).
        # Guard the 0 * log(0) term: with a near-singular covariance some
        # ratios underflow to exactly zero and numpy emits 0 * -inf = nan.
        # xlogy gives the correct limit (0) without the warning.
        if pos.size:
            q = pos / pos.sum()
            eff = float(np.exp(-np.sum(np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0))))
        else:
            eff = 0.0
        info = {
            "cov_eig_max": float(eigs[-1]),
            "cov_eig_min": float(eigs[0]),
            "cov_rank_numeric": int(np.sum(eigs > max(eigs[-1], 0.0) * 1e-12)),
            "cov_effective_rank": eff,
        }
        if weighting == "identity":
            info["omega_cond"] = 1.0
            return (lambda B: B), info
        if weighting == "diagonal":
            d = np.diag(C) + rho_rel * scale
            d = np.where(d > 0, d, scale)
            inv = 1.0 / d
            info["omega_cond"] = float(inv.max() / inv.min())
            return (lambda B: inv[:, None] * B if B.ndim == 2 else inv * B), info
        reg = C + rho_rel * scale * np.eye(m)
        eig_reg = eigs + rho_rel * scale
        info["omega_cond"] = float(eig_reg[-1] / max(eig_reg[0], 1e-300))
        try:
            chol = np.linalg.cholesky(reg)
        except np.linalg.LinAlgError:
            reg = reg + (abs(eig_reg[0]) + 1e-10 * scale) * np.eye(m)
            chol = np.linalg.cholesky(reg)
        return (lambda B: np.linalg.solve(chol.T, np.linalg.solve(chol, B))), info

    # -- fitting ------------------------------------------------------

    def _dict_config(self) -> DictionaryConfig:
        c = self.config
        return DictionaryConfig(n_loc=c.dict_n_loc, n_scale=c.dict_n_scale,
                                location_rule="grid",
                                scale_lo_mult=c.dict_lo_mult,
                                scale_hi_mult=c.dict_hi_mult)

    @staticmethod
    def eval_kernels(X: np.ndarray, bank: Atoms) -> np.ndarray:
        diff = X[:, None, :] - bank.means[None, :, :]
        v = bank.variances[None, :, :]
        return np.exp(-0.5 * np.sum(diff * diff / v + np.log(v) + gaussops.LOG_2PI,
                                    axis=2))

    def setup(self, X: np.ndarray) -> dict:
        """Everything that depends on the training sample but not on (rho, lam)."""
        cfg = self.config
        t0 = perf_counter()
        atoms = build_dictionary(X, self._dict_config())
        bank = build_measurement_bank(X, cfg.measurement, dictionary=atoms)
        D = gaussops.design_matrix(bank.means, bank.variances,
                                   atoms.means, atoms.variances)
        K = self.eval_kernels(X, bank)
        N = X.shape[0]
        y = K.mean(axis=0)
        Kc = K - y
        C = (Kc.T @ Kc) / (N * (N - 1.0)) if N > 1 else np.eye(bank.size)
        out = {"atoms": atoms, "bank": bank, "D": D, "y": y, "C": C,
               "N": N, "setup_time": perf_counter() - t0}
        if cfg.refit == "l2" or cfg.selection == "none":
            t1 = perf_counter()
            out["G"] = gaussops.gram_matrix(atoms.means, atoms.variances)
            out["s"] = self.eval_kernels(X, atoms).mean(axis=0)
            out["l2_time"] = perf_counter() - t1
        return out

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        cfg = self.config
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        S0 = self.setup(X)
        D, y, C, atoms = S0["D"], S0["y"], S0["C"], S0["atoms"]
        M = atoms.size

        out: list[FittedDensity | None] = [None] * len(cands)
        by_rho: dict[float, list[int]] = {}
        for i, c in enumerate(cands):
            by_rho.setdefault(float(c["rho"]), []).append(i)

        for rho, idxs in by_rho.items():
            apply_omega, cov_info = self.omega_apply(C, cfg.weighting, rho)
            t_h = perf_counter()
            OmD = apply_omega(D)
            H1 = D.T @ OmD
            H1 = 0.5 * (H1 + H1.T)
            g = D.T @ apply_omega(y)
            hess_time = perf_counter() - t_h

            if cfg.selection == "none":
                for slot in idxs:
                    out[slot] = self._finalise(
                        np.ones(M), None, S0, apply_omega, cov_info, rho,
                        float(cands[slot]["lam_ratio"]), hess_time, 0.0,
                        stage1_kkt=0.0, stage1_converged=True, no_selection=True)
                continue

            lam_max = solvers.lambda_max_nonneg(g)
            ratios = np.array([cands[i]["lam_ratio"] for i in idxs], dtype=float)
            lams = ratios * lam_max if lam_max > 0 else np.zeros_like(ratios)
            t1 = perf_counter()
            if cfg.stage1_solver == "fista":
                from .stage1 import nonneg_lasso_path_fista, omega_half
                half = omega_half(C, cfg.weighting, rho)
                A = half(D)
                bvec = half(y)
                ref = max(float(np.max(np.abs(g))) + float(np.mean(np.diag(H1))), 1e-300)
                path = nonneg_lasso_path_fista(
                    A, bvec, lams, ref=ref, kkt_tol=cfg.fista_kkt_tol,
                    max_iter=cfg.fista_max_iter)
            else:
                path = nonneg_lasso_path_v2(
                    H1, g, lams, max_iter=cfg.max_cd_iter,
                    stagnation_stop=cfg.stagnation_stop,
                    patience=cfg.stagnation_patience)
            per = (perf_counter() - t1) / max(len(idxs), 1)
            for slot, res in zip(idxs, path):
                out[slot] = self._finalise(
                    res.x, res, S0, apply_omega, cov_info, rho,
                    float(cands[slot]["lam_ratio"]), hess_time, per,
                    stage1_kkt=res.kkt_residual, stage1_converged=res.converged)
        return out

    # -- tuning with full diagnostics ---------------------------------

    def fit_logged(self, X: np.ndarray, rng: np.random.Generator,
                   oracle_ise=None) -> tuple[FittedDensity, dict]:
        """Like ``Estimator.fit`` but keeps the per-candidate record.

        ``oracle_ise(fit) -> float`` is optional and, when given, is evaluated
        on the *final-fit* path only, to log what an oracle tuner would have
        chosen.  It never influences selection: ``best_i`` comes from the
        held-out J-score alone.  The oracle column is written to a separate
        file by the runner and is labelled as an oracle solve.
        """
        from akde.estimators.base import kfold_indices                  # local
        from akde.metrics import j_score

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        cands = self.candidates(X)
        t0 = perf_counter()
        folds = kfold_indices(X.shape[0], self.tuning.n_folds, rng, self.tuning.shuffle)
        scores = np.full((len(folds), len(cands)), np.nan)
        domain = self._domain(X)
        fold_errors = 0
        for fi, (tr, va) in enumerate(folds):
            try:
                fits = self.fit_path(X[tr], cands, rng)
            except Exception:
                fold_errors += 1
                continue
            for ci, f in enumerate(fits):
                if f is None:
                    continue
                try:
                    scores[fi, ci] = j_score(f, X[va], domain=domain)
                except Exception:
                    scores[fi, ci] = np.nan
        with np.errstate(invalid="ignore"):
            mean_scores = np.nanmean(scores, axis=0)
        all_nan = bool(np.all(np.isnan(mean_scores)))
        best_i = 0 if all_nan else int(np.nanargmin(mean_scores))
        tune_time = perf_counter() - t0

        # The deployable final fit: the selected candidate only, timed on its
        # own so the reported cost is the cost a practitioner would pay.
        t1 = perf_counter()
        fit = self.fit_path(X, [cands[best_i]], rng)[0]
        fit_time = perf_counter() - t1

        # The full final path is computed only when an oracle diagnostic is
        # requested; it is a separately logged oracle solve, not part of the
        # estimator, and its cost is reported separately.
        full_path = None
        oracle_path_time = 0.0
        if oracle_ise is not None:
            t2 = perf_counter()
            full_path = self.fit_path(X, cands, rng)
            oracle_path_time = perf_counter() - t2

        cfg = self.config
        rhos = sorted({float(c["rho"]) for c in cands})
        lams = sorted({float(c["lam_ratio"]) for c in cands})
        br, bl = float(cands[best_i]["rho"]), float(cands[best_i]["lam_ratio"])
        log = {
            "n_candidates": len(cands),
            "n_folds": len(folds),
            "fold_errors": fold_errors,
            "all_candidates_failed": all_nan,
            "n_nan_scores": int(np.isnan(scores).sum()),
            "best_index": best_i,
            "best_cv_score": float(mean_scores[best_i]) if not all_nan else float("nan"),
            "cv_scores": mean_scores.tolist(),
            "sel_rho": br, "sel_lam_ratio": bl,
            "rho_at_grid_edge": bool(len(rhos) > 1 and br in (rhos[0], rhos[-1])),
            "lam_at_grid_edge": bool(len(lams) > 1 and bl in (lams[0], lams[-1])),
            "rho_edge_side": ("lo" if (len(rhos) > 1 and br == rhos[0])
                              else "hi" if (len(rhos) > 1 and br == rhos[-1]) else ""),
            "lam_edge_side": ("lo" if (len(lams) > 1 and bl == lams[0])
                              else "hi" if (len(lams) > 1 and bl == lams[-1]) else ""),
            "tune_time_s": tune_time,
            "fit_time_s": fit_time,
            "oracle_path_time_s": oracle_path_time,
        }
        if oracle_ise is not None:
            ises = np.array([oracle_ise(f) if f is not None else np.nan
                             for f in full_path], dtype=float)
            j = int(np.nanargmin(ises)) if np.isfinite(ises).any() else best_i
            log["oracle_best_index"] = j
            log["oracle_best_ise"] = float(ises[j])
            log["oracle_selected_ise"] = float(ises[best_i])
            log["oracle_tuning_gap_ratio"] = (float(ises[best_i] / ises[j])
                                              if ises[j] > 0 else float("nan"))
            log["path_ise"] = ises.tolist()
        fit.diagnostics.update(tune_time_s=tune_time, fit_time_s=fit_time,
                               total_time_s=tune_time + fit_time,
                               selected=dict(cands[best_i]),
                               provenance=self.provenance,
                               estimator=self.name)
        return fit, log

    def _finalise(self, u, res, S0, apply_omega, cov_info, rho, lam_ratio,
                  hess_time, stage1_time, *, stage1_kkt, stage1_converged,
                  no_selection=False) -> FittedDensity:
        cfg = self.config
        atoms, D, y = S0["atoms"], S0["D"], S0["y"]
        M = atoms.size
        fallback = capped = False

        if no_selection:
            S = np.arange(M)
        else:
            umax = float(u.max()) if u.size else 0.0
            S = np.flatnonzero(u > cfg.tau * umax) if umax > 0 else np.array([], int)
            if S.size > cfg.max_support:
                S = np.sort(S[np.argsort(-u[S])[: cfg.max_support]])
                capped = True
            if S.size == 0:
                fallback = True
                DtOD = np.einsum("ji,ji->i", D, apply_omega(D))
                S = np.array([int(np.argmin(0.5 * DtOD - D.T @ apply_omega(y)))])
        n_selected = int(S.size)

        t2 = perf_counter()
        if cfg.refit == "normalise":
            w = np.zeros(M)
            tot = u[S].sum()
            w[S] = u[S] / tot if tot > 0 else 1.0 / S.size
            kkt2 = float("nan")
        elif cfg.refit == "l2":
            G, s = S0["G"], S0["s"]
            H2 = 2.0 * G[np.ix_(S, S)]
            c2 = -2.0 * s[S]
            start = (u[S] / u[S].sum()) if (not no_selection and u[S].sum() > 0) else None
            r2 = solvers.simplex_qp(H2, c2, x0=start)
            w = np.zeros(M)
            w[S] = r2.x
            kkt2 = r2.kkt_residual
        else:                                   # 'moment'
            DS = D[:, S]
            H2 = DS.T @ apply_omega(DS)
            H2 = 0.5 * (H2 + H2.T)
            c2 = -DS.T @ apply_omega(y)
            start = (u[S] / u[S].sum()) if (not no_selection and u[S].sum() > 0) else None
            r2 = solvers.simplex_qp(H2, c2, x0=start)
            w = np.zeros(M)
            w[S] = r2.x
            kkt2 = r2.kkt_residual
        stage2_time = perf_counter() - t2

        keep = w > 0
        sd = np.sqrt(atoms.variances[:, 0])
        h_ref = float(np.asarray(atoms.meta["h_ref"]).ravel()[0])
        mass_fine = float(w[sd < cfg.measurement.lo_mult * h_ref].sum())
        return mixture_density(
            w[keep], atoms.means[keep], atoms.variances[keep], support=S,
            n_selected=n_selected, n_dictionary=M,
            n_measurements=int(S0["bank"].size),
            fallback_used=fallback, support_capped=capped,
            stage1_kkt=stage1_kkt, stage1_converged=stage1_converged,
            stage1_iters=(res.iterations if res is not None else 0),
            stage1_solver=cfg.stage1_solver,
            stage1_stop=(getattr(res, "stopped_reason", "") if res is not None else "n/a"),
            stage2_kkt=kkt2,
            setup_time_s=S0["setup_time"] + hess_time,
            stage1_time_s=stage1_time, stage2_time_s=stage2_time,
            rho=rho, lam_ratio=lam_ratio, weighting=cfg.weighting,
            variant=cfg.label, family=cfg.family,
            selection=cfg.selection, refit=cfg.refit,
            mass_below_meas_scale=mass_fine,
            mean_atom_sd_over_h=float(w[keep] @ sd[keep] / h_ref) if keep.any() else np.nan,
            **cov_info,
        )


# ---------------------------------------------------------------- solver with stop rule

def nonneg_lasso_path_v2(H, g, lambdas, tol=1e-9, max_iter=500,
                         stagnation_stop=True, patience=3):
    """V2's non-negative lasso path, plus an opt-in stagnation stop.

    V2 declares convergence only at relative KKT <= 1e-8 and otherwise runs all
    ``max_iter`` sweeps.  On these Hessians (cond ~ 1e19-1e21) that criterion is
    never met, so ~99% of the sweeps change nothing (audit_findings.md §3.1).

    With ``stagnation_stop=True`` the path also stops when the largest
    coordinate move has stayed below ``tol * ref`` for ``patience`` consecutive
    sweeps *and* the KKT residual has stopped improving.  The returned
    ``converged`` flag still means "KKT below 1e-8 relative"; a stagnation exit
    is reported separately in ``stopped_reason`` so it is never mistaken for
    convergence.  Setting ``stagnation_stop=False`` reproduces V2 exactly.
    """
    H = np.ascontiguousarray(H, dtype=float)
    g = np.ascontiguousarray(g, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float).ravel()
    n = g.size
    ref = max(float(np.max(np.abs(g))) + float(np.mean(np.diag(H))), 1e-300)
    eff_tol = tol * ref
    kkt_tol = 1e-8 * ref

    order = np.argsort(-lambdas)
    results: list[solvers.QPResult | None] = [None] * lambdas.size
    x = np.zeros(n)
    Hx = np.zeros(n)
    all_idx = np.arange(n, dtype=np.int64)

    for pos in order:
        lam = float(lambdas[pos])
        n_iter, converged, stalled = 0, False, 0
        best_kkt = np.inf
        reason = "max_iter"
        for _ in range(max_iter):
            grad0 = Hx - g + lam
            active = np.ascontiguousarray(all_idx[(x > 0) | (grad0 < 0)])
            if active.size == 0:
                converged, reason = True, "empty_active_set"
                break
            delta = solvers._cd_sweep_nonneg(H, g, lam, x, Hx, active, eff_tol)
            n_iter += 1
            if delta <= eff_tol or (n_iter % 5 == 0):
                k = solvers._kkt_nonneg(H, g, lam, x, Hx)
                if k <= kkt_tol:
                    converged, reason = True, "kkt"
                    break
                if stagnation_stop:
                    if delta <= eff_tol and k >= best_kkt * (1.0 - 1e-6):
                        stalled += 1
                        if stalled >= patience:
                            reason = "stagnation"
                            break
                    else:
                        stalled = 0
                    best_kkt = min(best_kkt, k)
        kkt = solvers._kkt_nonneg(H, g, lam, x, Hx)
        obj = float(0.5 * x @ Hx - g @ x + lam * x.sum())
        r = solvers.QPResult(x.copy(), obj, kkt / ref, n_iter, converged)
        r.stopped_reason = reason           # type: ignore[attr-defined]
        results[pos] = r
    return [r for r in results if r is not None]


# ---------------------------------------------------------------- variant catalogue

def _spec(n_loc, n_scale, lo, hi, source="independent", label=""):
    return MeasurementSpec(n_loc, n_scale, lo, hi, source, label)


#: Base measurement bank: V2's default geometry (100 x 3 over 0.25-2.0 h).
BASE_BANK = _spec(100, 3, 0.25, 2.0, label="base_100x3_0.25-2.0h")
#: Nested location refinement: 100 -> 199 locations, scale range UNCHANGED.
#: m = 597.  linspace(a,b,100) is a subset of linspace(a,b,199), so the added
#: features are strictly additional probes at the same three widths.
BANK_LOC_REFINED = _spec(199, 3, 0.25, 2.0, label="nested_loc_199x3_0.25-2.0h")
#: Nested scale refinement: 3 -> 5 scales over the SAME range, same locations.
#: geomspace(0.25,2,5) contains geomspace(0.25,2,3).  m = 500.
BANK_SCALE_REFINED = _spec(100, 5, 0.25, 2.0, label="nested_scale_100x5_0.25-2.0h")
#: Matched design: the measurement family IS the 60x5 dictionary.  m = 300.
BANK_MATCHED = _spec(60, 5, 0.05, 4.0, source="dictionary", label="matched_60x5")
#: Wider scale range reaching the dictionary's finest atoms (0.05h) and its
#: coarsest (4h).  NOT nested in BASE_BANK: this changes the range, not the
#: count, and is reported as a separate question.
BANK_FULLRANGE = _spec(100, 3, 0.05, 4.0, label="range_100x3_0.05-4h")
#: Range extension at matched count is impossible without changing spacing, so
#: the Stage-B candidate holds m near 300 while covering 0.05-4h with 5 scales
#: at 60 locations (m = 300 exactly, same count as BASE_BANK).
BANK_FULLRANGE_M300 = _spec(60, 5, 0.05, 4.0, label="range_60x5_0.05-4h_m300")


def make_variant(label: str, **over) -> FollowupConfig:
    base = dict(dict_n_loc=60, dict_n_scale=5, measurement=BASE_BANK,
                weighting="ridge", selection="l1", refit="moment",
                label=label, family="moment")
    base.update(over)
    return FollowupConfig(**base)


#: Stage A: eight diagnostic variants, one pinned dictionary per cell.
STAGE_A = {
    "A1_v2_moment_m300": make_variant(
        "A1_v2_moment_m300"),
    "A2_moment_full_nosel": make_variant(
        "A2_moment_full_nosel", selection="none"),
    "A3_hybrid_momentsel_l2refit": make_variant(
        "A3_hybrid_momentsel_l2refit", refit="l2", family="hybrid"),
    "A4_l2_full_simplex": make_variant(
        "A4_l2_full_simplex", selection="none", refit="l2", family="l2"),
    "A5_moment_m597_nested_loc": make_variant(
        "A5_moment_m597_nested_loc", measurement=BANK_LOC_REFINED),
    "A6_moment_matched_m300": make_variant(
        "A6_moment_matched_m300", measurement=BANK_MATCHED),
    "A7_moment_diagonal": make_variant(
        "A7_moment_diagonal", weighting="diagonal"),
    "A8_moment_identity": make_variant(
        "A8_moment_identity", weighting="identity"),
}

#: Sample-size-aware bank used by B2.  ONE rule, frozen here before Stage C:
#: below the threshold use the 60x5 wide-range bank (m=300); at or above it use
#: the nested 119x5 refinement (m=595), which literally contains the 60x5
#: locations (linspace(a,b,60) is a subset of linspace(a,b,119)).  The rule was
#: chosen on Stage-A development evidence (A5: more measurement features help
#: at N=5000 and hurt at N=500) and is not re-chosen on confirmation data.
BANK_WIDE_SMALL = _spec(60, 5, 0.05, 4.0, label="wide_60x5_0.05-4h_m300")
BANK_WIDE_LARGE = _spec(119, 5, 0.05, 4.0, label="wide_119x5_0.05-4h_m595")
N_AWARE_THRESHOLD = 2000


def n_aware_bank(N: int) -> MeasurementSpec:
    return BANK_WIDE_SMALL if N < N_AWARE_THRESHOLD else BANK_WIDE_LARGE


#: An honestly independent wide-range bank: same m = 300 and the same
#: 0.05-4h scale range as the dictionary, but 75 locations x 4 scales instead
#: of the dictionary's 60 x 5, so the measurement kernels are NOT the
#: dictionary atoms and D != G.  This exists because the first attempt at this
#: control (B1 below) turned out to be byte-identical to A6: asking
#: build_measurement_bank for an "independent" 60x5 bank over 0.05-4h
#: reconstructs exactly the 60x5 dictionary.  That is recorded in
#: diagnosis_report.md rather than quietly replaced.
BANK_WIDE_INDEP = _spec(75, 4, 0.05, 4.0, label="wide_indep_75x4_0.05-4h_m300")

#: Stage B: four variants, chosen from Stage-A evidence (see diagnosis_report.md).
STAGE_B = {
    # B1 was INTENDED to separate "the bank must reach the dictionary's fine
    # scales" from "the bank must BE the dictionary".  It does not: with
    # n_loc=60, n_scale=5 and range 0.05-4h it rebuilds the dictionary exactly,
    # so it reproduces A6 to the last bit (verified: max |ISE difference| = 0
    # over all 40 paired fits).  Kept in place, and labelled, so the wasted 40
    # fits are visible; B1b below is the control that actually answers it.
    "B1_moment_wide_indep_m300": make_variant(
        "B1_moment_wide_indep_m300", measurement=BANK_WIDE_SMALL),
    # B2 adds the one frozen sample-size rule on top of B1.
    "B2_moment_wide_naware": make_variant(
        "B2_moment_wide_naware", measurement=BANK_WIDE_SMALL,
        n_aware_measurement=True),
    # B3 asks whether the stage-1 solver artefact still matters once the bank
    # is fixed: same objective as B1, solved to the KKT tolerance.
    "B3_moment_wide_fista": make_variant(
        "B3_moment_wide_fista", measurement=BANK_WIDE_SMALL,
        stage1_solver="fista", fista_max_iter=4000),
    # B4 is the HYBRID: moment selection, empirical-L2 simplex refit.  Named
    # as a hybrid everywhere; never reported as an improvement to moment fitting.
    "B4_hybrid_wide_l2refit": make_variant(
        "B4_hybrid_wide_l2refit", measurement=BANK_WIDE_SMALL,
        refit="l2", family="hybrid"),
}

#: The replacement control for the failed B1.  Run as its own stage so the
#: budget accounting stays honest: Stage B used 4 x 40 = 160 fits of which 40
#: (B1) were a duplicate of A6, plus these 40.
STAGE_B_EXTRA = {
    "B1b_moment_wide_indep75x4": make_variant(
        "B1b_moment_wide_indep75x4", measurement=BANK_WIDE_INDEP),
}


def resolve_config(cfg: FollowupConfig, N: int) -> FollowupConfig:
    """Apply any N-dependent rule.  Called once per cell; the resolved bank is
    recorded on every fit, so the rule is auditable after the fact."""
    if not cfg.n_aware_measurement:
        return cfg
    from dataclasses import replace
    return replace(cfg, measurement=n_aware_bank(int(N)))
