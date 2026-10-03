"""The improved AKDE estimator (see package docstring for the switches)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular

from akde import gaussops, solvers
from akde.dictionary import Atoms, _reference_bandwidth
from akde.estimators.base import Estimator, TuningConfig, kfold_indices
from akde.metrics import j_score
from akde.model import FittedDensity, mixture_density

from .merge import merge_atoms

__all__ = ["ImprovedConfig", "ImprovedAKDE", "VARIANTS", "build_dictionary_abs",
           "SILVERMAN_SCALES", "CV_ANCHOR_GRID", "CV_ANCHOR_REL"]

SILVERMAN_SCALES = tuple(float(x) for x in np.geomspace(0.05, 4.0, 5))
CV_ANCHOR_GRID = tuple(float(x) for x in np.geomspace(0.05, 4.0, 9))   # base bandwidth candidates x h_ref
CV_ANCHOR_REL = (0.25, 0.5, 1.0, 2.0, 4.0)                              # relative scales around h_cv


@dataclass
class ImprovedConfig:
    label: str = "improved"
    # dictionary
    n_loc: int = 60
    pad_mult_h: float = 4.0
    anchor: str = "silverman"             # 'silverman' | 'cv'
    scales: tuple = SILVERMAN_SCALES      # used with anchor='silverman' (x h_ref)
    cv_anchor_grid: tuple = CV_ANCHOR_GRID
    cv_anchor_rel: tuple = CV_ANCHOR_REL
    # criterion
    mode: str = "two_stage"               # 'two_stage' | 'simplex_l2'
    weighting: str = "gram"               # 'gram' | 'identity' | 'ridge'
    gram_eps: float = 1e-8                # relative to mean diag(G)
    rho_grid: tuple = (1e-6, 1e-3, 1e-1, 1.0)
    # stage 1
    solver: str = "fista"                 # 'fista' | 'cd_v2'
    kkt_tol: float = 1e-8
    max_iter: int = 20000
    n_lambda: int = 6
    lam_max_ratio: float = 0.7
    lam_min_ratio: float = 3e-4
    penalty: str = "uniform"              # 'uniform' | 'colnorm'
    tau: float = 1e-3
    max_support: int = 400
    # tuning
    n_folds: int = 3
    select: str = "min"                   # 'min' | '1se'
    # post
    merge: bool = False
    merge_tol: float = 1e-2
    merge_reach: float = 2.0

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("scales", "cv_anchor_grid", "cv_anchor_rel", "rho_grid"):
            d[k] = [float(x) for x in d[k]]
        return d


# ---------------------------------------------------------------- dictionary
def build_dictionary_abs(X: np.ndarray, n_loc: int, sds: np.ndarray, pad: float) -> Atoms:
    """Grid of ``n_loc`` centres on [min-pad, max+pad] at absolute SDs ``sds``."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] == 1:
        X = X.T
    sds = np.asarray(sds, float)
    loc = np.linspace(X.min() - pad, X.max() + pad, n_loc)[:, None]
    means = np.tile(loc, (sds.size, 1))
    variances = np.repeat((sds ** 2)[:, None], n_loc, axis=0)          # (n_scale*n_loc, 1)
    return Atoms(means=means, variances=variances, n_loc=n_loc, n_scale=int(sds.size),
                 scales=sds, meta={})


# ---------------------------------------------------------------- stage-1 solver
def nonneg_lasso_fista(A, b, lams, pen, *, ref, kkt_tol=1e-8, max_iter=20000, check_every=25):
    """min_{u>=0} 1/2||Au - b||^2 + lam * pen'u, warm-started down the lambda path.

    ``pen`` is the per-atom penalty weight vector (ones for a uniform penalty).
    Returns a list of (u, kkt_rel, iters, converged, objective) in path order.
    """
    A = np.ascontiguousarray(A, float)
    b = np.ascontiguousarray(b, float).ravel()
    M = A.shape[1]
    v = np.ones(M) / np.sqrt(M)
    L = 1.0
    for _ in range(100):
        w = A.T @ (A @ v)
        nrm = float(np.linalg.norm(w))
        if nrm <= 0:
            break
        v = w / nrm
        L = nrm
    step = 1.0 / max(L, 1e-300)
    out = []
    u = np.zeros(M)

    def kkt(u, lam):
        g = A.T @ (A @ u - b) + lam * pen
        vp = np.abs(np.where(u > 0, g, 0.0))
        vz = np.maximum(0.0, -np.where(u <= 0, g, 0.0))
        return float(max(vp.max(initial=0.0), vz.max(initial=0.0)) / ref)

    for lam in lams:
        z = u.copy()
        t_k = 1.0
        u_prev = u.copy()
        k = 0
        converged = False
        for k in range(1, max_iter + 1):
            grad = A.T @ (A @ z - b)
            u_new = np.maximum(z - step * (grad + lam * pen), 0.0)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            z = u_new + ((t_k - 1.0) / t_new) * (u_new - u_prev)
            if float((z - u_new) @ (u_new - u_prev)) < 0:       # adaptive restart
                z = u_new.copy()
                t_new = 1.0
            u_prev, t_k = u_new, t_new
            if k % check_every == 0 and kkt(u_new, lam) <= kkt_tol:
                converged = True
                break
        u = u_prev
        r = A @ u - b
        obj = float(0.5 * r @ r + lam * pen @ u)
        out.append((u.copy(), kkt(u, lam), k, converged, obj))
    return out


# ---------------------------------------------------------------- estimator
class ImprovedAKDE(Estimator):
    name = "improved_akde"

    def __init__(self, config: ImprovedConfig | None = None) -> None:
        self.config = config or ImprovedConfig()
        super().__init__(TuningConfig(n_folds=self.config.n_folds))
        c = self.config
        self.name = c.label
        self.provenance = ("improved AKDE | mode=%s weighting=%s solver=%s penalty=%s "
                           "lambda=%d->%g select=%s folds=%d anchor=%s merge=%s"
                           % (c.mode, c.weighting, c.solver, c.penalty, c.n_lambda,
                              c.lam_min_ratio, c.select, c.n_folds, c.anchor, c.merge))

    # -- candidates ----------------------------------------------------
    def candidates(self, X: np.ndarray) -> list[dict]:
        c = self.config
        anchors = list(c.cv_anchor_grid) if c.anchor == "cv" else [None]
        if c.mode == "simplex_l2":
            return [{"anchor": a} for a in anchors]
        rhos = list(c.rho_grid) if c.weighting == "ridge" else [0.0]
        lams = np.geomspace(c.lam_max_ratio, c.lam_min_ratio, c.n_lambda).tolist()
        return [{"anchor": a, "rho": r, "lam_ratio": l} for a in anchors for r in rhos for l in lams]

    # -- setup per (sample, anchor) ------------------------------------
    def _setup(self, X: np.ndarray, anchor):
        c = self.config
        t0 = perf_counter()
        h = float(_reference_bandwidth(X)[0])
        if c.anchor == "cv":
            sds = np.asarray(c.cv_anchor_rel, float) * (float(anchor) * h)
        else:
            sds = np.asarray(c.scales, float) * h
        atoms = build_dictionary_abs(X, c.n_loc, sds, c.pad_mult_h * h)
        G = gaussops.gram_matrix(atoms.means, atoms.variances)
        G = 0.5 * (G + G.T)
        diff = X[:, None, :] - atoms.means[None, :, :]
        v = atoms.variances[None, :, :]
        Kmat = np.exp(-0.5 * np.sum(diff * diff / v + np.log(v) + gaussops.LOG_2PI, axis=2))
        s = Kmat.mean(axis=0)
        S0 = {"atoms": atoms, "G": G, "s": s, "h_ref": h, "sds": sds, "N": X.shape[0],
              "setup_time": perf_counter() - t0, "_X": X[:, 0]}
        if c.mode == "two_stage" and c.weighting == "ridge":
            Kc = Kmat - s
            S0["C"] = (Kc.T @ Kc) / (X.shape[0] * (X.shape[0] - 1.0))
        return S0

    # -- fitting --------------------------------------------------------
    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        c = self.config
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        out: list[FittedDensity | None] = [None] * len(cands)
        by_anchor: dict = {}
        for i, cd in enumerate(cands):
            by_anchor.setdefault(cd.get("anchor"), []).append(i)
        for anchor, idxs in by_anchor.items():
            S0 = self._setup(X, anchor)
            if c.mode == "simplex_l2":
                for i in idxs:
                    out[i] = self._simplex_l2(S0, cands[i])
                continue
            by_rho: dict = {}
            for i in idxs:
                by_rho.setdefault(float(cands[i]["rho"]), []).append(i)
            for rho, jdx in by_rho.items():
                A, bvec, half_info = self._whiten(S0, rho)
                G, s, atoms = S0["G"], S0["s"], S0["atoms"]
                # penalty weights
                if c.penalty == "colnorm":
                    pen = np.sqrt(np.diag(G))
                    pen = pen / pen.mean()
                else:
                    pen = np.ones(atoms.size)
                g = A.T @ bvec
                lam_max = float(max(np.max(g / pen), 0.0))
                ratios = np.array([cands[i]["lam_ratio"] for i in jdx], float)
                order = np.argsort(-ratios)
                lams = ratios[order] * lam_max
                H_diag_mean = float(np.mean(np.einsum("ji,ji->i", A, A)))
                ref = max(float(np.max(np.abs(g))) + H_diag_mean, 1e-300)
                t1 = perf_counter()
                if c.solver == "fista":
                    path = nonneg_lasso_fista(A, bvec, lams, pen, ref=ref, kkt_tol=c.kkt_tol,
                                              max_iter=c.max_iter)
                else:
                    from followup.akde_followup.variants import nonneg_lasso_path_v2
                    H = A.T @ A
                    H = 0.5 * (H + H.T)
                    # cd_v2 supports a uniform penalty only; fold pen into g via scaling
                    res = nonneg_lasso_path_v2(H, g, lams, max_iter=500, stagnation_stop=False)
                    path = [(r.x, r.kkt_residual, r.iterations, r.converged, r.objective) for r in res]
                per = (perf_counter() - t1) / max(len(jdx), 1)
                for pos, (u, kk, it, conv, obj) in zip(order, path):
                    i = jdx[pos]
                    out[i] = self._refit(S0, A, bvec, u, cands[i], rho, kk, it, conv, obj, per, half_info)
        return out

    def _whiten(self, S0, rho):
        """A, b with ||Au - b||^2 = (Gu - s)' Omega (Gu - s) + const."""
        c = self.config
        G, s = S0["G"], S0["s"]
        info = {}
        if c.weighting == "gram":
            eps = c.gram_eps * float(np.mean(np.diag(G)))
            Lc = np.linalg.cholesky(G + eps * np.eye(G.shape[0]))
            # Omega = (G+eps)^-1 = L^-T L^-1 ; A = L^-1 G, b = L^-1 s
            A = solve_triangular(Lc, G, lower=True)
            b = solve_triangular(Lc, s, lower=True)
            info["gram_eps_abs"] = eps
        elif c.weighting == "identity":
            A, b = G.copy(), s.copy()
        else:
            C = S0["C"]
            scale = float(np.mean(np.diag(C))) or 1.0
            reg = C + rho * scale * np.eye(C.shape[0])
            try:
                Lc = np.linalg.cholesky(reg)
            except np.linalg.LinAlgError:
                eig = np.linalg.eigvalsh(reg)
                Lc = np.linalg.cholesky(reg + (abs(min(eig[0], 0.0)) + 1e-10 * scale) * np.eye(C.shape[0]))
            A = solve_triangular(Lc, G, lower=True)
            b = solve_triangular(Lc, s, lower=True)
        return A, b, info

    def _refit(self, S0, A, bvec, u, cand, rho, kk, it, conv, obj, s1_time, half_info):
        c = self.config
        atoms, G, s = S0["atoms"], S0["G"], S0["s"]
        M = atoms.size
        umax = float(u.max()) if u.size else 0.0
        S = np.flatnonzero(u > c.tau * umax) if umax > 0 else np.array([], int)
        capped = fallback = False
        if S.size > c.max_support:
            S = np.sort(S[np.argsort(-u[S])[: c.max_support]])
            capped = True
        if S.size == 0:
            fallback = True
            S = np.array([int(np.argmin(0.5 * np.diag(G) - s))])
        t2 = perf_counter()
        AS = A[:, S]
        H2 = AS.T @ AS
        H2 = 0.5 * (H2 + H2.T)
        c2 = -AS.T @ bvec
        start = u[S] / u[S].sum() if u[S].sum() > 0 else None
        r2 = solvers.simplex_qp(H2, c2, x0=start)
        w = np.zeros(M)
        w[S] = r2.x
        s2_time = perf_counter() - t2
        return self._finalise(S0, w, S, dict(stage1_kkt=kk, stage1_iters=int(it),
                                              stage1_converged=bool(conv), stage1_objective=obj,
                                              stage1_time_s=s1_time, stage2_kkt=float(r2.kkt_residual),
                                              stage2_time_s=s2_time, rho=rho,
                                              lam_ratio=float(cand["lam_ratio"]),
                                              fallback_used=fallback, support_capped=capped,
                                              n_selected=int(S.size), **half_info))

    def _simplex_l2(self, S0, cand):
        G, s, atoms = S0["G"], S0["s"], S0["atoms"]
        t1 = perf_counter()
        r = solvers.simplex_qp(G, -s, direct_limit=0, fw_iter=400)
        w = r.x
        S = np.flatnonzero(w > 0)
        return self._finalise(S0, w, S, dict(stage1_kkt=np.nan, stage1_iters=int(r.iterations),
                                              stage1_converged=bool(r.converged),
                                              stage1_objective=float(r.objective),
                                              stage1_time_s=perf_counter() - t1,
                                              stage2_kkt=float(r.kkt_residual), stage2_time_s=0.0,
                                              rho=0.0, lam_ratio=np.nan, fallback_used=False,
                                              support_capped=False, n_selected=int(S.size)))

    def _finalise(self, S0, w, S, diag) -> FittedDensity:
        c = self.config
        atoms = S0["atoms"]
        keep = w > 0
        ww, mm, vv = w[keep], atoms.means[keep, 0], atoms.variances[keep, 0]
        sw = np.zeros(atoms.n_scale)
        sc_idx = (np.flatnonzero(keep) // atoms.n_loc)
        np.add.at(sw, sc_idx, ww)
        merged = {}
        if c.merge and ww.size > 1:
            t3 = perf_counter()
            ww, mm, vv, minfo = merge_atoms(S0["_X"], ww, mm, vv, tol=c.merge_tol, reach=c.merge_reach)
            merged = {"merge_" + k: v for k, v in minfo.items()}
            merged["merge_time_s"] = perf_counter() - t3
            merged["n_before_merge"] = int(keep.sum())
        fit = mixture_density(ww, mm, vv, support=S, n_dictionary=int(atoms.size),
                              n_measurements=int(atoms.size), dict_n_loc=int(atoms.n_loc),
                              dict_n_scale=int(atoms.n_scale), h_ref=S0["h_ref"],
                              dict_sds=[float(x) for x in S0["sds"]],
                              scale_weights=[float(x) for x in sw],
                              setup_time_s=S0["setup_time"], weighting=c.weighting, mode=c.mode,
                              solver=c.solver, penalty=c.penalty, selection=c.select,
                              variant=c.label, **diag, **merged)
        return fit

    # -- tuning with the 1-SE rule and full logging -------------------------
    def fit_logged(self, X: np.ndarray, rng: np.random.Generator) -> tuple[FittedDensity, dict]:
        c = self.config
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        cands = self.candidates(X)
        t0 = perf_counter()
        fold_errors = 0
        if len(cands) > 1:
            folds = kfold_indices(X.shape[0], c.n_folds, rng, True)
            scores = np.full((len(folds), len(cands)), np.nan)
            domain = self._domain(X)
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
                se_scores = np.nanstd(scores, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(scores), axis=0))
            all_nan = bool(np.all(np.isnan(mean_scores)))
            best_i = 0 if all_nan else int(np.nanargmin(mean_scores))
            if c.select == "1se" and not all_nan and "lam_ratio" in cands[0]:
                thr = mean_scores[best_i] + se_scores[best_i]
                # among candidates with the SAME anchor and rho as the best, pick the
                # largest lambda (sparsest) whose mean score is within one SE
                ba, br = cands[best_i].get("anchor"), cands[best_i].get("rho")
                elig = [i for i in range(len(cands)) if cands[i].get("anchor") == ba
                        and cands[i].get("rho") == br and np.isfinite(mean_scores[i])
                        and mean_scores[i] <= thr]
                if elig:
                    best_i = max(elig, key=lambda i: cands[i]["lam_ratio"])
            n_folds_done = len(folds)
        else:
            mean_scores = np.array([np.nan])
            se_scores = np.array([np.nan])
            best_i, all_nan, n_folds_done = 0, False, 0
        tune_time = perf_counter() - t0
        t1 = perf_counter()
        fit = self.fit_path(X, [cands[best_i]], rng)[0]
        fit_time = perf_counter() - t1
        sel = dict(cands[best_i])
        lams = sorted({float(cd["lam_ratio"]) for cd in cands if "lam_ratio" in cd})
        anchors = sorted({float(cd["anchor"]) for cd in cands if cd.get("anchor") is not None})
        log = {"n_candidates": len(cands), "n_folds": n_folds_done, "fold_errors": fold_errors,
               "all_candidates_failed": all_nan, "best_index": best_i,
               "best_cv_score": float(mean_scores[best_i]) if np.isfinite(mean_scores[best_i]) else np.nan,
               "best_cv_se": float(se_scores[best_i]) if np.isfinite(se_scores[best_i]) else np.nan,
               "cv_scores": [float(x) for x in mean_scores],
               "sel_lam_ratio": float(sel.get("lam_ratio", np.nan)),
               "sel_rho": float(sel.get("rho", np.nan)),
               "sel_anchor": float(sel["anchor"]) if sel.get("anchor") is not None else np.nan,
               "lam_at_grid_edge": bool(lams and "lam_ratio" in sel and sel["lam_ratio"] in (lams[0], lams[-1])),
               "lam_edge_side": ("lo" if lams and sel.get("lam_ratio") == lams[0]
                                 else "hi" if lams and sel.get("lam_ratio") == lams[-1] else ""),
               "anchor_at_grid_edge": bool(anchors and sel.get("anchor") is not None
                                           and sel["anchor"] in (anchors[0], anchors[-1])),
               "tune_time_s": tune_time, "fit_time_s": fit_time}
        fit.diagnostics.update(tune_time_s=tune_time, fit_time_s=fit_time,
                               total_time_s=tune_time + fit_time, selected=sel,
                               provenance=self.provenance, estimator=self.name)
        return fit, log


# ---------------------------------------------------------------- variant catalogue
def _v(label, **over):
    return ImprovedConfig(label=label, **over)


#: Development ablation.  I0 is the confirmed finalist re-run through the
#: FollowupAKDE class (see runner) so pairing and pipeline are identical.
VARIANTS = {
    "I1_gram_fista":        _v("I1_gram_fista", weighting="gram", solver="fista"),
    "I2_identity_fista":    _v("I2_identity_fista", weighting="identity", solver="fista"),
    "I3_ridge_fista":       _v("I3_ridge_fista", weighting="ridge", solver="fista"),
    "I4_simplex_l2":        _v("I4_simplex_l2", mode="simplex_l2"),
    "I5_gram_colnorm":      _v("I5_gram_colnorm", weighting="gram", penalty="colnorm"),
    "I6_gram_ext1se5f":     _v("I6_gram_ext1se5f", weighting="gram", n_lambda=8,
                               lam_min_ratio=3e-5, select="1se", n_folds=5),
    "I7_simplex_cvanchor":  _v("I7_simplex_cvanchor", mode="simplex_l2", anchor="cv"),
    "I8_simplex_merge":     _v("I8_simplex_merge", mode="simplex_l2", merge=True),
    "I9_simplex_cvanchor_merge": _v("I9_simplex_cvanchor_merge", mode="simplex_l2",
                                    anchor="cv", merge=True),
    "I10_gram_colnorm_cvanchor": _v("I10_gram_colnorm_cvanchor", weighting="gram",
                                    penalty="colnorm", anchor="cv"),
}
