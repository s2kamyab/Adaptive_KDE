"""Oracle diagnostics — ISOLATED. Nothing here may reach a deployable estimator.

Everything in this module needs the true density `f0`.  It exists only to
decompose an already-fitted estimator's error, and it is deliberately kept out
of `variants.py` so that no oracle quantity can be wired into a fit by
accident: `FollowupAKDE.fit` takes `X` and nothing else.

Quantities (all closed form for Gaussian mixtures)
--------------------------------------------------
    a_i  = int b_i(x) f0(x) dx
    c0   = int f0(x)^2 dx
    G_ij = int b_i b_j
    R(w) = w'G w - 2 a'w + c0          (exact ISE of the mixture with weights w)

    y0_j = int k_j(x) f0(x) dx         (population measurement vector)
    C0   = Cov_{f0}(k(X)) / N          (population covariance of the sample mean)

Decomposition
-------------
    R(w_hat) = R_full* + (R_S* - R_full*) + (R(w_hat) - R_S*)
               ^dictionary  ^support         ^excess within the support

The third term bundles sampling noise, criterion mismatch, regularisation and
optimisation error.  It is NOT "criterion bias"; the population-moment control
below is what separates sampling from criterion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PY = Path(__file__).resolve().parents[2]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from akde import gaussops, solvers                     # noqa: E402
from akde.densities import MixtureSpec                 # noqa: E402
from akde.dictionary import Atoms                      # noqa: E402

__all__ = ["exact_risk_parts", "risk", "min_risk_on", "population_moments",
           "population_covariance", "decompose", "spectra", "ambiguity_probe"]


def exact_risk_parts(atoms: Atoms, target: MixtureSpec):
    """``(G, a, c0)`` for the exact-risk quadratic ``R(w) = w'Gw - 2a'w + c0``."""
    G = gaussops.gram_matrix(atoms.means, atoms.variances)
    a = gaussops.overlap_diag(atoms.means, atoms.variances,
                              target.means, target.variances) @ target.weights
    c0 = float(target.self_overlap())
    return G, a, c0


def risk(w, G, a, c0) -> float:
    w = np.asarray(w, float).ravel()
    return float(w @ G @ w - 2.0 * a @ w + c0)


def min_risk_on(G, a, c0, support=None, x0=None):
    """Minimum exact risk over the simplex, optionally restricted to ``support``."""
    idx = np.arange(G.shape[0]) if support is None else np.asarray(support, int)
    H = 2.0 * G[np.ix_(idx, idx)]
    c = -2.0 * a[idx]
    res = solvers.simplex_qp(H, c, x0=x0)
    w = np.zeros(G.shape[0])
    w[idx] = res.x
    return risk(w, G, a, c0), w, res


def population_moments(bank: Atoms, target: MixtureSpec) -> np.ndarray:
    """``y0_j = int k_j(x) f0(x) dx`` in closed form."""
    return gaussops.overlap_diag(bank.means, bank.variances,
                                 target.means, target.variances) @ target.weights


def population_covariance(bank: Atoms, target: MixtureSpec, n: int) -> np.ndarray:
    """``Cov_{f0}(k(X)) / n``, closed form for a diagonal Gaussian bank in 1-D.

    ``E[k_i k_j] = int k_i k_j f0``.  The product of two Gaussian densities is
    ``N(mu_i; mu_j, v_i+v_j) * N(x; m_ij, v_ij)`` with
    ``v_ij = (1/v_i + 1/v_j)^-1`` and ``m_ij = v_ij (mu_i/v_i + mu_j/v_j)``, so

        E[k_i k_j] = N(mu_i; mu_j, v_i+v_j) * sum_c pi_c N(m_ij; mu_c, v_ij+s_c^2).
    """
    mu = bank.means[:, 0]
    v = bank.variances[:, 0]
    pref = gaussops.overlap_diag(bank.means, bank.variances, bank.means, bank.variances)
    vij = 1.0 / (1.0 / v[:, None] + 1.0 / v[None, :])
    mij = vij * (mu[:, None] / v[:, None] + mu[None, :] / v[None, :])
    tw, tm, tv = target.weights, target.means[:, 0], target.variances[:, 0]
    s = vij[:, :, None] + tv[None, None, :]
    d = mij[:, :, None] - tm[None, None, :]
    mix = np.exp(-0.5 * (d * d / s + np.log(s) + gaussops.LOG_2PI)) @ tw
    E2 = pref * mix
    e1 = population_moments(bank, target)
    C = (E2 - np.outer(e1, e1)) / float(n)
    return 0.5 * (C + C.T)


def decompose(w_hat, atoms: Atoms, target: MixtureSpec, support=None) -> dict:
    """The three-term exact decomposition, plus its residual against R(w_hat)."""
    G, a, c0 = exact_risk_parts(atoms, target)
    R_hat = risk(w_hat, G, a, c0)
    R_full, w_full, res_full = min_risk_on(G, a, c0)
    if support is None:
        support = np.flatnonzero(np.asarray(w_hat).ravel() > 0)
    R_S, w_S, res_S = min_risk_on(G, a, c0, support=support,
                                  x0=np.asarray(w_hat).ravel()[np.asarray(support, int)])
    return {
        "R_hat": R_hat,
        "R_full_star": R_full,
        "R_S_star": R_S,
        "term_dictionary": R_full,
        "term_support": R_S - R_full,
        "term_excess": R_hat - R_S,
        "identity_residual": R_hat - (R_full + (R_S - R_full) + (R_hat - R_S)),
        "kkt_full": res_full.kkt_residual,
        "kkt_S": res_S.kkt_residual,
        "support_size": int(np.asarray(support).size),
        "w_full_star": w_full,
        "w_S_star": w_S,
    }


def spectra(C: np.ndarray, D: np.ndarray, apply_omega) -> dict:
    """Singular spectra / effective ranks of Ĉ and of Omega^{1/2} D.

    ``Omega^{1/2} D`` is obtained without forming Omega^{1/2}: its singular
    values are the square roots of the eigenvalues of ``D' Omega D``.
    Also reports the same restricted to directions tangent to ``sum(w)=1``
    (i.e. after projecting out the all-ones direction in coefficient space).
    """
    ec = np.linalg.eigvalsh(0.5 * (C + C.T))
    ec = ec[::-1]
    pc = ec[ec > 0]
    H = D.T @ apply_omega(D)
    H = 0.5 * (H + H.T)
    eh = np.linalg.eigvalsh(H)[::-1]
    ph = eh[eh > 0]
    M = H.shape[0]
    ones = np.ones(M) / np.sqrt(M)
    P = np.eye(M) - np.outer(ones, ones)
    Ht = P @ H @ P
    et = np.linalg.eigvalsh(0.5 * (Ht + Ht.T))[::-1]
    pt = et[et > 0]

    def eff(p):
        if p.size == 0:
            return 0.0
        q = p / p.sum()
        return float(np.exp(-np.sum(q * np.log(q))))

    return {
        "cov_eigs_top10": ec[:10].tolist(),
        "cov_numeric_rank": int(np.sum(ec > max(ec[0], 0.0) * 1e-12)),
        "cov_effective_rank": eff(pc),
        "omD_sv_top10": np.sqrt(np.maximum(eh[:10], 0)).tolist(),
        "omD_numeric_rank": int(np.sum(eh > max(eh[0], 0.0) * 1e-12)),
        "omD_effective_rank": eff(ph),
        "omD_cond": float(eh[0] / max(eh[eh > 0].min(), 1e-300)) if ph.size else np.inf,
        "omD_tangent_numeric_rank": int(np.sum(et > max(et[0], 0.0) * 1e-12)),
        "omD_tangent_effective_rank": eff(pt),
    }


def ambiguity_probe(D, apply_omega, G, w_ref, n_dir: int = 5, step_frac: float = 1.0,
                    face_tol: float = 1e-12):
    """Feasible directions with small weighted moment response but large v'Gv.

    Searches the generalised eigenproblem ``G v = mu (D'Omega D) v`` for the
    largest ``mu`` — a large density change per unit measured discrepancy —
    **restricted to the face of the simplex that ``w_ref`` actually lies in**.

    Restricting to the face matters.  ``w_ref`` is a simplex optimum, so most
    of its coordinates are exactly zero; a direction with a negative component
    at any of those coordinates has maximum feasible step zero, and searching
    the whole space returns only such directions.  Reporting them as
    "indistinguishable densities" would be wrong: they are not reachable.
    Here the search is over ``{v : supp(v) subset supp(w_ref), sum(v) = 0}``,
    so ``w_ref + t v`` stays a probability vector for a strictly positive
    range of ``t``, and the achievable ISE change at the largest feasible step
    is what gets reported.

    Returns the two quadratic forms, the feasible step, the ISE change it
    buys, and the weighted moment discrepancy it costs — so "large
    coefficient ambiguity" is never reported as "large density ambiguity".
    """
    w_ref = np.asarray(w_ref, float).ravel()
    S = np.flatnonzero(w_ref > face_tol)
    out = []
    if S.size < 2:
        return out
    A = D.T @ apply_omega(D)
    A = 0.5 * (A + A.T)
    Gs = G[np.ix_(S, S)]
    As = A[np.ix_(S, S)]
    k = S.size
    ones = np.ones(k) / np.sqrt(k)
    P = np.eye(k) - np.outer(ones, ones)
    Gt = 0.5 * (P @ Gs @ P + (P @ Gs @ P).T)
    At = 0.5 * (P @ As @ P + (P @ As @ P).T)
    sA = float(np.mean(np.diag(At)))
    At = At + 1e-10 * (sA if sA > 0 else 1.0) * np.eye(k)
    try:
        vals, vecs = np.linalg.eigh(np.linalg.solve(At, Gt))
    except np.linalg.LinAlgError:                              # pragma: no cover
        return out
    order = np.argsort(-vals.real)
    wS = w_ref[S]
    for j in order[:n_dir]:
        v = vecs[:, j].real
        v = v - v.mean()
        nv = np.linalg.norm(v)
        if nv <= 0:
            continue
        v /= nv
        neg = v < 0
        t_max = float(np.min(wS[neg] / (-v[neg]))) if neg.any() else np.inf
        if not np.isfinite(t_max):
            t_max = float(np.max(wS))
        t = step_frac * t_max
        vG = float(v @ Gs @ v)
        vA = float(v @ As @ v)
        out.append({
            "face_size": int(k),
            "gen_eig_G_over_DtOmD": float(vals.real[j]),
            "v_G_v": vG, "v_DtOmD_v": vA,
            "t_max_feasible": t_max,
            "delta_ise_at_t_max": float(t * t * vG),
            "delta_weighted_moment_at_t_max": float(t * t * vA),
        })
    return out
