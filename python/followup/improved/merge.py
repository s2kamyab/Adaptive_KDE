"""Post-fit greedy merging of near-duplicate atoms.

Two Gaussian atoms (w_i, mu_i, v_i), (w_j, mu_j, v_j) are replaced by their
moment-matched single Gaussian

    w = w_i + w_j,   mu = (w_i mu_i + w_j mu_j) / w,
    v  = (w_i (v_i + mu_i^2) + w_j (v_j + mu_j^2)) / w - mu^2,

which preserves mass, mean and variance of the pair.  A merge is accepted only
if the TRAINING integrated-square loss  L(w) = 1/2 w'Gw - s'w  (the same loss
the estimator minimises; the truth is never consulted) rises by less than
``tol`` times its current magnitude.  Candidates are pairs whose centres are
within ``reach`` times the larger of the two SDs.  After the greedy pass the
weights are refitted on the simplex over the merged atoms.

Everything is analytic: G entries are Gaussian overlaps and s entries are
sample means of the atom at the data points.
"""
from __future__ import annotations

import numpy as np

from akde import gaussops, solvers

__all__ = ["merge_atoms"]


def _atom_means_at_data(X, mu, v):
    """s_k = (1/N) sum_n N(x_n; mu, v) for a batch of atoms (mu, v are 1-D)."""
    diff = X[:, None] - mu[None, :]
    return np.exp(-0.5 * (diff * diff / v[None, :] + np.log(v)[None, :] + gaussops.LOG_2PI)).mean(axis=0)


def _loss(w, G, s):
    return float(0.5 * w @ (G @ w) - s @ w)


def merge_atoms(X, w, mu, v, *, tol=1e-2, reach=2.0, max_merges=None, refit=True):
    """Return (w, mu, v, info) after greedy merging.  Inputs are 1-D arrays."""
    X = np.asarray(X, float).ravel()
    w = np.asarray(w, float).copy()
    mu = np.asarray(mu, float).copy()
    v = np.asarray(v, float).copy()
    keep = w > 0
    w, mu, v = w[keep], mu[keep], v[keep]
    G = gaussops.overlap_diag(mu[:, None], v[:, None], mu[:, None], v[:, None])
    s = _atom_means_at_data(X, mu, v)
    L0 = _loss(w, G, s)
    n_merged = 0
    history = []
    limit = max_merges if max_merges is not None else w.size
    while w.size > 1 and n_merged < limit:
        K = w.size
        sd = np.sqrt(v)
        best = None
        for i in range(K):
            for j in range(i + 1, K):
                if abs(mu[i] - mu[j]) > reach * max(sd[i], sd[j]):
                    continue
                wk = w[i] + w[j]
                mk = (w[i] * mu[i] + w[j] * mu[j]) / wk
                vk = (w[i] * (v[i] + mu[i] ** 2) + w[j] * (v[j] + mu[j] ** 2)) / wk - mk ** 2
                if vk <= 0:
                    continue
                # loss of the mixture with i, j replaced by k
                gk = gaussops.overlap_diag(np.array([[mk]]), np.array([[vk]]), mu[:, None], v[:, None]).ravel()
                gkk = float(gaussops.overlap_diag(np.array([[mk]]), np.array([[vk]]),
                                                   np.array([[mk]]), np.array([[vk]]))[0, 0])
                sk = float(_atom_means_at_data(X, np.array([mk]), np.array([vk]))[0])
                mask = np.ones(K, bool)
                mask[[i, j]] = False
                wr = w[mask]
                Lnew = (0.5 * wr @ (G[np.ix_(mask, mask)] @ wr) - s[mask] @ wr
                        + wk * (gk[mask] @ wr) + 0.5 * wk * wk * gkk - wk * sk)
                if best is None or Lnew < best[0]:
                    best = (Lnew, i, j, wk, mk, vk)
        if best is None:
            break
        Lnew, i, j, wk, mk, vk = best
        Lcur = _loss(w, G, s)
        if (Lnew - Lcur) > tol * max(abs(Lcur), 1e-300):
            break
        mask = np.ones(w.size, bool)
        mask[[i, j]] = False
        w = np.concatenate([w[mask], [wk]])
        mu = np.concatenate([mu[mask], [mk]])
        v = np.concatenate([v[mask], [vk]])
        G = gaussops.overlap_diag(mu[:, None], v[:, None], mu[:, None], v[:, None])
        s = _atom_means_at_data(X, mu, v)
        n_merged += 1
        history.append({"loss_before": Lcur, "loss_after": float(Lnew)})
    info = {"n_merged": n_merged, "loss_before_merge": L0, "loss_after_merge": _loss(w, G, s)}
    if refit and w.size > 1:
        r = solvers.simplex_qp(G, -s, x0=w / w.sum())
        w = r.x
        info["merge_refit_kkt"] = float(r.kkt_residual)
        info["loss_after_refit"] = _loss(w, G, s)
        keep = w > 0
        w, mu, v = w[keep], mu[keep], v[keep]
    return w, mu, v, info
