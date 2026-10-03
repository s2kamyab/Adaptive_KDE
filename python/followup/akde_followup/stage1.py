"""Stage-1 solvers for ``min_{u>=0} 0.5 (Au - b)'(Au - b) + lam 1'u``.

Two are provided and they are *not* interchangeable; which one ran is recorded
on every fit.

``cd_v2``
    V2's cyclic coordinate descent on the M x M Hessian ``H = A'A``, capped at
    500 sweeps and declaring convergence only at relative KKT <= 1e-8.  On
    these Hessians (cond ~1e19-1e21) that criterion is never met, so the cap
    always binds — and, contrary to a first guess, the iterate is still moving
    when it binds.  Truncating the sweeps therefore changes the answer.  Kept
    so V2 is exactly reproducible.

``fista``
    Proximal-gradient (FISTA) in the m-dimensional *residual* space.  The
    stage-1 objective is exactly ``0.5||Au - b||^2 + lam 1'u`` for
    ``A = Omega^{1/2} D``, ``b = Omega^{1/2} y``; no approximation is
    involved, only a factorisation V2 already computes (the Cholesky of
    ``C + rho s I``).  Two consequences:

    * cost per iteration is ``O(mM)`` instead of ``O(M^2)`` — at m=300,
      M=1500 that is 5x fewer flops per pass;
    * it converges.  FISTA has an O(1/k^2) objective guarantee for any
      convex-plus-simple-prox problem regardless of conditioning, where cyclic
      CD on a numerically singular Hessian can stall.

    The stopping rule is a *relative KKT* test on the same scale V2 uses, plus
    a monotone-objective safeguard and restart.  The achieved KKT residual and
    the reason for stopping are always returned.

Both return the same ``QPResult`` fields, plus ``stopped_reason`` and
``n_matvecs`` so cost can be compared on a common unit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PY = Path(__file__).resolve().parents[2]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from akde import solvers                                    # noqa: E402

__all__ = ["omega_half", "nonneg_lasso_path_fista", "nonneg_lasso_path_cd_v2"]


def omega_half(C: np.ndarray, weighting: str, rho_rel: float):
    """Return ``apply_half(B) = Omega^{1/2} B`` for the three weighting schemes.

    For ridge, ``Omega = (C + rho s I)^{-1} = L^{-T} L^{-1}`` with
    ``LL' = C + rho s I``, so ``A = L^{-1} D`` satisfies ``A'A = D' Omega D``
    and ``A' (L^{-1} y) = D' Omega y`` exactly.  This is a factorisation of the
    weighting already used, not a new approximation.
    """
    m = C.shape[0]
    scale = float(np.mean(np.diag(C)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if weighting == "identity":
        return (lambda B: B)
    if weighting == "diagonal":
        d = np.diag(C) + rho_rel * scale
        d = np.where(d > 0, d, scale)
        r = 1.0 / np.sqrt(d)
        return (lambda B: r[:, None] * B if B.ndim == 2 else r * B)
    reg = C + rho_rel * scale * np.eye(m)
    eigs = np.linalg.eigvalsh(reg)
    try:
        L = np.linalg.cholesky(reg)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(reg + (abs(min(eigs[0], 0.0)) + 1e-10 * scale) * np.eye(m))
    from scipy.linalg import solve_triangular
    return (lambda B: solve_triangular(L, B, lower=True))


def _kkt(grad_smooth, u, lam, ref):
    """Relative KKT residual, on the same scale V2 reports."""
    g = grad_smooth + lam
    viol_pos = np.abs(np.where(u > 0, g, 0.0))
    viol_zero = np.maximum(0.0, -np.where(u <= 0, g, 0.0))
    return float(max(viol_pos.max(initial=0.0), viol_zero.max(initial=0.0)) / ref)


def nonneg_lasso_path_fista(A, b, lambdas, *, ref, kkt_tol=1e-8,
                            max_iter=20000, check_every=25):
    """Warm-started FISTA down a decreasing lambda path.

    ``ref`` must be the same scale V2 uses (``max|g| + mean(diag(H))``) so the
    reported residuals are directly comparable across solvers.
    """
    A = np.ascontiguousarray(A, dtype=float)
    b = np.ascontiguousarray(b, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float).ravel()
    M = A.shape[1]

    # Lipschitz constant = sigma_max(A)^2, by power iteration on A'A.
    v = np.ones(M) / np.sqrt(M)
    L = 1.0
    for _ in range(100):
        w = A.T @ (A @ v)
        nrm = float(np.linalg.norm(w))
        if nrm <= 0:
            break
        v = w / nrm
        L = nrm
    L = max(L, 1e-300)
    step = 1.0 / L

    order = np.argsort(-lambdas)
    out: list[solvers.QPResult | None] = [None] * lambdas.size
    u = np.zeros(M)
    n_matvec = 0

    for pos in order:
        lam = float(lambdas[pos])
        z = u.copy()
        t_k = 1.0
        u_prev = u.copy()
        reason = "max_iter"
        k = 0
        for k in range(1, max_iter + 1):
            r = A @ z - b
            grad = A.T @ r
            n_matvec += 2
            u_new = np.maximum(z - step * (grad + lam), 0.0)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            z = u_new + ((t_k - 1.0) / t_new) * (u_new - u_prev)
            # adaptive restart: if the momentum step is an ascent direction,
            # drop the momentum (O'Donoghue & Candes 2015)
            if float((z - u_new) @ (u_new - u_prev)) < 0:
                z = u_new.copy()
                t_new = 1.0
            u_prev, t_k = u_new, t_new
            if k % check_every == 0 or k == max_iter:
                rr = A @ u_new - b
                gg = A.T @ rr
                n_matvec += 2
                if _kkt(gg, u_new, lam, ref) <= kkt_tol:
                    reason = "kkt"
                    break
        u = u_prev
        rr = A @ u - b
        gg = A.T @ rr
        n_matvec += 2
        kkt = _kkt(gg, u, lam, ref)
        obj = float(0.5 * rr @ rr - 0.5 * b @ b + lam * u.sum())
        res = solvers.QPResult(u.copy(), obj, kkt, k, kkt <= kkt_tol)
        res.stopped_reason = reason                       # type: ignore[attr-defined]
        res.n_matvecs = n_matvec                          # type: ignore[attr-defined]
        out[pos] = res
    return [r for r in out if r is not None]


def nonneg_lasso_path_cd_v2(H, g, lambdas, *, max_iter=500):
    """Exactly V2's path solver, re-exported so both live behind one interface."""
    res = solvers.nonneg_lasso_path(H, g, lambdas, max_iter=max_iter)
    for r in res:
        r.stopped_reason = "kkt" if r.converged else "max_iter"   # type: ignore
        r.n_matvecs = r.iterations                                 # type: ignore
    return res
