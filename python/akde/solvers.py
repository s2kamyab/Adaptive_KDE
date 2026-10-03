"""Convex solvers for the two fitting stages and for SPADES.

Three problems are needed, all small-to-medium dense QPs:

1. **Non-negative penalised selection** (manuscript eq:sparse-obj)::

       min_{u >= 0}  1/2 u^T H u + (lam * 1 - g)^T u

   Because ``u >= 0`` the penalty ``lam * 1^T u`` is linear, so this is a
   non-negative lasso.  Solved by cyclic coordinate descent with an active
   set, warm-started down a decreasing ``lam`` path.  Coordinate descent is
   the right tool here: fitting the whole path costs little more than one
   solve, which is what makes validating over ``lam`` affordable across
   thousands of replicate datasets.

2. **Signed lasso** for the original SPADES objective, identical machinery
   with a two-sided soft threshold.

3. **Simplex refit** (manuscript eq:refit-obj)::

       min_{alpha >= 0, 1^T alpha = 1}  1/2 alpha^T H alpha + c^T alpha

   Solved by FISTA with exact Euclidean simplex projection, then polished by
   an equality-constrained active-set step.  The support is small after
   selection, so this is cheap and can be driven to tight KKT residuals.

Every routine returns its KKT residual so the runner can record solver
failure rates, which the evaluation protocol asks for explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "QPResult",
    "nonneg_lasso_path",
    "signed_lasso_path",
    "simplex_qp",
    "lambda_max_nonneg",
    "lambda_max_signed",
    "project_simplex",
    "HAVE_NUMBA",
]

# Coordinate descent touches one dictionary atom at a time, so the sweep is an
# inherently scalar loop over up to M = 1500 coordinates.  In pure Python that
# dominates the entire experiment's runtime; JIT-compiling just the two sweep
# kernels is worth roughly two orders of magnitude.  A NumPy fallback keeps the
# package usable without numba, at a large speed cost.
try:  # pragma: no cover - exercised by whichever branch is installed
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco


# NOTE: both kernels index ``H[i, r]`` rather than ``H[r, i]`` in the rank-one
# update.  ``H`` is symmetric in every use here (``D^T Omega D`` is symmetrised
# on construction, and the SPADES Gram matrix is symmetric by definition), so
# the two are equal -- but for a C-contiguous array the row access is
# contiguous and the column access strides, which is a large difference at
# M = 1500.

@njit(cache=True, fastmath=False)
def _sweep_nonneg_jit(H, g, lam, x, Hx, active):
    """One cyclic non-negative CD pass. Mutates ``x`` and ``Hx`` in place."""
    max_delta = 0.0
    n = H.shape[0]
    for a in range(active.shape[0]):
        i = active[a]
        hii = H[i, i]
        if hii <= 0.0:
            continue
        rho = g[i] - (Hx[i] - hii * x[i]) - lam
        new = rho / hii
        if new < 0.0:
            new = 0.0
        delta = new - x[i]
        if delta != 0.0:
            x[i] = new
            for r in range(n):
                Hx[r] += delta * H[i, r]
            ad = delta if delta > 0.0 else -delta
            if ad > max_delta:
                max_delta = ad
    return max_delta


@njit(cache=True, fastmath=False)
def _sweep_signed_jit(H, g, lam, x, Hx, active):
    """One cyclic signed (soft-threshold) CD pass."""
    max_delta = 0.0
    n = H.shape[0]
    for a in range(active.shape[0]):
        i = active[a]
        hii = H[i, i]
        if hii <= 0.0:
            continue
        rho = g[i] - (Hx[i] - hii * x[i])
        if rho > lam:
            new = (rho - lam) / hii
        elif rho < -lam:
            new = (rho + lam) / hii
        else:
            new = 0.0
        delta = new - x[i]
        if delta != 0.0:
            x[i] = new
            for r in range(n):
                Hx[r] += delta * H[i, r]
            ad = delta if delta > 0.0 else -delta
            if ad > max_delta:
                max_delta = ad
    return max_delta


@dataclass
class QPResult:
    x: np.ndarray
    objective: float
    kkt_residual: float
    iterations: int
    converged: bool


def lambda_max_nonneg(g: np.ndarray) -> float:
    """Smallest ``lam`` for which ``u = 0`` solves the non-negative lasso.

    First-order optimality at the origin is ``g <= lam * 1`` componentwise
    (manuscript, discussion after eq:selected-support), so the path should
    start at ``max_i g_i``.  Anchoring the grid here is what makes the penalty
    comparable across weighting matrices whose scales differ by orders of
    magnitude -- the recommendation that a shared numerical ``gamma = 7``
    does not give AKDE and SPADES equivalent regularisation.
    """
    return float(max(np.max(g), 0.0))


def lambda_max_signed(g: np.ndarray) -> float:
    """Same anchor for the signed problem, where the condition is |g| <= lam."""
    return float(max(np.max(np.abs(g)), 0.0))


def _cd_sweep_nonneg(H, g, lam, x, Hx, active, tol):
    """One cyclic pass of non-negative coordinate descent. Returns max change."""
    return _sweep_nonneg_jit(H, g, float(lam), x, Hx, active)


def _kkt_nonneg(H, g, lam, x, Hx) -> float:
    """Max violation of the non-negative-lasso KKT conditions."""
    grad = Hx - g + lam
    viol_pos = np.abs(np.where(x > 0, grad, 0.0))
    viol_zero = np.maximum(0.0, -np.where(x <= 0, grad, 0.0))
    return float(max(viol_pos.max(initial=0.0), viol_zero.max(initial=0.0)))


def nonneg_lasso_path(
    H: np.ndarray,
    g: np.ndarray,
    lambdas: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 500,
    scale_tol: bool = True,
) -> list[QPResult]:
    """Solve ``min_{u>=0} 1/2 u^T H u - g^T u + lam 1^T u`` for each ``lam``.

    ``lambdas`` is sorted internally into decreasing order and results are
    returned in the order given.  Warm starts propagate along the path.
    """
    H = np.ascontiguousarray(H, dtype=float)
    g = np.ascontiguousarray(g, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float).ravel()
    n = g.size

    # Every tolerance here is RELATIVE to the problem scale.  The weighting
    # matrix Omega ~ Cov(ybar)^{-1} is O(N), so H and g are inflated by a
    # factor that varies by orders of magnitude across weighting schemes and
    # sample sizes; an absolute tolerance would mean "converge instantly" for
    # one configuration and "never converge" for another.
    ref = float(np.max(np.abs(g))) + float(np.mean(np.diag(H)))
    ref = max(ref, 1e-300)
    eff_tol = tol * ref if scale_tol else tol
    kkt_tol = 1e-8 * ref

    order = np.argsort(-lambdas)
    results: list[QPResult | None] = [None] * lambdas.size

    x = np.zeros(n, dtype=float)
    Hx = np.zeros(n, dtype=float)
    all_idx = np.arange(n, dtype=np.int64)

    for pos in order:
        lam = float(lambdas[pos])
        n_iter = 0
        converged = False
        for outer in range(max_iter):
            # Cheap screening: only coordinates that are active or whose
            # gradient at zero is negative can move.
            grad0 = Hx - g + lam
            active = np.ascontiguousarray(all_idx[(x > 0) | (grad0 < 0)])
            if active.size == 0:
                converged = True
                break
            delta = _cd_sweep_nonneg(H, g, lam, x, Hx, active, eff_tol)
            n_iter += 1
            if delta <= eff_tol or (n_iter % 5 == 0):
                if _kkt_nonneg(H, g, lam, x, Hx) <= kkt_tol:
                    converged = True
                    break
        kkt = _kkt_nonneg(H, g, lam, x, Hx)
        obj = float(0.5 * x @ Hx - g @ x + lam * x.sum())
        results[pos] = QPResult(x.copy(), obj, kkt / ref, n_iter, converged)

    return [r for r in results if r is not None]


def _cd_sweep_signed(H, g, lam, x, Hx, active):
    return _sweep_signed_jit(H, g, float(lam), x, Hx, active)


def _kkt_signed(H, g, lam, x, Hx) -> float:
    grad = Hx - g
    nz = x != 0
    viol_nz = np.abs(grad[nz] + lam * np.sign(x[nz])) if nz.any() else np.array([0.0])
    z = ~nz
    viol_z = np.maximum(0.0, np.abs(grad[z]) - lam) if z.any() else np.array([0.0])
    return float(max(viol_nz.max(initial=0.0), viol_z.max(initial=0.0)))


def signed_lasso_path(
    H: np.ndarray,
    g: np.ndarray,
    lambdas: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 500,
    scale_tol: bool = True,
) -> list[QPResult]:
    """Solve ``min_x 1/2 x^T H x - g^T x + lam ||x||_1`` for each ``lam``.

    This is the original SPADES objective once its ``w^T G w - 2 s^T w +
    gamma ||w||_1`` form is put in the ``1/2 x^T H x`` convention used here
    (``H = 2G``, ``g = 2s``, ``lam = gamma``) -- the factor-of-two conversion
    the manuscript flags in the quadratic-program implementation paragraph.
    """
    H = np.ascontiguousarray(H, dtype=float)
    g = np.ascontiguousarray(g, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float).ravel()
    n = g.size

    ref = float(np.max(np.abs(g))) + float(np.mean(np.diag(H)))
    ref = max(ref, 1e-300)
    eff_tol = tol * ref if scale_tol else tol
    kkt_tol = 1e-8 * ref

    order = np.argsort(-lambdas)
    results: list[QPResult | None] = [None] * lambdas.size

    x = np.zeros(n, dtype=float)
    Hx = np.zeros(n, dtype=float)
    all_idx = np.arange(n, dtype=np.int64)

    for pos in order:
        lam = float(lambdas[pos])
        n_iter = 0
        converged = False
        for outer in range(max_iter):
            grad0 = Hx - g
            active = np.ascontiguousarray(all_idx[(x != 0) | (np.abs(grad0) > lam)])
            if active.size == 0:
                converged = True
                break
            delta = _cd_sweep_signed(H, g, lam, x, Hx, active)
            n_iter += 1
            if delta <= eff_tol or (n_iter % 5 == 0):
                if _kkt_signed(H, g, lam, x, Hx) <= kkt_tol:
                    converged = True
                    break
        kkt = _kkt_signed(H, g, lam, x, Hx)
        obj = float(0.5 * x @ Hx - g @ x + lam * np.abs(x).sum())
        results[pos] = QPResult(x.copy(), obj, kkt / ref, n_iter, converged)

    return [r for r in results if r is not None]


def project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, sum x = z}`` (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - z
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    if not cond.any():
        out = np.zeros(n)
        out[int(np.argmax(v))] = z
        return out
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _simplex_kkt(H, c, x) -> float:
    """Max KKT violation for the simplex QP.

    Stationarity requires a single multiplier ``nu`` with ``(Hx + c)_i = nu``
    on the support and ``>= nu`` off it.  ``nu`` is estimated as the mean
    gradient over the support.
    """
    grad = H @ x + c
    supp = x > 1e-12
    if not supp.any():
        return float("inf")
    nu = float(np.mean(grad[supp]))
    viol_on = float(np.max(np.abs(grad[supp] - nu)))
    off = ~supp
    viol_off = float(np.max(np.maximum(0.0, nu - grad[off]))) if off.any() else 0.0
    scale = max(1.0, float(np.max(np.abs(grad))))
    return max(viol_on, viol_off) / scale


def _power_lipschitz(H: np.ndarray, iters: int = 50) -> float:
    """Largest eigenvalue of a PSD matrix by power iteration.

    Much cheaper than ``eigvalsh`` for the 1500-atom dictionaries used here,
    where a full eigendecomposition inside a cross-validation loop would
    dominate the runtime.
    """
    n = H.shape[0]
    v = np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(iters):
        w = H @ v
        nrm = float(np.linalg.norm(w))
        if nrm <= 0:
            return 1.0
        v = w / nrm
        lam = nrm
    return float(lam)


def _equality_qp(H: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    """Solve ``min 1/2 x'Hx + c'x`` s.t. ``1'x = 1`` (no bounds) via the KKT system."""
    k = c.size
    kkt = np.empty((k + 1, k + 1))
    kkt[:k, :k] = H
    kkt[:k, k] = 1.0
    kkt[k, :k] = 1.0
    kkt[k, k] = 0.0
    rhs = np.concatenate([-c, [1.0]])
    reg = 1e-11 * max(1.0, float(np.mean(np.abs(np.diag(H)))))
    for bump in (0.0, reg, reg * 1e3, reg * 1e6):
        try:
            m = kkt.copy()
            if bump:
                m[:k, :k] += bump * np.eye(k)
            sol = np.linalg.solve(m, rhs)
            if np.all(np.isfinite(sol)):
                return sol[:k]
        except np.linalg.LinAlgError:
            continue
    return None


def _active_set_simplex(H: np.ndarray, c: np.ndarray, max_iter: int = 300,
                        x0: np.ndarray | None = None):
    """Primal active-set method for the bounded simplex QP.

    Maintains a set of coordinates pinned at zero, solves the
    equality-constrained problem on the rest, drops the most negative
    coordinate when a solve leaves the feasible region, and releases a pinned
    coordinate whose multiplier says it should move.  Returns ``None`` if it
    cannot settle, in which case the caller falls back to FISTA.

    ``x0`` seeds the free set.  This matters for the refit stage: starting
    from the coordinates that stage 1 gave positive weight typically lands one
    or two drops from the answer, instead of working down from all ``n``
    coordinates free, which costs a dense ``(n+1)`` solve per iteration.
    """
    n = c.size
    if x0 is not None:
        free = np.asarray(x0, dtype=float).ravel() > 0
        if not free.any():
            free = np.ones(n, dtype=bool)
    else:
        free = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(free)
        if idx.size == 0:
            return None
        if idx.size == 1:
            x = np.zeros(n)
            x[idx[0]] = 1.0
        else:
            sub = _equality_qp(H[np.ix_(idx, idx)], c[idx])
            if sub is None:
                return None
            if np.any(sub < -1e-12):
                # pin the most negative coordinate and retry
                free[idx[int(np.argmin(sub))]] = False
                continue
            x = np.zeros(n)
            x[idx] = np.maximum(sub, 0.0)

        grad = H @ x + c
        on = x > 1e-13
        nu = float(np.mean(grad[on])) if on.any() else float(np.min(grad))
        pinned = ~free
        if pinned.any():
            viol = nu - grad[pinned]           # > 0 means releasing would help
            j = int(np.argmax(viol))
            if viol[j] > 1e-10 * max(1.0, abs(nu)):
                free[np.flatnonzero(pinned)[j]] = True
                continue
        return x
    return None


def _fista_simplex(H, c, x0, tol, max_iter):
    n = c.size
    x = project_simplex(x0) if x0 is not None else np.full(n, 1.0 / n)
    L = _power_lipschitz(H)
    if not np.isfinite(L) or L <= 0:
        L = max(float(np.max(np.abs(H))) * n, 1.0)
    step = 1.0 / L

    def obj(v):
        return float(0.5 * v @ (H @ v) + c @ v)

    y = x.copy()
    t = 1.0
    f_prev = obj(x)
    n_iter = 0
    for k in range(max_iter):
        grad = H @ y + c
        x_new = project_simplex(y - step * grad)
        f_new = obj(x_new)
        if f_new > f_prev:
            y = x_new.copy()
            t = 1.0
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        delta = float(np.max(np.abs(x_new - x)))
        x, f_prev = x_new, f_new
        n_iter = k + 1
        if delta <= tol:
            break
    return x, n_iter


def simplex_qp(
    H: np.ndarray,
    c: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-11,
    max_iter: int = 5000,
    direct_limit: int = 400,
    fw_iter: int = 200,
) -> QPResult:
    """Solve ``min 1/2 x^T H x + c^T x`` over the probability simplex.

    Small problems (the refit stage, where the support has already been
    selected) go straight to the exact active-set method.  Large ones -- the
    full-dictionary QP that a method without a sparsity term has to solve --
    use fully corrective Frank-Wolfe: each round adds the most violating atom
    and re-solves exactly on the working set.  Because simplex QP solutions
    concentrate on few atoms, the working set stays small and this is far
    cheaper than first-order iteration over all M coordinates.

    FISTA is retained as a fallback and the returned KKT residual always
    reflects the solution actually returned.
    """
    H = np.ascontiguousarray(H, dtype=float)
    c = np.ascontiguousarray(c, dtype=float).ravel()
    n = c.size
    if n == 0:
        raise ValueError("simplex_qp: empty problem")

    def obj(v):
        return float(0.5 * v @ (H @ v) + c @ v)

    if n == 1:
        return QPResult(np.array([1.0]), float(0.5 * H[0, 0] + c[0]), 0.0, 0, True)

    # ---- small problem: exact active set ----
    if n <= direct_limit:
        x = _active_set_simplex(H, c, x0=x0)
        if x is not None:
            res = _simplex_kkt(H, c, x)
            if res < 1e-7:
                return QPResult(x, obj(x), res, 1, True)
        if x0 is not None:
            # retry cold, in case the warm start pinned the wrong coordinates
            x = _active_set_simplex(H, c, x0=None)
            if x is not None:
                res = _simplex_kkt(H, c, x)
                if res < 1e-7:
                    return QPResult(x, obj(x), res, 2, True)
        xf, it = _fista_simplex(H, c, x0, tol, max_iter)
        return QPResult(xf, obj(xf), _simplex_kkt(H, c, xf), it, True)

    # ---- large problem: fully corrective Frank-Wolfe ----
    vertex_obj = 0.5 * np.diag(H) + c
    work = [int(np.argmin(vertex_obj))]
    x = np.zeros(n)
    x[work[0]] = 1.0
    best = x.copy()
    n_iter = 0
    for it in range(fw_iter):
        n_iter = it + 1
        idx = np.array(sorted(set(work)))
        sub = _active_set_simplex(H[np.ix_(idx, idx)], c[idx])
        if sub is None:
            sub, _ = _fista_simplex(H[np.ix_(idx, idx)], c[idx], None, tol, 2000)
        x = np.zeros(n)
        x[idx] = sub
        best = x

        grad = H @ x + c
        on = x > 1e-13
        nu = float(np.mean(grad[on])) if on.any() else float(np.min(grad))
        j = int(np.argmin(grad))
        gap = nu - grad[j]
        if gap <= 1e-10 * max(1.0, abs(nu)):
            break
        # keep the working set lean: drop atoms that carry no weight
        work = [i for i in idx.tolist() if x[i] > 1e-13]
        work.append(j)

    res = _simplex_kkt(H, c, best)
    return QPResult(best, obj(best), res, n_iter, res < 1e-6)

