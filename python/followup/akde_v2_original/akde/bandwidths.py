"""Univariate bandwidth selectors for the conventional-KDE references.

The evaluation asks whether adaptation beats a *well-tuned* ordinary
estimator, so these baselines have to be the real selectors rather than
rules of thumb.  Implemented here:

``silverman``
    Normal reference rule, reported only as a weak control.

``lscv``
    Least-squares cross-validation.  Computed from the exact closed-form
    leave-one-out risk for a Gaussian kernel rather than by binning, so it is
    the exact minimiser of the same integrated-square criterion used
    everywhere else in this framework.

``sheather_jones``
    The Sheather & Jones (1991) solve-the-equation plug-in rule, following
    R's ``stats::bw.SJ(method = "ste")`` including its pairwise-difference
    binning, its ``phi4``/``phi6`` functionals and its bracket expansion.
    ``dpi`` (direct plug-in, R's ``bw.SJ(method="dpi")``) is also available.

``botev_isj``
    The improved Sheather-Jones bandwidth of Botev, Grotowski & Kroese
    (2010), via their DCT fixed-point recursion.

    Note on what this actually is: the released ``kde.m`` accompanying that
    paper returns a *fixed-bandwidth* Gaussian kernel estimate evaluated
    through the DCT at the ISJ bandwidth.  That is what is reproduced here,
    and it is reported under a name that says so.  The fully adaptive linear
    diffusion estimator of their Section 5 is a different algorithm and is
    deliberately not claimed to be implemented -- the recommendations warn
    specifically against conflating the two.

All selectors return a scalar standard deviation (not a variance).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.fft import dct

__all__ = [
    "silverman",
    "scott",
    "lscv",
    "sheather_jones",
    "botev_isj",
    "select_bandwidth",
]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_DELMAX = 1000.0


def _robust_scale(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 0:
        return min(sd, iqr / 1.349)
    return sd


def silverman(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    s = _robust_scale(x)
    if s <= 0:
        s = 1.0
    return 0.9 * s * n ** (-0.2)


def scott(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    return 1.06 * float(np.std(x, ddof=1)) * x.size ** (-0.2)


# --------------------------------------------------------------------------
# Least-squares cross-validation
# --------------------------------------------------------------------------

def _pairwise_sq_diff_sums(x: np.ndarray, block: int = 2048):
    """Yield blocks of squared pairwise differences (upper triangle, i < j)."""
    n = x.size
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = x[start:stop, None] - x[None, :]
        # keep only j > i within the global indexing
        rows = np.arange(start, stop)[:, None]
        cols = np.arange(n)[None, :]
        mask = cols > rows
        yield (d * d)[mask]


def lscv_score(x: np.ndarray, h: float) -> float:
    """Exact least-squares CV criterion for a Gaussian kernel.

    ``J(h) = int fhat^2 - (2/n) sum_i fhat_{-i}(x_i)``, which for a Gaussian
    kernel equals

        (1/n^2) sum_{i,j} phi(x_i - x_j; 2h^2)
        - 2/(n(n-1)) sum_{i != j} phi(x_i - x_j; h^2).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if h <= 0:
        return np.inf
    h2 = h * h
    s_2h = 0.0  # sum over i<j of phi(d; 2h^2)
    s_h = 0.0   # sum over i<j of phi(d; h^2)
    for d2 in _pairwise_sq_diff_sums(x):
        s_2h += float(np.sum(np.exp(-d2 / (4.0 * h2))))
        s_h += float(np.sum(np.exp(-d2 / (2.0 * h2))))
    norm_2h = 1.0 / (_SQRT_2PI * np.sqrt(2.0) * h)
    norm_h = 1.0 / (_SQRT_2PI * h)
    # diagonal terms: i == j contributes n * norm_2h to the first sum
    int_f2 = (2.0 * s_2h * norm_2h + n * norm_2h) / (n * n)
    cross = 2.0 * (2.0 * s_h * norm_h) / (n * (n - 1.0))
    return float(int_f2 - cross)


def lscv(x: np.ndarray, n_grid: int = 60, lo_mult: float = 0.05, hi_mult: float = 3.0) -> float:
    """Minimise :func:`lscv_score` over a log grid, then refine.

    LSCV is notoriously multi-modal, so a coarse grid search precedes the
    local refinement rather than starting a solver from one point.
    """
    x = np.asarray(x, dtype=float).ravel()
    h_ref = silverman(x)
    if h_ref <= 0:
        return 1.0
    grid = np.logspace(np.log10(h_ref * lo_mult), np.log10(h_ref * hi_mult), n_grid)
    scores = np.array([lscv_score(x, h) for h in grid])
    scores = np.where(np.isfinite(scores), scores, np.inf)
    k = int(np.argmin(scores))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, n_grid - 1)]
    if hi <= lo:
        return float(grid[k])
    try:
        res = optimize.minimize_scalar(
            lambda h: lscv_score(x, h), bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-4 * h_ref},
        )
        if res.success and np.isfinite(res.fun) and res.fun <= scores[k]:
            return float(res.x)
    except Exception:
        pass
    return float(grid[k])


# --------------------------------------------------------------------------
# Sheather-Jones (R bw.SJ)
# --------------------------------------------------------------------------

def _bw_den(x: np.ndarray, nb: int = 1000) -> tuple[float, np.ndarray]:
    """Binned pairwise-difference counts, replicating R's ``C_bw_den``.

    R assigns each point the integer bin ``(int)(x_i / dd)`` (C truncation
    toward zero) and histograms ``|bin_i - bin_j|`` over all pairs ``i > j``.
    Rather than looping over O(n^2) pairs, the same counts are obtained as the
    autocorrelation of the bin-occupancy vector, which is what makes this
    affordable at N = 5000 inside a cross-validation loop.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    rang = (float(x.max()) - float(x.min())) * 1.01
    if rang <= 0:
        rang = 1.0
    dd = rang / nb
    idx = np.trunc(x / dd).astype(np.int64)
    idx -= idx.min()
    occ = np.bincount(idx, minlength=int(idx.max()) + 1).astype(float)

    # pairs at lag 0 are within-bin pairs; lag t>0 counts are the
    # autocorrelation of the occupancy vector
    full = np.correlate(occ, occ, mode="full")
    mid = occ.size - 1
    counts = full[mid:].copy()
    counts[0] = float(np.sum(occ * (occ - 1.0)) / 2.0)

    cnt = np.zeros(nb, dtype=float)
    take = min(nb, counts.size)
    cnt[:take] = counts[:take]
    return dd, cnt


def _phi4(n: int, dd: float, cnt: np.ndarray, h: float) -> float:
    if h <= 0:
        return np.nan
    i = np.arange(cnt.size, dtype=float)
    delta = (i * dd / h) ** 2
    ok = delta < _DELMAX
    term = np.exp(-delta[ok] / 2.0) * (delta[ok] ** 2 - 6.0 * delta[ok] + 3.0)
    s = float(np.sum(term * cnt[ok]))
    s = 2.0 * s + n * 3.0
    return s / (n * (n - 1.0) * h ** 5 * _SQRT_2PI)


def _phi6(n: int, dd: float, cnt: np.ndarray, h: float) -> float:
    if h <= 0:
        return np.nan
    i = np.arange(cnt.size, dtype=float)
    delta = (i * dd / h) ** 2
    ok = delta < _DELMAX
    d_ = delta[ok]
    term = np.exp(-d_ / 2.0) * (d_ ** 3 - 15.0 * d_ ** 2 + 45.0 * d_ - 15.0)
    s = float(np.sum(term * cnt[ok]))
    s = 2.0 * s - 15.0 * n
    return s / (n * (n - 1.0) * h ** 7 * _SQRT_2PI)


def sheather_jones(x: np.ndarray, nb: int = 1000, method: str = "ste") -> float:
    """Sheather-Jones plug-in bandwidth, following R's ``bw.SJ``.

    Falls back to :func:`silverman` (and says so via a warning-free return)
    if the ``phi6`` functional is non-positive or no sign change can be
    bracketed, matching the situations where R raises an error.  Falling back
    rather than failing keeps a single bad replicate from aborting a run;
    the fallback rate is recorded by the estimator wrapper.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        return silverman(x)
    dd, cnt = _bw_den(x, nb=nb)
    scale = _robust_scale(x)
    if scale <= 0:
        return silverman(x)

    a = 1.24 * scale * n ** (-1.0 / 7.0)
    b = 1.23 * scale * n ** (-1.0 / 9.0)
    c1 = 1.0 / (2.0 * np.sqrt(np.pi) * n)
    TD = -_phi6(n, dd, cnt, b)
    if not np.isfinite(TD) or TD <= 0:
        return silverman(x)

    if method == "dpi":
        val = _phi4(n, dd, cnt, (2.394 / (n * TD)) ** (1.0 / 7.0))
        if not np.isfinite(val) or val <= 0:
            return silverman(x)
        return float((c1 / val) ** 0.2)

    sd = float(np.std(x, ddof=1))
    hmax = 1.144 * sd * n ** (-0.2)
    if hmax <= 0:
        return silverman(x)
    lower, upper = 0.1 * hmax, hmax

    sd4 = _phi4(n, dd, cnt, a)
    if not np.isfinite(sd4) or sd4 <= 0:
        return silverman(x)
    alph2 = 1.357 * (sd4 / TD) ** (1.0 / 7.0)

    def fSD(h: float) -> float:
        val = _phi4(n, dd, cnt, alph2 * h ** (5.0 / 7.0))
        if not np.isfinite(val) or val <= 0:
            return np.nan
        return float((c1 / val) ** 0.2 - h)

    itry = 1
    flo, fhi = fSD(lower), fSD(upper)
    while not (np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0):
        if itry > 99:
            return silverman(x)
        if itry % 2:
            upper *= 1.2
        else:
            lower /= 1.2
        flo, fhi = fSD(lower), fSD(upper)
        itry += 1
    try:
        root = optimize.brentq(fSD, lower, upper, xtol=1e-6 * hmax, maxiter=200)
        return float(root)
    except Exception:
        return silverman(x)


# --------------------------------------------------------------------------
# Botev-Grotowski-Kroese improved Sheather-Jones
# --------------------------------------------------------------------------

def _isj_fixed_point(t: float, n: int, i_sq: np.ndarray, a2: np.ndarray, ell: int = 7) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        f = 2.0 * np.pi ** (2 * ell) * float(np.sum(i_sq ** ell * a2 * np.exp(-i_sq * np.pi ** 2 * t)))
        for s in range(ell - 1, 1, -1):
            k0 = float(np.prod(np.arange(1, 2 * s, 2))) / np.sqrt(2.0 * np.pi)
            const = (1.0 + 0.5 ** (s + 0.5)) / 3.0
            if f <= 0 or not np.isfinite(f):
                return np.nan
            time = (2.0 * const * k0 / (n * f)) ** (2.0 / (3.0 + 2.0 * s))
            f = 2.0 * np.pi ** (2 * s) * float(
                np.sum(i_sq ** s * a2 * np.exp(-i_sq * np.pi ** 2 * time))
            )
        if f <= 0 or not np.isfinite(f):
            return np.nan
        return float(t - (2.0 * n * np.sqrt(np.pi) * f) ** (-0.4))


def botev_isj(x: np.ndarray, n_grid: int = 1 << 12) -> float:
    """Improved Sheather-Jones bandwidth (Botev, Grotowski & Kroese, 2010).

    Returns the standard deviation of the Gaussian kernel.  Falls back to
    :func:`sheather_jones` if the fixed point cannot be bracketed.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_unique = int(np.unique(x).size)
    if n_unique < 3:
        return silverman(x)

    span = float(x.max() - x.min())
    if span <= 0:
        return silverman(x)
    lo = float(x.min()) - span / 10.0
    hi = float(x.max()) + span / 10.0
    rng_ = hi - lo

    hist, _ = np.histogram(x, bins=n_grid, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return silverman(x)
    w = hist / total

    a = dct(w, type=2, norm=None)
    i_sq = np.arange(1, n_grid, dtype=float) ** 2
    a2 = (a[1:] / 2.0) ** 2

    def fp(t):
        return _isj_fixed_point(t, n_unique, i_sq, a2)

    t_lo, t_hi = 1e-12, 0.1
    f_lo, f_hi = fp(t_lo), fp(t_hi)
    tries = 0
    while not (np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0):
        t_hi *= 2.0
        f_hi = fp(t_hi)
        tries += 1
        if tries > 30 or t_hi > 10.0:
            return sheather_jones(x)
    try:
        t_star = optimize.brentq(fp, t_lo, t_hi, xtol=1e-14, rtol=1e-12, maxiter=300)
    except Exception:
        return sheather_jones(x)
    h = float(np.sqrt(t_star) * rng_)
    if not np.isfinite(h) or h <= 0:
        return sheather_jones(x)
    return h


_SELECTORS = {
    "silverman": silverman,
    "scott": scott,
    "lscv": lscv,
    "sj": sheather_jones,
    "sj_dpi": lambda x: sheather_jones(x, method="dpi"),
    "botev_isj": botev_isj,
}


def select_bandwidth(x: np.ndarray, rule: str) -> float:
    if rule not in _SELECTORS:
        raise KeyError(f"unknown bandwidth rule {rule!r}; available: {sorted(_SELECTORS)}")
    h = float(_SELECTORS[rule](x))
    if not np.isfinite(h) or h <= 0:
        h = silverman(x)
    return h
