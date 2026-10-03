"""Bootstrap inference over replicate datasets.

What is resampled matters more than the number of resamples.  The unit of
replication here is the **independently generated dataset**, so that is what
gets resampled.  Multiple optimiser initialisations on one dataset are not
independent replications and are never treated as such.

Because every estimator sees the *same* data realisation within a replicate,
the informative quantity is the **paired** difference, whose bootstrap
distribution is far tighter than the difference of two marginal distributions.
Paired differences are resampled by replicate index, which preserves the
pairing exactly.

Provided:
  * percentile and BCa intervals for a statistic of a single sample,
  * paired-difference intervals with a bootstrap p-value,
  * median as well as mean summaries, since catastrophic failures on a few
    replicates dominate the mean (the historical ``f5`` entry is exactly this
    situation),
  * Holm adjustment across the density-by-density family of tests.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

__all__ = [
    "BootstrapResult",
    "bootstrap_statistic",
    "paired_bootstrap",
    "holm_adjust",
    "summarise",
]


@dataclass
class BootstrapResult:
    estimate: float
    lo: float
    hi: float
    se: float
    n: int
    n_boot: int
    method: str
    confidence: float
    p_value: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.estimate:.4g} [{self.lo:.4g}, {self.hi:.4g}]"


def _resample_idx(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def _bca_interval(theta_hat, boot, jack, alpha):
    """BCa endpoints from bootstrap replicates and jackknife values."""
    n_boot = boot.size
    prop = float(np.mean(boot < theta_hat))
    prop = min(max(prop, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = stats.norm.ppf(prop)

    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5)
    a = num / den if den > 0 else 0.0

    z_lo, z_hi = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)

    def adj(z):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return np.clip(stats.norm.cdf(z0 + z), 0.0, 1.0)
        return float(np.clip(stats.norm.cdf(z0 + (z0 + z) / denom), 0.0, 1.0))

    return adj(z_lo), adj(z_hi)


def bootstrap_statistic(
    x: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10000,
    confidence: float = 0.95,
    method: str = "bca",
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Bootstrap CI for ``statistic(x)`` over replicate datasets.

    ``method='bca'`` applies bias-correction and acceleration, which matters
    here because ISE distributions are strongly right-skewed -- a percentile
    interval on a skewed statistic is visibly off-centre.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    alpha = 1.0 - confidence
    if n == 0:
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan, 0, n_boot, method, confidence)
    if n == 1:
        v = float(statistic(x))
        return BootstrapResult(v, v, v, 0.0, 1, n_boot, method, confidence)

    theta_hat = float(statistic(x))
    idx = _resample_idx(n, n_boot, rng)
    boot = np.array([statistic(x[i]) for i in idx], dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return BootstrapResult(theta_hat, np.nan, np.nan, np.nan, n, n_boot, method, confidence)

    se = float(np.std(boot, ddof=1))
    if method == "bca":
        jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
        p_lo, p_hi = _bca_interval(theta_hat, boot, jack, alpha)
    else:
        p_lo, p_hi = alpha / 2.0, 1.0 - alpha / 2.0
    lo, hi = np.quantile(boot, [p_lo, p_hi])
    return BootstrapResult(theta_hat, float(lo), float(hi), se, n, int(boot.size),
                           method, confidence)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10000,
    confidence: float = 0.95,
    method: str = "bca",
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """CI and p-value for ``statistic(a - b)`` with replicates kept paired.

    ``a`` and ``b`` must be aligned: entry ``r`` of each is the score of the
    two methods on the *same* generated dataset.  Replicate indices are
    resampled jointly so that pairing survives resampling.

    The p-value is the usual two-sided bootstrap value for the null that the
    paired difference statistic is zero, obtained by centring the bootstrap
    distribution -- reported alongside the interval, not instead of it.
    """
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_bootstrap: shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    keep = np.isfinite(d)
    d = d[keep]
    n = d.size
    alpha = 1.0 - confidence
    if n == 0:
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan, 0, n_boot, method, confidence)

    theta_hat = float(statistic(d))
    idx = _resample_idx(n, n_boot, rng)
    boot = np.array([statistic(d[i]) for i in idx], dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return BootstrapResult(theta_hat, np.nan, np.nan, np.nan, n, n_boot, method, confidence)

    se = float(np.std(boot, ddof=1))
    if method == "bca" and n > 2:
        jack = np.array([statistic(np.delete(d, i)) for i in range(n)], dtype=float)
        p_lo, p_hi = _bca_interval(theta_hat, boot, jack, alpha)
    else:
        p_lo, p_hi = alpha / 2.0, 1.0 - alpha / 2.0
    lo, hi = np.quantile(boot, [p_lo, p_hi])

    centred = boot - float(np.mean(boot))
    p = 2.0 * min(
        float(np.mean(centred <= -abs(theta_hat))),
        float(np.mean(centred >= abs(theta_hat))),
    )
    p = min(1.0, max(p, 1.0 / (boot.size + 1)))

    return BootstrapResult(theta_hat, float(lo), float(hi), se, n, int(boot.size),
                           method, confidence, p_value=p)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment.

    Fifteen densities means fifteen simultaneous paired tests; reporting raw
    p-values across that family overstates significance.
    """
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    pf = p[finite]
    m = pf.size
    if m == 0:
        return out
    order = np.argsort(pf)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pf[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    out[finite] = adj
    return out


def summarise(x: np.ndarray, n_boot: int = 10000, rng=None) -> dict:
    """Mean and median with bootstrap intervals, plus robust spread.

    Both are reported because they answer different questions: the mean is what
    an MISE table conventionally shows, the median is what survives a handful
    of catastrophic fits.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    mean_ci = bootstrap_statistic(finite, np.mean, n_boot, rng=rng)
    med_ci = bootstrap_statistic(finite, np.median, n_boot, rng=rng)
    return {
        "n": int(finite.size),
        "n_nonfinite": int(x.size - finite.size),
        "mean": mean_ci.estimate, "mean_lo": mean_ci.lo, "mean_hi": mean_ci.hi,
        "median": med_ci.estimate, "median_lo": med_ci.lo, "median_hi": med_ci.hi,
        "iqr": float(np.subtract(*np.percentile(finite, [75, 25]))) if finite.size else np.nan,
        "max": float(finite.max()) if finite.size else np.nan,
    }
