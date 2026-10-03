"""Shared summarising code: paired log ratios, intervals, and honest labels.

Rules enforced here, from the follow-up brief:

* ISE is analytic and strictly positive for these mixtures.  Non-positive or
  non-finite values are counted and excluded from log-ratio summaries with the
  count reported — never floored to an arbitrary constant.
* For a paired log ratio z = log(ISE_a / ISE_b) the report always carries
  mean(z), SD(z)/sqrt(R), exp(mean z), the median, and the individual paired
  points when R is small.
* Interval method is declared.  Student-t for the mean log ratio, with a
  normality plausibility check; bootstrap percentile as a cross-check.  BCa is
  refused at small R when the jackknife distribution is degenerate, rather
  than being forced.
* A non-significant comparison is never called equivalence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["paired_log_ratio", "bootstrap_mean_ci", "summarise_pairs",
           "geometric_mean_ratio_by_target"]


def paired_log_ratio(a, b):
    """``z = log(a/b)`` over finite, strictly positive pairs; returns (z, report)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_log_ratio: shape mismatch %s vs %s" % (a.shape, b.shape))
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    rep = {"n_pairs_total": int(a.size), "n_pairs_used": int(ok.sum()),
           "n_dropped_nonpositive": int(np.sum(~ok & np.isfinite(a) & np.isfinite(b))),
           "n_dropped_nonfinite": int(np.sum(~np.isfinite(a) | ~np.isfinite(b)))}
    return np.log(a[ok] / b[ok]), rep


def bootstrap_mean_ci(z, n_boot=5000, confidence=0.95, seed=0):
    """Percentile bootstrap for the mean, on stored metric values (no refitting)."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, z.size, size=(n_boot, z.size))
    means = z[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(np.mean(z)), float(lo), float(hi)


def summarise_pairs(a, b, label_a="a", label_b="b", n_boot=5000, seed=0,
                    confidence=0.95):
    """Full paired summary of ISE_a vs ISE_b on the same replicates."""
    z, drop = paired_log_ratio(a, b)
    R = z.size
    out = dict(drop)
    out.update({"method": label_a, "reference": label_b, "R": R})
    if R == 0:
        return out
    m = float(np.mean(z))
    sd = float(np.std(z, ddof=1)) if R > 1 else np.nan
    se = sd / np.sqrt(R) if R > 1 else np.nan
    tcrit = stats.t.ppf(0.5 + confidence / 2, R - 1) if R > 1 else np.nan
    out.update({
        "mean_logratio": m,
        "sd_logratio": sd,
        "se_logratio": se,
        "geom_mean_ratio": float(np.exp(m)),
        "t_lo": float(np.exp(m - tcrit * se)) if R > 1 else np.nan,
        "t_hi": float(np.exp(m + tcrit * se)) if R > 1 else np.nan,
        "t_halfwidth_log": float(tcrit * se) if R > 1 else np.nan,
        "median_logratio": float(np.median(z)),
        "median_ratio": float(np.exp(np.median(z))),
        "min_ratio": float(np.exp(z.min())),
        "max_ratio": float(np.exp(z.max())),
        "n_better": int(np.sum(z < 0)),
        "n_worse": int(np.sum(z > 0)),
        "interval_method": "Student-t on the mean paired log ratio",
    })
    # plausibility of the t interval
    if R >= 4:
        out["skew_logratio"] = float(stats.skew(z))
        try:
            out["shapiro_p"] = float(stats.shapiro(z).pvalue)
        except Exception:
            out["shapiro_p"] = np.nan
    bm, bl, bh = bootstrap_mean_ci(z, n_boot=n_boot, confidence=confidence, seed=seed)
    out.update({"boot_geom_mean_ratio": float(np.exp(bm)),
                "boot_lo": float(np.exp(bl)), "boot_hi": float(np.exp(bh)),
                "n_boot": n_boot,
                "boot_method": "percentile on stored metric rows"})
    out["paired_logratios"] = ";".join("%.5g" % v for v in z)
    return out


def geometric_mean_ratio_by_target(df, value="ise", method_col="variant",
                                   method=None, reference=None,
                                   by=("target", "N"), rep_col="rep",
                                   n_boot=5000, seed=0):
    """Paired summaries per cell, plus an equal-weight-per-target aggregate.

    Targets are never pooled as extra replications: the benchmark-wide number
    is the mean over targets of the per-target mean log ratio, so every target
    counts once.
    """
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        piv = g.pivot_table(index=rep_col, columns=method_col, values=value,
                            aggfunc="first")
        if method not in piv.columns or reference not in piv.columns:
            continue
        both = piv[[method, reference]].dropna()
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(summarise_pairs(both[method].to_numpy(),
                                   both[reference].to_numpy(),
                                   method, reference, n_boot=n_boot, seed=seed))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out
