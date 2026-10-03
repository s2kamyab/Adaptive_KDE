"""Turn raw replicate results into bootstrapped tables.

The reporting rules follow the evaluation recommendations:

* per-density results, not just an average over the suite;
* confidence intervals for **paired** differences, since every estimator saw
  the same data realisation;
* medians alongside means, because a handful of catastrophic fits dominates a
  mean (the historical ``f5`` entry being the motivating example);
* multiplicity control across the 15-density family of tests;
* cost and validity reported next to accuracy, so an accuracy win bought with
  a hundredfold runtime or an invalid density is visible as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_statistic, holm_adjust, paired_bootstrap, summarise
from .densities import MARRON_WAND_MAPPING

__all__ = [
    "accuracy_table",
    "paired_table",
    "cost_table",
    "diagnostics_table",
    "win_loss_summary",
    "to_latex",
]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    return df


def accuracy_table(
    df: pd.DataFrame,
    metric: str = "ise",
    n_boot: int = 10000,
    seed: int = 0,
    by: tuple[str, ...] = ("target", "N", "estimator"),
) -> pd.DataFrame:
    """Mean and median of ``metric`` with bootstrap CIs, per cell."""
    df = _clean(df)
    rng = np.random.default_rng(seed)
    rows = []
    for key, grp in df.groupby(list(by), sort=False):
        vals = grp[metric].to_numpy(dtype=float)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(summarise(vals, n_boot=n_boot, rng=rng))
        rec["metric"] = metric
        if "target" in rec and rec["target"] in MARRON_WAND_MAPPING:
            num, nm = MARRON_WAND_MAPPING[rec["target"]]
            rec["mw_number"], rec["mw_name"] = num, nm
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_table(
    df: pd.DataFrame,
    reference: str = "akde",
    metric: str = "ise",
    statistic: str = "mean",
    n_boot: int = 10000,
    seed: int = 0,
    by: tuple[str, ...] = ("target", "N"),
    relative: bool = True,
) -> pd.DataFrame:
    """Paired ``reference - baseline`` differences with CIs and Holm-adjusted p.

    A negative difference means the reference is better (lower error).  Pairing
    is by replicate index within each cell, so the two arrays compared are
    scores on identical datasets.
    """
    df = _clean(df)
    stat = {"mean": np.mean, "median": np.median}[statistic]
    rng = np.random.default_rng(seed)

    rows = []
    for key, grp in df.groupby(list(by), sort=False):
        rec_base = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        pivot = grp.pivot_table(index="rep", columns="estimator", values=metric,
                                aggfunc="first")
        if reference not in pivot.columns:
            continue
        ref = pivot[reference]
        for est in pivot.columns:
            if est == reference:
                continue
            both = pd.concat([ref, pivot[est]], axis=1).dropna()
            if both.shape[0] < 3:
                continue
            a = both.iloc[:, 0].to_numpy(dtype=float)
            b = both.iloc[:, 1].to_numpy(dtype=float)
            res = paired_bootstrap(a, b, statistic=stat, n_boot=n_boot, rng=rng)
            rec = dict(rec_base)
            rec.update({
                "reference": reference, "estimator": est, "metric": metric,
                "statistic": statistic,
                "ref_value": float(stat(a)), "est_value": float(stat(b)),
                "diff": res.estimate, "diff_lo": res.lo, "diff_hi": res.hi,
                "p_raw": res.p_value, "n_pairs": res.n,
                "reference_better": bool(res.estimate < 0),
            })
            if relative:
                # paired log-ratio: scale-free, and the right summary when
                # errors span orders of magnitude across densities
                ok = (a > 0) & (b > 0)
                if ok.sum() >= 3:
                    lr = paired_bootstrap(np.log(a[ok]), np.log(b[ok]),
                                          statistic=stat, n_boot=n_boot, rng=rng)
                    rec["log_ratio"] = lr.estimate
                    rec["ratio"] = float(np.exp(lr.estimate))
                    rec["ratio_lo"] = float(np.exp(lr.lo))
                    rec["ratio_hi"] = float(np.exp(lr.hi))
                    rec["p_logratio"] = lr.p_value
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm adjustment within each (estimator, N) family across densities
    out["p_holm"] = np.nan
    group_cols = [c for c in ("estimator", "N") if c in out.columns]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        sub = out.loc[idx, "p_raw"].to_numpy(dtype=float)
        out.loc[idx, "p_holm"] = holm_adjust(sub)
    return out


def win_loss_summary(paired: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Count significant wins / losses / ties per baseline, after Holm."""
    if paired.empty:
        return paired
    rows = []
    group_cols = [c for c in ("estimator", "N") if c in paired.columns]
    for key, grp in paired.groupby(group_cols, sort=False):
        sig = grp["p_holm"] < alpha
        wins = int(((grp["diff"] < 0) & sig).sum())
        losses = int(((grp["diff"] > 0) & sig).sum())
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update({
            "reference": grp["reference"].iloc[0],
            "n_comparisons": int(len(grp)),
            "reference_wins": wins,
            "reference_losses": losses,
            "ties": int(len(grp) - wins - losses),
            "median_ratio": float(grp["ratio"].median()) if "ratio" in grp else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def cost_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Runtime, representation size and tuning cost per estimator and N."""
    df = _clean(df)
    rng = np.random.default_rng(seed)
    rows = []
    for key, grp in df.groupby(["estimator", "N"], sort=False):
        est, n = key
        rec = {"estimator": est, "N": n, "n_obs": int(len(grp))}
        for col, label in (("fit_time_s", "fit"), ("tune_time_s", "tune"),
                           ("wall_time_s", "wall"), ("n_active", "support")):
            if col in grp.columns:
                vals = grp[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    b = bootstrap_statistic(vals, np.median, n_boot=n_boot,
                                            method="percentile", rng=rng)
                    rec[f"{label}_median"] = b.estimate
                    rec[f"{label}_lo"] = b.lo
                    rec[f"{label}_hi"] = b.hi
        rows.append(rec)
    return pd.DataFrame(rows)


def diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validity, solver and conditioning diagnostics per estimator and N."""
    df = _clean(df)
    rows = []
    for key, grp in df.groupby(["estimator", "N"], sort=False):
        est, n = key
        rec = {"estimator": est, "N": n, "n_obs": int(len(grp))}
        if "is_valid_density" in grp:
            rec["frac_valid_density"] = float(grp["is_valid_density"].astype(float).mean())
        for col, agg, label in (
            ("mass_error", "max", "max_mass_error"),
            ("neg_mass", "mean", "mean_negative_mass"),
            ("fallback_used", "mean", "fallback_rate"),
            ("support_capped", "mean", "support_cap_rate"),
            ("stage1_converged", "mean", "stage1_converged_rate"),
            ("stage1_kkt", "max", "max_stage1_kkt"),
            ("stage2_kkt", "max", "max_stage2_kkt"),
            ("cov_rank_numeric", "median", "median_cov_rank"),
            ("cov_effective_rank", "median", "median_cov_eff_rank"),
            ("omega_cond", "median", "median_omega_cond"),
            ("n_measurements", "median", "n_measurements"),
            ("nll_floored", "sum", "nll_floor_hits"),
            ("em_converged", "mean", "em_converged_rate"),
            ("lam_clipped", "mean", "mean_lam_clipped"),
        ):
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(vals):
                    rec[label] = float(getattr(vals, agg)())
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt(v, sig=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.{sig - 1}e}".replace("e-0", "e-").replace("e+0", "e")
        return f"{v:.{sig}g}"
    return str(v)


def to_latex(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None,
             caption: str = "", label: str = "", bold_min: str | None = None,
             group_col: str | None = None) -> str:
    """Render a booktabs table, optionally bolding the minimum within a group."""
    headers = headers or columns
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        f"\\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
    ]
    best_idx = set()
    if bold_min and bold_min in df.columns:
        if group_col and group_col in df.columns:
            for _, g in df.groupby(group_col, sort=False):
                v = pd.to_numeric(g[bold_min], errors="coerce")
                if v.notna().any():
                    best_idx.add(v.idxmin())
        else:
            v = pd.to_numeric(df[bold_min], errors="coerce")
            if v.notna().any():
                best_idx.add(v.idxmin())

    for i, row in df.iterrows():
        cells = []
        for c in columns:
            s = _fmt(row.get(c))
            if c == bold_min and i in best_idx:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(l for l in lines if l)
