"""Figures for the manuscript.

    python experiments/make_figures.py --run results/main_full

Produces:
  ise_vs_n.png          error against sample size, per density, with bootstrap bands
  ise_vs_support.png    the accuracy / representation-size trade-off
  ise_vs_runtime.png    the accuracy / end-to-end-cost trade-off
  paired_forest.png     paired error ratios with confidence intervals
  fitted_densities.png  fitted curves against the truth, one panel per density
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from akde import registry  # noqa: E402
from akde.bootstrap import bootstrap_statistic  # noqa: E402
from akde.densities import MARRON_WAND_MAPPING, get_target  # noqa: E402

# A colourblind-safe qualitative palette, consistent across every figure so a
# method keeps its colour from panel to panel.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def colour_map(estimators):
    return {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(sorted(estimators))}


def fig_ise_vs_n(df, out, metric="ise"):
    targets = [t for t in [f"f{i}" for i in range(15)] if t in set(df["target"])]
    if not targets:
        targets = sorted(df["target"].unique())
    cmap = colour_map(df["estimator"].unique())
    ncol = 5
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow),
                             sharex=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, t in zip(axes.ravel(), targets):
        sub = df[df["target"] == t]
        for est, g in sub.groupby("estimator"):
            ns, med, lo, hi = [], [], [], []
            for n, gg in g.groupby("N"):
                v = gg[metric].to_numpy(float)
                v = v[np.isfinite(v)]
                if v.size < 2:
                    continue
                b = bootstrap_statistic(v, np.median, n_boot=2000,
                                        method="percentile", rng=rng)
                ns.append(n); med.append(b.estimate); lo.append(b.lo); hi.append(b.hi)
            if not ns:
                continue
            o = np.argsort(ns)
            ns = np.array(ns)[o]; med = np.array(med)[o]
            lo = np.array(lo)[o]; hi = np.array(hi)[o]
            ax.plot(ns, med, "-o", ms=3, lw=1.4, color=cmap[est], label=est)
            ax.fill_between(ns, lo, hi, color=cmap[est], alpha=0.16, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        lbl = MARRON_WAND_MAPPING.get(t, (None, t))[1]
        ax.set_title(f"{t}: {lbl}", fontsize=8)
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("N")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"median {metric.upper()}")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{metric.upper()} vs sample size (median over replicates, "
                 f"bootstrap 95% band)", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(df, out, xcol, xlabel, metric="ise", logx=True):
    """Accuracy against a cost axis, one panel per sample size."""
    if xcol not in df.columns:
        return
    sizes = sorted(df["N"].unique())
    cmap = colour_map(df["estimator"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.3 * len(sizes), 3.0),
                             squeeze=False, sharey=True)
    for ax, n in zip(axes[0], sizes):
        sub = df[df["N"] == n]
        for est, g in sub.groupby("estimator"):
            x = pd.to_numeric(g[xcol], errors="coerce").to_numpy(float)
            y = g[metric].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if m.sum() < 2:
                continue
            ax.scatter(np.median(x[m]), np.median(y[m]), s=48, color=cmap[est],
                       label=est, zorder=3, edgecolor="white", linewidth=0.6)
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"N = {n}", fontsize=9)
    axes[0][0].set_ylabel(f"median {metric.upper()}")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Accuracy versus {xlabel} (medians pooled over densities)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_paired_forest(paired_csv, out, alpha=0.05):
    if not Path(paired_csv).exists():
        return
    p = pd.read_csv(paired_csv)
    if p.empty or "ratio" not in p.columns:
        return
    sizes = sorted(p["N"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes), 4.2),
                             squeeze=False, sharey=True)
    for ax, n in zip(axes[0], sizes):
        sub = p[p["N"] == n]
        agg = sub.groupby("estimator").agg(
            ratio=("ratio", "median"), lo=("ratio_lo", "median"), hi=("ratio_hi", "median"),
            wins=("p_holm", lambda s: int((s < alpha).sum())),
        ).sort_values("ratio")
        ypos = np.arange(len(agg))
        colours = ["#55A868" if r < 1 else "#C44E52" for r in agg["ratio"]]
        ax.hlines(ypos, agg["lo"], agg["hi"], color=colours, lw=2.2, alpha=0.75)
        ax.plot(agg["ratio"], ypos, "o", ms=6, color="black", zorder=3)
        ax.axvline(1.0, color="0.35", ls="--", lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(agg.index)
        ax.set_xscale("log")
        ax.set_xlabel("ISE ratio (reference / baseline)")
        ax.set_title(f"N = {n}", fontsize=9)
    fig.suptitle("Paired error ratio, median across densities "
                 "(left of 1 favours the reference)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_fitted(out, n=500, estimators=("akde", "spades_simplex", "kde_sj", "abramson"),
               seed=1, n_loc=150, n_scale=10):
    """Fitted curves against the truth, one panel per Marron-Wand density."""
    from akde import metrics
    targets = [f"f{i}" for i in range(15)]
    cmap = colour_map(estimators)
    fig, axes = plt.subplots(3, 5, figsize=(16, 8.4), squeeze=False)
    for ax, t in zip(axes.ravel(), targets):
        target = get_target(t)
        X = target.sample(n, np.random.default_rng([seed, int(t[1:])]))
        lo, hi = target.plot_range(pad=3.5)
        grid = np.linspace(lo, hi, 1200)[:, None]
        ax.plot(grid, target.pdf(grid), color="0.15", lw=2.0, label="true density", zorder=5)
        ax.plot(X[:, 0], np.full(X.shape[0], -0.02), "|", color="0.6", ms=4, alpha=0.5)
        for est_name in estimators:
            try:
                est = registry.build(est_name, n_loc=n_loc, n_scale=n_scale)
                fit = est.fit(X, np.random.default_rng(7))
                ax.plot(grid, fit.pdf(grid), lw=1.3, color=cmap[est_name],
                        label=f"{est_name} (ISE {metrics.ise(fit, target):.2e})")
            except Exception:
                continue
        ax.set_title(f"{t}: {MARRON_WAND_MAPPING[t][1]}", fontsize=8)
        ax.legend(fontsize=5.5, loc="upper right")
        ax.set_yticks([])
    fig.suptitle(f"Fitted densities on one draw, N = {n} "
                 "(a single draw is illustrative only; see the bootstrap tables)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", default="akde")
    ap.add_argument("--fitted", action="store_true", help="also refit curves for the panel figure")
    ap.add_argument("--fitted-n", type=int, default=500)
    args = ap.parse_args()

    _style()
    run_dir = Path(args.run)
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    p = run_dir / "all_results.csv"
    df = pd.read_csv(p) if p.exists() else pd.concat(
        [pd.read_csv(s) for s in sorted(run_dir.glob("*_N*.csv"))], ignore_index=True)
    df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]

    fig_ise_vs_n(df, out / "ise_vs_n.png")
    fig_tradeoff(df, out / "ise_vs_support.png", "n_active", "active representation size")
    fig_tradeoff(df, out / "ise_vs_runtime.png", "wall_time_s", "end-to-end wall time (s)")
    fig_paired_forest(run_dir / "analysis" / f"paired_vs_{args.reference}.csv",
                      out / "paired_forest.png")
    if args.fitted:
        fig_fitted(out / "fitted_densities.png", n=args.fitted_n)

    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
