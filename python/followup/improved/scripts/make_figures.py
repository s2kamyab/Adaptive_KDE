"""Figures for the improved-AKDE study (PDF + SVG + 300-dpi PNG, plot data cached).

    python followup/improved/scripts/make_figures.py            # build plot data, draw
    python followup/improved/scripts/make_figures.py --plot-only

    improved_accuracy_vs_N      benchmark mean ISE vs N; exp(T) vs AKDE-M; per-target ratios
    improved_fitted_densities   truth + AKDE-M + AKDE-R + SPADES + KDE-SJ, 4 targets x N=200/5000
    improved_dev_ablation       development ablation: ratio vs finalist per variant
    improved_compactness        components vs ISE; query cost vs N; fit time
    improved_hr_error_vs_r      hr family: error vs r and paired ratios
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from akde import gaussops  # noqa: E402
from akde.densities import MARRON_WAND_MAPPING, get_target  # noqa: E402
from followup.akde_followup.plotting import apply_style, save_figure, save_plot_data, truth_range  # noqa: E402
from followup.improved import PY_ROOT  # noqa: E402
from followup.improved.runner import RESULTS  # noqa: E402

FIG = HERE / "figures"
DATA = FIG / "data"
A = RESULTS / "analysis"
FIN = json.load(open(HERE / "finalist_improved.json", encoding="utf-8"))
NEW = FIN["finalist"]
ABL = FIN.get("ablation")
STAGE_C_FITS = PY_ROOT / "followup" / "results" / "stage_c" / "fits"
CAPTIONS = {}

ST = {
    "truth":          ("Ground truth",      "#000000", "-",                   2.2, 3),
    NEW:              ("AKDE-R (repaired)", "#D81B60", "-",                   2.1, 8),
    "akde_m":         ("AKDE-M",            "#0072B2", (0, (6, 3)),           1.7, 6),
    "akde_v2":        ("AKDE-V2",           "#E69F00", (0, (2, 2)),           1.2, 2),
    "spades_simplex": ("SPADES-simplex",    "#785EF0", (0, (6, 4, 1, 4)),     1.5, 7),
    "kde_sj":         ("Sheather-Jones KDE", "#009E73", (0, (1.5, 3.5)),      1.8, 7),
    "abramson":       ("Abramson KDE",      "#D55E00", (0, (4, 1.5, 1, 1.5)), 1.4, 4),
    "gmm_em":         ("Gaussian mixture (EM)", "#56B4E9", (0, (3, 2, 1, 2)), 1.4, 4),
    "akde_single":    ("AKDE single-scale", "#E69F00", (0, (6, 3)),           1.4, 4),
}
if ABL:
    ST[ABL] = ("AKDE-R core (no anchor/merge)", "#F06292", (0, (4, 2)), 1.5, 5)
MARK = {NEW: "o", "akde_m": "s", "akde_v2": "x", "spades_simplex": "^", "kde_sj": "D", "abramson": "v",
        "gmm_em": "P", "akde_single": "s"}
if ABL:
    MARK[ABL] = "h"
ORDER = [NEW] + ([ABL] if ABL else []) + ["akde_m", "akde_v2", "spades_simplex", "kde_sj", "abramson", "gmm_em"]
SIZES = (200, 1000, 5000)
DIAG = ["f0", "f2", "f5", "f9"]


def line(ax, k, x, y, **kw):
    lab, c, ls, lw, z = ST[k]
    return ax.plot(x, y, color=c, linestyle=ls, linewidth=lw, zorder=z, label=kw.pop("label", lab), **kw)


def load_fit(method, target, N, rep):
    if method in (NEW, ABL):
        p = RESULTS / "confirm" / "fits" / ("%s_N%d_r%d_%s.npz" % (target, N, rep, method))
    else:
        key = {"akde_m": "finalist", "akde_v2": "v2_akde"}.get(method, method)
        p = STAGE_C_FITS / ("%s_N%d_r%d_%s.npz" % (target, N, rep, key))
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in ("weights", "means", "variances", "ise")}


def fpdf(f, x):
    return gaussops.mixture_pdf_diag(np.asarray(x, float)[:, None], f["weights"], f["means"], f["variances"])


# ------------------------------------------------------------------ F1
def fig_accuracy(plot_only):
    stem = "improved_accuracy_vs_N"
    bm = pd.read_csv(A / "benchmark_means.csv")
    fb = pd.read_csv(A / "fixed_benchmark.csv")
    per = pd.read_csv(A / "per_target_ratios.csv")
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])},
                   table=pd.concat([bm.assign(_t="benchmark_means"), fb.assign(_t="fixed_benchmark"),
                                    per.assign(_t="per_target")], ignore_index=True),
                   meta={"bootstrap": "B=10000 within (target,N) cell, targets never resampled"})
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    ax = axes[0]
    for m in ORDER:
        s = bm[bm.method == m].sort_values("N")
        if s.empty:
            continue
        ax.errorbar(s.N, s.benchmark_mean_ise, yerr=[s.benchmark_mean_ise - s.boot_ci_lo, s.boot_ci_hi - s.benchmark_mean_ise],
                    color=ST[m][1], ls=ST[m][2], lw=1.3, marker=MARK[m], ms=4, capsize=2, elinewidth=0.6, label=ST[m][0])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$"); ax.set_ylabel("benchmark mean ISE (equal weight per target)")
    ax.set_title("Benchmark mean ISE [95% bootstrap CI]", fontsize=8)
    ax.legend(fontsize=5.4, frameon=False, ncol=2)
    ax = axes[1]
    for m in [NEW] + ([ABL] if ABL else []):
        for ref, ls, lab in (("akde_m", "-", "vs AKDE-M"), ("spades_simplex", "--", "vs SPADES-simplex"), ("kde_sj", ":", "vs KDE-SJ")):
            s = fb[(fb.method == m) & (fb.reference == ref)].sort_values("N")
            if s.empty:
                continue
            ax.errorbar(s.N, s.exp_T, yerr=[s.exp_T - s.boot_ci_lo, s.boot_ci_hi - s.exp_T], color=ST[m][1], ls=ls,
                        lw=1.3, marker=MARK[m], ms=4, capsize=2, elinewidth=0.6, label="%s %s" % (ST[m][0].split(" (")[0], lab))
    ax.axhline(1, color="k", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$"); ax.set_ylabel("fixed-benchmark paired ratio $\\exp(T)$")
    ax.set_title("Paired ratios (< 1 favours the repaired estimator)", fontsize=8)
    ax.legend(fontsize=5.4, frameon=False)
    ax = axes[2]
    for i, N in enumerate(SIZES):
        s = per[(per.method == NEW) & (per.reference == "akde_m") & (per.N == N)].sort_values("mw")
        ax.errorbar(s.mw + (i - 1) * 0.22, s.geom_mean_ratio, yerr=[s.geom_mean_ratio - s.boot_ci_lo, s.boot_ci_hi - s.geom_mean_ratio],
                    ls="", marker="o", ms=3, capsize=1.2, elinewidth=0.5, color=["#999999", "#0072B2", "#D81B60"][i], label="$N=%d$" % N)
    ax.axhline(1, color="k", lw=0.7)
    ax.set_yscale("log"); ax.set_xticks(range(1, 16)); ax.set_xlabel("Marron-Wand target")
    ax.set_ylabel("per-target ratio AKDE-R / AKDE-M"); ax.set_title("Per-target paired ratios [95% CI]", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    p = save_figure(fig, FIG, stem); plt.close(fig)
    CAPTIONS[stem] = ("Accuracy of the repaired estimator on the 15-target confirmation benchmark (30 datasets per "
                      "target and N; comparator rows are the saved Stage-C fits on the identical datasets). Left: "
                      "equal-weight-per-target benchmark mean ISE with 95% bootstrap intervals. Middle: fixed-benchmark "
                      "paired geometric-mean ratios exp(T) of AKDE-R against AKDE-M, SPADES-simplex and the "
                      "Sheather-Jones KDE; values below one favour AKDE-R. Right: per-target paired ratios against "
                      "AKDE-M. Bootstrap: 10,000 resamples of replicate ids within each (target, N) cell; the target "
                      "index is never resampled.")
    print("F1", p["pdf"])


# ------------------------------------------------------------------ F2
def fig_fits(plot_only):
    stem = "improved_fitted_densities"
    cache = DATA / ("%s_curves.npz" % stem)
    methods = ["akde_m", NEW, "spades_simplex", "kde_sj"]
    cols = (200, 5000)
    if not plot_only or not cache.exists():
        grid, table = {}, []
        for t in DIAG:
            tgt = get_target(t)
            lo, hi, xg = truth_range(tgt)
            grid["%s_x" % t] = xg; grid["%s_truth" % t] = tgt.pdf(xg)
            for N in cols:
                for m in methods:
                    f = load_fit(m, t, N, 0)
                    if f is None:
                        table.append({"target": t, "N": N, "method": m, "status": "MISSING"}); continue
                    grid["%s_N%d_%s" % (t, N, m)] = fpdf(f, xg)
                    table.append({"target": t, "N": N, "method": m, "status": "ok", "ise": float(f["ise"]),
                                  "n_components": int(f["weights"].size)})
        tab = pd.DataFrame(table)
        save_plot_data(FIG, stem, grid=grid, table=tab, meta={"replicate": "confirmation replicate 0 in every cell"})
    else:
        grid = dict(np.load(cache)); tab = pd.read_csv(DATA / ("%s_metrics.csv" % stem))
    apply_style()
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.8))
    for r, t in enumerate(DIAG):
        xg = grid["%s_x" % t]
        for c, N in enumerate(cols):
            ax = axes[r, c]
            line(ax, "truth", xg, grid["%s_truth" % t])
            ymax = float(grid["%s_truth" % t].max())
            for m in methods:
                k = "%s_N%d_%s" % (t, N, m)
                if k in grid:
                    line(ax, m, xg, grid[k]); ymax = max(ymax, float(np.nanmax(grid[k])))
            sub = tab[(tab.target == t) & (tab.N == N) & (tab.status == "ok")]
            ann = ["%s: ISE %.2e, K=%d" % (ST[m][0].split(" (")[0], rr.ise, int(rr.n_components))
                   for m in methods for rr in [sub[sub.method == m].iloc[0]] if len(sub[sub.method == m])]
            ax.text(0.02, 0.97, "\n".join(ann), transform=ax.transAxes, va="top", fontsize=5.6,
                    bbox=dict(fc="white", ec="0.8", lw=0.4, alpha=0.9, pad=1.3))
            ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(0, 1.06 * ymax)
            if r == 0:
                ax.set_title("$N = %d$" % N, pad=4)
            if c == 0:
                ax.set_ylabel("density")
            if r == 3:
                ax.set_xlabel("$x$")
        n, nm = MARRON_WAND_MAPPING[t]
        axes[r, 0].text(-0.2, 0.5, "$f_{%s}$ (MW%d, %s)" % (t[1:], n, nm), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8)
    h = [Line2D([], [], color=ST[k][1], linestyle=ST[k][2], lw=ST[k][3], label=ST[k][0]) for k in ["truth"] + methods]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.54, 1.005), fontsize=7.5)
    fig.tight_layout(rect=(0.02, 0, 1, 0.965))
    p = save_figure(fig, FIG, stem); plt.close(fig)
    CAPTIONS[stem] = ("Fitted densities on confirmation replicate 0 (the first locked seed; no selection after "
                      "inspection) for four Marron-Wand targets at N = 200 and 5000. Within a panel every method is "
                      "fitted to the identical observations. Annotations give ISE and the number of mixture components K.")
    print("F2", p["pdf"])


# ------------------------------------------------------------------ F3
def fig_dev(plot_only):
    stem = "improved_dev_ablation"
    d = pd.read_csv(RESULTS / "dev" / "dev_ablation.csv")
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])}, table=d, meta={"dev": "development seeds only"})
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    d = d[d.variant != "I0_finalist"]
    y = np.arange(len(d))
    ax = axes[0]
    ax.errorbar(d.primary_ratio, y, xerr=[d.primary_ratio - d.primary_ratio_lo, d.primary_ratio_hi - d.primary_ratio],
                ls="", marker="o", ms=4, color="#0072B2", capsize=2, elinewidth=0.6, label="Marron-Wand dev cells (primary)")
    ax.errorbar(d.hr_ratio, y + 0.25, xerr=[d.hr_ratio - d.hr_ratio_lo, d.hr_ratio_hi - d.hr_ratio], ls="", marker="s",
                ms=3.5, color="#D81B60", capsize=2, elinewidth=0.6, label="hr dev cells (gate)")
    ax.axvline(1, color="k", lw=0.7); ax.set_xscale("log")
    ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5]); ax.set_xticklabels(["0.2", "0.3", "0.5", "0.7", "1", "1.5"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y); ax.set_yticklabels(d.variant, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("geometric-mean ISE ratio vs the confirmed finalist (AKDE-M)")
    ax.set_title("Development ablation (ratio < 1 = better than AKDE-M)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    ax.barh(y, d.mean_components, color="#0072B2", alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d.variant, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("mean number of components"); ax.set_title("Compactness", fontsize=8)
    ax = axes[2]
    ax.barh(y, d.mean_wall_s, color="#785EF0", alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d.variant, fontsize=6.5); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("mean end-to-end fit time (s)"); ax.set_title("Cost", fontsize=8)
    fig.tight_layout()
    p = save_figure(fig, FIG, stem); plt.close(fig)
    CAPTIONS[stem] = ("Development ablation on the development seeds (4 Marron-Wand targets x N in {500, 5000} x 5 "
                      "datasets, plus 2 hr targets x 2 N x 2 datasets). Left: equal-weight-per-cell geometric-mean "
                      "paired ISE ratio of each variant against the confirmed finalist AKDE-M, with 95% bootstrap "
                      "intervals; the finalist was chosen by the rule fixed in the locked protocol. Middle and "
                      "right: mean component count and mean fit time. Development results; not confirmation.")
    print("F3", p["pdf"])


# ------------------------------------------------------------------ F4
def fig_compact(plot_only):
    stem = "improved_compactness"
    pc = pd.read_csv(A / "per_cell_summary.csv")
    cost_new = pd.read_csv(RESULTS / "cost" / "query_cost.csv")
    cost_new = cost_new[cost_new.stage == "confirm"]
    old_paths = glob.glob(str(PY_ROOT / "followup" / "results" / "**" / "model_size_and_query_cost.csv"), recursive=True)
    frames = [cost_new[["method", "N", "n_components", "query_us_per_point_median"]]]
    if old_paths:
        oc = pd.read_csv(old_paths[0]); oc = oc[oc.status == "ok"]
        oc["method"] = oc["method"].map({"finalist": "akde_m", "v2_akde": "akde_v2"}).fillna(oc["method"])
        frames.append(oc.rename(columns={"query_us_per_query_median": "query_us_per_point_median"})
                      [["method", "N", "n_components", "query_us_per_point_median"]])
    cost = pd.concat(frames, ignore_index=True)
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])}, table=pd.concat([pc.assign(_t="per_cell"), cost.assign(_t="cost")], ignore_index=True),
                   meta={"query": "10000 identical points, 7 timing repeats, median within model"})
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    ax = axes[0]
    for m in ORDER:
        s = pc[pc.method == m]
        if s.empty:
            continue
        ax.scatter(s.mean_components, s.mean_ise, s=14, marker=MARK[m], color=ST[m][1], alpha=0.75, edgecolors="none", label=ST[m][0])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("mean number of components (per cell)"); ax.set_ylabel("mean ISE (per cell)")
    ax.set_title("Operating points: accuracy vs size", fontsize=8); ax.legend(fontsize=5.2, frameon=False, ncol=2)
    ax = axes[1]
    for m in ORDER:
        s = cost[cost.method == m].groupby("N").query_us_per_point_median.mean()
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=ST[m][1], ls=ST[m][2], lw=1.4, marker=MARK[m], ms=3.5, label=ST[m][0])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$"); ax.set_ylabel("$\\mu$s per density evaluation"); ax.set_title("Query cost (identical queries)", fontsize=8)
    ax = axes[2]
    for m in ORDER:
        s = pc[pc.method == m]
        if s.empty:
            continue
        ax.scatter(s.mean_wall_s, s.mean_ise, s=14, marker=MARK[m], color=ST[m][1], alpha=0.75, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("mean end-to-end fit time (s)"); ax.set_ylabel("mean ISE (per cell)")
    ax.set_title("Accuracy vs fitting cost", fontsize=8)
    fig.tight_layout()
    p = save_figure(fig, FIG, stem); plt.close(fig)
    CAPTIONS[stem] = ("Compactness and cost on the confirmation benchmark. Left: every (target, N) cell's validation-"
                      "selected operating point (mean ISE vs mean number of components over 30 datasets); not a Pareto "
                      "frontier. Middle: query latency per point (mean over timed models of the median over 7 timing "
                      "repeats; comparator latencies from the package's saved measurement). Right: mean ISE vs mean "
                      "end-to-end fit time including cross-validation.")
    print("F4", p["pdf"])


# ------------------------------------------------------------------ F5
def fig_hr(plot_only):
    stem = "improved_hr_error_vs_r"
    if not (A / "hr_per_cell.csv").exists():
        return
    pc = pd.read_csv(A / "hr_per_cell.csv"); rr = pd.read_csv(A / "hr_ratios.csv")
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])}, table=pd.concat([pc.assign(_t="per_cell"), rr.assign(_t="ratios")], ignore_index=True), meta={})
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
    meths = [NEW, "akde_m", "akde_single", "spades_simplex", "kde_sj", "abramson", "gmm_em"]
    for i, N in enumerate((500, 5000)):
        for c, (col, title) in enumerate((("mean_ise", "Total ISE"), ("mean_ise_narrow", "Narrow-region ISE"))):
            ax = axes[i, c]
            for m in meths:
                s = pc[(pc.method == m) & (pc.N == N)].sort_values("r")
                if s.empty or col not in s:
                    continue
                sd = s[col.replace("mean_", "sd_") if col != "mean_ise" else "sd_ise_ddof1"]
                ax.errorbar(s.r, s[col], yerr=sd, color=ST[m][1], ls=ST[m][2], lw=1.2, marker=MARK[m], ms=3.6, capsize=1.5, elinewidth=0.6, label=ST[m][0])
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
            ax.set_xlabel("$r$"); ax.set_ylabel("ISE, mean $\\pm$ SD over 20 datasets"); ax.set_title("%s, $N=%d$" % (title, N), fontsize=8)
        ax = axes[i, 2]
        for ref, ls, lab in (("akde_m", "-", "AKDE-R / AKDE-M"), ("akde_single", "--", "AKDE-R / single-scale"), ("kde_sj", ":", "AKDE-R / KDE-SJ")):
            s = rr[(rr.method == NEW) & (rr.reference == ref) & (rr.N == N) & (rr.metric == "ise")].sort_values("r")
            if s.empty:
                continue
            ax.errorbar(s.r, s.geom_mean_ratio, yerr=[s.geom_mean_ratio - s.boot_ci_lo, s.boot_ci_hi - s.geom_mean_ratio], color=ST[NEW][1], ls=ls, lw=1.3, marker="o", ms=3.6, capsize=2, elinewidth=0.6, label=lab)
        ax.axhline(1, color="k", lw=0.7); ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("$r$"); ax.set_ylabel("paired geometric-mean ISE ratio"); ax.set_title("Paired ratios, $N=%d$ [95%% CI]" % N, fontsize=8)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)
    h = [Line2D([], [], color=ST[m][1], ls=ST[m][2], lw=1.2, marker=MARK[m], ms=3.6, label=ST[m][0]) for m in meths]
    fig.legend(handles=h, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.4)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = save_figure(fig, FIG, stem); plt.close(fig)
    CAPTIONS[stem] = ("The repaired estimator on the scale-heterogeneity family f_r (20 datasets per cell; comparator "
                      "rows are the saved bandwidth-flexibility fits on the identical datasets). Left and middle: total "
                      "and narrow-region ISE, mean +- SD across datasets. Right: paired geometric-mean ratios of AKDE-R "
                      "against AKDE-M, the single-scale ablation and the Sheather-Jones KDE with 95% bootstrap intervals.")
    print("F5", p["pdf"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plot-only", action="store_true"); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    fig_accuracy(a.plot_only); fig_fits(a.plot_only); fig_dev(a.plot_only); fig_compact(a.plot_only); fig_hr(a.plot_only)
    with open(FIG / "captions.md", "w", encoding="utf-8") as fh:
        for k, v in CAPTIONS.items():
            fh.write("## %s\n\n%s\n\n" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
