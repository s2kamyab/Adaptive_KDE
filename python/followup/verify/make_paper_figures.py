"""Task 6 - publication figures, built from saved fits and saved metric rows.

    python followup/verify/make_paper_figures.py             # build everything
    python followup/verify/make_paper_figures.py --plot-only # redraw from cache

``--plot-only`` reads only ``followup/paper_figures/data/`` and performs no
fitting and no model loading.

Every figure is written as vector PDF, vector SVG and 300-dpi PNG, with its
plotted data alongside and a caption in ``captions.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402

from akde import gaussops                          # noqa: E402
from akde.densities import MARRON_WAND_MAPPING, get_target     # noqa: E402
from followup.akde_followup.plotting import (      # noqa: E402
    apply_style, save_figure, save_plot_data, truth_range)

from followup.paths import (RAW_ROOT, DIAG_ROOT, SUMMARY_ROOT,   # noqa: E402
                            FIGURES_ROOT)
RES = RAW_ROOT
FIG = FIGURES_ROOT
DATA = FIG / "data"
CAPTIONS = {}

DIAG = ["f0", "f2", "f5", "f9"]
ALL_TARGETS = ["f%d" % i for i in range(15)]
SIZES = (200, 1000, 5000)

ST = {
    "truth":          ("Ground truth",     "#000000", "-",                2.4, 3),
    "v2_akde":        ("AKDE-V2",          "#E69F00", (0, (6, 3)),        1.5, 4),
    "finalist":       ("AKDE-M",           "#0072B2", "-",                2.1, 5),
    "spades_simplex": ("SPADES-simplex",   "#785EF0", (0, (6, 4, 1, 4)),  1.5, 6),
    "kde_sj":         ("KDE (SJ)",         "#009E73", (0, (1.5, 3.5)),    1.8, 7),
    "abramson":       ("Abramson KDE",     "#D55E00", (0, (4, 1.5, 1, 1.5)), 1.4, 2),
    "gmm_em":         ("GMM (EM)",         "#56B4E9", (0, (3, 2, 1, 2)),  1.4, 2),
}
ORDER = ["finalist", "v2_akde", "spades_simplex", "kde_sj", "abramson", "gmm_em"]


def line(ax, key, x, y, **kw):
    lab, c, ls, lw, z = ST[key]
    return ax.plot(x, y, color=c, linestyle=ls, linewidth=lw, zorder=z,
                   label=kw.pop("label", lab), **kw)


def tname(t):
    n, nm = MARRON_WAND_MAPPING[t]
    return r"$f_{%s}$ (MW%d, %s)" % (t[1:], n, nm)


def load_fit(target, N, rep, variant):
    p = RES / "stage_c" / "fits" / ("%s_N%d_r%d_%s.npz" % (target, N, rep, variant))
    if not p.exists():
        return None
    z = np.load(p)
    return {"weights": z["weights"], "means": z["means"],
            "variances": z["variances"], "ise": float(z["ise"]), "path": str(p)}


def fpdf(fit, x):
    return gaussops.mixture_pdf_diag(np.asarray(x, float)[:, None],
                                     fit["weights"], fit["means"], fit["variances"])


# ==================================================================== F1
def fig1(plot_only=False):
    stem = "fig1_fitted_densities"
    cache = DATA / ("%s_curves.npz" % stem)
    methods = ["v2_akde", "finalist", "spades_simplex", "kde_sj"]
    cols = (200, 5000)
    if not plot_only or not cache.exists():
        grid, table = {}, []
        for t in DIAG:
            tgt = get_target(t)
            x_lo, x_hi, xg = truth_range(tgt)
            grid["%s_x" % t] = xg
            grid["%s_truth" % t] = tgt.pdf(xg)
            for N in cols:
                for m in methods:
                    f = load_fit(t, N, 0, m)
                    if f is None:
                        table.append({"target": t, "N": N, "method": m,
                                      "status": "MISSING"})
                        continue
                    grid["%s_N%d_%s" % (t, N, m)] = fpdf(f, xg)
                    inside = float(np.trapezoid(fpdf(f, xg), xg))
                    table.append({"target": t, "mw": MARRON_WAND_MAPPING[t][0],
                                  "N": N, "method": m, "status": "ok",
                                  "ise": f["ise"],
                                  "n_components": int(f["weights"].size),
                                  "mass_outside_plot_window": float(max(1.0 - inside, 0.0)),
                                  "x_lo": x_lo, "x_hi": x_hi, "fit_file": f["path"]})
        tab = pd.DataFrame(table)
        save_plot_data(FIG, stem, grid=grid, table=tab, meta={
            "replicate": "confirmation replicate 0 (first prespecified); identical "
                         "data for every method within a panel",
            "window": "truth quantiles [0.0005, 0.9995], visualisation only"})
    else:
        grid = dict(np.load(cache))
        tab = pd.read_csv(DATA / ("%s_metrics.csv" % stem))

    apply_style()
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.8))
    for r, t in enumerate(DIAG):
        xg = grid["%s_x" % t]
        ymax = float(np.nanmax(grid["%s_truth" % t]))
        for N in cols:
            for m in methods:
                k = "%s_N%d_%s" % (t, N, m)
                if k in grid:
                    ymax = max(ymax, float(np.nanmax(grid[k])))
        for c, N in enumerate(cols):
            ax = axes[r, c]
            line(ax, "truth", xg, grid["%s_truth" % t])
            miss = []
            for m in methods:
                k = "%s_N%d_%s" % (t, N, m)
                if k in grid:
                    line(ax, m, xg, grid[k])
                else:
                    miss.append(ST[m][0])
            sub = tab[(tab.target == t) & (tab.N == N)] if len(tab) else tab
            ann = []
            for m, lab in (("v2_akde", "V2"), ("finalist", "M")):
                rr = sub[sub.method == m]
                if len(rr) and rr.iloc[0].get("status") == "ok":
                    ann.append("%s: ISE %.3g, K=%d"
                               % (lab, rr.iloc[0]["ise"], int(rr.iloc[0]["n_components"])))
            if miss:
                ann.append("MISSING: " + ", ".join(miss))
            ax.text(0.02, 0.97, "\n".join(ann), transform=ax.transAxes, va="top",
                    ha="left", fontsize=6.2, linespacing=1.25,
                    bbox=dict(fc="white", ec="0.8", lw=0.4, alpha=0.85, pad=1.5))
            ax.set_xlim(xg[0], xg[-1])
            ax.set_ylim(0, 1.06 * ymax)
            if r == 0:
                ax.set_title("$N = %d$" % N, pad=4)
            if c == 0:
                ax.set_ylabel("density")
            if r == 3:
                ax.set_xlabel("$x$")
        axes[r, 0].text(-0.20, 0.5, tname(t), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8)
    h = [Line2D([], [], color=ST[k][1], linestyle=ST[k][2], lw=ST[k][3],
                label=ST[k][0]) for k in ["truth"] + methods]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.54, 1.005), fontsize=7.5)
    fig.tight_layout(rect=(0.02, 0, 1, 0.965))
    p = save_figure(fig, FIG, stem)
    plt.close(fig)
    maxtail = (float(tab["mass_outside_plot_window"].max())
               if "mass_outside_plot_window" in tab else float("nan"))
    CAPTIONS[stem] = (
        "Fitted densities on a single dataset per panel, in original data units. "
        "Rows: four Marron-Wand targets; columns: N = 200 and N = 5000. Within a "
        "panel every method is fitted to the identical observations, namely "
        "confirmation replicate 0 (the first prespecified seed); no seed was "
        "chosen after inspecting results. Annotations give the ISE and the number "
        "of mixture components K of AKDE-V2 and AKDE-M; complete per-curve "
        "metrics are in data/%s_metrics.csv. The plotting window is the "
        "predeclared truth quantile range [0.0005, 0.9995] and was never used in "
        "fitting; the largest fitted mass outside that window over all curves is "
        "%.4f. No y-limit cap is applied, so the AKDE-V2 spikes are shown at full "
        "height. Where a curve is not visible it coincides with another, not "
        "because it is missing." % (stem, maxtail))
    print("F1:", p["pdf"])


# ==================================================================== F2
def fig2(N=1000, plot_only=False):
    stem = "fig2_variability_N%d" % N
    cache = DATA / ("%s_curves.npz" % stem)
    methods = ["v2_akde", "finalist", "kde_sj"]
    if not plot_only or not cache.exists():
        rows = pd.read_csv(RES / "stage_c" / "all_rows.csv", low_memory=False)
        R = int(rows[rows.N == N].rep.nunique())
        grid, table = {"R": np.array([R])}, []
        for t in DIAG:
            tgt = get_target(t)
            _, _, xg = truth_range(tgt)
            grid["%s_x" % t] = xg
            grid["%s_truth" % t] = tgt.pdf(xg)
            for m in methods:
                curves = []
                for rep in range(R):
                    f = load_fit(t, N, rep, m)
                    if f is not None:
                        curves.append(fpdf(f, xg))
                if not curves:
                    table.append({"target": t, "method": m, "status": "MISSING"})
                    continue
                C = np.vstack(curves)
                grid["%s_%s_mean" % (t, m)] = C.mean(axis=0)
                grid["%s_%s_p10" % (t, m)] = np.percentile(C, 10, axis=0)
                grid["%s_%s_p90" % (t, m)] = np.percentile(C, 90, axis=0)
                table.append({
                    "target": t, "method": m, "status": "ok", "n_curves": len(curves),
                    "mass_of_mean_curve": float(np.trapezoid(C.mean(axis=0), xg)),
                    "mass_of_pointwise_median_curve": float(
                        np.trapezoid(np.percentile(C, 50, axis=0), xg))})
        tab = pd.DataFrame(table)
        save_plot_data(FIG, stem, grid=grid, table=tab, meta={
            "bands": "pointwise 10th-90th percentile across independent datasets",
            "not": "NOT confidence bands and NOT simultaneous coverage"})
    else:
        grid = dict(np.load(cache))
        tab = pd.read_csv(DATA / ("%s_metrics.csv" % stem))
    R = int(grid["R"][0])

    apply_style()
    fig, axes = plt.subplots(4, 3, figsize=(7.4, 8.4))
    for r, t in enumerate(DIAG):
        xg = grid["%s_x" % t]
        ymax = float(np.nanmax(grid["%s_truth" % t]))
        for m in methods:
            k = "%s_%s_p90" % (t, m)
            if k in grid:
                ymax = max(ymax, float(np.nanmax(grid[k])))
        for c, m in enumerate(methods):
            ax = axes[r, c]
            if "%s_%s_mean" % (t, m) in grid:
                ax.fill_between(xg, grid["%s_%s_p10" % (t, m)],
                                grid["%s_%s_p90" % (t, m)], color=ST[m][1],
                                alpha=0.22, lw=0, zorder=2)
                line(ax, m, xg, grid["%s_%s_mean" % (t, m)])
            else:
                ax.text(0.5, 0.5, "MISSING", transform=ax.transAxes,
                        ha="center", va="center", color="crimson")
            line(ax, "truth", xg, grid["%s_truth" % t])
            ax.set_xlim(xg[0], xg[-1])
            ax.set_ylim(0, 1.08 * ymax)
            if r == 0:
                ax.set_title(ST[m][0], pad=4, fontsize=8)
            if c == 0:
                ax.set_ylabel("density")
            if r == 3:
                ax.set_xlabel("$x$")
        axes[r, 0].text(-0.27, 0.5, tname(t), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=7.5)
    h = [Line2D([], [], color="k", lw=2.2, label="Ground truth"),
         Line2D([], [], color="0.4", lw=6, alpha=0.25,
                label="pointwise 10th-90th percentile across %d datasets" % R),
         Line2D([], [], color="0.2", lw=1.6, label="mean fitted density")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, 1.004), fontsize=7)
    fig.tight_layout(rect=(0.03, 0, 1, 0.962))
    p = save_figure(fig, FIG, stem)
    plt.close(fig)
    mm = tab[tab.status == "ok"] if "status" in tab else tab
    CAPTIONS[stem] = (
        "Variability of the fitted density across %d independent confirmation "
        "datasets at N = %d. Each panel shows the arithmetic mean fitted density "
        "and the pointwise 10th-90th percentile band across replicates, against "
        "the ground truth. These are EMPIRICAL VARIABILITY BANDS describing the "
        "spread of the estimator across datasets: they are not confidence bands "
        "for the true density and carry no simultaneous coverage statement. The "
        "arithmetic mean of normalised densities is itself a density (measured "
        "mass %.4f to %.4f across panels); the pointwise median curve is not "
        "constrained to integrate to one (measured mass %.4f to %.4f) and is "
        "therefore not plotted." % (
            R, N,
            float(mm.mass_of_mean_curve.min()) if len(mm) else float("nan"),
            float(mm.mass_of_mean_curve.max()) if len(mm) else float("nan"),
            float(mm.mass_of_pointwise_median_curve.min()) if len(mm) else float("nan"),
            float(mm.mass_of_pointwise_median_curve.max()) if len(mm) else float("nan")))
    print("F2:", p["pdf"])


# ==================================================================== F3
def fig3(plot_only=False):
    stem = "fig3_accuracy_vs_N"
    bm = pd.read_csv(SUMMARY_ROOT / "benchmark_means.csv")
    fb = pd.read_csv(SUMMARY_ROOT / "fixed_benchmark.csv")
    pe = pd.read_csv(SUMMARY_ROOT / "paired_effects.csv")
    ps = pd.read_csv(SUMMARY_ROOT / "per_cell_summary.csv")
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])},
                   table=bm.merge(fb, on=["N", "method"], how="outer"),
                   meta={"source": "followup/results/statistics/*.csv"})

    apply_style()
    fig = plt.figure(figsize=(7.4, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.45], hspace=0.45,
                          wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for m in ORDER:
        s = bm[bm.method == m].sort_values("N")
        if s.empty:
            continue
        ax.plot(s.N, s.benchmark_mean_ise_equal_weight, color=ST[m][1],
                linestyle=ST[m][2], lw=1.5, marker="o", ms=3.4, label=ST[m][0])
        ax.fill_between(s.N, s.boot_ci_lo, s.boot_ci_hi, color=ST[m][1],
                        alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$")
    ax.set_ylabel("benchmark mean ISE\n(equal weight per target)")
    ax.set_title("Accuracy, 95% bootstrap CIs", fontsize=8)
    ax.legend(fontsize=5.8, frameon=False, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    for m in [x for x in ORDER if x != "v2_akde"]:
        s = fb[fb.method == m].sort_values("N")
        if s.empty:
            continue
        ax.plot(s.N, s.exp_T_geometric_mean_ratio, color=ST[m][1],
                linestyle=ST[m][2], lw=1.5, marker="o", ms=3.4, label=ST[m][0])
        ax.fill_between(s.N, s.boot_ci_lo, s.boot_ci_hi, color=ST[m][1],
                        alpha=0.15, lw=0)
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\exp(T)$ vs AKDE-V2")
    ax.set_title("Fixed-benchmark paired ratio", fontsize=8)
    ax.legend(fontsize=5.8, frameon=False)

    ax = fig.add_subplot(gs[1, :])
    sub = pe[pe.method == "finalist"].copy()
    sub["mw"] = sub.target.map(lambda s: MARRON_WAND_MAPPING[s][0])
    xs = np.arange(15)
    for k, N in enumerate(SIZES):
        s = sub[sub.N == N].sort_values("mw")
        off = (k - 1) * 0.24
        ax.errorbar(xs + off, s.geom_mean_ratio,
                    yerr=[s.geom_mean_ratio - s.boot_ci_lo_geom,
                          s.boot_ci_hi_geom - s.geom_mean_ratio],
                    fmt="o", ms=3.2, lw=0.9, capsize=1.6,
                    color=["#0072B2", "#E69F00", "#009E73"][k], label="N = %d" % N)
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["$f_{%d}$" % i for i in range(15)], fontsize=6.5)
    ax.set_ylabel("AKDE-M / AKDE-V2\npaired ISE ratio")
    ax.set_title("Per-target paired ratios, 95% paired bootstrap CIs "
                 "(below 1 favours AKDE-M); all 15 targets shown", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower right")

    ax = fig.add_subplot(gs[2, :])
    H = []
    for m in ORDER:
        row = []
        for N in SIZES:
            for t in ALL_TARGETS:
                r = ps[(ps.method == m) & (ps.N == N) & (ps.target == t)]
                row.append(np.log10(float(r.mean_ise.iloc[0])) if len(r) else np.nan)
        H.append(row)
    H = np.array(H)
    im = ax.imshow(H, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([ST[m][0] for m in ORDER], fontsize=7)
    ax.set_xticks([i * 15 + 7 for i in range(3)])
    ax.set_xticklabels(["N = %d" % N for N in SIZES], fontsize=7.5)
    for i in (15, 30):
        ax.axvline(i - 0.5, color="w", lw=1.1)
    ax.set_title(r"$\log_{10}$ mean ISE, every target ($f_0$ to $f_{14}$ "
                 "left to right within each block)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label(
        r"$\log_{10}$ mean ISE", fontsize=7)
    p = save_figure(fig, FIG, stem)
    plt.close(fig)
    CAPTIONS[stem] = (
        "Accuracy over the locked confirmation (15 targets, N in {200, 1000, "
        "5000}, R = 30 independent datasets per cell). Top left: benchmark mean "
        "ISE giving each target equal weight, with 95% percentile bootstrap CIs "
        "obtained by resampling replicate IDs within each target-N cell "
        "(B = 10,000), carrying all methods together so pairing is preserved; "
        "the target index is never resampled because the 15 densities are a "
        "fixed benchmark. Top right: the fixed-benchmark paired effect exp(T), "
        "where T is the equal-weight mean over targets of the mean paired log "
        "ISE ratio against AKDE-V2, with the same bootstrap scheme. Middle: "
        "per-target paired ratios AKDE-M / AKDE-V2 with paired bootstrap CIs; "
        "all 15 targets are shown at all three sample sizes and none is omitted. "
        "Bottom: log10 mean ISE for every method and every target. Any apparent "
        "trend in N describes the observed range only; three sample sizes cannot "
        "support a convergence-rate claim.")
    print("F3:", p["pdf"])


# ==================================================================== F4
def fig4(plot_only=False):
    stem = "fig4_mechanism"
    lad = pd.read_csv(DIAG_ROOT / "oracle_ladder_check.csv")
    amb = pd.read_csv(DIAG_ROOT / "ambiguity_cone_witnesses.csv")
    solp = DIAG_ROOT / "solver_verification.csv"
    sol = pd.read_csv(solp) if solp.exists() else None
    fcp = PY / "followup" / "figures" / "data" / "figureC_metrics.csv"
    fc = pd.read_csv(fcp) if fcp.exists() else None
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])}, table=lad,
                   meta={"source": "oracle ladder + ambiguity cone + solver checks"})

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.2))

    ax = axes[0, 0]
    w = 0.36
    cells = lad[lad.bank == "base"][["target", "N"]].drop_duplicates()
    xs = np.arange(len(cells))
    for bi, bank in enumerate(("base", "matched")):
        dictb, supb, exb = [], [], []
        for _, cc in cells.iterrows():
            r = lad[(lad.bank == bank) & (lad.target == cc.target) &
                    (lad.N == cc.N)].iloc[0]
            dictb.append(max(r.R_full_star, 1e-18))
            exb.append(max(r.abs_excess_y0_C0, 1e-18))
            supb.append(max(r.sampling_gap_abs, 1e-18))
        off = (bi - 0.5) * w
        ax.bar(xs + off, dictb, w, color="0.75",
               label=r"dictionary $R^*_{\rm full}$" if bi == 0 else None)
        ax.bar(xs + off, exb, w, bottom=dictb, color="#CC5500",
               label="criterion excess (population moments)" if bi == 0 else None)
        ax.bar(xs + off, supb, w, bottom=np.array(dictb) + np.array(exb),
               color="#0072B2", label="sampling gap" if bi == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["$f_{%s}$\n%d" % (cc.target[1:], cc.N)
                        for _, cc in cells.iterrows()], fontsize=5.8)
    ax.set_ylabel("exact ISE (log scale)")
    ax.set_title("Oracle ladder: V2 bank (left) vs matched (right)", fontsize=7.5)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left")

    ax = axes[0, 1]
    if fc is not None:
        for meth, mk, col, lab in (("v2_akde", "o", "#E69F00", "AKDE-V2"),
                                   ("finalist", "^", "#0072B2", "AKDE-M")):
            s = fc[fc.method == meth]
            ax.scatter(s.max_abs_std_emp_residual, s.max_abs_density_error, s=32,
                       marker=mk, facecolor="none", edgecolor=col, label=lab)
            for r in s.itertuples():
                ax.annotate("$f_{%s}$,%d" % (r.target[1:], r.N),
                            (r.max_abs_std_emp_residual, r.max_abs_density_error),
                            fontsize=5, xytext=(3, 2), textcoords="offset points")
        ax.axvline(2.0, color="0.6", lw=0.7, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel(r"max $|(Du-y)_j / \widehat{\rm sd}_j|$")
        ax.set_ylabel(r"max $|\hat f(x)-f_0(x)|$")
        ax.set_title("Small measured discrepancy, large shape error", fontsize=7.5)
        ax.legend(fontsize=6.5, frameon=False)
    else:
        ax.text(0.5, 0.5, "residual metrics unavailable", ha="center",
                va="center", transform=ax.transAxes, color="crimson")

    ax = axes[1, 0]
    for bank, col, lab in (("base", "#E69F00", "V2 bank"),
                           ("matched", "#0072B2", "matched bank")):
        s = amb[amb.bank == bank]
        g = s.groupby("moment_tol").abs_delta_ise.max()
        ax.plot(g.index, g.values, marker="o", ms=4, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("relative weighted-moment budget")
    ax.set_ylabel(r"largest witnessed $|\Delta\,$ISE$|$")
    ax.set_title("Ambiguity witnesses in the feasible cone (lower bounds)",
                 fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1, 1]
    if sol is not None and len(sol):
        s = sol[sol.stage1_solver == "cd"]
        for b, col in zip(sorted(s.stage1_budget.unique()),
                          ["#999999", "#0072B2", "#CC5500"]):
            v = s[s.stage1_budget == b].stage1_kkt_rel.dropna()
            v = v[v > 0]
            if len(v):
                ax.hist(np.log10(v), bins=18, histtype="step", color=col, lw=1.3,
                        label="CD budget %d" % b)
        v = sol[sol.stage1_solver == "fista"].stage1_kkt_rel.dropna()
        v = v[v > 0]
        if len(v):
            ax.hist(np.log10(v), bins=18, histtype="step", color="#009E73",
                    lw=1.3, ls="--", label="FISTA")
        ax.axvline(-8, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"$\log_{10}$ relative stage-1 KKT residual")
        ax.set_ylabel("instances")
        ax.set_title(r"Stage-1 optimisation accuracy (dotted: $10^{-8}$ target)",
                     fontsize=7.5)
        ax.legend(fontsize=6, frameon=False)
    else:
        ax.text(0.5, 0.5, "solver verification not available", ha="center",
                va="center", transform=ax.transAxes, color="crimson")
    fig.tight_layout()
    p = save_figure(fig, FIG, stem)
    plt.close(fig)
    CAPTIONS[stem] = (
        "Mechanism diagnostics, all from the frozen 16-dataset development "
        "subset. (a) Oracle ladder: the exact risk is split into the dictionary "
        "optimum, the excess incurred by the moment criterion when sampling "
        "error is removed by substituting population moments and the population "
        "covariance, and the further gap on moving to empirical quantities. Left "
        "bar of each pair is the V2 measurement bank, right bar the matched "
        "bank; log scale. The population control removes sampling error IN THE "
        "MOMENTS conditional on a dictionary that is still built from the "
        "sample; it does not remove approximation, constraint, weighting or "
        "numerical effects, so the middle band is not criterion bias alone. "
        "(b) Measured moment discrepancy against fitted density error: AKDE-V2 "
        "satisfies its own moment conditions to within their sampling noise "
        "(dotted line at two standard errors) while leaving a large shape error. "
        "(c) Largest ISE change achieved by a feasible perturbation of the "
        "exact-risk optimum within a stated relative weighted-moment budget, "
        "searching the full feasible tangent cone {v : 1'v = 0, v_i >= 0 where "
        "w_i = 0}. These are WITNESSES, i.e. lower bounds; no global upper bound "
        "on ambiguity is claimed. (d) Stage-1 optimisation accuracy: relative "
        "KKT residual for coordinate descent at three iteration budgets and for "
        "an independent proximal-gradient solver. Neither reliably reaches the "
        "1e-8 target.")
    print("F4:", p["pdf"])


# ==================================================================== F5
def fig5(plot_only=False):
    stem = "fig5_cost_and_compactness"
    cost = pd.read_csv(DIAG_ROOT / "model_size_and_query_cost.csv")
    cost = cost[cost.status == "ok"]
    ps = pd.read_csv(SUMMARY_ROOT / "per_cell_summary.csv")
    save_plot_data(FIG, stem, grid={"_": np.array([0.0])}, table=cost,
                   meta={"source": "saved Stage-C models; timing repeats are not "
                                   "statistical replications"})

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.8))

    ax = axes[0, 0]
    for m in ORDER:
        s = ps[ps.method == m]
        if s.empty:
            continue
        ax.scatter(s.mean_support, s.mean_ise, s=13, color=ST[m][1], alpha=0.75,
                   edgecolors="none", label=ST[m][0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean number of mixture components")
    ax.set_ylabel("mean ISE")
    ax.set_title("Accuracy vs representation size", fontsize=8)
    ax.legend(fontsize=5.6, frameon=False, ncol=2)

    ax = axes[0, 1]
    for m in ORDER:
        s = ps[ps.method == m]
        if s.empty:
            continue
        ax.scatter(s.mean_wall_time_s, s.mean_ise, s=13, color=ST[m][1],
                   alpha=0.75, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean cold end-to-end fit time (s)")
    ax.set_ylabel("mean ISE")
    ax.set_title("Accuracy vs fitting cost", fontsize=8)

    ax = axes[1, 0]
    for m in ORDER:
        s = cost[cost.method == m].groupby("N")["query_us_per_query_median"].median()
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=ST[m][1], linestyle=ST[m][2], lw=1.5,
                marker="o", ms=3.4, label=ST[m][0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\mu$s per density evaluation")
    ax.set_title("Query cost (identical queries, 7 timing repeats)", fontsize=8)
    ax.legend(fontsize=5.6, frameon=False, ncol=2)

    ax = axes[1, 1]
    for m in ORDER:
        s = cost[cost.method == m].groupby("N")["canonical_model_bytes"].median()
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=ST[m][1], linestyle=ST[m][2], lw=1.5,
                marker="o", ms=3.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("$N$")
    ax.set_ylabel("serialized model size (bytes)")
    ax.set_title("Model size (float64 parameters)", fontsize=8)
    fig.tight_layout()
    p = save_figure(fig, FIG, stem)
    plt.close(fig)
    CAPTIONS[stem] = (
        "Cost and compactness. (a) mean ISE against the mean number of mixture "
        "components and (b) against the mean cold end-to-end fit time, one point "
        "per (target, N) cell. (c) density-evaluation latency, measured on the "
        "SAVED fitted models with identical query points (10,000 per model, "
        "batch 2,000), identical warmup and 7 timing repeats per model, in the "
        "same process on the same hardware with BLAS pinned to one thread; the "
        "reported value is the median over repeats and then over models. Timing "
        "repeats reduce timer noise for a fixed model and are NOT independent "
        "statistical replications. (d) serialized model size counting float64 "
        "parameters (one weight, one mean and one variance per component in one "
        "dimension). Fitting cost and query cost are reported separately because "
        "they trade off in opposite directions: the kernel estimators fit almost "
        "instantly but carry the whole sample at query time, whereas the "
        "dictionary methods pay a fitting cost once and then evaluate in time "
        "independent of N.")
    print("F5:", p["pdf"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--figure", default="all")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    todo = {"1": fig1, "2": lambda po: fig2(1000, po), "3": fig3,
            "4": fig4, "5": fig5}
    keys = list(todo) if a.figure == "all" else [a.figure]
    failed = []
    for k in keys:
        try:
            todo[k](a.plot_only)
        except Exception as exc:
            failed.append((k, "%s: %s" % (type(exc).__name__, exc)))
            print("  figure %s NOT built: %s: %s" % (k, type(exc).__name__, exc))
    if CAPTIONS:
        cap = FIG / "captions.md"
        prev = {}
        if cap.exists():
            for ch in cap.read_text(encoding="utf-8").split("\n## ")[1:]:
                kk = ch.split("\n", 1)[0].strip()
                prev[kk] = ch.split("\n", 1)[1].strip()
        prev.update(CAPTIONS)
        cap.write_text("# Figure captions\n" + "".join(
            "\n## %s\n\n%s\n" % (k, prev[k]) for k in sorted(prev)), encoding="utf-8")
        print("captions ->", cap)
    if failed:
        print("\nNOT BUILT:", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
