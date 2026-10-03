"""Build the follow-up figures from recorded fits.  No estimator is refitted.

    python followup/scripts/make_figures.py --figure A --which dev
    python followup/scripts/make_figures.py --figure all
    python followup/scripts/make_figures.py --plot-only          # redraw from cache

Every figure writes PDF + SVG + 300 dpi PNG, plus the plotted grids, truth
values, fitted curves and metrics as NPZ/CSV under ``figures/data/``.
``--plot-only`` redraws from that cache and touches no fit.

Truth is used for evaluation and for drawing the reference curve only; it
never entered any fit.  Replicate selection is frozen: the single-fit panels
always use rep 0, the first prespecified seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402

from akde.densities import MARRON_WAND_MAPPING, get_target   # noqa: E402
from followup.akde_followup import figdata                   # noqa: E402
from followup.akde_followup.plotting import (                # noqa: E402
    STYLE, apply_style, curve, save_figure, save_plot_data,
    tail_mass_outside, truth_range)

RES = ROOT / "followup" / "results"
FIG = ROOT / "followup" / "figures"
DIAG_TARGETS = ["f0", "f2", "f5", "f9"]
CAPTIONS: dict[str, str] = {}


def tlabel(t: str) -> str:
    num, name = MARRON_WAND_MAPPING[t]
    return "$f_{%s}$ (MW%d, %s)" % (t[1:], num, name)


def _finalist_name() -> str:
    p = RES / "finalist.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))["variant"]
    return "B1_moment_wide_indep_m300"


def _finalist_stage() -> str:
    p = RES / "finalist.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))["stage"]
    return "stage_b"


# ===================================================================== FIG A
def figure_A(which="dev", plot_only=False):
    finalist, fstage = _finalist_name(), _finalist_stage()
    stem = "figureA_%s" % which
    cache = FIG / "data" / ("%s_curves.npz" % stem)
    metrics_csv = FIG / "data" / ("%s_metrics.csv" % stem)

    if which == "dev":
        sizes, stage_v2, stage_fin, ns = (500, 5000), "stage_a", fstage, "development"
        v2key, refstage = "A1_v2_moment_m300", "dev_references"
        subtitle = ("Diagnostic / exploratory: development replicate 0, "
                    "Stage-A/B development study")
    else:
        sizes, stage_v2, stage_fin, ns = (200, 5000), "stage_c", "stage_c", "confirmation"
        v2key, refstage = "v2_akde", "stage_c"
        # Stage-C fits are stored under the comparator key, not the
        # development variant name.
        finalist = "finalist"
        subtitle = "Confirmation counterpart: Stage-C locked run, replicate 0"

    if not plot_only or not cache.exists():
        grid, table = {}, []
        for t in DIAG_TARGETS:
            tgt = get_target(t)
            x_lo, x_hi, xg = truth_range(tgt)
            grid["%s_x" % t] = xg
            grid["%s_truth" % t] = tgt.pdf(xg)
            for N in sizes:
                X = figdata.dataset(ns, t, N, 0)
                grid["%s_N%d_rug" % (t, N)] = X[:, 0]
                for key, stage, var in (("v2_akde", stage_v2, v2key),
                                        ("finalist", stage_fin, finalist),
                                        ("spades_simplex", refstage, "spades_simplex"),
                                        ("kde_sj", refstage, "kde_sj")):
                    f = figdata.load_fit(stage, t, N, 0, var)
                    if f is None:
                        table.append({"target": t, "N": N, "method": key,
                                      "variant": var, "status": "MISSING"})
                        continue
                    y = figdata.fit_pdf(f, xg)
                    grid["%s_N%d_%s" % (t, N, key)] = y
                    table.append({
                        "target": t, "mw": MARRON_WAND_MAPPING[t][0],
                        "mw_name": MARRON_WAND_MAPPING[t][1], "N": N, "rep": 0,
                        "method": key, "variant": var, "status": "ok",
                        "ise": f["ise"], "n_support": int(f["weights"].size),
                        "total_mass": float(f["weights"].sum()),
                        "mass_outside_plot_range": tail_mass_outside(
                            lambda z, f=f: figdata.fit_pdf(f, np.atleast_1d(z)),
                            x_lo, x_hi),
                        "x_lo": x_lo, "x_hi": x_hi, "fit_file": f["path"]})
        tab = pd.DataFrame(table)
        save_plot_data(FIG, stem, grid=grid, table=tab, meta={
            "figure": "A", "which": which, "sizes": list(sizes),
            "replicate_rule": "rep 0, the first prespecified seed",
            "plot_range_rule": "truth quantiles [0.0005, 0.9995]",
            "truth_use": "evaluation and visualisation only"})
    else:
        grid = dict(np.load(cache))
        tab = pd.read_csv(metrics_csv)

    apply_style()
    fig, axes = plt.subplots(4, 2, figsize=(7.3, 9.4))
    for r, t in enumerate(DIAG_TARGETS):
        xg = grid["%s_x" % t]
        ymax = 0.0
        for c, N in enumerate(sizes):
            for k in ("truth", "v2_akde", "finalist", "spades_simplex", "kde_sj"):
                nm = "%s_truth" % t if k == "truth" else "%s_N%d_%s" % (t, N, k)
                if nm in grid:
                    ymax = max(ymax, float(np.nanmax(grid[nm])))
        for c, N in enumerate(sizes):
            ax = axes[r, c]
            rug = grid.get("%s_N%d_rug" % (t, N))
            if rug is not None and rug.size <= 1200:
                ax.plot(rug, np.full(rug.size, -0.018 * ymax), "|",
                        color="0.35", ms=3, mew=0.5, alpha=0.5, zorder=1)
            curve(ax, "truth", xg, grid["%s_truth" % t])
            missing = []
            for k in ("kde_sj", "spades_simplex", "v2_akde", "finalist"):
                nm = "%s_N%d_%s" % (t, N, k)
                if nm in grid:
                    curve(ax, k, xg, grid[nm])
                else:
                    missing.append(STYLE[k][0])
            sub = tab[(tab.target == t) & (tab.N == N)] if len(tab) else tab
            ann = []
            for k, short in (("v2_akde", "V2"), ("finalist", "Fin")):
                row = sub[sub.method == k]
                if len(row) and row.iloc[0].get("status") == "ok":
                    ann.append("%s ISE %.3g, |S|=%d"
                               % (short, row.iloc[0]["ise"], int(row.iloc[0]["n_support"])))
            if missing:
                ann.append("MISSING: " + ", ".join(missing))
            ax.text(0.02, 0.97, "\n".join(ann), transform=ax.transAxes,
                    va="top", ha="left", fontsize=6.4, linespacing=1.25,
                    bbox=dict(fc="white", ec="0.8", lw=0.4, alpha=0.85, pad=1.6))
            ax.set_ylim(-0.045 * ymax, 1.06 * ymax)
            ax.set_xlim(xg[0], xg[-1])
            if r == 0:
                ax.set_title("$N = %d$" % N, pad=4)
            if c == 0:
                ax.set_ylabel("density")
            if r == 3:
                ax.set_xlabel("$x$ (original data units)")
        axes[r, 0].text(-0.20, 0.5, tlabel(t), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8.5)

    handles = [Line2D([], [], color=STYLE[k][1], linestyle=STYLE[k][2],
                      linewidth=STYLE[k][3], label=STYLE[k][0])
               for k in ("truth", "v2_akde", "finalist", "spades_simplex", "kde_sj")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, 1.005))
    fig.suptitle(subtitle, y=0.955, fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.02, 0, 1, 0.925))
    paths = save_figure(fig, FIG, stem)
    plt.close(fig)
    CAPTIONS[stem] = _caption_A(which, sizes, tab)
    print("Figure A (%s):" % which, paths["pdf"])
    return paths


def _caption_A(which, sizes, tab):
    fin = _finalist_name()
    n_missing = int((tab.get("status", pd.Series(dtype=str)) == "MISSING").sum()) \
        if len(tab) else 0
    max_out = (float(tab["mass_outside_plot_range"].max())
               if "mass_outside_plot_range" in tab else float("nan"))
    return (
        "Figure A%s. Fitted densities on a single dataset per panel, in original "
        "data units. Rows: the four diagnostic targets f0 (MW1 Gaussian), f2 "
        "(MW3 strongly skewed), f5 (MW6 bimodal), f9 (MW10 claw). Columns: "
        "N = %d and N = %d. All five curves in a panel are fitted to the "
        "identical observations, shown as a faint rug. Methods: ground truth "
        "(black solid); original V2 AKDE (orange dashed); the frozen finalist "
        "%s (blue solid); adapted SPADES, i.e. density-valid non-negative "
        "selection plus simplex refit (purple dash-dot); Sheather-Jones KDE "
        "(green dotted). Selection rule: hyperparameters chosen by 3-fold "
        "cross-validation of the held-out integrated-square score on training "
        "data only; the dictionary is pinned at 60x5 = 300 atoms for every "
        "dictionary-based method. Replicate selection was frozen before any "
        "outcome was inspected: rep 0, the first prespecified seed, not the "
        "best-looking replicate. %s"
        "The plotting window is the predeclared truth quantile range "
        "[0.0005, 0.9995]; it was never used in training. The largest fitted "
        "mass falling outside that window, over all curves and panels, is "
        "%.4f, so no curve is materially truncated and no tail panel is "
        "needed; the per-curve values are in the companion CSV. No y-limit cap "
        "is applied: where the original AKDE spikes far above the truth, the "
        "spike is shown at full height. Where a reference curve is not visible "
        "it coincides with the finalist, not because it is missing. "
        "Annotations "
        "give the ISE and active support size of the original and finalist "
        "fits; complete metrics are in figures/data/%s_metrics.csv. Truth was "
        "used only for evaluation and for drawing the reference curve. "
        "%s" % (
            " (development)" if which == "dev" else " (confirmation counterpart)",
            sizes[0], sizes[1], fin,
            "%d required fit(s) are missing and are labelled in the affected "
            "panels rather than replaced by a different seed. " % n_missing
            if n_missing else "",
            max_out,
            "figureA_%s" % which,
            "THIS FIGURE IS DIAGNOSTIC / EXPLORATORY: it comes from the "
            "5-replicate development study and carries no confirmatory weight."
            if which == "dev" else
            "Drawn from the locked Stage-C confirmation run."))


# ===================================================================== FIG B
def figure_B(N=1000, plot_only=False, reference="kde_sj"):
    stem = "figureB_N%d" % N
    cache = FIG / "data" / ("%s_curves.npz" % stem)
    # Stage-C fits are stored under the comparator key, not the development
    # variant name, so the on-disk name for the finalist is "finalist".
    methods = [("v2_akde", "v2_akde"), ("finalist", "finalist"),
               (reference, reference)]

    if not plot_only or not cache.exists():
        rows = pd.read_csv(RES / "stage_c" / "all_rows.csv", low_memory=False)
        R = int(rows[rows.N == N].rep.nunique())
        grid, table = {"R": np.array([R])}, []
        for t in DIAG_TARGETS:
            tgt = get_target(t)
            x_lo, x_hi, xg = truth_range(tgt)
            grid["%s_x" % t] = xg
            grid["%s_truth" % t] = tgt.pdf(xg)
            for key, var in methods:
                curves, used = [], []
                for rep in range(R):
                    f = figdata.load_fit("stage_c", t, N, rep, var)
                    if f is None:
                        continue
                    curves.append(figdata.fit_pdf(f, xg))
                    used.append(rep)
                if not curves:
                    table.append({"target": t, "method": key, "status": "MISSING"})
                    continue
                Cv = np.vstack(curves)
                grid["%s_%s_mean" % (t, key)] = Cv.mean(axis=0)
                grid["%s_%s_p10" % (t, key)] = np.percentile(Cv, 10, axis=0)
                grid["%s_%s_p90" % (t, key)] = np.percentile(Cv, 90, axis=0)
                table.append({"target": t, "method": key, "variant": var,
                              "status": "ok", "n_curves": len(curves),
                              "reps_used": ";".join(map(str, used)),
                              "mass_of_mean_curve": float(np.trapezoid(
                                  Cv.mean(axis=0), xg))})
        tab = pd.DataFrame(table)
        save_plot_data(FIG, stem, grid=grid, table=tab, meta={
            "figure": "B", "N": N, "R": R,
            "bands": "pointwise 10th-90th percentile over independent datasets",
            "not": "these are NOT confidence bands for the true density"})
    else:
        grid = dict(np.load(cache))
        tab = pd.read_csv(FIG / "data" / ("%s_metrics.csv" % stem))
    R = int(grid["R"][0])

    apply_style()
    fig, axes = plt.subplots(4, 3, figsize=(7.5, 8.6), sharex="row")
    for r, t in enumerate(DIAG_TARGETS):
        xg = grid["%s_x" % t]
        ymax = float(np.nanmax(grid["%s_truth" % t]))
        for key, _ in methods:
            k = "%s_%s_p90" % (t, key)
            if k in grid:
                ymax = max(ymax, float(np.nanmax(grid[k])))
        for c, (key, _) in enumerate(methods):
            ax = axes[r, c]
            if "%s_%s_mean" % (t, key) in grid:
                ax.fill_between(xg, grid["%s_%s_p10" % (t, key)],
                                grid["%s_%s_p90" % (t, key)],
                                color=STYLE[key][1], alpha=0.22, lw=0, zorder=2)
                curve(ax, key, xg, grid["%s_%s_mean" % (t, key)])
            else:
                ax.text(0.5, 0.5, "MISSING", transform=ax.transAxes,
                        ha="center", va="center", color="crimson")
            curve(ax, "truth", xg, grid["%s_truth" % t])
            ax.set_ylim(0, 1.08 * ymax)
            ax.set_xlim(xg[0], xg[-1])
            if r == 0:
                ax.set_title(STYLE[key][0], pad=4, fontsize=8)
            if c == 0:
                ax.set_ylabel("density")
            if r == 3:
                ax.set_xlabel("$x$")
        axes[r, 0].text(-0.28, 0.5, tlabel(t), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8)
    handles = [Line2D([], [], color=STYLE["truth"][1], linestyle="-", lw=2.2,
                      label="Ground truth"),
               Line2D([], [], color="0.4", lw=6, alpha=0.25,
                      label="pointwise 10th-90th percentile band over %d datasets" % R),
               Line2D([], [], color="0.2", lw=1.6, label="arithmetic mean fitted density")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, 1.004), fontsize=7.5)
    fig.tight_layout(rect=(0.03, 0, 1, 0.955))
    paths = save_figure(fig, FIG, stem)
    plt.close(fig)
    CAPTIONS[stem] = (
        "Figure B. Variability of the fitted density across %d independent "
        "confirmation datasets at N = %d, for the four diagnostic targets. "
        "Each panel shows the arithmetic mean fitted density (coloured line) "
        "and the pointwise 10th-90th percentile band across replicates "
        "(shaded), against the ground truth (black). These are EMPIRICAL "
        "VARIABILITY BANDS describing the spread of the estimator across "
        "datasets; they are NOT confidence bands for the true density, and at "
        "R = %d a 10th-90th percentile band is resolved by only the 2nd and "
        "9th order statistics, so its edges are themselves noisy. The mean of "
        "normalised densities is itself a density; the ISE of this mean curve "
        "is a different quantity from the mean ISE reported in Table form and "
        "the two must not be interchanged. Columns: original V2 AKDE, the "
        "frozen finalist, and one conventional reference. Selection rule and "
        "dictionary as in Figure A. Truth used only for evaluation and "
        "display." % (R, N, R))
    print("Figure B:", paths["pdf"])
    return paths


# ===================================================================== FIG C
def figure_C(plot_only=False, stage="stage_a", N_list=(500, 5000)):
    from followup.akde_followup.variants import BANK_MATCHED, BASE_BANK
    finalist, fstage = _finalist_name(), _finalist_stage()
    stem = "figureC"
    targets = ["f5", "f9"]
    cache = FIG / "data" / ("%s_curves.npz" % stem)

    if not plot_only or not cache.exists():
        grid, table = {}, []
        for t in targets:
            tgt = get_target(t)
            x_lo, x_hi, xg = truth_range(tgt)
            grid["%s_x" % t] = xg
            grid["%s_truth" % t] = tgt.pdf(xg)
            for N in N_list:
                for key, st, var, spec in (
                        ("v2_akde", stage, "A1_v2_moment_m300", BASE_BANK),
                        ("finalist", fstage, finalist, BANK_MATCHED)):
                    f = figdata.load_fit(st, t, N, 0, var)
                    if f is None:
                        table.append({"target": t, "N": N, "method": key,
                                      "status": "MISSING"})
                        continue
                    grid["%s_N%d_%s_err" % (t, N, key)] = \
                        figdata.fit_pdf(f, xg) - tgt.pdf(xg)
                    rp = figdata.residual_panel(st, t, N, 0, var, spec)
                    for nm in ("centres", "scales", "emp_residual",
                               "pop_residual", "std_emp_residual",
                               "sampling_sd", "std_usable"):
                        grid["%s_N%d_%s_%s" % (t, N, key, nm)] = rp[nm]
                    table.append({"target": t, "N": N, "method": key, "variant": var,
                                  "status": "ok", "ise": f["ise"], "m": rp["m"],
                                  "n_support": rp["n_support"],
                                  "n_probes_unresolved": rp["n_probes_unresolved"],
                                  "sd_floor": rp["sd_floor"],
                                  "max_abs_std_emp_residual":
                                      float(np.nanmax(np.abs(rp["std_emp_residual"]))),
                                  "rms_emp_residual":
                                      float(np.sqrt(np.mean(rp["emp_residual"] ** 2))),
                                  "rms_pop_residual":
                                      float(np.sqrt(np.mean(rp["pop_residual"] ** 2))),
                                  "max_abs_density_error":
                                      float(np.max(np.abs(
                                          grid["%s_N%d_%s_err" % (t, N, key)])))})
        # measurement-count sweep, per target, from the V2 archive
        sweep = pd.read_csv(ROOT / "results" / "moment_count" / "raw.csv")
        for t in targets:
            s = sweep[sweep.target == t]
            for N in sorted(s.N.unique()):
                g = s[s.N == N].groupby("m").ise.median()
                grid["sweep_%s_N%d_m" % (t, N)] = g.index.to_numpy(float)
                grid["sweep_%s_N%d_ise" % (t, N)] = g.to_numpy(float)
        tab = pd.DataFrame(table)
        save_plot_data(FIG, stem, grid=grid, table=tab, meta={
            "figure": "C", "targets": targets, "N": list(N_list),
            "sweep_source": str(ROOT / "results" / "moment_count" / "raw.csv"),
            "sweep_note": "V2 archive, per target (NOT pooled)"})
    else:
        grid = dict(np.load(cache))
        tab = pd.read_csv(FIG / "data" / ("%s_metrics.csv" % stem))

    apply_style()
    fig, axes = plt.subplots(4, 2, figsize=(7.3, 9.0))
    for c, t in enumerate(targets):
        xg = grid["%s_x" % t]
        ax = axes[0, c]
        for key in ("v2_akde", "finalist"):
            for N, alpha in zip(N_list, (0.45, 1.0)):
                nm = "%s_N%d_%s_err" % (t, N, key)
                if nm in grid:
                    ax.plot(xg, grid[nm], color=STYLE[key][1],
                            linestyle=STYLE[key][2], lw=STYLE[key][3],
                            alpha=alpha, label="%s, N=%d" % (STYLE[key][0], N))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("%s — fitted-density error" % tlabel(t), fontsize=8)
        ax.set_ylabel(r"$\hat f(x) - f_0(x)$")
        ax.set_xlim(xg[0], xg[-1])
        if c == 1:
            ax.legend(fontsize=6, frameon=False, loc="upper right")

        ax = axes[1, c]
        notes = []
        for key, mk in (("v2_akde", "o"), ("finalist", "^")):
            nm = "%s_N%d_%s_std_emp_residual" % (t, N_list[-1], key)
            if nm not in grid:
                continue
            cen = grid["%s_N%d_%s_centres" % (t, N_list[-1], key)]
            sc = grid["%s_N%d_%s_scales" % (t, N_list[-1], key)]
            z = grid[nm]
            good = np.isfinite(z)
            n_bad = int((~good).sum())
            if n_bad:
                notes.append("%s: %d of %d probes dropped ($n_{\\mathrm{eff}}<5$)"
                             % (STYLE[key][0], n_bad, z.size))
            uniq = np.unique(np.round(sc, 10))
            for si, sv in enumerate(uniq):
                sel = np.isclose(sc, sv) & good
                if not sel.any():
                    continue
                ax.scatter(cen[sel], z[sel], s=7, marker=mk, facecolor="none",
                           edgecolor=STYLE[key][1],
                           alpha=0.30 + 0.70 * si / max(len(uniq) - 1, 1),
                           linewidths=0.6,
                           label=("%s (%d probe widths; darker = wider)"
                                  % (STYLE[key][0], len(uniq)))
                           if (si == 0 and c == 1) else None)
        ax.axhline(0, color="k", lw=0.6)
        for lv in (-2, 2):
            ax.axhline(lv, color="0.6", lw=0.5, ls=":")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.SymmetricalLogLocator(base=10.0, linthresh=1.0,
                                                    subs=[1.0]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title("standardised training residual, N=%d" % N_list[-1], fontsize=8)
        ax.set_ylabel(r"$(Du-y)_j\,/\,\widehat{\mathrm{sd}}_j$" "\n"
                      r"symlog, dotted at $\pm2$")
        if notes:
            ax.text(0.5, -0.34, " · ".join(notes), transform=ax.transAxes,
                    fontsize=5.4, va="top", ha="center", color="0.3")
        if c == 1:
            ax.legend(fontsize=5.6, frameon=False, ncol=1, loc="upper left",
                      bbox_to_anchor=(0.0, 1.02))

        ax = axes[2, c]
        for key, mk in (("v2_akde", "o"), ("finalist", "^")):
            nm = "%s_N%d_%s_pop_residual" % (t, N_list[-1], key)
            if nm not in grid:
                continue
            cen = grid["%s_N%d_%s_centres" % (t, N_list[-1], key)]
            ax.scatter(cen, grid[nm], s=7, marker=mk, facecolor="none",
                       edgecolor=STYLE[key][1], linewidths=0.6, alpha=0.6,
                       label="%s: $Du-y_0$" % STYLE[key][0])
            nm2 = "%s_N%d_%s_emp_residual" % (t, N_list[-1], key)
            ax.scatter(cen, grid[nm2], s=5, marker="x",
                       color=STYLE[key][1], alpha=0.35, linewidths=0.5,
                       label="%s: $Du-y$" % STYLE[key][0])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("population vs training measurement residual, N=%d" % N_list[-1],
                     fontsize=8)
        ax.set_ylabel("residual")
        ax.set_xlabel("measurement centre")
        if c == 1:
            ax.legend(fontsize=5.6, frameon=False, loc="upper right")

        ax = axes[3, c]
        for N, mk in zip((500, 5000), ("o", "s")):
            km = "sweep_%s_N%d_m" % (t, N)
            if km in grid:
                ax.plot(grid[km], grid["sweep_%s_N%d_ise" % (t, N)], marker=mk,
                        ms=4, lw=1.2, label="N = %d" % N)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("measurement count $m$")
        ax.set_ylabel("median ISE")
        ax.set_title("measurement-count sweep, per target (V2 archive)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, FIG, stem)
    plt.close(fig)
    CAPTIONS[stem] = (
        "Figure C. Why the moment fit misses. Columns: f5 (MW6 bimodal) and f9 "
        "(MW10 claw). Row 1: fitted-density error f_hat(x) - f0(x) for the "
        "original V2 AKDE and the frozen finalist, at N = %s (faint) and "
        "N = %s (solid), on development replicate 0. Row 2: standardised "
        "training measurement residuals (Du - y)_j divided by the estimated "
        "sampling standard deviation of y_j, plotted against measurement "
        "centre and separated by probe width (darker = wider), on a symlog "
        "scale with dotted lines at +/- 2. A narrow probe centred where the "
        "sample has no points has an estimated sampling variance that "
        "underflows to numerically zero, so its standardised residual is 0/0; "
        "those probes are DROPPED and counted in the panel rather than "
        "clipped, and the counts are also in the companion CSV "
        "(n_probes_unresolved). A probe is kept only when its effective "
        "sample count, Kish's (sum_n k_j(x_n))^2 / sum_n k_j(x_n)^2, is at "
        "least 5. Row 3: "
        "the same fits' population residual Du - y0 (open markers, y0 computed "
        "in closed form from the true density — an oracle quantity used for "
        "diagnosis only) against the training residual Du - y (crosses). "
        "Rows 2-3 together show whether a measured discrepancy that is small "
        "on its own sampling scale coexists with a large density error in "
        "row 1. Row 4: the measurement-count sweep from the V2 archive, shown "
        "PER TARGET rather than pooled over {f5, f9} as the manuscript's "
        "Table 6 does; 10 replicates per point, median ISE. Truth was used "
        "only for evaluation and for the oracle residual."
        % (N_list[0], N_list[-1]))
    print("Figure C:", paths["pdf"])
    return paths


# ===================================================================== FIG D
def figure_D(plot_only=False):
    finalist = _finalist_name()
    stem = "figureD"
    cache = FIG / "data" / ("%s_metrics.csv" % stem)
    if not plot_only or not cache.exists():
        rows = pd.read_csv(RES / "stage_c" / "all_rows.csv", low_memory=False)
        rows["error"] = rows["error"].fillna("").astype(str)
        rows = rows[rows["error"] == ""]
        save_plot_data(FIG, stem, grid={"dummy": np.array([0.0])}, table=rows,
                       meta={"figure": "D", "source": "stage_c/all_rows.csv"})
    else:
        rows = pd.read_csv(cache, low_memory=False)

    methods = [m for m in ("v2_akde", "finalist", "spades_simplex", "kde_sj",
                           "abramson", "gmm_em") if m in set(rows.variant)]
    targets = sorted(set(rows.target), key=lambda s: int(s[1:]))
    sizes = sorted(rows.N.unique())

    apply_style(8.0)
    fig = plt.figure(figsize=(7.5, 10.4))
    gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 1.25, 1.15], hspace=0.62,
                          wspace=0.35)

    # (a) per-density ISE vs N
    for i, t in enumerate(targets):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        for m in methods:
            g = rows[(rows.target == t) & (rows.variant == m)]
            if g.empty:
                continue
            med = g.groupby("N").ise.median()
            ax.plot(med.index, med.values, color=STYLE[m][1],
                    linestyle=STYLE[m][2], lw=1.3, marker="o", ms=2.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=5.6)
        ax.tick_params(labelsize=5.6)
        num, name = MARRON_WAND_MAPPING[t]
        ax.set_title("$f_{%s}$ · MW%d\n%s" % (t[1:], num, name),
                     fontsize=5.8, pad=2, linespacing=1.15)
        if i % 5 == 0:
            ax.set_ylabel("median ISE", fontsize=6.4)
        if i // 5 == 2:
            ax.set_xlabel("$N$", fontsize=6.4)
    fig.legend(handles=[Line2D([], [], color=STYLE[m][1], linestyle=STYLE[m][2],
                               lw=1.4, label=STYLE[m][0]) for m in methods],
               loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.975), fontsize=6.6)

    # (b) paired ISE ratios finalist / v2
    axr = fig.add_subplot(gs[3, :])
    if "finalist" in methods and "v2_akde" in methods:
        xs, labels = [], []
        for j, t in enumerate(targets):
            for k, N in enumerate(sizes):
                piv = rows[(rows.target == t) & (rows.N == N)].pivot_table(
                    index="rep", columns="variant", values="ise", aggfunc="first")
                if "finalist" not in piv or "v2_akde" not in piv:
                    continue
                d = piv[["finalist", "v2_akde"]].dropna()
                r = (d["finalist"] / d["v2_akde"]).to_numpy()
                pos = j + (k - 1) * 0.26
                axr.scatter(np.full(r.size, pos), r, s=8, alpha=0.55,
                            color=["#0072B2", "#E69F00", "#009E73"][k],
                            edgecolors="none")
                axr.plot([pos - 0.09, pos + 0.09],
                         [np.exp(np.mean(np.log(r)))] * 2, color="k", lw=1.2)
                xs.append(pos)
            labels.append(t)
        axr.axhline(1.0, color="k", lw=0.8)
        axr.axhline(0.8, color="0.5", lw=0.7, ls="--")
        axr.set_yscale("log")
        axr.set_xticks(range(len(targets)))
        axr.set_xticklabels(targets, fontsize=6.5)
        axr.set_ylabel("paired ISE ratio\nfinalist / original V2", fontsize=7)
        axr.set_title("Paired ISE ratios, one point per replicate; bars are the "
                      "geometric mean. Dashed line: the 0.8 development objective.",
                      fontsize=7, pad=3)
        axr.legend(handles=[Line2D([], [], marker="o", ls="", color=c,
                                   label="N = %d" % n, markersize=4)
                            for c, n in zip(["#0072B2", "#E69F00", "#009E73"], sizes)],
                   fontsize=6.2, frameon=False, ncol=3, loc="upper right")

    # (c) accuracy vs cost
    for col, (xcol, xlabel, logx) in enumerate(
            [("n_active", "final active support size", True),
             ("wall_time_s", "cold end-to-end fit time (s)", True)]):
        ax = fig.add_subplot(gs[4, col * 3:(col * 3 + 3) if col == 0 else 5])
        for m in methods:
            g = rows[rows.variant == m]
            if g.empty or xcol not in g:
                continue
            agg = g.groupby(["target", "N"]).agg(x=(xcol, "median"),
                                                 y=("ise", "median")).reset_index()
            ax.scatter(agg.x, agg.y, s=11, color=STYLE[m][1], alpha=0.75,
                       edgecolors="none", label=STYLE[m][0])
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("median ISE", fontsize=7)
        ax.tick_params(labelsize=6)
        if col == 1:
            ax.legend(fontsize=5.8, frameon=False, loc="upper right")
    fig.suptitle("Accuracy and cost, Stage-C confirmation", y=1.0, fontsize=9)
    paths = save_figure(fig, FIG, stem)
    plt.close(fig)
    R = int(rows.rep.nunique())
    CAPTIONS[stem] = (
        "Figure D. Accuracy and cost over the locked Stage-C confirmation "
        "(15 Marron-Wand targets, N in {%s}, R = %d independent datasets per "
        "cell, six methods). Top block: median ISE against N, one small panel "
        "per target, log-log. Any apparent slope is a DESCRIPTIVE finite-range "
        "summary over three sample sizes; no convergence rate is inferred from "
        "it. Middle: paired ISE ratios finalist / original V2 AKDE, one point "
        "per replicate, black bar at the geometric mean, dashed line at the "
        "0.8 development objective (a proposed decision threshold, not a "
        "target the confirmation was required to meet). Bottom: median ISE "
        "against the final active support size and against cold end-to-end fit "
        "time, one point per (target, N). Selection rule: 3-fold "
        "cross-validated held-out integrated-square score, training data only; "
        "pinned 60x5 dictionary for every dictionary-based method. Truth used "
        "only for evaluation." % (", ".join(map(str, sizes)), R))
    print("Figure D:", paths["pdf"])
    return paths


# ===================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="all",
                    choices=["A", "Adev", "Aconf", "B", "C", "D", "all"])
    ap.add_argument("--which", default="dev", choices=["dev", "conf"])
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--figure-b-N", type=int, default=1000)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    todo = []
    if a.figure in ("A", "all"):
        todo.append(("A", lambda: figure_A(a.which, a.plot_only)))
    if a.figure == "Adev":
        todo.append(("A", lambda: figure_A("dev", a.plot_only)))
    if a.figure in ("Aconf", "all"):
        todo.append(("Aconf", lambda: figure_A("conf", a.plot_only)))
    if a.figure in ("B", "all"):
        todo.append(("B", lambda: figure_B(a.figure_b_N, a.plot_only)))
    if a.figure in ("C", "all"):
        todo.append(("C", lambda: figure_C(a.plot_only)))
    if a.figure in ("D", "all"):
        todo.append(("D", lambda: figure_D(a.plot_only)))

    failed = []
    for name, fn in todo:
        try:
            fn()
        except Exception as exc:
            failed.append((name, "%s: %s" % (type(exc).__name__, exc)))
            print("  Figure %s NOT built: %s: %s" % (name, type(exc).__name__, exc))

    if CAPTIONS:
        cap = FIG / "captions.md"
        prev = cap.read_text(encoding="utf-8") if cap.exists() else ""
        blocks = {}
        for chunk in prev.split("\n## ")[1:]:
            k = chunk.split("\n", 1)[0].strip()
            blocks[k] = chunk.split("\n", 1)[1].strip()
        blocks.update({k: v for k, v in CAPTIONS.items()})
        with open(cap, "w", encoding="utf-8") as fh:
            fh.write("# Figure captions\n")
            for k in sorted(blocks):
                fh.write("\n## %s\n\n%s\n" % (k, blocks[k]))
        print("captions ->", cap)
    if failed:
        print("\nNOT BUILT:", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
