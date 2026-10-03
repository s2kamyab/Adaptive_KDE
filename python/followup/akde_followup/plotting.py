"""Figure style and IO for the follow-up.

Colour and line style are chosen together so every figure is readable in
greyscale and for the common colour-vision deficiencies: the palette is
Okabe-Ito plus one IBM-palette violet, and no two curves share a dash pattern.

Every figure is written three ways (vector PDF, vector SVG, 300 dpi PNG) and
every plotted curve is also written to disk as CSV and NPZ, so the figures can
be redrawn with ``--plot-only`` and no refitting.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

__all__ = ["STYLE", "apply_style", "save_figure", "save_plot_data",
           "truth_range", "METHOD_ORDER"]

#: method key -> (label, colour, linestyle, linewidth, z-order)
# Draw order matters as much as colour here: the finalist, the adapted SPADES
# and the Sheather-Jones KDE often agree closely enough to overplot, so the
# two references are given the HIGHER z-order and dashed patterns with wide
# gaps, and the finalist is drawn beneath them as a solid line.  Where a
# reference is invisible it is because it coincides with the finalist, not
# because it is missing; the companion CSV carries every curve's metrics.
STYLE = {
    "truth":          ("Ground truth",        "#000000", "-",              2.6, 3),
    "v2_akde":        ("Original AKDE (V2)",  "#E69F00", (0, (6, 3)),      1.5, 4),
    "finalist":       ("Finalist",            "#0072B2", "-",              2.2, 5),
    "spades_simplex": ("Adapted SPADES",      "#785EF0", (0, (6, 4, 1, 4)), 1.5, 6),
    "kde_sj":         ("KDE (Sheather-Jones)", "#009E73", (0, (1.5, 3.5)), 1.8, 7),
    "abramson":       ("Abramson KDE",        "#D55E00", (0, (4, 1.5, 1, 1.5)), 1.5, 2),
    "gmm_em":         ("Gaussian mixture (EM)", "#56B4E9", (0, (3, 2, 1, 2)), 1.5, 2),
}
METHOD_ORDER = ["truth", "v2_akde", "finalist", "spades_simplex", "kde_sj",
                "abramson", "gmm_em"]


def apply_style(base_font: float = 9.0) -> None:
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font + 0.5,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 0.5,
        "figure.titlesize": base_font + 2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3
        "ps.fonttype": 42,
        "svg.fonttype": "none",      # keep text as text in the SVG
        "figure.dpi": 110,
    })


def curve(ax, key, x, y, **kw):
    label, colour, ls, lw, z = STYLE[key]
    return ax.plot(x, y, color=colour, linestyle=ls, linewidth=lw,
                   zorder=z, label=kw.pop("label", label), **kw)


def save_figure(fig, out_dir, stem: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        p = out_dir / ("%s.%s" % (stem, ext))
        fig.savefig(p, **kw)
        paths[ext] = str(p)
    return paths


def save_plot_data(out_dir, stem: str, *, grid: dict, table: pd.DataFrame | None = None,
                   meta: dict | None = None) -> dict:
    """Write the plotted curves (NPZ), the metric table (CSV) and metadata."""
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    npz = out_dir / "data" / ("%s_curves.npz" % stem)
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in grid.items()})
    written = {"npz": str(npz)}
    if table is not None:
        csv = out_dir / "data" / ("%s_metrics.csv" % stem)
        table.to_csv(csv, index=False)
        written["csv"] = str(csv)
    if meta is not None:
        import json
        j = out_dir / "data" / ("%s_meta.json" % stem)
        with open(j, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        written["meta"] = str(j)
    return written


def truth_range(target, q_lo: float = 0.0005, q_hi: float = 0.9995,
                n_grid: int = 4000):
    """Plotting window from predeclared truth quantiles.  Visualisation only.

    Computed by inverting the mixture CDF on a fine bracketing grid, so it does
    not depend on any sample and can be declared before any fit is inspected.
    """
    from scipy.stats import norm
    w = np.asarray(target.weights, float)
    mu = np.asarray(target.means, float)[:, 0]
    sd = np.sqrt(np.asarray(target.variances, float)[:, 0])
    lo = float((mu - 12 * sd).min())
    hi = float((mu + 12 * sd).max())
    t = np.linspace(lo, hi, 200001)
    cdf = (w[None, :] * norm.cdf((t[:, None] - mu[None, :]) / sd[None, :])).sum(axis=1)
    x_lo = float(np.interp(q_lo, cdf, t))
    x_hi = float(np.interp(q_hi, cdf, t))
    return x_lo, x_hi, np.linspace(x_lo, x_hi, n_grid)


def tail_mass_outside(fit_pdf, x_lo, x_hi, pad=40.0):
    """Fitted mass falling outside the plotting window (reported in captions)."""
    from akde.metrics import integrate_1d
    left = integrate_1d(fit_pdf, x_lo - pad, x_lo, tol=1e-9, max_panels=4096)
    right = integrate_1d(fit_pdf, x_hi, x_hi + pad, tol=1e-9, max_panels=4096)
    return float(max(left, 0.0) + max(right, 0.0))
