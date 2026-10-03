"""LaTeX tables and numerical macros for the improved-AKDE study.

    python followup/improved/scripts/make_tables.py

    improved_benchmark_mean_sd.tex   mean +- pooled SD over the 450 fits per (method, N)
    improved_fixed_benchmark.tex     exp(T) [95% bootstrap CI] of AKDE-R vs each reference
    improved_dev_ablation.tex        development ablation
    improved_cost.tex                components, storage, query, fit time: mean +- SD
    improved_hr.tex                  hr family mean +- SD
    improved_numbers.tex             \\newcommand macros (digit-free names)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from followup.improved.runner import RESULTS  # noqa: E402

TAB = HERE / "tables"
A = RESULTS / "analysis"
FIN = json.load(open(HERE / "finalist_improved.json", encoding="utf-8"))
NEW, ABL = FIN["finalist"], FIN.get("ablation")
LAB = {NEW: "AKDE-R (repaired)", "akde_m": "AKDE-M", "akde_v2": "AKDE-V2", "spades_simplex": "SPADES-simplex",
       "kde_sj": "Sheather--Jones KDE", "abramson": "Abramson KDE", "gmm_em": "Gaussian mixture (EM)",
       "akde_single": "AKDE single-scale"}
if ABL:
    LAB[ABL] = "AKDE-R core (no anchor/merge)"
ORDER = [NEW] + ([ABL] if ABL else []) + ["akde_m", "akde_v2", "spades_simplex", "kde_sj", "abramson", "gmm_em"]
_D = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def fmt(v, sig=3):
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    if float(v).is_integer() and abs(v) < 1e7:
        return "%d" % int(v)
    if 1e3 <= abs(v) < 1e5:
        return "%.0f" % v
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        m, e = ("%.*e" % (sig - 1, v)).split("e")
        return r"%s\times10^{%d}" % (m, int(e))
    return "%.*g" % (sig, v)


def pm(m, s, sig=3):
    return "$%s \\pm %s$" % (fmt(m, sig), fmt(s, sig))


def write(stem, text, df=None):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / (stem + ".tex")).write_text(text, encoding="utf-8")
    if df is not None:
        df.to_csv(TAB / (stem + ".csv"), index=False)
    print("wrote", TAB / (stem + ".tex"))


TAG = {NEW: "R", "akde_m": "M", "spades_simplex": "Spades", "kde_sj": "Sj", "abramson": "Abr",
       "gmm_em": "Gmm", "akde_v2": "Vtwo", "akde_single": "Single"}
if ABL:
    TAG[ABL] = "C"
_TAG_ORDER = sorted(TAG, key=len, reverse=True)


def macro_name(s):
    for k in _TAG_ORDER:
        s = s.replace(k, TAG[k])
    pieces = [p for p in re.split(r"[^A-Za-z0-9]+", s) if p]
    s = "".join(p[0].upper() + p[1:] for p in pieces)
    for a, b in (("N5000", "NLarge"), ("N1000", "NMid"), ("N200", "NSmall"), ("N500", "NFiveH")):
        s = s.replace(a, b)
    return "ir" + "".join(_D.get(c, c) for c in s)


def main():
    po = pd.read_csv(A / "pooled_mean_sd.csv"); fb = pd.read_csv(A / "fixed_benchmark.csv")
    bm = pd.read_csv(A / "benchmark_means.csv"); rel = pd.read_csv(A / "reliability.csv")
    dev = pd.read_csv(RESULTS / "dev" / "dev_ablation.csv"); cost = pd.read_csv(RESULTS / "cost" / "query_cost.csv")
    tidy = pd.read_csv(A / "tidy_confirm.csv", low_memory=False)
    mac = {}

    def add(k, v, kind="g3"):
        if kind == "g3":
            s = fmt(v, 3)
        elif kind == "f2":
            s = "%.2f" % v
        elif kind == "f1":
            s = "%.1f" % v
        elif kind == "int":
            s = "%d" % int(round(v))
        elif kind == "pct1":
            s = "%.1f" % (100 * v)
        else:
            s = str(v)
        mac[macro_name(k)] = s

    # 1 benchmark mean +- pooled SD
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & $N=200$ & $N=1000$ & $N=5000$ \\", r"\midrule"]
    rows = []
    for m in ORDER:
        cells = []
        for N in (200, 1000, 5000):
            s = po[(po.method == m) & (po.N == N)]
            cells.append(pm(s.iloc[0].mean_ise, s.iloc[0].pooled_sd_ddof1) if len(s) else "--")
            if len(s):
                rows.append({"method": m, "N": N, "mean": s.iloc[0].mean_ise, "pooled_sd": s.iloc[0].pooled_sd_ddof1, "n": s.iloc[0].n_fits})
                add("Pooled%sN%d" % (m, N), s.iloc[0].mean_ise); add("PooledSd%sN%d" % (m, N), s.iloc[0].pooled_sd_ddof1)
                add("Median%sN%d" % (m, N), s.iloc[0].median_ise)
        lines.append("%s & %s \\\\" % (LAB[m], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("improved_benchmark_mean_sd", "\n".join(lines), pd.DataFrame(rows))

    # 2 fixed benchmark ratios
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Method & Reference & $N=200$ & $N=1000$ & $N=5000$ \\", r"\midrule"]
    for m in [NEW] + ([ABL] if ABL else []):
        for ref in ("akde_m", "spades_simplex", "kde_sj", "abramson", "gmm_em"):
            cells = []
            for N in (200, 1000, 5000):
                s = fb[(fb.method == m) & (fb.reference == ref) & (fb.N == N)]
                if s.empty:
                    cells.append("--"); continue
                s = s.iloc[0]
                cells.append("%.3f [%.3f, %.3f] (%d/15)" % (s.exp_T, s.boot_ci_lo, s.boot_ci_hi, s.n_targets_favouring_method))
                add("ExpT%sVs%sN%d" % (m, ref, N), s.exp_T, "f2"); add("ExpTLo%sVs%sN%d" % (m, ref, N), s.boot_ci_lo, "f2")
                add("ExpTHi%sVs%sN%d" % (m, ref, N), s.boot_ci_hi, "f2"); add("Fav%sVs%sN%d" % (m, ref, N), s.n_targets_favouring_method, "int")
            lines.append("%s & %s & %s \\\\" % (LAB[m], LAB[ref], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("improved_fixed_benchmark", "\n".join(lines), fb)

    # 3 dev ablation
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Variant & tuned h.p. & ratio vs AKDE-M (MW dev) [95\% CI] & ratio (hr dev) & stage 1 / 2 at tol & components & fit time (s) \\", r"\midrule"]
    for r in dev.itertuples():
        s1 = "--" if not np.isfinite(r.stage1_at_tol) else "%.0f\\%%" % (100 * r.stage1_at_tol)
        lines.append("%s & %d & %.3f [%.3f, %.3f] & %.3f & %s / %.0f\\%% & %.1f & %.1f \\\\" % (
            r.variant.replace("_", "\\_"), r.n_hyper, r.primary_ratio, r.primary_ratio_lo, r.primary_ratio_hi, r.hr_ratio,
            s1, 100 * r.stage2_at_tol, r.mean_components, r.mean_wall_s))
        add("Dev%s" % r.variant, r.primary_ratio, "f2"); add("DevHr%s" % r.variant, r.hr_ratio, "f2")
        add("DevLo%s" % r.variant, r.primary_ratio_lo, "f2"); add("DevHi%s" % r.variant, r.primary_ratio_hi, "f2")
        add("DevComp%s" % r.variant, r.mean_components, "f1"); add("DevWall%s" % r.variant, r.mean_wall_s, "f1")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("improved_dev_ablation", "\n".join(lines), dev)

    # 4 cost
    t = tidy[tidy.valid]
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"$N$ & Method & components & query ($\mu$s/point) & fit time incl.\ tuning (s) \\", r"\midrule"]
    for N in (200, 1000, 5000):
        for m in [NEW] + ([ABL] if ABL else []) + ["akde_m", "spades_simplex", "kde_sj", "gmm_em"]:
            s = t[(t.method == m) & (t.N == N)]
            c = cost[(cost.stage == "confirm") & (cost.method == m) & (cost.N == N)]
            k, w = s.n_components.astype(float), s.wall_time_s.astype(float)
            q = c.query_us_per_point_median.astype(float) if len(c) else pd.Series(dtype=float)
            lines.append("%d & %s & %s & %s & %s \\\\" % (N, LAB[m], pm(k.mean(), k.std(ddof=1)),
                                                          pm(q.mean(), q.std(ddof=1)) if len(q) else "see Table~\\ref{tab:cost}",
                                                          pm(w.mean(), w.std(ddof=1))))
            add("Comp%sN%d" % (m, N), k.mean(), "f1"); add("Wall%sN%d" % (m, N), w.mean())
            if len(q):
                add("Query%sN%d" % (m, N), q.mean())
        lines.append(r"\midrule" if N != 5000 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("improved_cost", "\n".join(lines))

    # 5 hr
    if (A / "hr_per_cell.csv").exists():
        hp = pd.read_csv(A / "hr_per_cell.csv"); hr_r = pd.read_csv(A / "hr_ratios.csv")
        lines = [r"\begin{tabular}{lcccc}", r"\toprule"]
        for N in (500, 5000):
            lines += [r"\multicolumn{5}{l}{\emph{$N = %d$}} \\" % N, r"Method & $r=1$ & $r=2$ & $r=4$ & $r=8$ \\", r"\midrule"]
            for m in [NEW, "akde_m", "akde_single", "spades_simplex", "kde_sj", "abramson", "gmm_em"]:
                cells = []
                for r in (1, 2, 4, 8):
                    s = hp[(hp.method == m) & (hp.N == N) & (hp.r == r)]
                    cells.append(pm(s.iloc[0].mean_ise, s.iloc[0].sd_ise_ddof1) if len(s) else "--")
                lines.append("%s & %s \\\\" % (LAB[m], " & ".join(cells)))
            lines.append(r"\midrule" if N == 500 else r"\bottomrule")
        lines.append(r"\end{tabular}")
        write("improved_hr", "\n".join(lines), hp)
        for s in hr_r[hr_r.metric == "ise"].itertuples():
            add("Hr%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.geom_mean_ratio, "f2")
            add("HrLo%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.boot_ci_lo, "f2")
            add("HrHi%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.boot_ci_hi, "f2")
            add("HrFav%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.n_pairs_favouring_method, "int")
        for s in hr_r[hr_r.metric == "ise_narrow"].itertuples():
            add("HrNarrow%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.geom_mean_ratio, "f2")
            add("HrNarrowLo%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.boot_ci_lo, "f2")
            add("HrNarrowHi%sVs%sR%dN%d" % (s.method, s.reference, s.r, s.N), s.boot_ci_hi, "f2")

    # reliability + benchmark means
    for s in rel.itertuples():
        add("Large%s" % s.method, s.n_large_error, "int"); add("LargeFrac%s" % s.method, s.frac_large_error, "pct1")
        add("Worst%s" % s.method, s.worst_ise_over_cell_median, "f1"); add("Err%s" % s.method, s.n_errors, "int")
        add("NValid%s" % s.method, s.n_valid_density, "int")
        for c in ("stage1_kkt_frac_at_tol", "stage2_kkt_frac_at_tol", "stage2_kkt_max", "lam_at_grid_edge_frac"):
            v = getattr(s, c, np.nan)
            if isinstance(v, float) and np.isfinite(v):
                add("%s%s" % (c, s.method), v, "pct1" if "frac" in c else "g3")
    for s in bm.itertuples():
        add("Bm%sN%d" % (s.method, s.N), s.benchmark_mean_ise); add("BmLo%sN%d" % (s.method, s.N), s.boot_ci_lo); add("BmHi%sN%d" % (s.method, s.N), s.boot_ci_hi)
    add("Finalist", NEW.replace("_", " "), "s")
    add("Ablation", (ABL or "none").replace("_", " "), "s")
    text = "% generated by followup/improved/scripts/make_tables.py\n" + "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in mac.items()) + "\n"
    write("improved_numbers", text, pd.DataFrame({"macro": list(mac), "value": list(mac.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
