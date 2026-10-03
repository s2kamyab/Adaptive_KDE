"""Task 5 - statistical reporting from saved per-replication rows. No refitting.

Deliverables (followup/results/statistics/):

    tidy_metrics.csv          one row per (target, N, rep, method): the raw metrics
    per_cell_summary.csv      mean +- SD (ddof=1), n_valid/n_attempted, median,
                              and a SEPARATELY LABELLED bootstrap CI for the mean
    paired_effects.csv        per-cell paired log ratios with paired bootstrap CIs
    fixed_benchmark.csv       T = (1/15) sum_d mean_r z_dr and exp(T) per N,
                              bootstrapped WITHIN each fixed target
    heterogeneity.csv         between-target spread of the per-target effects
    bootstrap_config.json     seeds, B, scheme, limitations
    tables/*.tex              generated LaTeX tables

Key distinctions enforced throughout:
  * mean +- SD describes variation BETWEEN fitted datasets. It is not a CI.
  * a bootstrap CI for the mean is labelled as such; if a bootstrap SE is
    reported it is labelled bootstrap SE, never "replicate SD".
  * the 15 targets are a FIXED benchmark, so the target index is never
    resampled; only replicate IDs are, within each target-N cell, carrying all
    methods together so pairing is preserved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from akde.densities import MARRON_WAND_MAPPING     # noqa: E402

from followup.paths import RAW_ROOT, SUMMARY_ROOT    # noqa: E402
RES = RAW_ROOT
OUT = SUMMARY_ROOT
TAB = OUT / "tables"

B_BOOT = 10000
SEED = 20260928
REFERENCE = "v2_akde"
PRIMARY = "finalist"
METHOD_ORDER = ["finalist", "v2_akde", "spades_simplex", "kde_sj", "abramson", "gmm_em"]
METHOD_LABEL = {
    "finalist": r"AKDE-M (matched)", "v2_akde": r"AKDE-V2 (ablation)",
    "spades_simplex": r"SPADES-simplex", "kde_sj": r"KDE (SJ)",
    "abramson": r"Abramson KDE", "gmm_em": r"GMM (EM)",
}


# ---------------------------------------------------------------- loading
def load_tidy():
    d = pd.read_csv(RES / "stage_c" / "all_rows.csv", low_memory=False)
    d["error"] = d["error"].fillna("").astype(str)
    keep = ["target", "N", "rep", "variant", "ise", "wall_time_s", "n_active",
            "is_valid_density", "mass_error", "nll", "tune_time_s", "fit_time_s",
            "stage1_converged", "stage1_kkt", "stage2_kkt", "error"]
    keep = [c for c in keep if c in d.columns]
    t = d[keep].copy()
    t["mw"] = t.target.map(lambda s: MARRON_WAND_MAPPING[s][0])
    t["mw_name"] = t.target.map(lambda s: MARRON_WAND_MAPPING[s][1])
    t["attempted"] = 1
    t["valid"] = (t["error"] == "") & np.isfinite(t.ise) & (t.ise > 0)
    return t


# ---------------------------------------------------------------- bootstrap
def cell_rep_matrix(t, target, N):
    """reps x methods matrix of ISE for one (target, N) cell, preserving pairing."""
    sub = t[(t.target == target) & (t.N == N)]
    return sub.pivot_table(index="rep", columns="variant", values="ise",
                           aggfunc="first")


def bootstrap_indices(n_reps, B, rng):
    return rng.integers(0, n_reps, size=(B, n_reps))


def per_cell_summary(t, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for (target, N), sub in t.groupby(["target", "N"], sort=False):
        piv = cell_rep_matrix(t, target, N)
        n_reps = piv.shape[0]
        idx = bootstrap_indices(n_reps, B, rng)
        for m in piv.columns:
            v = piv[m].to_numpy(dtype=float)
            att = int((sub.variant == m).sum())
            val = int(sub[(sub.variant == m)].valid.sum())
            finite = v[np.isfinite(v)]
            boots = finite[idx].mean(axis=1) if finite.size == n_reps else np.array([])
            rows.append({
                "target": target, "mw": MARRON_WAND_MAPPING[target][0],
                "mw_name": MARRON_WAND_MAPPING[target][1], "N": N, "method": m,
                "n_attempted": att, "n_valid": val,
                "mean_ise": float(finite.mean()),
                "sd_ise_ddof1": float(finite.std(ddof=1)),
                "median_ise": float(np.median(finite)),
                "iqr_ise": float(np.subtract(*np.percentile(finite, [75, 25]))),
                "min_ise": float(finite.min()), "max_ise": float(finite.max()),
                "boot_ci_lo_mean": float(np.percentile(boots, 2.5)) if boots.size else np.nan,
                "boot_ci_hi_mean": float(np.percentile(boots, 97.5)) if boots.size else np.nan,
                "boot_se_mean": float(boots.std(ddof=1)) if boots.size else np.nan,
                "mean_support": float(sub[sub.variant == m].n_active.mean()),
                "mean_wall_time_s": float(sub[sub.variant == m].wall_time_s.mean()),
            })
    return pd.DataFrame(rows)


def paired_effects(t, reference=REFERENCE, B=B_BOOT, seed=SEED):
    """Per-cell paired log ratios z_dr = log(ISE_method / ISE_reference)."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for (target, N), _ in t.groupby(["target", "N"], sort=False):
        piv = cell_rep_matrix(t, target, N)
        n_reps = piv.shape[0]
        idx = bootstrap_indices(n_reps, B, rng)
        if reference not in piv.columns:
            continue
        ref = piv[reference].to_numpy(dtype=float)
        for m in piv.columns:
            if m == reference:
                continue
            cur = piv[m].to_numpy(dtype=float)
            ok = np.isfinite(cur) & np.isfinite(ref) & (cur > 0) & (ref > 0)
            n_drop = int((~ok).sum())
            z = np.log(cur[ok] / ref[ok])
            if z.size == 0:
                continue
            bz = z[idx[:, :z.size] % z.size].mean(axis=1) if z.size else np.array([])
            rows.append({
                "target": target, "mw": MARRON_WAND_MAPPING[target][0],
                "mw_name": MARRON_WAND_MAPPING[target][1], "N": N,
                "method": m, "reference": reference,
                "n_pairs": int(z.size), "n_dropped_nonpositive_or_missing": n_drop,
                "mean_logratio": float(z.mean()),
                "sd_logratio_ddof1": float(z.std(ddof=1)),
                "geom_mean_ratio": float(np.exp(z.mean())),
                "median_ratio": float(np.exp(np.median(z))),
                "boot_ci_lo_geom": float(np.exp(np.percentile(bz, 2.5))),
                "boot_ci_hi_geom": float(np.exp(np.percentile(bz, 97.5))),
                "boot_se_mean_logratio": float(bz.std(ddof=1)),
                "n_replicates_better": int((z < 0).sum()),
            })
    return pd.DataFrame(rows)


def fixed_benchmark(t, reference=REFERENCE, B=B_BOOT, seed=SEED):
    """T = (1/D) sum_d mean_r z_dr for the FIXED 15-target benchmark.

    Bootstrap: resample replicate IDs WITHIN each target-N cell (carrying all
    methods together, so pairing is preserved), then recompute T with equal
    weight on every target. The target index is never resampled: the benchmark
    is a fixed finite set of densities, not a sample from a population of
    densities.
    """
    rng = np.random.default_rng(seed + 2)
    rows = []
    for N, subN in t.groupby("N", sort=False):
        targets = sorted(subN.target.unique(), key=lambda s: MARRON_WAND_MAPPING[s][0])
        mats = {}
        for d in targets:
            piv = cell_rep_matrix(t, d, N)
            mats[d] = piv
        n_reps = mats[targets[0]].shape[0]
        idx = bootstrap_indices(n_reps, B, rng)
        methods = [m for m in mats[targets[0]].columns if m != reference]
        for m in methods:
            per_target_mean = []
            boot_T = np.zeros(B)
            for d in targets:
                piv = mats[d]
                cur = piv[m].to_numpy(dtype=float)
                ref = piv[reference].to_numpy(dtype=float)
                ok = np.isfinite(cur) & np.isfinite(ref) & (cur > 0) & (ref > 0)
                z = np.full(n_reps, np.nan)
                z[ok] = np.log(cur[ok] / ref[ok])
                per_target_mean.append(np.nanmean(z))
                boot_T += np.nanmean(z[idx], axis=1)
            boot_T /= len(targets)
            pt = np.array(per_target_mean, dtype=float)
            T = float(pt.mean())
            rows.append({
                "N": N, "method": m, "reference": reference,
                "n_targets": len(targets), "n_replicates_per_cell": n_reps,
                "T_mean_of_per_target_mean_logratio": T,
                "exp_T_geometric_mean_ratio": float(np.exp(T)),
                "boot_ci_lo": float(np.exp(np.percentile(boot_T, 2.5))),
                "boot_ci_hi": float(np.exp(np.percentile(boot_T, 97.5))),
                "boot_se_T": float(boot_T.std(ddof=1)),
                "n_targets_favouring_method": int((pt < 0).sum()),
                "between_target_sd_of_per_target_mean": float(pt.std(ddof=1)),
                "min_per_target_ratio": float(np.exp(pt.min())),
                "max_per_target_ratio": float(np.exp(pt.max())),
            })
    return pd.DataFrame(rows)


def benchmark_means(t, B=B_BOOT, seed=SEED):
    """Equal-weight-per-target benchmark arithmetic mean ISE with bootstrap CI."""
    rng = np.random.default_rng(seed + 3)
    rows = []
    for N, subN in t.groupby("N", sort=False):
        targets = sorted(subN.target.unique(), key=lambda s: MARRON_WAND_MAPPING[s][0])
        mats = {d: cell_rep_matrix(t, d, N) for d in targets}
        n_reps = mats[targets[0]].shape[0]
        idx = bootstrap_indices(n_reps, B, rng)
        for m in mats[targets[0]].columns:
            per_target = np.array([np.nanmean(mats[d][m].to_numpy(dtype=float))
                                   for d in targets])
            boot = np.zeros(B)
            for d in targets:
                boot += np.nanmean(mats[d][m].to_numpy(dtype=float)[idx], axis=1)
            boot /= len(targets)
            rows.append({
                "N": N, "method": m, "n_targets": len(targets),
                "benchmark_mean_ise_equal_weight": float(per_target.mean()),
                "boot_ci_lo": float(np.percentile(boot, 2.5)),
                "boot_ci_hi": float(np.percentile(boot, 97.5)),
                "boot_se": float(boot.std(ddof=1)),
                "between_target_sd": float(per_target.std(ddof=1)),
                "note_between_target_sd": (
                    "spread of the 15 per-target means; NOT the uncertainty of "
                    "the benchmark mean"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LaTeX
def fmt(v, sig=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
        s = "%.*e" % (sig - 1, v)
        m, e = s.split("e")
        return r"$%s\times10^{%d}$" % (m, int(e))
    return ("%.*g" % (sig, v))


def tex_main_benchmark(bm, fb):
    lines = [r"\begin{tabular}{l" + "rrr" * 1 + r"}", r"\toprule",
             r"Method & \multicolumn{3}{c}{Benchmark mean ISE, equal weight per target"
             r" [95\% bootstrap CI]} \\",
             r"\cmidrule(lr){2-4}",
             r" & $N=200$ & $N=1000$ & $N=5000$ \\", r"\midrule"]
    for m in METHOD_ORDER:
        cells = []
        for N in (200, 1000, 5000):
            r = bm[(bm.method == m) & (bm.N == N)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            cells.append("%s [%s, %s]" % (fmt(r.benchmark_mean_ise_equal_weight),
                                          fmt(r.boot_ci_lo), fmt(r.boot_ci_hi)))
        lines.append("%s & %s \\\\" % (METHOD_LABEL[m], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tex_fixed_benchmark(fb):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Comparison (vs.\ AKDE-V2) & $N=200$ & $N=1000$ & $N=5000$ \\",
             r"\midrule"]
    for m in [x for x in METHOD_ORDER if x != REFERENCE]:
        cells = []
        for N in (200, 1000, 5000):
            r = fb[(fb.method == m) & (fb.N == N)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            cells.append("%.3f [%.3f, %.3f]" % (r.exp_T_geometric_mean_ratio,
                                                r.boot_ci_lo, r.boot_ci_hi))
        lines.append("%s & %s \\\\" % (METHOD_LABEL[m], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tex_per_target_supplement(ps, N):
    sub = ps[ps.N == N].copy()
    sub = sub.sort_values(["mw", "method"])
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Target & Method & mean ISE & SD & median & $n$ valid/attempted \\",
             r"\midrule"]
    last = None
    for r in sub.itertuples():
        if r.method not in METHOD_LABEL:
            continue
        name = "" if last == r.target else r"$f_{%d}$ (MW%d)" % (int(r.target[1:]), r.mw)
        last = r.target
        lines.append("%s & %s & %s & %s & %s & %d/%d \\\\"
                     % (name, METHOD_LABEL[r.method], fmt(r.mean_ise),
                        fmt(r.sd_ise_ddof1), fmt(r.median_ise),
                        r.n_valid, r.n_attempted))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tex_cost(ps):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Mean fit time (s)} & \multicolumn{3}{c}{Mean active support} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & 200 & 1000 & 5000 & 200 & 1000 & 5000 \\", r"\midrule"]
    for m in METHOD_ORDER:
        tt, ss = [], []
        for N in (200, 1000, 5000):
            r = ps[(ps.method == m) & (ps.N == N)]
            if r.empty:
                tt.append("--")
                ss.append("--")
                continue
            tt.append(fmt(float(r.mean_wall_time_s.mean())))
            ss.append("%.0f" % float(r.mean_support.mean()))
        lines.append("%s & %s & %s \\\\" % (METHOD_LABEL[m], " & ".join(tt),
                                            " & ".join(ss)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    t = load_tidy()
    t.to_csv(OUT / "tidy_metrics.csv", index=False)
    print("tidy rows: %d  (valid %d, attempted %d)"
          % (len(t), int(t.valid.sum()), int(t.attempted.sum())))

    ps = per_cell_summary(t)
    ps.to_csv(OUT / "per_cell_summary.csv", index=False)
    pe = paired_effects(t)
    pe.to_csv(OUT / "paired_effects.csv", index=False)
    fb = fixed_benchmark(t)
    fb.to_csv(OUT / "fixed_benchmark.csv", index=False)
    bm = benchmark_means(t)
    bm.to_csv(OUT / "benchmark_means.csv", index=False)

    het = pe.groupby(["method", "N"]).agg(
        n_targets=("target", "size"),
        min_ratio=("geom_mean_ratio", "min"),
        median_ratio=("geom_mean_ratio", "median"),
        max_ratio=("geom_mean_ratio", "max"),
        sd_of_per_target_mean_logratio=("mean_logratio", "std"),
        n_targets_favouring=("mean_logratio", lambda s: int((s < 0).sum())),
    ).reset_index()
    het.to_csv(OUT / "heterogeneity.csv", index=False)

    with open(OUT / "bootstrap_config.json", "w", encoding="utf-8") as fh:
        json.dump({
            "B": B_BOOT, "master_seed": SEED,
            "streams": {"per_cell_summary": SEED, "paired_effects": SEED + 1,
                        "fixed_benchmark": SEED + 2, "benchmark_means": SEED + 3},
            "unit_resampled": "replicate ID within each (target, N) cell",
            "pairing": "all methods carried together on the resampled replicate IDs",
            "target_index": "NEVER resampled - the 15 targets are a fixed benchmark",
            "interval_type": "percentile",
            "limitations": [
                "A percentile bootstrap CI is not a p-value and no test is "
                "derived from it here.",
                "Resampling 30 replicate IDs cannot represent uncertainty about "
                "the choice of benchmark densities; between-target spread is "
                "reported separately and is not uncertainty of the benchmark mean.",
                "ISE distributions are right-skewed; medians are reported "
                "alongside means as a robustness summary.",
                "The bootstrap resamples stored metric rows only. Nothing is refitted.",
            ],
        }, fh, indent=2)

    (TAB / "tab_benchmark_mean.tex").write_text(tex_main_benchmark(bm, fb), encoding="utf-8")
    (TAB / "tab_fixed_benchmark_ratio.tex").write_text(tex_fixed_benchmark(fb), encoding="utf-8")
    (TAB / "tab_cost.tex").write_text(tex_cost(ps), encoding="utf-8")
    for N in (200, 1000, 5000):
        (TAB / ("tab_per_target_N%d.tex" % N)).write_text(
            tex_per_target_supplement(ps, N), encoding="utf-8")

    print("\n== benchmark mean ISE, equal weight per target [95% bootstrap CI] ==")
    print(bm.pivot_table(index="method", columns="N",
                         values="benchmark_mean_ise_equal_weight")
          .reindex(METHOD_ORDER).to_string(float_format=lambda v: "%.5g" % v))
    print("\n== fixed-benchmark geometric-mean ISE ratio vs AKDE-V2, exp(T) ==")
    for N in (200, 1000, 5000):
        s = fb[fb.N == N].set_index("method")
        print("  N=%-5d %s" % (N, "  ".join(
            "%s %.3f [%.3f,%.3f]" % (m, s.loc[m, "exp_T_geometric_mean_ratio"],
                                     s.loc[m, "boot_ci_lo"], s.loc[m, "boot_ci_hi"])
            for m in METHOD_ORDER if m in s.index)))
    print("\n== between-target heterogeneity of the finalist vs V2 ==")
    print(het[het.method == PRIMARY].to_string(index=False,
                                               float_format=lambda v: "%.4g" % v))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
