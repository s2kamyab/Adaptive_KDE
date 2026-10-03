"""Stage-C analysis, exactly as locked in protocol_followup_locked.yaml.

Formal testing is confined to ONE family: the primary comparison
(finalist vs original V2 AKDE) across the 15 targets within each N,
Holm-Bonferroni at alpha = 0.05, three families.  Everything else is reported
as estimated effects with declared intervals and no significance labels.
A non-significant comparison is never reported as equivalence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
from scipy import stats                                      # noqa: E402

from akde.densities import MARRON_WAND_MAPPING               # noqa: E402
from followup.akde_followup.report import summarise_pairs    # noqa: E402

RES = ROOT / "followup" / "results"
SRC = RES / "stage_c"
OUT = SRC / "analysis"
PRIMARY, REFERENCE = "finalist", "v2_akde"
SECONDARY = ["spades_simplex", "kde_sj", "abramson", "gmm_em"]


def holm(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    m = q.size
    if m == 0:
        return out
    order = np.argsort(q)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * q[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    df = pd.read_csv(SRC / "all_rows.csv", low_memory=False)
    err = df["error"].fillna("").astype(str)
    fails = df[err != ""]
    ok = df[err == ""].copy()
    R = int(ok.rep.nunique())
    print("rows %d  failures %d  replicates %d  methods %s"
          % (len(df), len(fails), R, sorted(ok.variant.unique())))
    if len(fails):
        fails.to_csv(OUT / "failures.csv", index=False)
        print(fails.groupby(["variant", "target", "N"]).size().to_string())

    # ---- ISE positivity, stated rather than floored ----------------------
    bad = ok[~np.isfinite(ok.ise) | (ok.ise <= 0)]
    print("\nnon-positive or non-finite ISE rows: %d (excluded from log ratios, "
          "listed in analysis/nonpositive_ise.csv)" % len(bad))
    if len(bad):
        bad.to_csv(OUT / "nonpositive_ise.csv", index=False)

    # ---- headline accuracy ----------------------------------------------
    med = ok.pivot_table(index="variant", columns="N", values="ise", aggfunc="median")
    mean = ok.pivot_table(index="variant", columns="N", values="ise", aggfunc="mean")
    med.to_csv(OUT / "median_ise_by_N.csv")
    mean.to_csv(OUT / "mean_ise_by_N.csv")
    print("\n== median ISE pooled over 15 targets ==")
    print(med.to_string(float_format=lambda v: "%.5g" % v))
    print("\n== mean ISE pooled over 15 targets ==")
    print(mean.to_string(float_format=lambda v: "%.5g" % v))
    per = ok.pivot_table(index=["target", "N"], columns="variant",
                         values="ise", aggfunc="median")
    per.to_csv(OUT / "median_ise_per_cell.csv")

    # ---- paired summaries ------------------------------------------------
    rows = []
    for ref in [REFERENCE] + SECONDARY:
        if ref == PRIMARY:
            continue
        for (t, N), g in ok.groupby(["target", "N"]):
            piv = g.pivot_table(index="rep", columns="variant", values="ise",
                                aggfunc="first")
            if PRIMARY not in piv or ref not in piv:
                continue
            d = piv[[PRIMARY, ref]].dropna()
            rec = {"target": t, "mw": MARRON_WAND_MAPPING[t][0],
                   "mw_name": MARRON_WAND_MAPPING[t][1], "N": N,
                   "is_primary": ref == REFERENCE}
            rec.update(summarise_pairs(d[PRIMARY].to_numpy(), d[ref].to_numpy(),
                                       PRIMARY, ref, n_boot=a.boot, seed=0))
            # paired t test on the log ratio, the one declared test statistic
            z = np.log(d[PRIMARY].to_numpy() / d[ref].to_numpy())
            z = z[np.isfinite(z)]
            rec["p_raw"] = float(stats.ttest_1samp(z, 0.0).pvalue) if z.size > 1 else np.nan
            rows.append(rec)
    paired = pd.DataFrame(rows)

    # Holm within the ONE predefined family: primary comparison, 15 targets per N
    paired["p_holm"] = np.nan
    prim = paired[paired.is_primary]
    for N, g in prim.groupby("N"):
        paired.loc[g.index, "p_holm"] = holm(g["p_raw"].to_numpy())
    paired.to_csv(OUT / "paired_summaries.csv", index=False)

    print("\n== PRIMARY: finalist vs original V2 AKDE, per (target, N) ==")
    cols = ["target", "mw", "mw_name", "N", "R", "geom_mean_ratio", "t_lo", "t_hi",
            "mean_logratio", "se_logratio", "median_ratio", "n_better",
            "p_raw", "p_holm"]
    pr = paired[paired.is_primary].sort_values(["N", "mw"])[cols]
    print(pr.to_string(index=False, float_format=lambda v: "%.4g" % v))
    pr.to_csv(OUT / "primary_per_cell.csv", index=False)

    # equal weight per target
    agg_rows = []
    for ref in [REFERENCE] + SECONDARY:
        sub = paired[paired.reference == ref] if "reference" in paired else \
            paired[paired.method == PRIMARY]
        sub = paired[(paired.reference == ref)]
        for N, g in sub.groupby("N"):
            zt = g["mean_logratio"].to_numpy(float)
            zt = zt[np.isfinite(zt)]
            se = float(np.std(zt, ddof=1) / np.sqrt(zt.size)) if zt.size > 1 else np.nan
            tc = stats.t.ppf(0.975, zt.size - 1) if zt.size > 1 else np.nan
            agg_rows.append({
                "reference": ref, "N": N, "n_targets": int(zt.size),
                "geom_mean_ratio_equal_weight": float(np.exp(zt.mean())),
                "ci_lo": float(np.exp(zt.mean() - tc * se)),
                "ci_hi": float(np.exp(zt.mean() + tc * se)),
                "n_targets_better": int((zt < 0).sum()),
                "worst_target_ratio": float(np.exp(zt.max())),
                "best_target_ratio": float(np.exp(zt.min())),
                "interval": "Student-t over the 15 per-target mean log ratios"})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT / "aggregate_equal_weight.csv", index=False)
    print("\n== benchmark-wide, equal weight per target (ratio < 1 favours the finalist) ==")
    print(agg.to_string(index=False, float_format=lambda v: "%.4g" % v))

    # significance count, primary family only
    sig = paired[paired.is_primary].copy()
    sig["significant"] = sig["p_holm"] < 0.05
    tab = sig.groupby("N").apply(
        lambda g: pd.Series({
            "n_comparisons": len(g),
            "finalist_better_and_significant":
                int(((g.mean_logratio < 0) & g.significant).sum()),
            "finalist_worse_and_significant":
                int(((g.mean_logratio > 0) & g.significant).sum()),
            "not_significant": int((~g.significant).sum())}),
        include_groups=False)
    tab.to_csv(OUT / "primary_holm_counts.csv")
    print("\n== primary family, Holm-adjusted within each N (alpha=0.05) ==")
    print(tab.to_string())
    print("A non-significant cell is NOT evidence of equivalence; it means the "
          "study at R=%d cannot resolve that cell." % R)

    # ---- cost -------------------------------------------------------------
    cost = ok.groupby(["variant", "N"]).agg(
        cold_wall_median=("wall_time_s", "median"),
        cold_wall_p90=("wall_time_s", lambda s: float(np.percentile(s, 90))),
        tune_median=("tune_time_s", "median"),
        fit_median=("fit_time_s", "median"),
        support_median=("n_active", "median")).reset_index()
    cost.to_csv(OUT / "cost.csv", index=False)
    print("\n== cold end-to-end wall time (s) and final support size, medians ==")
    print(cost.pivot_table(index="variant", columns="N",
                           values=["cold_wall_median", "support_median"]).to_string(
        float_format=lambda v: "%.3f" % v))

    # ---- validity and diagnostics ----------------------------------------
    diag = ok.groupby(["variant", "N"]).agg(
        n=("ise", "size"),
        frac_valid=("is_valid_density",
                    lambda s: float(s.astype(str).isin(["True"]).mean())),
        max_mass_error=("mass_error", "max")).reset_index()
    for c, lab in (("stage1_converged", "stage1_conv_rate"),
                   ("fallback_used", "fallback_rate"),
                   ("support_capped", "support_cap_rate")):
        if c in ok.columns:
            diag[lab] = ok.groupby(["variant", "N"])[c].apply(
                lambda s: float(s.astype(str).isin(["True"]).mean())).values
    for c in ("stage1_kkt", "stage2_kkt", "cov_rank_numeric",
              "cov_effective_rank", "omega_cond"):
        if c in ok.columns:
            diag["med_" + c] = ok.groupby(["variant", "N"])[c].median().values
    diag.to_csv(OUT / "diagnostics.csv", index=False)
    print("\n== validity / solver diagnostics ==")
    print(diag.to_string(index=False, float_format=lambda v: "%.4g" % v))

    # ---- interval plausibility -------------------------------------------
    pl = paired[paired.is_primary][["target", "N", "skew_logratio", "shapiro_p",
                                    "sd_logratio", "t_halfwidth_log",
                                    "geom_mean_ratio", "boot_geom_mean_ratio",
                                    "t_lo", "t_hi", "boot_lo", "boot_hi"]].copy()
    pl["t_vs_boot_lo_rel"] = (pl.t_lo - pl.boot_lo).abs() / pl.boot_lo
    pl["t_vs_boot_hi_rel"] = (pl.t_hi - pl.boot_hi).abs() / pl.boot_hi
    pl["meets_log125_halfwidth"] = pl.t_halfwidth_log <= np.log(1.25)
    pl.to_csv(OUT / "interval_plausibility.csv", index=False)
    print("\n== interval plausibility (primary family) ==")
    print("median |skew| of the paired log ratio : %.3f" % pl.skew_logratio.abs().median())
    print("cells with Shapiro p < 0.05           : %d of %d"
          % (int((pl.shapiro_p < 0.05).sum()), len(pl)))
    print("median relative gap t vs bootstrap    : lo %.3f, hi %.3f"
          % (pl.t_vs_boot_lo_rel.median(), pl.t_vs_boot_hi_rel.median()))
    print("cells meeting the log(1.25) half-width target: %d of %d (%.0f%%)"
          % (int(pl.meets_log125_halfwidth.sum()), len(pl),
             100 * pl.meets_log125_halfwidth.mean()))

    json.dump({"R": R, "n_rows": int(len(df)), "n_failures": int(len(fails)),
               "n_nonpositive_ise": int(len(bad)), "n_boot": a.boot},
              open(OUT / "run_summary.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
