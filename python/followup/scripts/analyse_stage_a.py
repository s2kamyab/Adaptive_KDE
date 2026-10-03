"""Stage-A analysis: descriptive, paired, and explicitly exploratory.

Five replicates per cell.  Nothing here is a significance claim; the outputs
are per-cell paired outcomes, descriptive effects, and the solver /
conditioning diagnostics that decide what Stage B should try.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from followup.akde_followup.report import (                  # noqa: E402
    geometric_mean_ratio_by_target, summarise_pairs)

RES = ROOT / "followup" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage_a")
    ap.add_argument("--reference", default="A1_v2_moment_m300")
    a = ap.parse_args()
    src = RES / a.stage
    out = src / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src / "all_rows.csv", low_memory=False)
    df["error"] = df["error"].fillna("").astype(str)
    bad = df[df["error"] != ""]
    ok = df[df["error"] == ""].copy()
    print("rows %d, errors %d" % (len(df), len(bad)))
    if len(bad):
        print(bad[["target", "N", "rep", "variant", "error"]].to_string(index=False))

    pd.set_option("display.width", 200)

    # ---------------- accuracy -------------------------------------------
    med = ok.pivot_table(index="variant", columns="N", values="ise", aggfunc="median")
    mean = ok.pivot_table(index="variant", columns="N", values="ise", aggfunc="mean")
    per_cell = ok.pivot_table(index="variant", columns=["target", "N"],
                              values="ise", aggfunc="median")
    med.to_csv(out / "median_ise_by_N.csv")
    mean.to_csv(out / "mean_ise_by_N.csv")
    per_cell.to_csv(out / "median_ise_per_cell.csv")
    print("\n== median ISE by variant and N (5 reps x 4 targets per column) ==")
    print(med.to_string(float_format=lambda v: "%.5g" % v))
    print("\n== median ISE per (target, N) ==")
    print(per_cell.to_string(float_format=lambda v: "%.4g" % v))

    # ---------------- paired vs the V2 variant ----------------------------
    frames = []
    for v in sorted(ok.variant.unique()):
        if v == a.reference:
            continue
        t = geometric_mean_ratio_by_target(ok, value="ise", method_col="variant",
                                           method=v, reference=a.reference,
                                           n_boot=2000, seed=0)
        frames.append(t)
    paired = pd.concat(frames, ignore_index=True)
    paired.to_csv(out / "paired_vs_reference.csv", index=False)

    # equal weight per target, within N
    agg = []
    for (v, N), g in paired.groupby(["method", "N"]):
        zbar = g["mean_logratio"].to_numpy(dtype=float)
        agg.append({"variant": v, "N": N, "n_targets": len(g),
                    "geom_mean_ratio_equal_weight": float(np.exp(np.mean(zbar))),
                    "min_target_ratio": float(np.exp(zbar.min())),
                    "max_target_ratio": float(np.exp(zbar.max())),
                    "n_targets_better": int(np.sum(zbar < 0))})
    aggdf = pd.DataFrame(agg).pivot_table(index="variant", columns="N",
                                          values="geom_mean_ratio_equal_weight")
    pd.DataFrame(agg).to_csv(out / "paired_aggregate.csv", index=False)
    print("\n== geometric-mean ISE ratio vs %s (equal weight per target; <1 is better) =="
          % a.reference)
    print(aggdf.to_string(float_format=lambda v: "%.3f" % v))
    print("\n== per-target ratios (5 paired replicates each; exploratory) ==")
    show = paired.pivot_table(index="method", columns=["target", "N"],
                              values="geom_mean_ratio")
    print(show.to_string(float_format=lambda v: "%.3f" % v))
    print("\n== per-target: how many of the 5 replicates improved ==")
    print(paired.pivot_table(index="method", columns=["target", "N"],
                             values="n_better").to_string())

    # ---------------- cost -------------------------------------------------
    cost = ok.groupby("variant").agg(
        median_total_s=("wall_time_s", "median"),
        median_tune_s=("tune_time_s", "median"),
        median_fit_s=("fit_time_s", "median"),
        median_support=("n_active", "median"),
        median_m=("n_measurements", "median")).sort_values("median_total_s")
    cost.to_csv(out / "cost.csv")
    print("\n== cost (median over all 40 datasets; wall includes the oracle path) ==")
    print(cost.to_string(float_format=lambda v: "%.3f" % v))

    # ---------------- solver / conditioning diagnostics -------------------
    diag_cols = [c for c in ("stage1_kkt", "stage2_kkt", "stage1_iters",
                             "cov_rank_numeric", "cov_effective_rank",
                             "omega_cond", "n_selected", "mass_below_meas_scale",
                             "mean_atom_sd_over_h") if c in ok.columns]
    diag = ok.groupby("variant")[diag_cols].median(numeric_only=True)
    diag["stage1_converged_rate"] = ok.groupby("variant")["stage1_converged"].apply(
        lambda s: float(pd.to_numeric(s.astype(str).map({"True": 1, "False": 0}),
                                      errors="coerce").mean()))
    diag["max_stage2_kkt"] = ok.groupby("variant")["stage2_kkt"].max()
    diag["fallback_rate"] = ok.groupby("variant")["fallback_used"].apply(
        lambda s: float(pd.to_numeric(s.astype(str).map({"True": 1, "False": 0}),
                                      errors="coerce").mean()))
    diag.to_csv(out / "diagnostics.csv")
    print("\n== solver / conditioning diagnostics (medians) ==")
    print(diag.to_string(float_format=lambda v: "%.4g" % v))

    # ---------------- tuning-grid boundary audit --------------------------
    edge = ok.groupby("variant").agg(
        rho_edge_rate=("rho_at_grid_edge", lambda s: float(
            pd.to_numeric(s.astype(str).map({"True": 1, "False": 0}),
                          errors="coerce").mean())),
        lam_edge_rate=("lam_at_grid_edge", lambda s: float(
            pd.to_numeric(s.astype(str).map({"True": 1, "False": 0}),
                          errors="coerce").mean())))
    for side in ("lo", "hi"):
        edge["rho_%s" % side] = ok.groupby("variant")["rho_edge_side"].apply(
            lambda s, side=side: float((s.astype(str) == side).mean()))
        edge["lam_%s" % side] = ok.groupby("variant")["lam_edge_side"].apply(
            lambda s, side=side: float((s.astype(str) == side).mean()))
    edge.to_csv(out / "grid_boundary.csv")
    print("\n== how often the reduced tuning grid selected an endpoint ==")
    print(edge.to_string(float_format=lambda v: "%.3f" % v))

    # ---------------- oracle tuning gap -----------------------------------
    ofiles = sorted(src.glob("*_oracle.csv"))
    if ofiles:
        od = pd.concat([pd.read_csv(f) for f in ofiles], ignore_index=True)
        od.to_csv(out / "oracle_all.csv", index=False)
        cols = ["R_full_star", "term_support", "term_excess", "R_hat",
                "oracle_tuning_gap_ratio", "identity_residual"]
        have = [c for c in cols if c in od.columns]
        summ = od.groupby(["variant", "N"])[have].median()
        summ.to_csv(out / "oracle_summary.csv")
        print("\n== oracle decomposition, medians (R_hat = R_full* + support + excess) ==")
        print(summ.to_string(float_format=lambda v: "%.4g" % v))
        share = od.copy()
        share["share_dictionary"] = share["R_full_star"] / share["R_hat"]
        share["share_support"] = share["term_support"] / share["R_hat"]
        share["share_excess"] = share["term_excess"] / share["R_hat"]
        sh = share.groupby(["variant"])[["share_dictionary", "share_support",
                                         "share_excess"]].median()
        sh.to_csv(out / "oracle_shares.csv")
        print("\n== share of total exact risk, medians ==")
        print(sh.to_string(float_format=lambda v: "%.4f" % v))
        print("\n== max |identity residual| of the decomposition: %.3e =="
              % od["identity_residual"].abs().max())

    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
