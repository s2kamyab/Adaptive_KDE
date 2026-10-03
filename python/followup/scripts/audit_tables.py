"""Reconstruct the V2 manuscript's Tables 2-7 from the saved per-replicate rows.

Nothing here refits anything: it reads results/main_full/all_results.csv and
results/moment_count/raw.csv and recomputes every number the manuscript
reports, so that each claimed value can be classified as reproduced, close, or
not reproduced.  Output goes to followup/results/audit_tables/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]      # .../python
sys.path.insert(0, str(ROOT))

from akde.analysis import paired_table, win_loss_summary  # noqa: E402

OUT = ROOT / "followup" / "results" / "audit_tables"
OUT.mkdir(parents=True, exist_ok=True)

MAIN = ROOT / "results" / "main_full" / "all_results.csv"
MOM = ROOT / "results" / "moment_count" / "raw.csv"

df = pd.read_csv(MAIN, low_memory=False)
df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
print(f"main_full rows: {len(df)}  estimators: {df.estimator.nunique()}  "
      f"targets: {df.target.nunique()}  N: {sorted(df.N.unique())}  reps: {df.rep.nunique()}")

report: dict = {"source": str(MAIN), "n_rows": int(len(df))}

# ---------------------------------------------------------------- Table 1/2
# tab:validity (N=1000) and tab:main-ise
val = df[df.N == 1000].groupby("estimator").agg(
    n=("ise", "size"),
    frac_valid=("is_valid_density", lambda s: float(s.astype(str).isin(["True", "1", "1.0"]).mean())),
    median_mass=("total_mass", "median"),
    worst_mass_err=("mass_error", "max"),
).reset_index()
val.to_csv(OUT / "recon_tab_validity_N1000.csv", index=False)
print("\n== tab:validity  (N=1000) ==\n", val.to_string(index=False))

# AKDE-only validity across the WHOLE grid, and the correct denominator
akde = df[df.estimator == "akde"]
report["akde_fits_total"] = int(len(akde))
report["all_fits_total"] = int(len(df))
report["akde_frac_valid"] = float(akde["is_valid_density"].astype(str).isin(["True"]).mean())
report["akde_worst_mass_error"] = float(akde["mass_error"].max())
report["akde_fallback_count"] = int(pd.to_numeric(akde["fallback_used"], errors="coerce").fillna(0).sum())
report["all_frac_valid"] = float(df["is_valid_density"].astype(str).isin(["True"]).mean())

pivot_ise = df.pivot_table(index="estimator", columns="N", values="ise", aggfunc="median")
pivot_ise.to_csv(OUT / "recon_tab_main_ise.csv")
print("\n== tab:main-ise  median ISE pooled over 15 densities ==\n",
      pivot_ise.to_string(float_format=lambda v: f"{v:.5g}"))

pivot_mean = df.pivot_table(index="estimator", columns="N", values="ise", aggfunc="mean")
pivot_mean.to_csv(OUT / "recon_main_ise_mean.csv")

# ---------------------------------------------------------------- Table 5
# log-log slopes of median ISE in N, small (50-200) and large (500-5000)
def slope(sub: pd.Series, sizes) -> float:
    s = sub.reindex(sizes).dropna()
    if len(s) < 2:
        return np.nan
    return float(np.polyfit(np.log(s.index.values.astype(float)), np.log(s.values), 1)[0])

slopes = pd.DataFrame({
    "slope_50_200": {e: slope(pivot_ise.loc[e], [50, 100, 200]) for e in pivot_ise.index},
    "slope_500_5000": {e: slope(pivot_ise.loc[e], [500, 1000, 5000]) for e in pivot_ise.index},
})
slopes.to_csv(OUT / "recon_tab_slopes.csv")
print("\n== tab:slopes ==\n", slopes.round(3).to_string())

# ---------------------------------------------------------------- Table 7
cost = df.groupby(["estimator", "N"]).agg(
    fit_time_median=("total_time_s", "median"),
    wall_median=("wall_time_s", "median"),
    support_median=("n_active", "median"),
).reset_index()
cost.to_csv(OUT / "recon_tab_cost.csv", index=False)
c = cost[cost.N.isin([500, 1000, 5000])]
print("\n== tab:cost (median total_time_s / n_active) ==\n",
      c.pivot_table(index="estimator", columns="N",
                    values=["fit_time_median", "support_median"]).round(3).to_string())

# ---------------------------------------------------------------- Table 6
mom = pd.read_csv(MOM, low_memory=False)
print("\n== moment_count raw columns ==", list(mom.columns)[:20])
print(mom.groupby(["N", "m"]).ise.agg(["median", "count"]).to_string()
      if "m" in mom.columns else mom.head().to_string())
mom.to_csv(OUT / "recon_moment_count_copy.csv", index=False)

# ---------------------------------------------------------------- Tables 3/4
# Holm family as actually implemented: within (estimator, N) across 15 densities
paired_mean = paired_table(df, reference="akde", statistic="mean", n_boot=2000, seed=0)
paired_mean.to_csv(OUT / "recon_paired_mean.csv", index=False)
wl = win_loss_summary(paired_mean)
wl.to_csv(OUT / "recon_winloss_by_baseline_N.csv", index=False)
byN = wl.groupby("N")[["reference_wins", "reference_losses", "ties"]].sum()
byN["n_comparisons"] = wl.groupby("N")["n_comparisons"].sum()
byN.to_csv(OUT / "recon_tab_winloss.csv")
print("\n== tab:winloss  (Holm within (baseline,N) over 15 densities) ==\n", byN.to_string())

ratio = paired_mean.pivot_table(index="estimator", columns="N", values="ratio", aggfunc="median")
ratio.to_csv(OUT / "recon_tab_ratio_medianOfPerDensity.csv")
print("\n== tab:ratio  (median over densities of the paired geometric-mean ratio) ==\n",
      ratio.round(3).to_string())

# pooled geometric-mean ratio across all density x rep pairs, for comparison
rows = []
for (n,), g in df.groupby(["N"]):
    p = g.pivot_table(index=["target", "rep"], columns="estimator", values="ise", aggfunc="first")
    if "akde" not in p:
        continue
    for est in p.columns:
        if est == "akde":
            continue
        ok = (p["akde"] > 0) & (p[est] > 0)
        rows.append({"N": n, "estimator": est,
                     "pooled_geom_ratio": float(np.exp(np.mean(np.log(p.loc[ok, "akde"] / p.loc[ok, est])))),
                     "n_pairs": int(ok.sum())})
pooled = pd.DataFrame(rows).pivot_table(index="estimator", columns="N", values="pooled_geom_ratio")
pooled.to_csv(OUT / "recon_tab_ratio_pooled.csv")
print("\n== pooled geometric-mean ratio (all density x rep pairs) ==\n", pooled.round(3).to_string())

# ---------------------------------------------------------------- diagnostics
diag_cols = ["cov_rank_numeric", "cov_effective_rank", "omega_cond", "n_measurements",
             "n_dictionary", "n_selected", "stage1_converged", "stage1_kkt", "stage2_kkt",
             "sel_rho", "sel_lam_ratio", "fallback_used", "support_capped"]
have = [c for c in diag_cols if c in akde.columns]
d = akde.groupby("N")[have].median(numeric_only=True)
d["stage1_conv_rate"] = akde.groupby("N")["stage1_converged"].apply(
    lambda s: float(s.astype(str).isin(["True"]).mean()))
d["max_stage1_kkt"] = akde.groupby("N")["stage1_kkt"].max()
d["max_stage2_kkt"] = akde.groupby("N")["stage2_kkt"].max()
d.to_csv(OUT / "akde_diagnostics_by_N.csv")
print("\n== AKDE diagnostics by N ==\n", d.to_string())

# selected dictionary size distribution per method (audit Q1)
for est in ["akde", "spades", "spades_simplex", "song"]:
    sub = df[df.estimator == est]
    if "n_dictionary" in sub:
        print(f"\nn_dictionary distribution, {est}:")
        print(sub.groupby("N")["n_dictionary"].value_counts().unstack(fill_value=0).to_string())

with open(OUT / "audit_numbers.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print("\nwrote", OUT)
