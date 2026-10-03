"""Choose R for the confirmation, before any confirmation result is opened.

Rule (fixed in protocol_followup_locked.yaml):
  target = a 95% half-width of log(1.25) = 0.2231 for the mean paired log ISE
  ratio z = log(ISE_finalist / ISE_v2) in the primary comparison, per
  (target, N) cell.  Half-width is approximated by t_{0.975,R-1} * sd / sqrt(R).

The SD of z is estimated two ways and both are reported:
  * from the Stage-A/B development replicates for the candidate finalist;
  * from the archived V2 benchmark, as the SD of the paired log ratio between
    two comparable dictionary-based methods (spades_simplex vs akde) at 30
    replicates, which is the only 30-replicate estimate available.

R is then chosen from {10, 20, 30}: the smallest value meeting the target in
a stated fraction of cells; if none does, the budgeted cap is kept and the
shortfall is reported as a limitation.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
from scipy import stats                                      # noqa: E402

RES = ROOT / "followup" / "results"
OUT = RES / "power"
TARGET_HALFWIDTH = float(np.log(1.25))
CANDIDATES = (10, 20, 30)


def sd_from_dev(finalist_variant: str, ref="A1_v2_moment_m300"):
    a = pd.read_csv(RES / "stage_a" / "all_rows.csv", low_memory=False)
    frames = [a]
    for extra in ("stage_b", "stage_b_extra"):
        p = RES / extra / "all_rows.csv"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for (t, N), g in df.groupby(["target", "N"]):
        piv = g.pivot_table(index="rep", columns="variant", values="ise",
                            aggfunc="first")
        if finalist_variant not in piv or ref not in piv:
            continue
        d = piv[[finalist_variant, ref]].dropna()
        z = np.log(d[finalist_variant].to_numpy() / d[ref].to_numpy())
        rows.append({"target": t, "N": N, "R_dev": int(z.size),
                     "mean_z": float(z.mean()),
                     "sd_z": float(np.std(z, ddof=1))})
    return pd.DataFrame(rows)


def sd_from_v2_archive():
    p = ROOT / "results" / "main_full" / "all_results.csv"
    df = pd.read_csv(p, low_memory=False)
    df = df[df["error"].fillna("").astype(str) == ""]
    rows = []
    for (t, N), g in df[df.N.isin([200, 1000, 5000])].groupby(["target", "N"]):
        piv = g.pivot_table(index="rep", columns="estimator", values="ise",
                            aggfunc="first")
        if "spades_simplex" not in piv or "akde" not in piv:
            continue
        d = piv[["spades_simplex", "akde"]].dropna()
        ok = (d > 0).all(axis=1)
        z = np.log(d.loc[ok, "spades_simplex"].to_numpy() / d.loc[ok, "akde"].to_numpy())
        rows.append({"target": t, "N": N, "R_archive": int(z.size),
                     "sd_z": float(np.std(z, ddof=1))})
    return pd.DataFrame(rows)


def halfwidth(sd, R):
    return stats.t.ppf(0.975, R - 1) * sd / np.sqrt(R)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    finalist = json.load(open(RES / "finalist.json", encoding="utf-8"))["variant"]

    dev = sd_from_dev(finalist)
    arc = sd_from_v2_archive()
    dev.to_csv(OUT / "sd_from_development.csv", index=False)
    arc.to_csv(OUT / "sd_from_v2_archive.csv", index=False)

    sd_dev = dev.sd_z.to_numpy()
    sd_arc = arc.sd_z.to_numpy()
    lines = []
    lines.append("# Replication count for the Stage-C confirmation\n")
    lines.append("Decided %s, before any confirmation seed was drawn.\n"
                 % datetime.now(timezone.utc).isoformat(timespec="seconds"))
    lines.append("Primary outcome: z = log(ISE_finalist / ISE_v2_akde), paired "
                 "within replicate.\n")
    lines.append("Precision target: 95%% half-width <= log(1.25) = %.4f for the "
                 "mean of z, per (target, N) cell.\n" % TARGET_HALFWIDTH)

    lines.append("\n## SD(z) estimates\n")
    lines.append("| source | cells | median SD | 75th pct | 90th pct | max |")
    lines.append("|---|---|---|---|---|---|")
    for nm, sd, n in (("Stage A+B development (%s vs V2 AKDE), R=5" % finalist,
                       sd_dev, len(dev)),
                      ("V2 archive (spades_simplex vs akde), R=30, N in {200,1000,5000}",
                       sd_arc, len(arc))):
        if len(sd) == 0:
            continue
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f |"
                     % (nm, n, np.median(sd), np.percentile(sd, 75),
                        np.percentile(sd, 90), sd.max()))

    lines.append("\nThe development estimate uses only 5 replicates per cell, so "
                 "each SD is itself very uncertain (a chi distribution with 4 "
                 "degrees of freedom: the 95% interval for a single such SD "
                 "spans roughly 0.6x to 2.2x the truth). The archive estimate is "
                 "from 30 replicates and a *different* pair of methods, so it "
                 "describes the general scale of paired log-ratio variability in "
                 "this benchmark rather than this specific contrast. Both are "
                 "reported; neither is treated as exact.\n")

    lines.append("\n## Achievable half-width\n")
    lines.append("| R | half-width at median SD (dev) | at 90th pct (dev) | "
                 "at median SD (archive) | at 90th pct (archive) | "
                 "cells meeting target (dev) | cells meeting target (archive) |")
    lines.append("|---|---|---|---|---|---|---|")
    table = []
    for R in CANDIDATES:
        hd_med = halfwidth(np.median(sd_dev), R) if len(sd_dev) else np.nan
        hd_90 = halfwidth(np.percentile(sd_dev, 90), R) if len(sd_dev) else np.nan
        ha_med = halfwidth(np.median(sd_arc), R) if len(sd_arc) else np.nan
        ha_90 = halfwidth(np.percentile(sd_arc, 90), R) if len(sd_arc) else np.nan
        fd = float(np.mean(halfwidth(sd_dev, R) <= TARGET_HALFWIDTH)) if len(sd_dev) else np.nan
        fa = float(np.mean(halfwidth(sd_arc, R) <= TARGET_HALFWIDTH)) if len(sd_arc) else np.nan
        table.append({"R": R, "hw_dev_median": hd_med, "hw_dev_p90": hd_90,
                      "hw_archive_median": ha_med, "hw_archive_p90": ha_90,
                      "frac_cells_dev": fd, "frac_cells_archive": fa})
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | %.0f%% | %.0f%% |"
                     % (R, hd_med, hd_90, ha_med, ha_90, 100 * fd, 100 * fa))
    pd.DataFrame(table).to_csv(OUT / "halfwidth_by_R.csv", index=False)

    ok = [r for r in table if r["frac_cells_dev"] >= 0.8 and
          r["frac_cells_archive"] >= 0.8]
    chosen = ok[0]["R"] if ok else max(CANDIDATES)
    met = bool(ok)
    lines.append("\n## Decision\n")
    if met:
        lines.append("**R = %d.** It is the smallest candidate meeting the "
                     "log(1.25) half-width target in at least 80%% of cells under "
                     "both SD estimates.\n" % chosen)
    else:
        lines.append("**R = %d (the cap).** No candidate in {10, 20, 30} meets the "
                     "log(1.25) half-width target in 80%% of cells under both SD "
                     "estimates, so the budgeted cap is kept and the shortfall is "
                     "reported as a limitation: per-cell intervals for the mean "
                     "log ratio will be wider than log(1.25) in a substantial "
                     "fraction of cells, and per-cell conclusions are "
                     "correspondingly weak. The benchmark-wide summary, which "
                     "averages one number per target over 15 targets, is much "
                     "better resolved than any single cell.\n" % chosen)
    lines.append("\nR is fixed here and not revisited. Replicates are not added "
                 "until significance is reached, and the run is not stopped early "
                 "because one method wins. No sequential procedure is specified, "
                 "so none is used.\n")

    (OUT / "replication_choice.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump({"chosen_R": chosen, "target_met": met,
               "target_halfwidth_log": TARGET_HALFWIDTH,
               "decided_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
              open(OUT / "replication_choice.json", "w", encoding="utf-8"), indent=2)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
