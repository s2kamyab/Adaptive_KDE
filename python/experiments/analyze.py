"""Bootstrap analysis and table generation.

    python experiments/analyze.py --run results/main_full
    python experiments/analyze.py --run results/main_full --reference akde --boot 10000

Writes, into ``<run>/analysis/``:

  accuracy_ise.csv        per (density, N, estimator): mean and median ISE with BCa CIs
  paired_vs_<ref>.csv     paired differences and error ratios, Holm-adjusted
  winloss.csv             significant wins / losses per baseline
  cost.csv                runtime and representation size
  diagnostics.csv         validity, solver and conditioning diagnostics
  tables.tex              LaTeX tables ready to include in the manuscript
  summary.md              a readable summary of what the numbers say
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akde import analysis
from akde.densities import MARRON_WAND_MAPPING


def load(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "all_results.csv"
    if p.exists():
        return pd.read_csv(p)
    shards = sorted(run_dir.glob("*_N*.csv"))
    if not shards:
        raise SystemExit(f"no results found in {run_dir}")
    return pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", default="akde")
    ap.add_argument("--metric", default="ise")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run)
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load(run_dir)
    ok = df["error"].isna() | (df["error"].astype(str).str.len() == 0)
    n_fail = int((~ok).sum())
    df = df[ok]
    print(f"loaded {len(df):,} successful rows ({n_fail} failures) from {run_dir}")

    print("bootstrapping accuracy ...")
    acc = analysis.accuracy_table(df, metric=args.metric, n_boot=args.boot, seed=args.seed)
    acc.to_csv(out / f"accuracy_{args.metric}.csv", index=False)

    print("bootstrapping paired differences ...")
    paired = analysis.paired_table(df, reference=args.reference, metric=args.metric,
                                   statistic="mean", n_boot=args.boot, seed=args.seed)
    paired.to_csv(out / f"paired_vs_{args.reference}.csv", index=False)

    paired_med = analysis.paired_table(df, reference=args.reference, metric=args.metric,
                                       statistic="median", n_boot=args.boot, seed=args.seed)
    paired_med.to_csv(out / f"paired_median_vs_{args.reference}.csv", index=False)

    wl = analysis.win_loss_summary(paired, alpha=args.alpha)
    wl.to_csv(out / "winloss.csv", index=False)

    cost = analysis.cost_table(df, seed=args.seed)
    cost.to_csv(out / "cost.csv", index=False)

    diag = analysis.diagnostics_table(df)
    diag.to_csv(out / "diagnostics.csv", index=False)

    # ---------------- LaTeX ----------------
    tex_parts = []
    for n in sorted(df["N"].unique()):
        sub = acc[acc["N"] == n].copy()
        pivot = sub.pivot_table(index="target", columns="estimator", values="mean")
        pivot = pivot.reindex([f"f{i}" for i in range(15)]).dropna(how="all")
        if pivot.empty:
            continue
        pivot.insert(0, "Density", [
            f"$f_{{{t[1:]}}}$ (MW{MARRON_WAND_MAPPING[t][0]})" if t in MARRON_WAND_MAPPING else t
            for t in pivot.index
        ])
        tex_parts.append(analysis.to_latex(
            pivot.reset_index(drop=True),
            columns=list(pivot.columns),
            headers=["Density"] + [c.replace("_", r"\_") for c in pivot.columns[1:]],
            caption=(f"Mean integrated squared error over replicate datasets at $N={n}$, "
                     f"computed analytically from Gaussian overlaps. "
                     f"Bootstrap intervals are in the accompanying CSV."),
            label=f"tab:ise-N{n}",
        ))
    (out / "tables.tex").write_text("\n\n".join(tex_parts), encoding="utf-8")

    # ---------------- summary ----------------
    lines = [f"# Analysis: {run_dir.name}", ""]
    lines.append(f"- successful fits: {len(df):,}  (failures: {n_fail})")
    lines.append(f"- densities: {df['target'].nunique()}, "
                 f"sample sizes: {sorted(df['N'].unique())}, "
                 f"replicates per cell: {df.groupby(['target','N','estimator']).size().median():.0f}")
    lines.append(f"- bootstrap resamples: {args.boot:,} (BCa, resampling replicate datasets)")
    lines.append("")

    lines.append("## Overall accuracy (ISE), pooled over densities")
    lines.append("")
    lines.append("| estimator | N | median ISE | mean ISE | valid density |")
    lines.append("|---|---|---|---|---|")
    for (est, n), g in df.groupby(["estimator", "N"]):
        valid = g["is_valid_density"].astype(float).mean() if "is_valid_density" in g else np.nan
        lines.append(f"| {est} | {n} | {g['ise'].median():.4g} | {g['ise'].mean():.4g} | "
                     f"{valid:.0%} |")
    lines.append("")

    if not wl.empty:
        lines.append(f"## Paired comparison against `{args.reference}` "
                     f"(Holm-adjusted, alpha={args.alpha})")
        lines.append("")
        lines.append(f"A 'win' means `{args.reference}` had significantly lower ISE on that "
                     "density; comparisons are paired on identical datasets.")
        lines.append("")
        lines.append("| baseline | N | wins | losses | ties | median error ratio |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in wl.sort_values(["N", "estimator"]).iterrows():
            lines.append(f"| {r['estimator']} | {r.get('N','')} | {r['reference_wins']} | "
                         f"{r['reference_losses']} | {r['ties']} | {r.get('median_ratio', float('nan')):.3g} |")
        lines.append("")
        lines.append(f"'median error ratio' is the paired geometric mean of "
                     f"ISE({args.reference}) / ISE(baseline): below 1 favours "
                     f"{args.reference}.")
        lines.append("")

    if not diag.empty:
        lines.append("## Validity and solver diagnostics")
        lines.append("")
        keep = [c for c in ("estimator", "N", "frac_valid_density", "max_mass_error",
                            "mean_negative_mass", "fallback_rate", "max_stage1_kkt",
                            "max_stage2_kkt", "median_cov_rank", "median_omega_cond")
                if c in diag.columns]
        lines.append(diag[keep].to_markdown(index=False, floatfmt=".4g"))
        lines.append("")

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote analysis to {out}")
    print((out / "summary.md").read_text(encoding="utf-8")[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
