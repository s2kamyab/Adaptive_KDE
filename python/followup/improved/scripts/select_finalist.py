"""Apply the LOCKED selection rule to the development rows and freeze the finalist.

    python followup/improved/scripts/select_finalist.py

Writes followup/improved/finalist_improved.json and results/improved/dev/dev_ablation.csv.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from followup.improved.estimator import VARIANTS  # noqa: E402
from followup.improved.runner import RESULTS  # noqa: E402

N_HYPER = {  # tuned hyperparameters per variant (grid searched by CV)
    "I0_finalist": 2, "I1_gram_fista": 1, "I2_identity_fista": 1, "I3_ridge_fista": 2,
    "I4_simplex_l2": 0, "I5_gram_colnorm": 1, "I6_gram_ext1se5f": 1, "I7_simplex_cvanchor": 1,
    "I8_simplex_merge": 0, "I9_simplex_cvanchor_merge": 1, "I10_gram_colnorm_cvanchor": 2,
}
B, SEED = 10000, 20261002


def main():
    d = pd.read_csv(RESULTS / "dev" / "all_rows.csv", low_memory=False)
    d["error"] = d["error"].fillna("").astype(str)
    P = json.load(open(HERE / "protocol_improved.json"))
    mw = d[d.namespace == "development"]
    hr = d[d.namespace == "bf_dev"]
    rng = np.random.default_rng(SEED)
    rows = []
    for m in ["I0_finalist"] + list(VARIANTS):
        rec = {"variant": m, "n_hyper": N_HYPER[m]}
        sub = d[d.method == m]
        rec["n_fits"] = len(sub)
        rec["n_errors"] = int((sub.error != "").sum())
        rec["all_valid"] = bool(sub.is_valid_density.astype(bool).all()) if len(sub) else False
        s1 = sub.stage1_kkt.astype(float)
        s2 = sub.stage2_kkt.astype(float)
        if sub["mode"].fillna("two_stage").iloc[0] == "simplex_l2" if len(sub) else False:
            rec["stage1_at_tol"] = np.nan
        else:
            rec["stage1_at_tol"] = float((s1 <= 1e-8).mean())
        rec["stage2_at_tol"] = float((s2 < 1e-7).mean())
        rec["mean_wall_s"] = float(sub.wall_time_s.mean())
        rec["mean_components"] = float(sub.n_components.mean())
        for fam, frame, key in (("mw", mw, "primary_ratio"), ("hr", hr, "hr_ratio")):
            zs, cells = [], []
            for (t, N), cell in frame.groupby(["target", "N"]):
                piv = cell.pivot_table(index="rep", columns="method", values="ise", aggfunc="first")
                if m not in piv or "I0_finalist" not in piv:
                    continue
                z = np.log(piv[m].to_numpy(float) / piv["I0_finalist"].to_numpy(float))
                z = z[np.isfinite(z)]
                if z.size:
                    zs.append(z.mean()); cells.append(z)
            if zs:
                rec[key] = float(np.exp(np.mean(zs)))
                boot = np.zeros(B)
                for z in cells:
                    idx = rng.integers(0, z.size, size=(B, z.size))
                    boot += z[idx].mean(axis=1)
                boot /= len(cells)
                rec[key + "_lo"] = float(np.exp(np.percentile(boot, 2.5)))
                rec[key + "_hi"] = float(np.exp(np.percentile(boot, 97.5)))
                rec[key + "_n_cells"] = len(cells)
        for N in (500, 5000):
            s = mw[(mw.method == m) & (mw.N == N)]
            rec["mw_median_ise_N%d" % N] = float(s.ise.median()) if len(s) else np.nan
            rec["mw_mean_ise_N%d" % N] = float(s.ise.mean()) if len(s) else np.nan
            s = hr[(hr.method == m) & (hr.N == N)]
            rec["hr_median_ise_N%d" % N] = float(s.ise.median()) if len(s) else np.nan
        rows.append(rec)
    tab = pd.DataFrame(rows)
    # gates
    tab["gate_hr"] = tab.hr_ratio <= 1.0
    tab["gate_kkt"] = (tab.stage1_at_tol.fillna(1.0) >= 0.9) & (tab.stage2_at_tol >= 0.9)
    tab["gate_ok"] = tab.gate_hr & tab.gate_kkt & (tab.n_errors == 0) & tab.all_valid
    cand = tab[tab.gate_ok & (tab.variant != "I0_finalist")].copy()
    tab.to_csv(RESULTS / "dev" / "dev_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab[["variant", "n_hyper", "primary_ratio", "primary_ratio_lo", "primary_ratio_hi", "hr_ratio",
               "stage1_at_tol", "stage2_at_tol", "mean_wall_s", "mean_components", "gate_ok"]]
          .to_string(index=False, float_format=lambda v: "%.3f" % v))
    if cand.empty:
        raise SystemExit("no variant passes the gates")
    best = cand.primary_ratio.min()
    within = cand[cand.primary_ratio <= 1.10 * best].sort_values(["n_hyper", "mean_wall_s"])
    fin = within.iloc[0].variant
    ablation = "I4_simplex_l2" if fin != "I4_simplex_l2" else None
    out = {"finalist": fin, "ablation": ablation, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rule": P["selection_rule"], "config": VARIANTS[fin].as_dict(),
           "primary_ratio_vs_I0": float(within.iloc[0].primary_ratio),
           "candidates_within_10pct": within.variant.tolist(),
           "best_primary_ratio": float(best), "table": tab.to_dict(orient="records")}
    (HERE / "finalist_improved.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nFINALIST:", fin, "ablation:", ablation, "| candidates within 10%%: %s" % within.variant.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
