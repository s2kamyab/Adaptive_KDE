"""Statistics for the improved-AKDE confirmation.  No refitting.

    python followup/improved/scripts/analyse.py

Combines the new confirmation rows with the saved Stage-C comparator rows
(identical datasets) and the new hr rows with the saved bandwidth-flexibility
rows, then writes results/improved/analysis/:

    tidy_confirm.csv        all methods x 1350 datasets (7 or 8 methods)
    per_cell_summary.csv    mean +- SD (ddof=1), median per (target, N, method)
    benchmark_means.csv     equal-weight-per-target mean ISE [bootstrap CI]
    pooled_mean_sd.csv      mean +- pooled SD over the 450 fits per (method, N)
    fixed_benchmark.csv     exp(T) paired ratios vs AKDE-M and vs each comparator
    per_target_ratios.csv   per-(target,N) paired geometric-mean ratios
    reliability.csv         errors, validity, convergence, large-error counts
    hr_per_cell.csv, hr_ratios.csv   the hr family
    bootstrap_config.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from akde.densities import MARRON_WAND_MAPPING  # noqa: E402
from followup.improved import PY_ROOT  # noqa: E402
from followup.improved.runner import RESULTS  # noqa: E402

OUT = RESULTS / "analysis"
B, SEED = 10000, 20261002
STAGE_C = PY_ROOT / "followup" / "results" / "stage_c" / "all_rows.csv"
BF_MAIN = PY_ROOT.parents[1] / "final_package" / "experiments" / "bandwidth_flexibility" / "results" / "main" / "all_rows.csv"
BASE = {"finalist": "akde_m", "v2_akde": "akde_v2", "spades_simplex": "spades_simplex", "kde_sj": "kde_sj",
        "abramson": "abramson", "gmm_em": "gmm_em"}
BF_BASE = {"multiscale_akde": "akde_m", "single_scale_akde": "akde_single", "spades_simplex": "spades_simplex",
           "kde_sj": "kde_sj", "abramson": "abramson", "gmm_em": "gmm_em"}


def sd1(v):
    return float(np.std(np.asarray(v, float), ddof=1)) if len(v) > 1 else np.nan


def load_confirm():
    new = pd.read_csv(RESULTS / "confirm" / "all_rows.csv", low_memory=False)
    new["error"] = new["error"].fillna("").astype(str)
    old = pd.read_csv(STAGE_C, low_memory=False)
    old["error"] = old["error"].fillna("").astype(str)
    old = old.rename(columns={"variant": "method", "n_active": "n_active_rel1e-8"})
    old["method"] = old["method"].map(BASE)
    old["n_components"] = old["n_active_rel1e-8"]
    cols = ["target", "N", "rep", "method", "ise", "wall_time_s", "tune_time_s", "fit_time_s", "n_components",
            "n_active_rel1e-8", "is_valid_density", "mass_error", "nll", "stage1_converged", "stage1_kkt",
            "stage2_kkt", "error", "lam_at_grid_edge"]
    for c in cols:
        for f in (new, old):
            if c not in f:
                f[c] = np.nan
    t = pd.concat([new[cols].assign(source="improved"), old[cols].assign(source="stage_c")], ignore_index=True)
    t["mw"] = t.target.map(lambda s: MARRON_WAND_MAPPING[s][0])
    t["mw_name"] = t.target.map(lambda s: MARRON_WAND_MAPPING[s][1])
    t["valid"] = (t.error == "") & np.isfinite(t.ise) & (t.ise > 0)
    med = t[t.valid].groupby(["method", "target", "N"]).ise.median().rename("cell_median_ise")
    t = t.merge(med, on=["method", "target", "N"], how="left")
    t["large_error"] = t.valid & (t.ise > 5 * t.cell_median_ise)
    return t


def per_cell(t, extra=()):
    rows = []
    for (tg, N, m), s in t.groupby(["target", "N", "method"]):
        ok = s[s.valid]
        r = {"target": tg, "N": N, "method": m, "n_attempted": len(s), "n_valid": int(s.valid.sum()),
             "n_large_error": int(s.large_error.sum()), "mean_ise": ok.ise.mean(), "sd_ise_ddof1": sd1(ok.ise),
             "median_ise": ok.ise.median(), "mean_components": ok.n_components.astype(float).mean(),
             "sd_components": sd1(ok.n_components.astype(float)), "mean_wall_s": ok.wall_time_s.astype(float).mean(),
             "sd_wall_s": sd1(ok.wall_time_s.astype(float))}
        if "mw" in s:
            r["mw"] = int(s.mw.iloc[0])
        for c in extra:
            if c in ok:
                r["mean_" + c] = ok[c].astype(float).mean(); r["sd_" + c] = sd1(ok[c].astype(float))
        rows.append(r)
    return pd.DataFrame(rows)


def cell_matrix(t, tg, N):
    return t[(t.target == tg) & (t.N == N) & t.valid].pivot_table(index="rep", columns="method", values="ise", aggfunc="first")


def fixed_benchmark(t, methods, references, rng, targets_order):
    rows, per = [], []
    for N in sorted(t.N.unique()):
        mats = {tg: cell_matrix(t, tg, N) for tg in targets_order}
        n_reps = min(p.shape[0] for p in mats.values())
        idx = rng.integers(0, n_reps, size=(B, n_reps))
        for m in methods:
            for ref in references:
                if m == ref:
                    continue
                pt, boot = [], np.zeros(B)
                for tg in targets_order:
                    p = mats[tg]
                    if m not in p or ref not in p:
                        continue
                    z = np.log(p[m].to_numpy(float) / p[ref].to_numpy(float))
                    z = np.where(np.isfinite(z), z, np.nan)[:n_reps]
                    pt.append(np.nanmean(z)); boot += np.nanmean(z[idx], axis=1)
                    zz = z[np.isfinite(z)]
                    bi = rng.integers(0, zz.size, size=(B, zz.size))
                    bz = zz[bi].mean(axis=1)
                    per.append({"N": N, "target": tg, "mw": MARRON_WAND_MAPPING[tg][0], "method": m, "reference": ref,
                                "n_pairs": int(zz.size), "geom_mean_ratio": float(np.exp(zz.mean())),
                                "boot_ci_lo": float(np.exp(np.percentile(bz, 2.5))),
                                "boot_ci_hi": float(np.exp(np.percentile(bz, 97.5))),
                                "n_pairs_favouring_method": int((zz < 0).sum())})
                if not pt:
                    continue
                boot /= len(pt)
                pt = np.array(pt)
                rows.append({"N": N, "method": m, "reference": ref, "n_targets": len(pt),
                             "exp_T": float(np.exp(pt.mean())), "boot_ci_lo": float(np.exp(np.percentile(boot, 2.5))),
                             "boot_ci_hi": float(np.exp(np.percentile(boot, 97.5))),
                             "n_targets_favouring_method": int((pt < 0).sum()),
                             "min_per_target_ratio": float(np.exp(pt.min())), "max_per_target_ratio": float(np.exp(pt.max()))})
    return pd.DataFrame(rows), pd.DataFrame(per)


def benchmark_means(t, rng, targets_order):
    rows = []
    for N in sorted(t.N.unique()):
        mats = {tg: cell_matrix(t, tg, N) for tg in targets_order}
        n_reps = min(p.shape[0] for p in mats.values())
        idx = rng.integers(0, n_reps, size=(B, n_reps))
        for m in sorted(t.method.unique()):
            per, boot = [], np.zeros(B)
            for tg in targets_order:
                p = mats[tg]
                if m not in p:
                    continue
                v = p[m].to_numpy(float)[:n_reps]
                per.append(np.nanmean(v)); boot += np.nanmean(v[idx], axis=1)
            boot /= len(per)
            rows.append({"N": N, "method": m, "benchmark_mean_ise": float(np.mean(per)),
                         "boot_ci_lo": float(np.percentile(boot, 2.5)), "boot_ci_hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows)


def pooled(t):
    rows = []
    for (m, N), s in t[t.valid].groupby(["method", "N"]):
        rows.append({"method": m, "N": N, "n_fits": len(s), "mean_ise": s.ise.mean(), "pooled_sd_ddof1": sd1(s.ise),
                     "median_ise": s.ise.median(), "mean_components": s.n_components.astype(float).mean(),
                     "sd_components": sd1(s.n_components.astype(float)), "mean_wall_s": s.wall_time_s.astype(float).mean(),
                     "sd_wall_s": sd1(s.wall_time_s.astype(float))})
    return pd.DataFrame(rows)


def reliability(t):
    rows = []
    for m, s in t.groupby("method"):
        ok = s[s.valid]
        r = {"method": m, "n_attempted": len(s), "n_errors": int((s.error != "").sum()), "n_valid": int(s.valid.sum()),
             "n_valid_density": int(ok.is_valid_density.astype(bool).sum()), "n_large_error": int(s.large_error.sum()),
             "frac_large_error": float(s.large_error.mean()), "worst_ise_over_cell_median": float((s.ise / s.cell_median_ise).max())}
        for c in ("stage1_kkt", "stage2_kkt"):
            k = ok[c].astype(float).dropna()
            if len(k):
                r[c + "_median"] = float(k.median()); r[c + "_max"] = float(k.max())
                r[c + "_frac_at_tol"] = float((k <= (1e-8 if c == "stage1_kkt" else 1e-7)).mean())
        if "stage1_converged" in ok and ok.stage1_converged.notna().any():
            r["stage1_converged_frac"] = float(ok.stage1_converged.dropna().astype(bool).mean())
        if "lam_at_grid_edge" in ok and ok.lam_at_grid_edge.notna().any():
            r["lam_at_grid_edge_frac"] = float(ok.lam_at_grid_edge.dropna().astype(bool).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def hr_analysis(rng):
    p = RESULTS / "hr" / "all_rows.csv"
    if not p.exists() or not BF_MAIN.exists():
        return None, None
    new = pd.read_csv(p, low_memory=False); new["error"] = new["error"].fillna("").astype(str)
    old = pd.read_csv(BF_MAIN, low_memory=False); old["error"] = old["error"].fillna("").astype(str)
    old["method"] = old["method"].map(BF_BASE)
    cols = ["target", "r", "N", "rep", "method", "ise", "ise_narrow", "n_components", "wall_time_s", "is_valid_density", "error"]
    t = pd.concat([new[cols], old[cols]], ignore_index=True)
    t["valid"] = (t.error == "") & np.isfinite(t.ise) & (t.ise > 0)
    med = t[t.valid].groupby(["method", "target", "N"]).ise.median().rename("cell_median_ise")
    t = t.merge(med, on=["method", "target", "N"], how="left")
    t["large_error"] = t.valid & (t.ise > 5 * t.cell_median_ise)
    pc = per_cell(t, extra=("ise_narrow",))
    pc["r"] = pc.target.map(lambda s: int(s[2:]))
    rows = []
    new_methods = sorted(new.method.unique())
    for (tg, N), s in t.groupby(["target", "N"]):
        for col in ("ise", "ise_narrow"):
            piv = s[s.valid].pivot_table(index="rep", columns="method", values=col, aggfunc="first")
            for m in new_methods:
                for ref in ("akde_m", "akde_single", "spades_simplex", "kde_sj", "abramson", "gmm_em"):
                    if m not in piv or ref not in piv:
                        continue
                    z = np.log(piv[m].to_numpy(float) / piv[ref].to_numpy(float)); z = z[np.isfinite(z)]
                    bi = rng.integers(0, z.size, size=(B, z.size)); bz = z[bi].mean(axis=1)
                    rows.append({"target": tg, "r": int(tg[2:]), "N": N, "metric": col, "method": m, "reference": ref,
                                 "n_pairs": int(z.size), "geom_mean_ratio": float(np.exp(z.mean())),
                                 "boot_ci_lo": float(np.exp(np.percentile(bz, 2.5))),
                                 "boot_ci_hi": float(np.exp(np.percentile(bz, 97.5))),
                                 "n_pairs_favouring_method": int((z < 0).sum())})
    t.to_csv(OUT / "tidy_hr.csv", index=False)
    return pc, pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t = load_confirm()
    t.to_csv(OUT / "tidy_confirm.csv", index=False)
    new_methods = sorted(t[t.source == "improved"].method.unique())
    print("rows %d; new methods %s; valid %d" % (len(t), new_methods, int(t.valid.sum())))
    targets = sorted(t.target.unique(), key=lambda s: MARRON_WAND_MAPPING[s][0])
    pc = per_cell(t); pc.to_csv(OUT / "per_cell_summary.csv", index=False)
    bm = benchmark_means(t, rng, targets); bm.to_csv(OUT / "benchmark_means.csv", index=False)
    po = pooled(t); po.to_csv(OUT / "pooled_mean_sd.csv", index=False)
    fb, per = fixed_benchmark(t, new_methods + ["spades_simplex", "kde_sj"], ["akde_m", "spades_simplex", "kde_sj", "abramson", "gmm_em"], rng, targets)
    fb.to_csv(OUT / "fixed_benchmark.csv", index=False); per.to_csv(OUT / "per_target_ratios.csv", index=False)
    rel = reliability(t); rel.to_csv(OUT / "reliability.csv", index=False)
    hpc, hr_r = hr_analysis(rng)
    if hpc is not None:
        hpc.to_csv(OUT / "hr_per_cell.csv", index=False); hr_r.to_csv(OUT / "hr_ratios.csv", index=False)
    json.dump({"B": B, "seed": SEED, "unit": "replicate id within (target, N) cell, all methods carried together",
               "targets_never_resampled": True, "interval": "percentile 95%",
               "comparators": "Stage-C rows (confirmation namespace) and bandwidth-flexibility main rows reused; "
                              "datasets identical by seed, verified by dataset sha256 where stored"},
              open(OUT / "bootstrap_config.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print("\n== benchmark mean ISE (equal weight per target) ==")
    print(bm.pivot_table(index="method", columns="N", values="benchmark_mean_ise").to_string(float_format=lambda v: "%.4g" % v))
    print("\n== exp(T) paired ratios ==")
    print(fb.pivot_table(index=["method", "reference"], columns="N", values="exp_T").to_string(float_format=lambda v: "%.3f" % v))
    print("\n== reliability ==")
    print(rel.to_string(index=False))
    if hr_r is not None:
        print("\n== hr ratios (total ISE) ==")
        print(hr_r[hr_r.metric == "ise"].pivot_table(index=["method", "reference"], columns=["N", "r"], values="geom_mean_ratio").to_string(float_format=lambda v: "%.2f" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
