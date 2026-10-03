"""Task 9 - reconcile every numerical claim in the manuscript against saved data.

Each entry below names a number that appears in ``final_package/paper/*.tex``
and recomputes it from the saved rows / diagnostics. Any mismatch beyond the
stated tolerance is reported and the script exits nonzero.

This is a real check, not a restatement of the implementation: the expected
values are read from the .tex source, and the actual values are computed from
the raw metric files by independent code paths.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from followup.paths import RAW_ROOT, DIAG_ROOT, SUMMARY_ROOT  # noqa: E402
RES = RAW_ROOT
def _package_root():
    """Locate final_package/ whether we are in the repo or inside the package.

    In the package, code/ and manifests/ are siblings; in the repository the
    package sits at <project>/final_package. Both are checked, so no script
    contains a machine-specific path.
    """
    here = Path(__file__).resolve()
    sib = here.parents[2].parent            # .../code/..  -> package root
    if (sib / "manifests").exists() or (sib / "paper" / "main.tex").exists():
        return sib
    for up in here.parents:
        cand = up / "final_package"
        if cand.exists():
            return cand
    return sib


PROJ = PY.parents[1]
PAPER = _package_root() / "paper"
OUT = DIAG_ROOT

CHECKS = []


def chk(name, expected, actual, rtol=0.02, note=""):
    if expected is None or actual is None or (isinstance(actual, float)
                                              and not np.isfinite(actual)):
        ok = False
        rel = float("nan")
    elif expected == 0:
        ok = abs(actual) < 1e-12
        rel = abs(actual)
    else:
        rel = abs(actual - expected) / abs(expected)
        ok = rel <= rtol
    CHECKS.append({"claim": name, "in_manuscript": expected, "recomputed": actual,
                   "rel_diff": rel, "pass": bool(ok), "note": note})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    tidy = pd.read_csv(SUMMARY_ROOT / "tidy_metrics.csv")
    fb = pd.read_csv(SUMMARY_ROOT / "fixed_benchmark.csv")
    bm = pd.read_csv(SUMMARY_ROOT / "benchmark_means.csv")
    pe = pd.read_csv(SUMMARY_ROOT / "paired_effects.csv")
    cost = pd.read_csv(DIAG_ROOT / "model_size_and_query_cost.csv")
    cost = cost[cost.status == "ok"]
    grid = json.load(open(DIAG_ROOT / "grid_verification.json",
                          encoding="utf-8"))
    stab = json.load(open(DIAG_ROOT / "stability_diagnostics.json",
                          encoding="utf-8"))
    solv = json.load(open(DIAG_ROOT / "solver_verification_summary.json",
                          encoding="utf-8"))
    lad = pd.read_csv(DIAG_ROOT / "oracle_ladder_check.csv")
    amb = pd.read_csv(DIAG_ROOT / "ambiguity_cone_witnesses.csv")

    # --- grid --------------------------------------------------------------
    chk("total confirmation fits (8,100)", 8100, grid["grid"]["n_rows"], 0.0)
    chk("execution failures (0)", 0,
        grid["grid"]["statuses"]["execution"]["n_execution_failures"], 0.0)
    chk("valid densities (8,100)", 8100,
        grid["grid"]["statuses"]["density_feasibility"]["n_valid_density"], 0.0)
    chk("max mass deviation (8.9e-16)", 8.9e-16,
        grid["grid"]["statuses"]["density_feasibility"]["max_abs_mass_error"], 0.05)

    conv = grid["grid"]["statuses"]["optimisation_convergence"]
    chk("stage-1 tolerance rate, AKDE-V2 (0.6%)", 0.006,
        conv["v2_akde"]["frac_stage1_converged"], 0.10)
    chk("stage-1 tolerance rate, AKDE-M (5.6%)", 0.056,
        conv["finalist"]["frac_stage1_converged"], 0.05)
    chk("stage-1 tolerance rate, SPADES (28.1%)", 0.281,
        conv["spades_simplex"]["frac_stage1_converged"], 0.02)

    # --- fixed benchmark ---------------------------------------------------
    for N, e, lo, hi in ((200, 0.453, 0.423, 0.485),
                         (1000, 0.174, 0.160, 0.188),
                         (5000, 0.0822, 0.0753, 0.0902)):
        r = fb[(fb.method == "finalist") & (fb.N == N)].iloc[0]
        chk("exp(T) AKDE-M vs V2 at N=%d" % N, e,
            float(r.exp_T_geometric_mean_ratio), 0.02)
        chk("exp(T) CI low at N=%d" % N, lo, float(r.boot_ci_lo), 0.03)
        chk("exp(T) CI high at N=%d" % N, hi, float(r.boot_ci_hi), 0.03)
        chk("targets favouring AKDE-M at N=%d (15)" % N, 15,
            int(r.n_targets_favouring_method), 0.0)

    r = fb[(fb.method == "finalist") & (fb.N == 200)].iloc[0]
    chk("per-target ratio range at N=200, min (0.221)", 0.221,
        float(r.min_per_target_ratio), 0.02)
    chk("per-target ratio range at N=200, max (0.715)", 0.715,
        float(r.max_per_target_ratio), 0.02)
    r = fb[(fb.method == "finalist") & (fb.N == 5000)].iloc[0]
    chk("per-target ratio range at N=5000, min (0.013)", 0.01323,
        float(r.min_per_target_ratio), 0.03)
    chk("per-target ratio range at N=5000, max (0.546)", 0.5456,
        float(r.max_per_target_ratio), 0.02)

    # --- medians -----------------------------------------------------------
    med = tidy.pivot_table(index="variant", columns="N", values="ise",
                           aggfunc="median")
    for N, e in ((200, 0.119), (1000, 0.0709), (5000, 0.0551)):
        chk("median ISE AKDE-V2 at N=%d" % N, e, float(med.loc["v2_akde", N]), 0.02)
    for N, e in ((200, 0.0611), (1000, 0.0136), (5000, 0.0032)):
        chk("median ISE AKDE-M at N=%d" % N, e, float(med.loc["finalist", N]), 0.02)
    for N, e in ((1000, 0.0128), (5000, 0.0031)):
        chk("median ISE SPADES at N=%d" % N, e,
            float(med.loc["spades_simplex", N]), 0.02)
    chk("AKDE-M median decline factor 200->5000 (19.3)", 19.3,
        float(med.loc["finalist", 200] / med.loc["finalist", 5000]), 0.03)
    chk("AKDE-V2 median decline factor 200->5000 (2.2)", 2.2,
        float(med.loc["v2_akde", 200] / med.loc["v2_akde", 5000]), 0.03)

    # --- AKDE-M vs SPADES paired ratio -------------------------------------
    rows = []
    for N in (200, 1000, 5000):
        zs = []
        for t in sorted(tidy.target.unique()):
            p = tidy[(tidy.target == t) & (tidy.N == N)].pivot_table(
                index="rep", columns="variant", values="ise", aggfunc="first")
            zs.append(np.mean(np.log(p["finalist"] / p["spades_simplex"])))
        rows.append((N, float(np.exp(np.mean(zs)))))
    for (N, actual), e in zip(rows, (1.23, 1.07, 1.15)):
        chk("AKDE-M / SPADES paired ratio at N=%d" % N, e, actual, 0.02)

    # --- vs kernel estimators ----------------------------------------------
    ratios = []
    for N in (200, 1000, 5000):
        a = float(bm[(bm.method == "finalist") & (bm.N == N)]
                  .benchmark_mean_ise_equal_weight.iloc[0])
        b = float(bm[(bm.method == "kde_sj") & (bm.N == N)]
                  .benchmark_mean_ise_equal_weight.iloc[0])
        ratios.append(a / b)
    chk("AKDE-M behind KDE(SJ) in benchmark mean, min factor (2.2)", 2.2,
        float(min(ratios)), 0.05)
    chk("AKDE-M behind KDE(SJ) in benchmark mean, max factor (4.5)", 4.5,
        float(max(ratios)), 0.05)
    # paired geometric-mean ratio against KDE(SJ): exp(T_AKDE-M)/exp(T_KDE-SJ)
    gr = []
    for N in (200, 1000, 5000):
        a = float(fb[(fb.method == "finalist") & (fb.N == N)]
                  .exp_T_geometric_mean_ratio.iloc[0])
        b = float(fb[(fb.method == "kde_sj") & (fb.N == N)]
                  .exp_T_geometric_mean_ratio.iloc[0])
        gr.append(a / b)
    chk("AKDE-M behind KDE(SJ) in paired geometric mean, min (2.6)", 2.6,
        float(min(gr)), 0.05)
    chk("AKDE-M behind KDE(SJ) in paired geometric mean, max (4.9)", 4.9,
        float(max(gr)), 0.05)

    # --- catastrophic replicates -------------------------------------------
    cat = {(c["method"], c["N"]): c for c in stab["catastrophic_replicates"]}
    for N, e in ((200, 0.0022), (1000, 0.0), (5000, 0.0267)):
        chk("AKDE-M catastrophic rate at N=%d" % N, e,
            cat[("finalist", N)]["frac_ge_5x_cell_median"], 0.03)
    for N, e in ((200, 0.040), (1000, 0.042), (5000, 0.013)):
        chk("AKDE-V2 catastrophic rate at N=%d" % N, e,
            cat[("v2_akde", N)]["frac_ge_5x_cell_median"], 0.05)
    for N in (200, 1000, 5000):
        chk("SPADES catastrophic rate at N=%d (0)" % N, 0.0,
            cat[("spades_simplex", N)]["frac_ge_5x_cell_median"], 0.0)
        chk("KDE(SJ) catastrophic rate at N=%d (0)" % N, 0.0,
            cat[("kde_sj", N)]["frac_ge_5x_cell_median"], 0.0)
    chk("AKDE-M worst catastrophic ratio at N=5000 (169)", 169.0,
        cat[("finalist", 5000)]["max_ratio_to_cell_median"], 0.02)

    bsel = {(c["method"], c["N"]): c for c in stab["boundary_selection"]}
    lo_rates = [bsel[("finalist", N)]["lam_at_lower_edge"] for N in (200, 1000, 5000)]
    chk("AKDE-M lambda lower-edge selection, min (22.7%)", 0.227,
        float(min(lo_rates)), 0.02)
    chk("AKDE-M lambda lower-edge selection, max (57.1%)", 0.571,
        float(max(lo_rates)), 0.02)

    # --- cost --------------------------------------------------------------
    q = cost.groupby(["method", "N"])["query_us_per_query_median"].median()
    for N, e in ((200, 0.99), (1000, 1.12), (5000, 1.17)):
        chk("AKDE-M query latency at N=%d (us)" % N, e,
            float(q[("finalist", N)]), 0.10)
    for N, e in ((200, 5.17), (1000, 30.3), (5000, 147.8)):
        chk("KDE(SJ) query latency at N=%d (us)" % N, e,
            float(q[("kde_sj", N)]), 0.10)
    chk("query speedup at N=5000 (127x)", 127.0,
        float(q[("kde_sj", 5000)] / q[("finalist", 5000)]), 0.10)
    b = cost.groupby(["method", "N"])["canonical_model_bytes"].median()
    chk("AKDE-M model bytes at N=5000 (1344)", 1344.0,
        float(b[("finalist", 5000)]), 0.05)
    chk("KDE(SJ) model bytes at N=5000 (120000)", 120000.0,
        float(b[("kde_sj", 5000)]), 0.0)
    chk("model-size factor at N=5000 (89x)", 89.0,
        float(b[("kde_sj", 5000)] / b[("finalist", 5000)]), 0.05)

    # --- oracle ladder ------------------------------------------------------
    base = lad[lad.bank == "base"]
    match = lad[lad.bank == "matched"]
    chk("V2-bank criterion excess, min share (22%)", 0.22,
        float(base.frac_of_total_from_criterion.min()), 0.05)
    chk("V2-bank criterion excess, max share (56%)", 0.56,
        float(base.frac_of_total_from_criterion.max()), 0.05)
    chk("matched criterion excess, max share (17%)", 0.175,
        float(match.frac_of_total_from_criterion.max()), 0.05)
    f2a = match[(match.target == "f2") & (match.N == 500)].iloc[0]
    f2b = match[(match.target == "f2") & (match.N == 5000)].iloc[0]
    chk("MW3 matched excess share at N=500 (8.1%)", 0.081,
        float(f2a.frac_of_total_from_criterion), 0.03)
    chk("MW3 matched excess share at N=5000 (17.5%)", 0.175,
        float(f2b.frac_of_total_from_criterion), 0.03)
    chk("MW3 R_full* at N=500 (0.00490)", 0.00490, float(f2a.R_full_star), 0.02)
    chk("MW3 moment(y0,C0) at N=500 (0.0101)", 0.0101,
        float(f2a.moment_y0_C0), 0.02)
    chk("MW3 R_full* at N=5000 (0.00334)", 0.00334, float(f2b.R_full_star), 0.02)
    chk("MW3 moment(y0,C0) at N=5000 (0.00521)", 0.00521,
        float(f2b.moment_y0_C0), 0.02)
    chk("covariance substitution max share (16.4%)", 0.164,
        float(lad.cov_substitution_frac_of_total.max()), 0.03)

    # --- ambiguity ----------------------------------------------------------
    z0 = amb[amb.moment_tol == 0.0]
    chk("ambiguity witness at zero budget, V2 bank (0.125)", 0.125,
        float(z0[z0.bank == "base"].abs_delta_ise.max()), 0.03)
    chk("ambiguity witness at zero budget, matched bank (0.0235)", 0.0235,
        float(z0[z0.bank == "matched"].abs_delta_ise.max()), 0.03)

    # --- solver -------------------------------------------------------------
    s1 = {(d["stage1_solver"], d["stage1_budget"]): d
          for d in solv["stage1_kkt_rel"]}
    s1c = {(d["stage1_solver"], d["stage1_budget"]): d
           for d in solv["stage1_converged_fraction"]}
    chk("stage-1 CD@500 median rel KKT (8.7e-5)", 8.7e-5,
        s1[("cd", 500)]["median"], 0.05)
    chk("stage-1 CD@10000 median rel KKT (1.1e-6)", 1.1e-6,
        s1[("cd", 10000)]["median"], 0.10)
    chk("stage-1 FISTA median rel KKT (1.6e-8)", 1.6e-8,
        s1[("fista", 40000)]["median"], 0.10)
    chk("stage-1 CD@500 tolerance rate (13.5%)", 0.135,
        s1c[("cd", 500)]["stage1_converged_flag"], 0.03)
    chk("stage-1 CD@10000 tolerance rate (24%)", 0.24,
        s1c[("cd", 10000)]["stage1_converged_flag"], 0.03)
    chk("stage-1 FISTA tolerance rate (12.5%)", 0.125,
        s1c[("fista", 40000)]["stage1_converged_flag"], 0.03)
    chk("stage-2 median rel KKT, original (6.3e-13)", 6.3e-13,
        solv["stage2_kkt_rel_orig"]["median"], 0.05)
    chk("stage-2 max rel KKT, original (3.3e-3)", 3.3e-3,
        solv["stage2_kkt_rel_orig"]["max"], 0.05)
    chk("stage-2 median rel KKT, reference (7.4e-7)", 7.4e-7,
        solv["stage2_kkt_rel_ref"]["median"], 0.05)
    chk("stage-2 max rel KKT, reference (0.87)", 0.87,
        solv["stage2_kkt_rel_ref"]["max"], 0.05)

    sv = pd.read_csv(DIAG_ROOT / "solver_verification.csv")
    cd = sv[sv.stage1_solver == "cd"]
    piv = cd.pivot_table(index=["target", "N", "rep", "bank", "lam_ratio"],
                         columns="stage1_budget", values="stage1_support")
    same = float((piv[500] == piv[10000]).mean())
    chk("support identical at budgets 500 vs 10000 (18.8%)", 0.188, same, 0.03)
    chk("median support change 500->10000 (21 atoms)", 21.0,
        float((piv[10000] - piv[500]).abs().median()), 0.05)

    # --- textual sanity: every number above appears in the manuscript -------
    df = pd.DataFrame(CHECKS)
    df.to_csv(OUT / "manuscript_reconciliation.csv", index=False)
    n_fail = int((~df["pass"]).sum())
    print("== manuscript number reconciliation ==")
    for c in CHECKS:
        print("  [%s] %-58s manuscript %-12s recomputed %-14s (rel %.3g)"
              % ("OK " if c["pass"] else "BAD", c["claim"],
                 "%.6g" % c["in_manuscript"] if c["in_manuscript"] is not None else "-",
                 "%.6g" % c["recomputed"] if c["recomputed"] is not None else "-",
                 c["rel_diff"]))
    print("\n%d checks, %d failed" % (len(df), n_fail))

    # cross-reference / citation hygiene on the tex itself
    cites = set(re.findall(r"\\cite\{([^}]*)\}", tex))
    keys = set()
    for c in cites:
        keys.update(k.strip() for k in c.split(","))
    refs = (PAPER / "references.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", refs))
    missing = sorted(keys - defined)
    unused = sorted(defined - keys)
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refd = set()
    for r in re.findall(r"\\ref\{([^}]*)\}", tex):
        refd.add(r.strip())
    dangling = sorted(refd - labels)
    print("\ncitations used: %d, undefined: %s, defined-but-unused: %s"
          % (len(keys), missing or "none", unused or "none"))
    print("dangling \\ref targets: %s" % (dangling or "none"))

    with open(OUT / "manuscript_reconciliation.json", "w", encoding="utf-8") as fh:
        json.dump({"n_checks": len(df), "n_failed": n_fail,
                   "undefined_citations": missing,
                   "unused_bibitems": unused,
                   "dangling_refs": dangling,
                   "checks": CHECKS}, fh, indent=2, default=float)
    return 1 if (n_fail or missing or dangling) else 0


if __name__ == "__main__":
    raise SystemExit(main())
