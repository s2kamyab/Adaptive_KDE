"""Task 2 - recompute every reported summary from raw rows and grade the claims.

Writes ``followup/results/verification/``:

    claim_evidence.csv        one row per claim: statement, source report,
                              recomputed value, status, evidence path
    oracle_ladder_check.csv   the oracle ladder with ABSOLUTE excesses as well
                              as ratios, and a numerical-resolution flag
    report_recomputation.json every headline number, recomputed

Statuses: supported | qualified | contradicted | unresolved.
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

from followup.paths import RAW_ROOT, DIAG_ROOT   # noqa: E402
RES = RAW_ROOT
OUT = DIAG_ROOT
PROJ = PY.parents[1]

CLAIMS = []


def claim(cid, statement, source, recomputed, status, note, evidence):
    CLAIMS.append({"id": cid, "statement": statement, "source": source,
                   "recomputed": recomputed, "status": status, "note": note,
                   "evidence": evidence})


def load(name):
    d = pd.read_csv(RES / name / "all_rows.csv", low_memory=False)
    d["error"] = d["error"].fillna("").astype(str)
    return d[d["error"] == ""].copy()


# =====================================================================
def check_oracle_ladder():
    """Claim group A: the oracle controls.

    Reports ABSOLUTE excess over the dictionary optimum as well as the ratio,
    and flags cells where the dictionary optimum is at or below the level at
    which the analytic ISE (a difference of three large closed-form terms) can
    be resolved in double precision.
    """
    frames = []
    for bank in ("base", "matched"):
        p = RES / "oracle_controls" / ("ladder_%s_rho0.001.csv" % bank)
        d = pd.read_csv(p)
        d["bank"] = bank
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)

    # int f0^2 sets the scale of the cancellation in ISE = ff - 2fg + gg.
    # The double-precision resolution of that difference was MEASURED against
    # mpmath at 50 digits (followup/verify/check_centres.py): the worst
    # realised cancellation factor over a generic weight vector, a small
    # simplex optimum, and the two full-dictionary optima that produced the
    # smallest reported risks was 2.44 x eps.  A factor of 10 is carried as
    # headroom, giving floor = 25 * eps * int f0^2.  The conservative a-priori
    # floor (1e4 * eps) is also recorded so the sensitivity is visible.
    EPS = float(np.finfo(float).eps)
    d["ise_resolution_floor"] = 25.0 * EPS * d["c0_int_f0_sq"]
    d["ise_resolution_floor_conservative"] = 1e4 * EPS * d["c0_int_f0_sq"]
    g = d.groupby(["bank", "target", "N"]).agg(
        R_full_star=("R_full_star", "median"),
        moment_y0_C0=("ise_moment_y0_C0", "median"),
        moment_y0_Chat=("ise_moment_y0_Chat", "median"),
        moment_y_C0=("ise_moment_y_C0", "median"),
        moment_y_Chat=("ise_moment_y_Chat", "median"),
        l2_empirical=("ise_l2_empirical", "median"),
        c0=("c0_int_f0_sq", "median"),
        floor=("ise_resolution_floor", "median"),
        floor_cons=("ise_resolution_floor_conservative", "median"),
        n=("R_full_star", "size"),
    ).reset_index()
    g["abs_excess_y0_C0"] = g.moment_y0_C0 - g.R_full_star
    g["ratio_y0_C0"] = g.moment_y0_C0 / g.R_full_star
    g["abs_excess_y_Chat"] = g.moment_y_Chat - g.R_full_star
    g["sampling_gap_abs"] = g.moment_y_Chat - g.moment_y0_C0
    g["frac_of_total_from_criterion"] = g.abs_excess_y0_C0 / g.moment_y_Chat
    g["R_full_below_resolution"] = g.R_full_star < g.floor
    g["R_full_below_conservative_floor"] = g.R_full_star < g.floor_cons
    g["ratio_is_meaningful"] = ~g.R_full_below_resolution
    g["cov_substitution_factor"] = np.maximum(
        g.moment_y0_Chat / g.moment_y0_C0, g.moment_y0_C0 / g.moment_y0_Chat)
    g.to_csv(OUT / "oracle_ladder_check.csv", index=False)

    m = g[g.bank == "matched"]
    b = g[g.bank == "base"]

    # A1: "all remaining error is sampling error" / "population floor vanishes"
    worst = m.loc[m.abs_excess_y0_C0.idxmax()]
    claim("A1",
          "With a matched bank the criterion's population floor essentially "
          "disappears and all remaining error is sampling error.",
          "diagnosis_report.md 3.2 / results_summary.md 3.5",
          "matched-bank oracle excess over the dictionary optimum is NOT zero: "
          "absolute excess ranges %.3g..%.3g; the worst cell is %s N=%d with "
          "R_full*=%.4g, moment(y0,C0)=%.4g, absolute excess %.4g = %.0f%% of "
          "that cell's total sampled moment error %.4g"
          % (m.abs_excess_y0_C0.min(), m.abs_excess_y0_C0.max(),
             worst.target, worst.N, worst.R_full_star, worst.moment_y0_C0,
             worst.abs_excess_y0_C0, 100 * worst.frac_of_total_from_criterion,
             worst.moment_y_Chat),
          "contradicted",
          "Replace with: matching reduces the population-level excess by orders "
          "of magnitude but does not remove it. On f2 (strongly skewed) the "
          "excess remains %.0f%% (N=500) and %.0f%% (N=5000) of the sampled "
          "moment error."
          % (100 * m[(m.target == 'f2') & (m.N == 500)].frac_of_total_from_criterion.iloc[0],
             100 * m[(m.target == 'f2') & (m.N == 5000)].frac_of_total_from_criterion.iloc[0]),
          "followup/results/verification/oracle_ladder_check.csv")

    # A2: "within a factor of 1-2 of R_full* on every cell"
    ok = m[m.ratio_is_meaningful]
    claim("A2",
          "moment(y0,C0) lands within a factor of 1-2 of R_full* on every cell "
          "(matched bank).",
          "diagnosis_report.md 3.2",
          "ratios over the %d matched cells where R_full* is above the analytic "
          "ISE resolution floor: min %.2f, median %.2f, max %.2f; %d of %d cells "
          "exceed 2. In the %d cells where R_full* is BELOW the resolution floor "
          "(%.1g..%.1g) the ratio is not a meaningful quantity at all."
          % (len(ok), ok.ratio_y0_C0.min(), ok.ratio_y0_C0.median(),
             ok.ratio_y0_C0.max(), int((ok.ratio_y0_C0 > 2).sum()), len(ok),
             int(m.R_full_below_resolution.sum()),
             m[m.R_full_below_resolution].R_full_star.min(),
             m[m.R_full_below_resolution].R_full_star.max()),
          "contradicted",
          "Report absolute excess, not ratios, wherever R_full* is at or below "
          "the resolution floor. Ratios to near-zero oracle risks are unstable.",
          "followup/results/verification/oracle_ladder_check.csv")

    # A3: the "fourteen orders of magnitude" claim
    f5 = b[(b.target == "f5") & (b.N == 5000)].iloc[0]
    claim("A3",
          "On f5 at N=5000 the V2-bank criterion floor is 0.0248 against a "
          "dictionary capability of 8e-17 - fourteen orders of magnitude.",
          "diagnosis_report.md 3.1",
          "the double-precision R_full* there is %.3g, which sits AT the "
          "validated analytic-ISE resolution floor %.3g for that target. A "
          "50-digit mpmath recomputation of the same optimum gives 2.11e-17, "
          "so the dictionary optimum is genuinely below 1e-16 - but the "
          "double-precision figure carries ~75%% relative error and must not "
          "be quoted to two significant figures, let alone turned into an "
          "orders-of-magnitude statement. Report the absolute excess %.4g and "
          "an upper bound on R_full*."
          % (f5.R_full_star, f5.floor, f5.abs_excess_y0_C0),
          "contradicted",
          "Replace with: 'the 300-atom dictionary represents this target to an "
          "exact risk below 1e-16 (verified at 50-digit precision), while the "
          "V2-bank population-moment fit leaves 0.0248'. Give the absolute "
          "excess; do not quote a ratio against a value at the noise floor.",
          "followup/results/verification/oracle_ladder_check.csv")

    # A4: covariance substitution "< 2x in every cell"
    worst_cov = g.loc[g.cov_substitution_factor.idxmax()]
    # The ratio is meaningless when both rungs sit at roundoff level.  The
    # decision-relevant quantity is the absolute change relative to the total
    # error actually incurred in that cell.
    g["cov_substitution_abs"] = (g.moment_y0_Chat - g.moment_y0_C0).abs()
    g["cov_substitution_frac_of_total"] = g.cov_substitution_abs / g.moment_y_Chat
    meaningful = g[(g.moment_y0_C0 > g.floor) & (g.moment_y0_Chat > g.floor)]
    claim("A4",
          "Replacing the estimated covariance by the population covariance "
          "changes results by well under 2x in every cell.",
          "results_summary.md 3.2",
          "as a RATIO the max over all %d cells is %.1f (%s N=%d, %s bank) and "
          "%d of %d cells exceed 2 - but the largest ratios occur where both "
          "rungs are at roundoff level (%.1e vs %.1e there). Restricted to the "
          "%d cells where both rungs exceed the analytic-ISE resolution floor, "
          "the max ratio is %.2f. As an ABSOLUTE change relative to that cell's "
          "total sampled moment error the substitution moves the result by at "
          "most %.1f%% (median %.2f%%)."
          % (len(g), worst_cov.cov_substitution_factor, worst_cov.target,
             worst_cov.N, worst_cov.bank,
             int((g.cov_substitution_factor > 2).sum()), len(g),
             worst_cov.moment_y0_C0, worst_cov.moment_y0_Chat,
             len(meaningful), meaningful.cov_substitution_factor.max(),
             100 * g.cov_substitution_frac_of_total.max(),
             100 * g.cov_substitution_frac_of_total.median()),
          "contradicted",
          "Drop the blanket '<2x in every cell'. Report the absolute effect: "
          "estimating Omega rather than knowing it moves the population-moment "
          "fit by at most %.0f%% of the cell's total error. Ratios computed at "
          "roundoff level must not be quoted."
          % (100 * g.cov_substitution_frac_of_total.max()),
          "followup/results/verification/oracle_ladder_check.csv")
    # re-save with the derived columns added above
    g.to_csv(OUT / "oracle_ladder_check.csv", index=False)
    return g


# =====================================================================
def check_confirmation():
    """Claim group B: the confirmation numbers."""
    d = load("stage_c")
    med = d.pivot_table(index="variant", columns="N", values="ise", aggfunc="median")
    mean = d.pivot_table(index="variant", columns="N", values="ise", aggfunc="mean")

    # B1: median ISE table
    claim("B1", "Median ISE table (Stage C).", "results_summary.md 2",
          med.round(6).to_dict(), "supported",
          "Recomputed from all_rows.csv; matches the reported table.",
          "followup/results/stage_c/analysis/median_ise_by_N.csv")

    # B2: slopes - descriptive only
    sl = {}
    for e in med.index:
        s = med.loc[e]
        sl[e] = float(np.polyfit(np.log(s.index.values.astype(float)),
                                 np.log(s.values), 1)[0])
    claim("B2",
          "The finalist's descriptive log-log slope is -0.92 against V2's -0.24.",
          "results_summary.md 2",
          {k: round(v, 3) for k, v in sl.items()}, "qualified",
          "Values reproduce, but three sample sizes cannot support a rate. "
          "Must be described as the observed finite-range decline over "
          "N in {200,1000,5000}, not a convergence rate, and no asymptotic "
          "floor may be inferred from it.",
          "followup/results/verification/report_recomputation.json")

    # B3: 'supersets of sample sizes'
    v2_sizes = sorted(pd.read_csv(PY / "results" / "main_full" / "config.json"
                                  .replace("config.json", "config.json"),
                                  nrows=0).columns) if False else None
    cfg = json.load(open(PY / "results" / "main_full" / "config.json", encoding="utf-8"))
    conf_sizes = sorted(int(x) for x in d.N.unique())
    is_superset = set(conf_sizes) >= set(cfg["sample_sizes"])
    claim("B3",
          "The confirmation runs on a superset of the V2 sample sizes.",
          "results_summary.md 2 / manuscript_patch.md B2",
          "V2 grid %s; confirmation grid %s; superset=%s (the confirmation is a "
          "SUBSET of three of the six V2 sizes)"
          % (cfg["sample_sizes"], conf_sizes, is_superset),
          "contradicted" if not is_superset else "supported",
          "Say 'on three of the six V2 sample sizes' and drop 'superset'.",
          "Adaptive_KDE/python/results/main_full/config.json")

    # B4: 'essentially tied' with SPADES
    ag = pd.read_csv(RES / "stage_c" / "analysis" / "aggregate_equal_weight.csv")
    sp = ag[ag.reference == "spades_simplex"]
    claim("B4",
          "Against the density-valid SPADES the finalist is essentially tied.",
          "results_summary.md 1 (headline table)",
          "equal-weight geometric-mean ISE ratio finalist/SPADES = "
          + ", ".join("%.3f [%.3f, %.3f] at N=%d"
                      % (r.geom_mean_ratio_equal_weight, r.ci_lo, r.ci_hi, r.N)
                      for r in sp.itertuples()),
          "contradicted",
          "A 7-23% disadvantage whose interval excludes 1 at two of three sample "
          "sizes is not a tie. No equivalence margin was predefined, so no "
          "equivalence claim is available. State it as a 7-23% disadvantage.",
          "followup/results/stage_c/analysis/aggregate_equal_weight.csv")

    # B5: AKDE-family row count
    fam = d[d.variant.isin(["v2_akde", "finalist"])]
    claim("B5",
          "Diagnostic columns are finite in all 900 AKDE-family rows.",
          "method_changes.md (post-run fixes)",
          "there are %d AKDE-family rows in Stage C (2 methods x 15 targets x 3 "
          "sizes x 30 reps), not 900; non-finite cov_effective_rank entries: %d"
          % (len(fam), int((~np.isfinite(pd.to_numeric(
              fam.cov_effective_rank, errors="coerce"))).sum())),
          "contradicted",
          "Correct the count to 2,700 AKDE-family rows.",
          "followup/results/stage_c/all_rows.csv")

    # B6: three status families
    err = d["error"].fillna("").astype(str)
    conv = {m: float(d[d.variant == m]["stage1_converged"].astype(str)
                     .isin(["True"]).mean())
            for m in ("v2_akde", "finalist", "spades_simplex")}
    claim("B6",
          "8,100 fits, 0 failures, all valid densities.",
          "results_summary.md 2",
          "execution failures %d; invalid densities %d; stage-1 optimiser "
          "convergence rate v2_akde %.3f, finalist %.3f, spades_simplex %.3f - "
          "these are THREE DIFFERENT statuses and only the first two are zero/clean"
          % (int((err != "").sum()),
             int((~d.is_valid_density.astype(str).isin(["True"])).sum()),
             conv["v2_akde"], conv["finalist"], conv["spades_simplex"]),
          "qualified",
          "Always report the three statuses separately; 'zero failures' must "
          "never be read as 'the optimiser converged'.",
          "followup/results/verification/grid_verification.json")
    return {"median": med.to_dict(), "mean": mean.to_dict(), "slopes": sl,
            "conv": conv}


# =====================================================================
def check_solver():
    """Claim group C: solver claims."""
    s = pd.read_csv(RES / "solver_audit" / "stage1_solver_comparison.csv")
    claim("C1",
          "Residual-space FISTA reaches the 1e-8 tolerance in 46% (M=300) / "
          "19% (M=1500) of instances.",
          "audit_findings.md 3.1 / method_changes.md 2",
          "recomputed: %s"
          % s.groupby("M")[["conv_cd", "conv_fista"]].mean().round(4).to_dict(),
          "qualified",
          "FISTA is BETTER but is not converged: it misses its own tolerance in "
          "the majority of instances. It must not be described as a converged "
          "reference, and 'finds a lower objective' is not by itself a "
          "convergence certificate.",
          "followup/results/solver_audit/stage1_solver_comparison.csv")

    dev = pd.concat([load("stage_a"), load("stage_b")], ignore_index=True)
    piv = dev.pivot_table(index=["target", "N", "rep"], columns="variant",
                          values="ise")
    if {"B3_moment_wide_fista", "A6_moment_matched_m300"} <= set(piv.columns):
        z = np.log(piv["B3_moment_wide_fista"] / piv["A6_moment_matched_m300"]).dropna()
        r = float(np.exp(z.mean()))
        claim("C2",
              "Solving stage 1 exactly buys almost nothing (0.213/0.050 vs "
              "0.214/0.063), so it is not the accuracy defect.",
              "diagnosis_report.md 5 / method_changes.md 2",
              "paired over the %d development fits, geometric-mean ISE ratio "
              "FISTA/finalist = %.3f (%.1f%% lower error for FISTA); by N: %s"
              % (len(z), r, 100 * (1 - r),
                 {int(n): round(float(np.exp(np.log(
                     g['B3_moment_wide_fista'] / g['A6_moment_matched_m300']).mean())), 3)
                  for n, g in piv.groupby(level=1)}),
              "qualified",
              "A ~%.0f%% mean reduction is not 'almost nothing'. State it as a "
              "modest accuracy gain at 7.5x the tuning cost, and note that the "
              "alternative solver is itself not certified converged."
              % (100 * (1 - r)),
              "followup/results/stage_a|stage_b/all_rows.csv")
    return s


# =====================================================================
def check_documentation_claims():
    """Claim group D: the audit's own documentation findings, re-verified."""
    v2 = pd.read_csv(PY / "results" / "main_full" / "all_results.csv", low_memory=False)
    v2 = v2[v2["error"].fillna("").astype(str) == ""]
    ak = v2[v2.estimator == "akde"]

    claim("D1", "AKDE was valid at all 27,000 fits.", "V2 manuscript 5.4",
          "AKDE rows = %d of %d total; AKDE valid fraction %.4f; pooled valid "
          "fraction over all estimators %.4f (spades is 0%% valid by construction)"
          % (len(ak), len(v2),
             float(ak.is_valid_density.astype(str).isin(["True"]).mean()),
             float(v2.is_valid_density.astype(str).isin(["True"]).mean())),
          "contradicted", "Denominator must be 2,700 AKDE fits.",
          "Adaptive_KDE/python/results/main_full/all_results.csv")

    dist = ak.groupby("N")["n_dictionary"].value_counts().unstack(fill_value=0)
    base = v2[v2.estimator.isin(["spades", "spades_simplex", "song"])]
    claim("D2", "akde, spades, spades_simplex and song share one dictionary.",
          "V2 manuscript 5.1",
          "AKDE n_dictionary distribution by N: %s; baselines use n_dictionary "
          "= %s in %d of %d rows"
          % (dist.to_dict(), sorted(base.n_dictionary.unique()),
             int((base.n_dictionary == 1500).sum()), len(base)),
          "contradicted",
          "AKDE cross-validated the size; the baselines were pinned at 1500.",
          "Adaptive_KDE/python/results/main_full/all_results.csv")

    claim("D3", "tau is selected by cross-validation.", "V2 manuscript 4 and 5.1",
          "no sel_tau column exists in the V2 rows (columns containing 'tau': %s); "
          "AKDEConfig.tau is a constant 1e-3 and AKDE.candidates enumerates only "
          "dict x rho x lam_ratio"
          % [c for c in v2.columns if "tau" in c.lower()],
          "contradicted", "tau was fixed, not tuned.",
          "Adaptive_KDE/python/akde/estimators/akde.py")

    er = ak.groupby("N")["cov_effective_rank"].median()
    nr = ak.groupby("N")["cov_rank_numeric"].median()
    claim("D4",
          "The regularized covariance is substantially rank deficient "
          "(numerical rank 42 -> 152).",
          "V2 manuscript 5.4",
          "numerical rank of the EMPIRICAL covariance by N: %s; effective rank "
          "by N: %s; the regularised matrix C+rho I is full rank for any rho>0"
          % (nr.round(1).to_dict(), er.round(2).to_dict()),
          "qualified",
          "Wording must say 'empirical covariance'; effective rank should also "
          "be reported since it is far smaller.",
          "Adaptive_KDE/python/results/main_full/all_results.csv")

    cost = pd.read_csv(RES / "cost" / "phase_share.csv", index_col=0)
    claim("D5",
          "The dominant cost is building, factorizing and cross-validating the "
          "dense m x m covariance.",
          "V2 manuscript 5.7",
          "phase shares: %s" % (100 * cost).round(1).to_dict(),
          "contradicted",
          "Covariance assembly + factorisation is a few per cent; the two "
          "quadratic programs dominate.",
          "followup/results/cost/phase_share.csv")
    return None


# =====================================================================
def check_ambiguity():
    """Claim group E: the ambiguity probe's scope."""
    a = pd.read_csv(RES / "oracle_controls" / "ambiguity_base_rho0.001.csv")
    claim("E1",
          "On the face of the simplex containing the exact-risk optimum, the "
          "most density-changing feasible direction buys at most 2.5e-7 of ISE, "
          "so unidentifiability is not the failure mode.",
          "diagnosis_report.md 7 / results_summary.md 3.3",
          "the search covers only directions supported on the ACTIVE face "
          "(supp(v) subset supp(w_ref), sum v = 0); it cannot activate any of "
          "the %d..%d coordinates that are zero at the optimum. Largest "
          "delta_ise found over the %d recorded directions: %.3g. This is a "
          "WITNESS / lower bound on achievable ambiguity, not a maximum."
          % (int(300 - a.face_size.max()), int(300 - a.face_size.min()),
             len(a), float(a.delta_ise_at_t_max.max())),
          "qualified",
          "Either restrict the claim to the active face, or extend the search "
          "to the full feasible tangent cone (sum v = 0, v_i >= 0 where w_i = 0). "
          "No global upper bound is available from a directional search.",
          "followup/results/oracle_controls/ambiguity_base_rho0.001.csv")


# =====================================================================
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ladder = check_oracle_ladder()
    conf = check_confirmation()
    check_solver()
    check_documentation_claims()
    check_ambiguity()

    df = pd.DataFrame(CLAIMS)
    df.to_csv(OUT / "claim_evidence.csv", index=False)
    with open(OUT / "report_recomputation.json", "w", encoding="utf-8") as fh:
        json.dump({"confirmation": conf,
                   "oracle_ladder": ladder.to_dict(orient="records")},
                  fh, indent=2, default=float)

    print("== claim ledger ==")
    for c in CLAIMS:
        print("\n[%s] %s" % (c["status"].upper(), c["id"]))
        print("  claim     : %s" % c["statement"])
        print("  source    : %s" % c["source"])
        print("  recomputed: %s" % c["recomputed"])
        print("  action    : %s" % c["note"])
    print("\n%d claims: %s" % (len(df), df.status.value_counts().to_dict()))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
