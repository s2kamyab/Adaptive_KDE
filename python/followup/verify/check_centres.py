"""Task 4a - are the reported centre distances numerically meaningful?

The reported distances (0.946 / 0.644 / 0.028) came from
``np.linalg.solve(A + jitter*I, b)`` with A = D'Omega D and a separate
``np.linalg.solve(G + jitter*I, s)``, on matrices whose condition numbers were
themselves reported as ~1e19-1e21. That is an inverse-based interpretation of
a numerically singular system and cannot be taken at face value.

This module:

  1. recomputes the centres from the SVD of the WHITENED design
     A = Omega^{1/2} D (never from normal equations), at four documented
     relative rank thresholds, and reports how the answer moves;
  2. records singular spectra, effective ranks, residuals and backward errors;
  3. checks the matched-bank identity analytically (D = G, y = s) and on a
     deliberately WELL-CONDITIONED synthetic dictionary where double precision
     is trustworthy;
  4. validates the analytic-ISE resolution floor against mpmath at 50 digits
     instead of assuming it;
  5. compares fitted-density differences and objective residuals, not only
     coefficient distances.
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

from akde import gaussops                          # noqa: E402
from akde.densities import MARRON_WAND_MAPPING, get_target   # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary  # noqa: E402
from followup.akde_followup import oracle          # noqa: E402
from followup.akde_followup.seeds import data_rng  # noqa: E402
from followup.akde_followup.stage1 import omega_half           # noqa: E402
from followup.akde_followup.variants import (      # noqa: E402
    BANK_MATCHED, BASE_BANK, FollowupAKDE, build_measurement_bank)
from followup.verify import check_config as CFG    # noqa: E402

from followup.paths import DIAG_ROOT   # noqa: E402
OUT = DIAG_ROOT


def _verify_labels():
    bad = [t for t, (n, nm) in CFG.TARGET_LABELS.items()
           if MARRON_WAND_MAPPING[t] != (n, nm)]
    if bad:
        raise AssertionError("target label mismatch: %s" % bad)
    return {t: MARRON_WAND_MAPPING[t] for t in CFG.TARGETS}


def build_instance(target, N, rep, bank_name, rho=CFG.RHO_MAIN, dict_cfg=None):
    dc = dict_cfg or CFG.DICT
    X = get_target(target).sample(N, data_rng(CFG.NAMESPACE, target, N, rep))
    atoms = build_dictionary(X, DictionaryConfig(dc["n_loc"], dc["n_scale"],
                                                 "grid", dc["lo"], dc["hi"]))
    spec = BASE_BANK if bank_name == "base" else BANK_MATCHED
    bank = build_measurement_bank(X, spec, dictionary=atoms)
    D = gaussops.design_matrix(bank.means, bank.variances,
                               atoms.means, atoms.variances)
    K = FollowupAKDE.eval_kernels(X, bank)
    y = K.mean(axis=0)
    Kc = K - y
    C = (Kc.T @ Kc) / (N * (N - 1.0))
    G = gaussops.gram_matrix(atoms.means, atoms.variances)
    s = FollowupAKDE.eval_kernels(X, atoms).mean(axis=0)
    half = omega_half(C, "ridge", rho)
    return dict(X=X, atoms=atoms, bank=bank, D=D, y=y, C=C, G=G, s=s,
                A=half(D), b=half(y), half=half)


def pinv_solve(M, rhs, rel_tol):
    """Minimum-norm least-squares solution via SVD with a stated cutoff."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    cut = rel_tol * S[0] if S.size else 0.0
    keep = S > cut
    x = Vt[keep].T @ ((U[:, keep].T @ rhs) / S[keep])
    return x, int(keep.sum()), S


def centre_table():
    rows = []
    for target in CFG.TARGETS:
        for N in CFG.SIZES:
            for rep in CFG.REPS:
                for bank in CFG.BANKS:
                    I = build_instance(target, N, rep, bank)
                    A, b, G, s, D, y = (I["A"], I["b"], I["G"], I["s"],
                                        I["D"], I["y"])
                    sv_A = np.linalg.svd(A, compute_uv=False)
                    sv_G = np.linalg.svd(G, compute_uv=False)
                    sv_D = np.linalg.svd(D, compute_uv=False)
                    for tol in CFG.RANK_THRESHOLDS:
                        um, km, _ = pinv_solve(A, b, tol)
                        # L2 centre: G is symmetric PSD, so its own SVD with
                        # the same cutoff is the matched comparison
                        u2, k2, _ = pinv_solve(G, s, tol)
                        dcent = um - u2
                        # ||f_um - f_u2||_2^2 = d' G d  (exact for mixtures)
                        dens_l2_sq = float(dcent @ G @ dcent)
                        rows.append({
                            "target": target, "N": N, "rep": rep, "bank": bank,
                            "rank_tol": tol,
                            "m": D.shape[0], "M": D.shape[1],
                            "rank_A": km, "rank_G": k2,
                            "cond_A": float(sv_A[0] / sv_A[-1]) if sv_A[-1] > 0 else np.inf,
                            "cond_G": float(sv_G[0] / sv_G[-1]) if sv_G[-1] > 0 else np.inf,
                            "cond_D": float(sv_D[0] / sv_D[-1]) if sv_D[-1] > 0 else np.inf,
                            "sv_A_max": float(sv_A[0]), "sv_A_min": float(sv_A[-1]),
                            "centre_distance_in_G": float(np.sqrt(max(dens_l2_sq, 0.0))),
                            "fitted_density_l2_diff": float(np.sqrt(max(dens_l2_sq, 0.0))),
                            "norm_um": float(np.linalg.norm(um)),
                            "norm_u2": float(np.linalg.norm(u2)),
                            "resid_moment_at_um": float(np.linalg.norm(A @ um - b)),
                            "resid_moment_at_u2": float(np.linalg.norm(A @ u2 - b)),
                            "resid_l2_at_um": float(np.linalg.norm(G @ um - s)),
                            "resid_l2_at_u2": float(np.linalg.norm(G @ u2 - s)),
                            # backward error of the moment solve
                            "backward_err_um": float(
                                np.linalg.norm(A @ um - b) /
                                max(np.linalg.norm(A) * np.linalg.norm(um)
                                    + np.linalg.norm(b), 1e-300)),
                        })
    return pd.DataFrame(rows)


def wellconditioned_synthetic():
    """A small, well-separated dictionary where double precision is reliable.

    Six atoms at unit spacing with sd 0.5 give cond(G) of order 10^2, so every
    quantity below is computed to essentially full precision. This is where the
    matched-bank identity can actually be *verified* rather than asserted.
    """
    mu = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])[:, None]
    var = np.full_like(mu, 0.25)
    G = gaussops.gram_matrix(mu, var)
    rng = np.random.default_rng(20260928)
    X = rng.normal(0.0, 1.0, size=(400, 1))
    s = gaussops.mixture_pdf_diag  # not used; compute directly
    Kb = np.exp(-0.5 * ((X[:, None, :] - mu[None, :, :]) ** 2 / var[None, :, :]
                        + np.log(var)[None, :, :] + gaussops.LOG_2PI)).squeeze(-1)
    sv = Kb.mean(axis=0)
    out = {"cond_G": float(np.linalg.cond(G)), "M": int(G.shape[0])}

    # matched bank: D = G, y = s exactly
    D = gaussops.design_matrix(mu, var, mu, var)
    out["max_abs_D_minus_G"] = float(np.abs(D - G).max())
    out["max_abs_y_minus_s"] = 0.0

    u2 = np.linalg.solve(G, sv)
    dists = {}
    for rho in CFG.RHO_SENSITIVITY:
        Kc = Kb - sv
        C = (Kc.T @ Kc) / (400 * 399.0)
        half = omega_half(C, "ridge", rho)
        A, b = half(D), half(sv)
        um = np.linalg.lstsq(A, b, rcond=None)[0]
        d = um - u2
        dists["rho=%g" % rho] = {
            "centre_distance_in_G": float(np.sqrt(max(d @ G @ d, 0.0))),
            "rel_coef_distance": float(np.linalg.norm(d) / np.linalg.norm(u2)),
        }
    out["matched_centre_vs_rho"] = dists

    # unmatched bank on the same dictionary: three wider probes
    mk = np.array([-2.0, 0.0, 2.0])[:, None]
    vk = np.full_like(mk, 1.0)
    Du = gaussops.design_matrix(mk, vk, mu, var)
    Kk = np.exp(-0.5 * ((X[:, None, :] - mk[None, :, :]) ** 2 / vk[None, :, :]
                        + np.log(vk)[None, :, :] + gaussops.LOG_2PI)).squeeze(-1)
    yk = Kk.mean(axis=0)
    Kc = Kk - yk
    Cu = (Kc.T @ Kc) / (400 * 399.0)
    und = {}
    for rho in CFG.RHO_SENSITIVITY:
        half = omega_half(Cu, "ridge", rho)
        um = np.linalg.lstsq(half(Du), half(yk), rcond=None)[0]
        d = um - u2
        und["rho=%g" % rho] = float(np.sqrt(max(d @ G @ d, 0.0)))
    out["unmatched_underdetermined_centre_vs_rho"] = und
    out["unmatched_note"] = (
        "m=3 < M=6, so the moment system is UNDERDETERMINED: the minimum-norm "
        "least-squares solution depends on Omega, and the distance to the L2 "
        "centre varies with rho accordingly. This is the regime in which "
        "'Omega cannot move the centre' is FALSE.")

    # overdetermined, inconsistent: 12 probes, 6 atoms
    mk2 = np.linspace(-3.5, 3.5, 12)[:, None]
    vk2 = np.full_like(mk2, 0.36)
    Do = gaussops.design_matrix(mk2, vk2, mu, var)
    Ko = np.exp(-0.5 * ((X[:, None, :] - mk2[None, :, :]) ** 2 / vk2[None, :, :]
                        + np.log(vk2)[None, :, :] + gaussops.LOG_2PI)).squeeze(-1)
    yo = Ko.mean(axis=0)
    Kc = Ko - yo
    Co = (Kc.T @ Kc) / (400 * 399.0)
    over = {}
    sols = []
    for rho in CFG.RHO_SENSITIVITY:
        half = omega_half(Co, "ridge", rho)
        um = np.linalg.lstsq(half(Do), half(yo), rcond=None)[0]
        sols.append(um)
        d = um - u2
        over["rho=%g" % rho] = float(np.sqrt(max(d @ G @ d, 0.0)))
    out["overdetermined_centre_vs_rho"] = over
    out["overdetermined_max_pairwise_coef_spread"] = float(
        max(np.linalg.norm(a - b) for a in sols for b in sols))
    out["overdetermined_residual_norm"] = float(
        np.linalg.norm(Do @ sols[0] - yo))
    out["overdetermined_note"] = (
        "m=12 > M=6 and the system is inconsistent (nonzero residual), so the "
        "weighted least-squares solution genuinely DEPENDS on Omega. The "
        "coefficient spread across the four rho values quantifies that.")
    return out



def _hp_one(mp, atoms, tgt, w):
    """Analytic ISE of one weight vector, in double and in mpmath."""
    M = atoms.size
    am = [mp.mpf(float(v)) for v in atoms.means[:, 0]]
    av = [mp.mpf(float(v)) for v in atoms.variances[:, 0]]
    tw = [mp.mpf(float(v)) for v in tgt.weights]
    tm = [mp.mpf(float(v)) for v in tgt.means[:, 0]]
    tv = [mp.mpf(float(v)) for v in tgt.variances[:, 0]]
    wm = [mp.mpf(float(v)) for v in w]

    def ov(m1, v1, m2, v2):
        s = v1 + v2
        d = m1 - m2
        return mp.e ** (-(d * d / s + mp.log(s) + mp.log(2 * mp.pi)) / 2)

    ff = mp.mpf(0)
    for i in range(M):
        if wm[i] == 0:
            continue
        for j in range(M):
            if wm[j] == 0:
                continue
            ff += wm[i] * wm[j] * ov(am[i], av[i], am[j], av[j])
    fg = mp.mpf(0)
    for i in range(M):
        if wm[i] == 0:
            continue
        for c in range(len(tw)):
            fg += wm[i] * tw[c] * ov(am[i], av[i], tm[c], tv[c])
    gg = mp.mpf(0)
    for c in range(len(tw)):
        for e in range(len(tw)):
            gg += tw[c] * tw[e] * ov(tm[c], tv[c], tm[e], tv[e])
    ise_hp = ff - 2 * fg + gg
    ise_dbl = gaussops.ise_mixtures(w, atoms.means, atoms.variances,
                                    tgt.weights, tgt.means, tgt.variances)
    c0 = float(gg)
    err = abs(float(ise_hp) - ise_dbl)
    import numpy as _np
    return {"ise_double": float(ise_dbl), "ise_mpmath": float(ise_hp),
            "abs_error": err, "int_f0_squared": c0,
            "abs_error_over_c0": err / c0,
            "implied_cancellation_factor_x_eps": err / (_np.finfo(float).eps * c0),
            "n_nonzero_weights": int((_np.asarray(w) > 0).sum())}


def hp_resolution_check():
    """Validate the analytic-ISE resolution floor against mpmath at 50 digits."""
    import mpmath as mp
    mp.mp.dps = CFG.HP_DIGITS

    tgt = get_target("f5")
    X = tgt.sample(CFG.HP_N, data_rng(CFG.NAMESPACE, "f5", CFG.HP_N, 0))
    atoms = build_dictionary(X, DictionaryConfig(
        CFG.HP_DICT["n_loc"], CFG.HP_DICT["n_scale"], "grid",
        CFG.HP_DICT["lo"], CFG.HP_DICT["hi"]))
    M = atoms.size
    rng = np.random.default_rng(7)
    w = rng.random(M)
    w /= w.sum()

    # Two regimes: a generic weight vector (mild cancellation) and the
    # simplex-optimal weight vector (maximal cancellation - this is exactly
    # where R_full* is evaluated, so it is the regime that matters).
    G_a, a_a, c0_a = oracle.exact_risk_parts(atoms, tgt)
    _, w_opt, _ = oracle.min_risk_on(G_a, a_a, c0_a)
    results = {}
    for tag, wv in (("generic_weights_M36", w), ("simplex_optimal_M36", w_opt)):
        results[tag] = _hp_one(mp, atoms, tgt, wv)

    # The decisive case: the FULL 300-atom dictionary optimum on f5 at
    # N = 5000, which is where the double-precision pipeline reported
    # R_full* = 8.3e-17.  w_opt is sparse, so mpmath only touches its support.
    for tname, Nfull in (("f5", 5000), ("f0", 5000)):
        tf = get_target(tname)
        Xf = tf.sample(Nfull, data_rng(CFG.NAMESPACE, tname, Nfull, 0))
        af = build_dictionary(Xf, DictionaryConfig(CFG.DICT["n_loc"],
                                                   CFG.DICT["n_scale"], "grid",
                                                   CFG.DICT["lo"], CFG.DICT["hi"]))
        Gf, aa, c0f = oracle.exact_risk_parts(af, tf)
        _, wf, resf = oracle.min_risk_on(Gf, aa, c0f)
        results["full_dict_optimum_%s_N%d" % (tname, Nfull)] = _hp_one(
            mp, af, tf, wf)
        results["full_dict_optimum_%s_N%d" % (tname, Nfull)]["simplex_kkt"] =             float(resf.kkt_residual)

    factors = [results[k]["implied_cancellation_factor_x_eps"] for k in results]
    import numpy as _np
    worst = max(factors)
    return {
        "M": atoms.size, "N": CFG.HP_N, "target": "f5", "digits": CFG.HP_DIGITS,
        "regimes": results,
        "worst_implied_cancellation_factor_x_eps": worst,
        "configured_cancellation_factor": CFG.CANCELLATION_FACTOR,
        "measured_resolution_floor_x_c0": worst * _np.finfo(float).eps,
        "configured_resolution_floor_x_c0": CFG.CANCELLATION_FACTOR * _np.finfo(float).eps,
        "note": ("The resolution floor is CANCELLATION_FACTOR * eps * int f0^2. "
                 "The configured factor is deliberately conservative; the "
                 "measured worst case over a generic and a simplex-optimal "
                 "weight vector is reported so the floor is validated rather "
                 "than assumed. Any analytic ISE at or below the floor is "
                 "numerically indistinguishable from zero and must not be "
                 "quoted as an orders-of-magnitude comparison."),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = _verify_labels()
    print("target labels verified:", labels)

    tab = centre_table()
    tab.to_csv(OUT / "centre_sensitivity.csv", index=False)
    print("\n== centre distance vs relative rank threshold (median over the 16 datasets) ==")
    piv = tab.pivot_table(index="bank", columns="rank_tol",
                          values="centre_distance_in_G", aggfunc="median")
    print(piv.to_string(float_format=lambda v: "%.4g" % v))
    print("\n== ratio max/min of the centre distance across thresholds, per instance ==")
    rr0 = tab.groupby(["bank", "target", "N", "rep"])["centre_distance_in_G"].agg(
        ["min", "max"])
    rr0["spread_factor"] = rr0["max"] / rr0["min"].replace(0, np.nan)
    spread = rr0.groupby(level=0)["spread_factor"].describe()
    print(spread[["min", "50%", "max"]].to_string(float_format=lambda v: "%.3g" % v))
    print("\n== conditioning (median) ==")
    print(tab.groupby("bank")[["cond_D", "cond_G", "cond_A"]].median().to_string(
        float_format=lambda v: "%.3g" % v))
    print("\n== numerical rank of the whitened design A, by threshold (median) ==")
    print(tab.pivot_table(index="bank", columns="rank_tol", values="rank_A",
                          aggfunc="median").to_string())

    syn = wellconditioned_synthetic()
    print("\n== well-conditioned synthetic dictionary (M=6, cond(G)=%.1f) ==" % syn["cond_G"])
    print("  max|D - G| on the matched bank: %.3e (exact identity)" % syn["max_abs_D_minus_G"])
    print("  matched centre distance to the L2 centre, by rho:")
    for k, v in syn["matched_centre_vs_rho"].items():
        print("    %-10s  ||u_m - u_2||_G = %.3e   relative coef distance %.3e"
              % (k, v["centre_distance_in_G"], v["rel_coef_distance"]))
    print("  UNDERdetermined bank (m=3 < M=6), distance by rho:")
    for k, v in syn["unmatched_underdetermined_centre_vs_rho"].items():
        print("    %-10s  %.4f" % (k, v))
    print("  OVERdetermined inconsistent bank (m=12 > M=6), distance by rho:")
    for k, v in syn["overdetermined_centre_vs_rho"].items():
        print("    %-10s  %.6f" % (k, v))
    print("  max pairwise coefficient spread across rho (overdetermined): %.3e"
          % syn["overdetermined_max_pairwise_coef_spread"])

    hp = hp_resolution_check()
    print("\n== analytic-ISE resolution, validated against mpmath (%d digits) ==" % hp["digits"])
    for tag, rr in hp["regimes"].items():
        print("  %-28s double %.17g  mpmath %.17g  abs err %.3e  "
              "(|err|/int f0^2 = %.3g x eps, %d nonzero weights)"
              % (tag, rr["ise_double"], rr["ise_mpmath"], rr["abs_error"],
                 rr["implied_cancellation_factor_x_eps"], rr["n_nonzero_weights"]))
    print("  worst measured cancellation factor: %.3g x eps  ->  resolution "
          "floor %.3g x int f0^2" % (hp["worst_implied_cancellation_factor_x_eps"],
                                     hp["measured_resolution_floor_x_c0"]))
    print("  configured (conservative) factor %g -> floor %.3g x int f0^2"
          % (hp["configured_cancellation_factor"],
             hp["configured_resolution_floor_x_c0"]))

    with open(OUT / "centre_checks.json", "w", encoding="utf-8") as fh:
        json.dump({"config_version": CFG.CONFIG_VERSION, "labels": labels,
                   "synthetic": syn, "hp_resolution": hp,
                   "centre_table_summary": {
                       "median_by_bank_and_tol": piv.to_dict(),
                       "spread_factor_by_bank": spread.to_dict()}},
                  fh, indent=2, default=float)
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
