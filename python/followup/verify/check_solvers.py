"""Task 4b - verify BOTH optimisation stages on the frozen 16-dataset subset.

For every instance this independently re-derives the objective and the KKT
system from the definition (not from the solver's own bookkeeping) and reports:

  * absolute and scaled KKT residuals for stage 1 and stage 2;
  * feasibility of the stage-2 iterate;
  * objective gap against the best point found by ANY solver on that instance
    (a lower bound on suboptimality, not a certificate);
  * support overlap, fitted-density L2 difference, ISE change, runtime;
  * sensitivity to the stage-1 iteration budget (500 / 2000 / 10000).

An alternative solver is not treated as a reference because it returns a
smaller objective: its own convergence status is reported alongside.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from akde import gaussops, solvers                 # noqa: E402
from akde.densities import get_target              # noqa: E402
from followup.akde_followup import oracle          # noqa: E402
from followup.akde_followup.stage1 import (        # noqa: E402
    nonneg_lasso_path_fista, omega_half)
from followup.verify import check_config as CFG    # noqa: E402
from followup.verify.check_centres import build_instance      # noqa: E402

from followup.paths import DIAG_ROOT   # noqa: E402
OUT = DIAG_ROOT


# ---------------------------------------------------------------- objectives
def stage1_obj(u, H, g, lam):
    """0.5 u'Hu - g'u + lam 1'u, re-derived from the definition."""
    return float(0.5 * u @ (H @ u) - g @ u + lam * u.sum())


def stage1_kkt(u, H, g, lam):
    """Return (absolute, scaled) KKT residuals for min_{u>=0} of stage1_obj."""
    grad = H @ u - g + lam
    viol_pos = np.abs(np.where(u > 0, grad, 0.0))
    viol_zero = np.maximum(0.0, -np.where(u <= 0, grad, 0.0))
    absr = float(max(viol_pos.max(initial=0.0), viol_zero.max(initial=0.0)))
    ref = float(np.max(np.abs(g))) + float(np.mean(np.diag(H)))
    return absr, absr / max(ref, 1e-300), ref


def stage2_obj(al, H2, c2):
    return float(0.5 * al @ (H2 @ al) + c2 @ al)


def stage2_kkt(al, H2, c2):
    """Simplex QP stationarity: grad = nu on the support, >= nu off it."""
    grad = H2 @ al + c2
    supp = al > 1e-12
    if not supp.any():
        return np.inf, np.inf, 1.0
    nu = float(np.mean(grad[supp]))
    on = float(np.max(np.abs(grad[supp] - nu)))
    off = ~supp
    offv = float(np.max(np.maximum(0.0, nu - grad[off]))) if off.any() else 0.0
    absr = max(on, offv)
    scale = max(1.0, float(np.max(np.abs(grad))))
    return absr, absr / scale, scale


def stage2_feasibility(al):
    return {"sum_minus_one": float(abs(al.sum() - 1.0)),
            "min_coefficient": float(al.min())}


# ---------------------------------------------------------------- stage 2 reference
def simplex_fista(H, c, max_iter=200000, tol=1e-14):
    """Independent accelerated projected-gradient reference for the simplex QP.

    Reported with its own achieved KKT residual; it is a second opinion, not a
    certificate.
    """
    n = c.size
    x = np.full(n, 1.0 / n)
    L = float(np.linalg.eigvalsh(0.5 * (H + H.T))[-1])
    if not np.isfinite(L) or L <= 0:
        L = max(float(np.abs(H).max()) * n, 1.0)
    step = 1.0 / L
    y = x.copy()
    t = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        g = H @ y + c
        xn = solvers.project_simplex(y - step * g)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = xn + ((t - 1.0) / tn) * (xn - x)
        if float((z - xn) @ (xn - x)) < 0:
            z, tn = xn.copy(), 1.0
        if np.max(np.abs(xn - x)) <= tol:
            x, y, t = xn, z, tn
            break
        x, y, t = xn, z, tn
    return x, it


# ---------------------------------------------------------------- per instance
def run_instance(target, N, rep, bank, rho, lam_ratio, budgets):
    I = build_instance(target, N, rep, bank, rho=rho)
    D, y, C, atoms = I["D"], I["y"], I["C"], I["atoms"]
    half = omega_half(C, "ridge", rho)
    A, b = half(D), half(y)
    H = A.T @ A
    H = 0.5 * (H + H.T)
    g = A.T @ b
    lam_max = solvers.lambda_max_nonneg(g)
    lam = lam_ratio * lam_max
    tgt = get_target(target)
    G, a_vec, c0 = oracle.exact_risk_parts(atoms, tgt)

    rows = []
    solutions = {}
    for budget in budgets:
        t0 = perf_counter()
        u = solvers.nonneg_lasso_path(H, g, np.array([lam]), max_iter=budget)[0]
        dt = perf_counter() - t0
        solutions[("cd", budget)] = (u.x, dt, u.iterations, bool(u.converged))
    t0 = perf_counter()
    ref = float(np.max(np.abs(g))) + float(np.mean(np.diag(H)))
    fi = nonneg_lasso_path_fista(A, b, np.array([lam]), ref=ref,
                                 kkt_tol=1e-10, max_iter=40000)[0]
    solutions[("fista", 40000)] = (fi.x, perf_counter() - t0, fi.iterations,
                                   bool(fi.converged))

    best_obj = min(stage1_obj(v[0], H, g, lam) for v in solutions.values())
    for (solver, budget), (u, dt, iters, conv) in solutions.items():
        absr, relr, scale = stage1_kkt(u, H, g, lam)
        obj = stage1_obj(u, H, g, lam)
        umax = float(u.max()) if u.size else 0.0
        S = np.flatnonzero(u > 1e-3 * umax) if umax > 0 else np.array([], int)
        rows.append({
            "target": target, "N": N, "rep": rep, "bank": bank, "rho": rho,
            "lam_ratio": lam_ratio, "stage1_solver": solver,
            "stage1_budget": budget, "stage1_iters": iters,
            "stage1_converged_flag": conv,
            "stage1_obj": obj, "stage1_obj_gap_vs_best": obj - best_obj,
            "stage1_rel_obj_gap": (obj - best_obj) / max(abs(best_obj), 1e-300),
            "stage1_kkt_abs": absr, "stage1_kkt_rel": relr,
            "stage1_kkt_scale": scale,
            "stage1_time_s": dt, "stage1_support": int(S.size),
            "_S": S, "_u": u, "_H": H, "_g": g, "_lam": lam,
        })
    return rows, I, half, G, a_vec, c0


def stage2_on_support(I, half, S, u):
    """Stage-2 Hessian and linear term, built from the WHITENED design.

    Omega = half' half with half = L^{-1}; half is not symmetric, so the
    weighted forms are built as A_S' A_S and -A_S' b with A_S = half(D_S),
    which is algebraically identical to D_S' Omega D_S and -D_S' Omega y.
    """
    D, y = I["D"], I["y"]
    DS = D[:, S]
    AS = half(DS)
    bS = half(y)
    H2 = AS.T @ AS
    H2 = 0.5 * (H2 + H2.T)
    c2 = -AS.T @ bS
    return H2, c2


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    lam_ratios = (0.7, 0.0316, 3e-4)      # top, middle, bottom of the frozen path
    for target in CFG.TARGETS:
        for N in CFG.SIZES:
            for rep in CFG.REPS:
                for bank in CFG.BANKS:
                    for lr in lam_ratios:
                        rows, I, half, G, a_vec, c0 = run_instance(
                            target, N, rep, bank, CFG.RHO_MAIN, lr,
                            CFG.STAGE1_BUDGETS)
                        for r in rows:
                            S, u = r.pop("_S"), r.pop("_u")
                            r.pop("_H"), r.pop("_g"), r.pop("_lam")
                            if S.size == 0:
                                r.update(stage2_status="empty_support")
                                recs.append(r)
                                continue
                            H2, c2 = stage2_on_support(I, half, S, u)
                            st = perf_counter()
                            r2 = solvers.simplex_qp(
                                H2, c2, x0=u[S] / u[S].sum() if u[S].sum() > 0 else None)
                            t_orig = perf_counter() - st
                            st = perf_counter()
                            xr, it_r = simplex_fista(H2, c2)
                            t_ref = perf_counter() - st

                            o_orig, o_ref = stage2_obj(r2.x, H2, c2), stage2_obj(xr, H2, c2)
                            a_orig = stage2_kkt(r2.x, H2, c2)
                            a_ref = stage2_kkt(xr, H2, c2)
                            fo, fr = np.zeros(G.shape[0]), np.zeros(G.shape[0])
                            fo[S], fr[S] = r2.x, xr
                            d = fo - fr
                            r.update(
                                stage2_status="ok",
                                stage2_obj_orig=o_orig, stage2_obj_ref=o_ref,
                                stage2_obj_gap=o_orig - min(o_orig, o_ref),
                                stage2_kkt_abs_orig=a_orig[0],
                                stage2_kkt_rel_orig=a_orig[1],
                                stage2_kkt_abs_ref=a_ref[0],
                                stage2_kkt_rel_ref=a_ref[1],
                                stage2_ref_iters=it_r,
                                stage2_time_orig_s=t_orig, stage2_time_ref_s=t_ref,
                                **{("stage2_feas_orig_" + k): v
                                   for k, v in stage2_feasibility(r2.x).items()},
                                **{("stage2_feas_ref_" + k): v
                                   for k, v in stage2_feasibility(xr).items()},
                                fitted_density_l2_orig_vs_ref=float(
                                    np.sqrt(max(d @ G @ d, 0.0))),
                                ise_orig=oracle.risk(fo, G, a_vec, c0),
                                ise_ref=oracle.risk(fr, G, a_vec, c0),
                                support_orig=int((r2.x > 1e-12).sum()),
                                support_ref=int((xr > 1e-12).sum()),
                                support_jaccard=float(
                                    len(set(np.flatnonzero(r2.x > 1e-12)) &
                                        set(np.flatnonzero(xr > 1e-12))) /
                                    max(len(set(np.flatnonzero(r2.x > 1e-12)) |
                                             set(np.flatnonzero(xr > 1e-12))), 1)),
                            )
                            recs.append(r)
                    print("  done %s N=%d rep=%d %s" % (target, N, rep, bank), flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "solver_verification.csv", index=False)

    ok = df[df.stage2_status == "ok"].copy()
    ok["ise_rel_change"] = (ok.ise_orig - ok.ise_ref).abs() / ok.ise_ref

    print("\n== STAGE 1: relative KKT residual by solver and budget (median / max) ==")
    print(ok.groupby(["stage1_solver", "stage1_budget"])["stage1_kkt_rel"]
          .agg(["median", "max", "size"]).to_string(float_format=lambda v: "%.3e" % v))
    print("\n== STAGE 1: fraction reaching the solver's own tolerance ==")
    print(ok.groupby(["stage1_solver", "stage1_budget"])["stage1_converged_flag"]
          .mean().to_string(float_format=lambda v: "%.3f" % v))
    print("\n== STAGE 1: relative objective gap vs the best point found ==")
    print(ok.groupby(["stage1_solver", "stage1_budget"])["stage1_rel_obj_gap"]
          .agg(["median", "max"]).to_string(float_format=lambda v: "%.3e" % v))
    print("\n== STAGE 1: budget sensitivity of the selected support (CD only) ==")
    cd = ok[ok.stage1_solver == "cd"]
    piv = cd.pivot_table(index=["target", "N", "rep", "bank", "lam_ratio"],
                         columns="stage1_budget", values="stage1_support")
    if set(CFG.STAGE1_BUDGETS) <= set(piv.columns):
        same = (piv[CFG.STAGE1_BUDGETS[0]] == piv[CFG.STAGE1_BUDGETS[-1]]).mean()
        print("  support identical at budget %d vs %d in %.1f%% of instances; "
              "median |change| = %.1f atoms"
              % (CFG.STAGE1_BUDGETS[0], CFG.STAGE1_BUDGETS[-1], 100 * same,
                 float((piv[CFG.STAGE1_BUDGETS[-1]] - piv[CFG.STAGE1_BUDGETS[0]])
                       .abs().median())))

    print("\n== STAGE 2: relative KKT residual, original vs independent reference ==")
    print(ok[["stage2_kkt_rel_orig", "stage2_kkt_rel_ref"]]
          .agg(["median", "max"]).to_string(float_format=lambda v: "%.3e" % v))
    print("\n== STAGE 2: feasibility ==")
    print(ok[["stage2_feas_orig_sum_minus_one", "stage2_feas_orig_min_coefficient",
              "stage2_feas_ref_sum_minus_one", "stage2_feas_ref_min_coefficient"]]
          .agg(["max", "min"]).to_string(float_format=lambda v: "%.3e" % v))
    print("\n== STAGE 2: agreement between the two solvers ==")
    print("  fitted-density L2 difference: median %.3e, max %.3e"
          % (ok.fitted_density_l2_orig_vs_ref.median(),
             ok.fitted_density_l2_orig_vs_ref.max()))
    print("  relative ISE change: median %.3e, max %.3e"
          % (ok.ise_rel_change.median(), ok.ise_rel_change.max()))
    print("  support Jaccard: median %.3f, min %.3f"
          % (ok.support_jaccard.median(), ok.support_jaccard.min()))
    print("  runtime original %.4f s vs reference %.4f s (median)"
          % (ok.stage2_time_orig_s.median(), ok.stage2_time_ref_s.median()))

    pe = CFG.PRACTICAL_EQUIVALENCE
    meets = bool((ok.ise_rel_change.max() <= pe["max_rel_ise_change"]) and
                 (ok.fitted_density_l2_orig_vs_ref.max() <= pe["max_fitted_density_l2"]))
    print("\n== predefined practical-equivalence criterion (frozen before the run) ==")
    print("  max relative ISE change <= %.3f  -> observed %.3e"
          % (pe["max_rel_ise_change"], ok.ise_rel_change.max()))
    print("  max fitted-density L2  <= %.1e -> observed %.3e"
          % (pe["max_fitted_density_l2"], ok.fitted_density_l2_orig_vs_ref.max()))
    print("  VERDICT: stage-2 solvers are %s on this subset"
          % ("PRACTICALLY EQUIVALENT" if meets else "NOT practically equivalent"))

    with open(OUT / "solver_verification_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "config_version": CFG.CONFIG_VERSION,
            "n_instances": int(len(ok)),
            "stage1_kkt_rel": ok.groupby(["stage1_solver", "stage1_budget"])
            ["stage1_kkt_rel"].agg(["median", "max"]).reset_index().to_dict("records"),
            "stage1_converged_fraction": ok.groupby(
                ["stage1_solver", "stage1_budget"])["stage1_converged_flag"]
            .mean().reset_index().to_dict("records"),
            "stage2_kkt_rel_orig": {"median": float(ok.stage2_kkt_rel_orig.median()),
                                    "max": float(ok.stage2_kkt_rel_orig.max())},
            "stage2_kkt_rel_ref": {"median": float(ok.stage2_kkt_rel_ref.median()),
                                   "max": float(ok.stage2_kkt_rel_ref.max())},
            "stage2_practical_equivalence": {"criterion": pe, "meets": meets,
                                             "max_rel_ise_change": float(ok.ise_rel_change.max()),
                                             "max_density_l2": float(ok.fitted_density_l2_orig_vs_ref.max())},
        }, fh, indent=2)
    print("\nwrote", OUT / "solver_verification.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
