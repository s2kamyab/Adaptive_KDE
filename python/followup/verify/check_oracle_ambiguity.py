"""Task 4c - oracle optima with stated numerical uncertainty, and an ambiguity
search over the FULL feasible tangent cone (not just the active face).

Two things the earlier probe could not do:

  * it searched only directions supported on the active face, so it could not
    activate any coordinate that is zero at the optimum;
  * it reported the largest direction it found as if it were a maximum.

Here the search covers the feasible tangent cone at a simplex point `w`:

    T(w) = { v : 1'v = 0,  v_i >= 0 for all i with w_i = 0 }

via three explicit families, each with an exact largest feasible step:

  (F1) face directions  (supp(v) subset supp(w)) - generalised eigenvectors of
       G against D'Omega D restricted to the face, as before;
  (F2) transfer directions  v = e_i - e_j  with i inactive and j active, which
       activate a zero coordinate; step t in [0, w_j];
  (F3) greedy multi-atom transfers built from the best (F2) pairs.

Every result is a WITNESS: a feasible perturbation whose ISE change and
weighted moment-discrepancy change are both computed exactly. Witnesses give
LOWER bounds on achievable ambiguity. No global upper bound is claimed.
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

from akde.densities import get_target              # noqa: E402
from followup.akde_followup import oracle          # noqa: E402
from followup.akde_followup.stage1 import omega_half           # noqa: E402
from followup.verify import check_config as CFG    # noqa: E402
from followup.verify.check_centres import build_instance       # noqa: E402

from followup.paths import DIAG_ROOT   # noqa: E402
OUT = DIAG_ROOT

#: Tolerance on the RELATIVE increase of the weighted moment objective that a
#: perturbation is allowed to incur.  Frozen with the rest of the check config.
MOMENT_TOL = (0.0, 0.01, 0.10)


def weighted_moment_obj(w, A, b):
    r = A @ w - b
    return float(r @ r)


def search_cone(w, G, a_vec, c0, A, b, moment_tol, n_face=6, n_pairs_keep=40):
    """Witness search over T(w). Returns a list of witness records."""
    M = G.shape[0]
    S = np.flatnonzero(w > 1e-12)
    Z = np.flatnonzero(w <= 1e-12)
    base_R = oracle.risk(w, G, a_vec, c0)
    base_Q = weighted_moment_obj(w, A, b)
    out = []

    def evaluate(v, t_max, family, extra=None):
        """Largest ISE change over t in [0, t_max] subject to the moment budget."""
        if t_max <= 0:
            return None
        # both objectives are quadratics in t
        gR = float(2.0 * (G @ w - a_vec) @ v)
        hR = float(v @ G @ v)
        Av = A @ v
        r0 = A @ w - b
        gQ = float(2.0 * r0 @ Av)
        hQ = float(Av @ Av)
        best = None
        for tol in moment_tol:
            budget = tol * base_Q
            # largest t with gQ*t + hQ*t^2 <= budget
            if hQ <= 0:
                t_ok = t_max if gQ <= 0 else min(t_max, budget / max(gQ, 1e-300))
            else:
                disc = gQ * gQ + 4.0 * hQ * budget
                t_ok = t_max if disc < 0 else min(
                    t_max, max(0.0, (-gQ + np.sqrt(max(disc, 0.0))) / (2.0 * hQ)))
            t_ok = max(0.0, min(t_ok, t_max))
            if t_ok <= 0:
                continue
            # |Delta R| is maximised at an endpoint or the vertex of the parabola
            cands = [t_ok]
            if hR > 0:
                tv = -gR / (2.0 * hR)
                if 0 < tv < t_ok:
                    cands.append(tv)
            for t in cands:
                dR = gR * t + hR * t * t
                dQ = gQ * t + hQ * t * t
                recd = {"family": family, "moment_tol": tol, "t": float(t),
                        "t_max_feasible": float(t_max),
                        "delta_ise": float(dR), "abs_delta_ise": float(abs(dR)),
                        "delta_weighted_moment": float(dQ),
                        "rel_delta_weighted_moment": float(dQ / max(base_Q, 1e-300)),
                        "base_ise": base_R, "base_weighted_moment": base_Q,
                        "rel_delta_ise": float(abs(dR) / max(base_R, 1e-300))}
                if extra:
                    recd.update(extra)
                if best is None or recd["abs_delta_ise"] > best["abs_delta_ise"]:
                    best = recd
                out.append(recd)
        return best

    # ---- F1: face directions -------------------------------------------
    if S.size >= 2:
        Gs, As = G[np.ix_(S, S)], (A[:, S].T @ A[:, S])
        k = S.size
        P = np.eye(k) - np.outer(np.ones(k) / np.sqrt(k), np.ones(k) / np.sqrt(k))
        Gt, At = P @ Gs @ P, P @ As @ P
        At = 0.5 * (At + At.T) + 1e-10 * max(float(np.mean(np.diag(At))), 1.0) * np.eye(k)
        try:
            vals, vecs = np.linalg.eigh(np.linalg.solve(At, 0.5 * (Gt + Gt.T)))
            order = np.argsort(-vals.real)
            for j in order[:n_face]:
                vv = vecs[:, j].real
                vv = vv - vv.mean()
                nrm = np.linalg.norm(vv)
                if nrm <= 0:
                    continue
                vv /= nrm
                v = np.zeros(M)
                v[S] = vv
                neg = v < 0
                t_max = float(np.min(w[neg] / (-v[neg]))) if neg.any() else float(w.max())
                evaluate(v, t_max, "F1_face",
                         {"activates_zero_coordinate": False})
        except np.linalg.LinAlgError:
            pass

    # ---- F2: transfer directions e_i - e_j, i inactive, j active --------
    # delta ISE for v = e_i - e_j at step t:  2t(Gw-a)_{i-j} + t^2 (G_ii - 2G_ij + G_jj)
    if S.size and Z.size:
        grad = G @ w - a_vec
        Gd = np.diag(G)
        pairs = []
        for i in Z:
            for j in S:
                hR = Gd[i] - 2.0 * G[i, j] + Gd[j]
                gR = 2.0 * (grad[i] - grad[j])
                t = w[j]
                pairs.append((abs(gR * t + hR * t * t), i, j))
        pairs.sort(reverse=True)
        for _, i, j in pairs[:n_pairs_keep]:
            v = np.zeros(M)
            v[i], v[j] = 1.0, -1.0
            evaluate(v, float(w[j]), "F2_transfer",
                     {"activates_zero_coordinate": True,
                      "enter": int(i), "leave": int(j)})

        # ---- F3: greedy multi-atom transfer -----------------------------
        take = [(i, j) for _, i, j in pairs[:8]]
        v = np.zeros(M)
        for i, j in take:
            v[i] += 1.0
            v[j] -= 1.0
        if np.abs(v).sum() > 0:
            v = v - v.mean()
            neg = v < 0
            t_max = float(np.min(w[neg] / (-v[neg]))) if neg.any() else float(w.max())
            evaluate(v, t_max, "F3_greedy_multi",
                     {"activates_zero_coordinate": True, "n_pairs": len(take)})
    return out, {"support_size": int(S.size), "zero_size": int(Z.size),
                 "base_ise": base_R, "base_weighted_moment": base_Q}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, meta_rows, orc_rows = [], [], []
    for target in CFG.TARGETS:
        for N in CFG.SIZES:
            for bank in CFG.BANKS:
                rep = CFG.REPS[0]
                I = build_instance(target, N, rep, bank, rho=CFG.RHO_MAIN)
                tgt = get_target(target)
                G, a_vec, c0 = oracle.exact_risk_parts(I["atoms"], tgt)
                A, b = I["A"], I["b"]

                # --- oracle optimum with stated numerical uncertainty ------
                R_full, w_full, res_full = oracle.min_risk_on(G, a_vec, c0)
                # independent re-solve from a different start, as a second opinion
                M = G.shape[0]
                from akde import solvers as _s
                x0 = np.full(M, 1.0 / M)
                r2 = _s.simplex_qp(2.0 * G, -2.0 * a_vec, x0=x0)
                R_alt = oracle.risk(r2.x, G, a_vec, c0)
                floor = 25.0 * float(np.finfo(float).eps) * c0
                orc_rows.append({
                    "target": target, "N": N, "bank": bank,
                    "R_full_star": R_full, "R_full_star_alt_start": R_alt,
                    "abs_difference_between_solves": abs(R_full - R_alt),
                    "kkt_rel_primary": res_full.kkt_residual,
                    "kkt_rel_alt": r2.kkt_residual,
                    "int_f0_squared": c0,
                    "analytic_ise_resolution_floor": floor,
                    "R_full_at_or_below_floor": bool(R_full <= floor),
                    "numerical_uncertainty_abs": max(
                        abs(R_full - R_alt), floor),
                })

                w = w_full
                witnesses, meta = search_cone(w, G, a_vec, c0, A, b, MOMENT_TOL)
                for rec in witnesses:
                    rec.update(target=target, N=N, bank=bank, rep=rep)
                    rows.append(rec)
                meta.update(target=target, N=N, bank=bank, rep=rep)
                meta_rows.append(meta)
                print("  %s N=%-5d %-8s |S|=%d |Z|=%d  R_full*=%.4g  witnesses=%d"
                      % (target, N, bank, meta["support_size"], meta["zero_size"],
                         R_full, len(witnesses)), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ambiguity_cone_witnesses.csv", index=False)
    orc = pd.DataFrame(orc_rows)
    orc.to_csv(OUT / "oracle_optima_uncertainty.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(OUT / "ambiguity_cone_meta.csv", index=False)

    print("\n== oracle optima: agreement between two independent simplex solves ==")
    print(orc[["target", "N", "bank", "R_full_star", "R_full_star_alt_start",
               "abs_difference_between_solves", "R_full_at_or_below_floor"]]
          .to_string(index=False, float_format=lambda v: "%.4g" % v))

    print("\n== ambiguity witnesses: largest |Delta ISE| found, by family and "
          "moment budget ==")
    piv = df.pivot_table(index=["bank", "family"], columns="moment_tol",
                         values="abs_delta_ise", aggfunc="max")
    print(piv.to_string(float_format=lambda v: "%.3e" % v))
    print("\n== relative to the exact risk at the optimum ==")
    piv2 = df.pivot_table(index=["bank", "family"], columns="moment_tol",
                          values="rel_delta_ise", aggfunc="max")
    print(piv2.to_string(float_format=lambda v: "%.3e" % v))

    best = df.loc[df.groupby(["bank", "moment_tol"])["abs_delta_ise"].idxmax()]
    print("\n== best witness per (bank, moment budget) ==")
    print(best[["bank", "moment_tol", "target", "N", "family",
                "activates_zero_coordinate", "abs_delta_ise", "base_ise",
                "rel_delta_ise", "rel_delta_weighted_moment"]]
          .to_string(index=False, float_format=lambda v: "%.3e" % v))

    with open(OUT / "ambiguity_cone_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "config_version": CFG.CONFIG_VERSION,
            "moment_tolerances": list(MOMENT_TOL),
            "interpretation": (
                "Every entry is a WITNESS: a feasible direction in the tangent "
                "cone T(w) = {v : 1'v = 0, v_i >= 0 where w_i = 0} together with "
                "an explicit nonnegative step. The reported |Delta ISE| is a "
                "LOWER BOUND on the ambiguity achievable within the stated "
                "weighted-moment budget. No global upper bound is claimed; the "
                "search is not exhaustive over T(w)."),
            "max_abs_delta_ise_by_bank_tol": df.pivot_table(
                index="bank", columns="moment_tol", values="abs_delta_ise",
                aggfunc="max").to_dict(),
            "max_rel_delta_ise_by_bank_tol": df.pivot_table(
                index="bank", columns="moment_tol", values="rel_delta_ise",
                aggfunc="max").to_dict(),
            "fraction_of_best_witnesses_activating_a_zero_coordinate": float(
                best.activates_zero_coordinate.mean()),
        }, fh, indent=2, default=float)
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
