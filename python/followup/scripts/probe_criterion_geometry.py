"""Why does the measurement bank matter so much?  The geometry of the criterion.

Two quadratics are compared on the same dictionary:

    moment   Q_m(u) = (D u - y)' Omega (D u - y)
    L2       Q_2(u) = u' G u - 2 s' u          (the SPADES / empirical-L2 loss)

Unconstrained, each is a quadratic with a centre:

    u_m = (D' Omega D)^{-1} D' Omega y      u_2 = G^{-1} s

When the measurement family equals the basis family, D = G and y = s, so

    Q_m(u) = (u - G^{-1}s)' G Omega G (u - G^{-1}s) + const
    Q_2(u) = (u - G^{-1}s)' G       (u - G^{-1}s) + const

i.e. the two criteria have the SAME centre and differ only in the metric they
measure deviation from it.  When the families differ, the centres differ too,
and the distance between them is error that no weighting matrix can remove.

This script measures, for each bank: cond(D), the distance between the two
centres in the exact-risk metric (which is the part of the error attributable
to the measurement design), and the alignment between the criterion's metric
D'Omega D and the ISE metric G.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from akde import gaussops                                    # noqa: E402
from akde.densities import get_target                        # noqa: E402
from followup.akde_followup import oracle                    # noqa: E402
from followup.akde_followup.figdata import pinned_dictionary  # noqa: E402
from followup.akde_followup.seeds import data_rng            # noqa: E402
from followup.akde_followup.variants import (                # noqa: E402
    BANK_MATCHED, BANK_WIDE_INDEP, BASE_BANK, FollowupAKDE, build_measurement_bank)

OUT = ROOT / "followup" / "results" / "criterion_geometry"
BANKS = [("A1_base_100x3_0.25-2h", BASE_BANK),
         ("B1b_indep_75x4_0.05-4h", BANK_WIDE_INDEP),
         ("A6_matched_60x5_0.05-4h", BANK_MATCHED)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tname in ["f0", "f2", "f5", "f9"]:
        tgt = get_target(tname)
        for N in [500, 5000]:
            X = data_rng, None
            X = get_target(tname).sample(N, data_rng("development", tname, N, 0))
            atoms = pinned_dictionary(X)
            G, avec, c0 = oracle.exact_risk_parts(atoms, tgt)
            s = FollowupAKDE.eval_kernels(X, atoms).mean(axis=0)
            # L2 centre, regularised only enough to be solvable
            jitter = 1e-12 * float(np.mean(np.diag(G)))
            Gj = G + jitter * np.eye(G.shape[0])
            u2 = np.linalg.solve(Gj, s)
            for bname, spec in BANKS:
                bank = build_measurement_bank(X, spec, dictionary=atoms)
                D = gaussops.design_matrix(bank.means, bank.variances,
                                           atoms.means, atoms.variances)
                K = FollowupAKDE.eval_kernels(X, bank)
                y = K.mean(axis=0)
                Kc = K - y
                C = (Kc.T @ Kc) / (N * (N - 1.0))
                ap, _ = FollowupAKDE.omega_apply(C, "ridge", 1e-3)
                A = D.T @ ap(D)
                A = 0.5 * (A + A.T)
                b = D.T @ ap(y)
                Aj = A + 1e-12 * float(np.mean(np.diag(A))) * np.eye(A.shape[0])
                um = np.linalg.solve(Aj, b)
                d = um - u2
                # metric alignment: correlation between the two quadratic forms
                # over random directions
                rng = np.random.default_rng(0)
                V = rng.standard_normal((G.shape[0], 400))
                V /= np.linalg.norm(V, axis=0, keepdims=True)
                qg = np.einsum("ij,ij->j", V, G @ V)
                qa = np.einsum("ij,ij->j", V, A @ V)
                rows.append({
                    "target": tname, "N": N, "bank": bname, "m": int(bank.size),
                    "M": int(atoms.size),
                    "cond_D": float(np.linalg.cond(D)),
                    "cond_DtOmD": float(np.linalg.cond(A)),
                    "cond_G": float(np.linalg.cond(G)),
                    "centre_distance_in_G": float(np.sqrt(max(d @ G @ d, 0.0))),
                    "risk_at_moment_centre": oracle.risk(um, G, avec, c0),
                    "risk_at_l2_centre": oracle.risk(u2, G, avec, c0),
                    "metric_rank_corr_G_vs_DtOmD": float(
                        pd.Series(np.log(np.maximum(qg, 1e-300))).corr(
                            pd.Series(np.log(np.maximum(qa, 1e-300))), method="spearman")),
                    "D_equals_G": bool(np.allclose(D, G)),
                    "y_equals_s": bool(np.allclose(y, s)),
                })
                print("  %-5s N=%-5d %-26s cond(D)=%.3e  ||u_m-u_2||_G=%.4g  "
                      "D==G:%s" % (tname, N, bname, rows[-1]["cond_D"],
                                   rows[-1]["centre_distance_in_G"],
                                   rows[-1]["D_equals_G"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "criterion_geometry.csv", index=False)
    print("\n== medians by bank ==")
    print(df.groupby("bank")[["cond_D", "cond_DtOmD", "centre_distance_in_G",
                              "metric_rank_corr_G_vs_DtOmD"]].median().to_string(
        float_format=lambda v: "%.4g" % v))
    print("\nwrote", OUT / "criterion_geometry.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
