"""Does stage 1 actually solve its own problem?  Two solvers, one objective.

For a grid of (target, N, dictionary, rho, lambda) this compares V2's capped
coordinate descent against residual-space FISTA on the *identical* objective
and reports, for each:

  * the achieved relative KKT residual (same scale V2 reports),
  * the objective value,
  * the selected support and how much it differs,
  * wall time and matrix-vector work.

No estimator decision is made here; this is a solver audit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akde import gaussops, solvers                                   # noqa: E402
from akde.densities import get_target                                # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary       # noqa: E402
from followup.akde_followup.stage1 import (                          # noqa: E402
    nonneg_lasso_path_fista, omega_half)
from followup.akde_followup.variants import (                        # noqa: E402
    BASE_BANK, FollowupAKDE, build_measurement_bank)

OUT = ROOT / "followup" / "results" / "solver_audit"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for tname in ["f0", "f2", "f5", "f9"]:
    for N in [500, 5000]:
        for (nl, ns) in [(60, 5), (150, 10)]:
            tgt = get_target(tname)
            X = tgt.sample(N, np.random.default_rng(1))
            atoms = build_dictionary(X, DictionaryConfig(nl, ns, "grid", 0.05, 4.0))
            bank = build_measurement_bank(X, BASE_BANK)
            D = gaussops.design_matrix(bank.means, bank.variances,
                                       atoms.means, atoms.variances)
            K = FollowupAKDE.eval_kernels(X, bank)
            y = K.mean(axis=0)
            Kc = K - y
            C = (Kc.T @ Kc) / (N * (N - 1.0))
            for rho in [1e-3, 1.0]:
                apply_omega, _ = FollowupAKDE.omega_apply(C, "ridge", rho)
                H = D.T @ apply_omega(D)
                H = 0.5 * (H + H.T)
                g = D.T @ apply_omega(y)
                ref = max(float(np.max(np.abs(g))) + float(np.mean(np.diag(H))), 1e-300)
                lam_max = solvers.lambda_max_nonneg(g)
                lams = np.geomspace(0.7, 3e-4, 6) * lam_max

                t0 = perf_counter()
                p_cd = solvers.nonneg_lasso_path(H, g, lams)
                t_cd = perf_counter() - t0

                half = omega_half(C, "ridge", rho)
                A = half(D)
                b = half(y)
                t0 = perf_counter()
                p_fi = nonneg_lasso_path_fista(A, b, lams, ref=ref)
                t_fi = perf_counter() - t0

                # objective on a common scale: 0.5 u'Hu - g'u + lam*sum(u)
                def obj(u, lam):
                    return float(0.5 * u @ (H @ u) - g @ u + lam * u.sum())

                for li, lam in enumerate(lams):
                    uc, uf = p_cd[li].x, p_fi[li].x
                    Sc = set(np.flatnonzero(uc > 1e-3 * max(uc.max(), 0)).tolist())
                    Sf = set(np.flatnonzero(uf > 1e-3 * max(uf.max(), 0)).tolist())
                    oc, of = obj(uc, lam), obj(uf, lam)
                    rows.append({
                        "target": tname, "N": N, "M": atoms.size, "m": bank.size,
                        "rho": rho, "lam_ratio": float(lam / lam_max) if lam_max else 0.0,
                        "cond_H": float(np.linalg.cond(H)) if li == 0 else np.nan,
                        "kkt_cd": p_cd[li].kkt_residual, "kkt_fista": p_fi[li].kkt_residual,
                        "conv_cd": bool(p_cd[li].converged),
                        "conv_fista": bool(p_fi[li].converged),
                        "iters_cd": p_cd[li].iterations, "iters_fista": p_fi[li].iterations,
                        "obj_cd": oc, "obj_fista": of,
                        "obj_gap_rel": (oc - of) / max(abs(of), 1e-300),
                        "supp_cd": len(Sc), "supp_fista": len(Sf),
                        "supp_jaccard": (len(Sc & Sf) / max(len(Sc | Sf), 1)),
                        "path_time_cd_s": t_cd, "path_time_fista_s": t_fi,
                    })
            print("done %s N=%d M=%d  cd %.2fs  fista %.2fs"
                  % (tname, N, atoms.size, t_cd, t_fi), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "stage1_solver_comparison.csv", index=False)
print("\n== achieved relative KKT residual ==")
print(df.groupby(["M"])[["kkt_cd", "kkt_fista"]].median().to_string())
print("\n== fraction reaching the 1e-8 tolerance ==")
print(df.groupby(["M"])[["conv_cd", "conv_fista"]].mean().to_string())
print("\n== objective gap (cd - fista) / |fista|  : positive means FISTA found a lower objective ==")
print(df.groupby(["M", "rho"])["obj_gap_rel"].describe()[["50%", "max"]].to_string())
print("\n== support agreement (Jaccard) and sizes ==")
print(df.groupby(["M"])[["supp_jaccard", "supp_cd", "supp_fista"]].median().to_string())
print("\n== path wall time (s), 6 lambdas ==")
print(df.groupby(["M", "N"])[["path_time_cd_s", "path_time_fista_s"]].median().to_string())
print("\nwrote", OUT / "stage1_solver_comparison.csv")
