"""Oracle and identification diagnostics for the proposed estimator.

These are **separately labelled oracle diagnostics**: they use the true density
to answer questions about the estimator's structure.  They must never be mixed
into the main comparison, and none of them is an estimator anyone could run in
practice.  They exist to attribute error to a cause.

    python experiments/run_diagnostics.py --targets f5,f9,f13 --out results/diagnostics

Three questions:

1. **Can the dictionary represent the target at all?**
   The oracle bound is the best simplex combination of dictionary atoms
   measured directly against the truth.  No data-driven fit on that dictionary
   can beat it.  If the bound is tiny, approximation error is not the problem.

2. **Is the limitation support selection, or the fitting criterion?**
   Fix the support the estimator actually selected and refit it two ways:
   L2-optimally (oracle) and moment-optimally (what stage 2 does).  A large gap
   between them means the moment loss is nearly flat along directions that
   carry real L2 error -- i.e. the criterion, not the support, is binding.

3. **How ill-posed is the moment system?**
   Reports the condition number and effective rank of the selected design
   ``D_S``, and the rank of the empirical covariance against its ceiling
   ``min(m, N-1)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akde import gaussops, metrics, solvers
from akde.densities import get_target
from akde.dictionary import build_dictionary, build_measurements
from akde.estimators.akde import AKDE, AKDEConfig
from akde.model import mixture_density
from akde.registry import shared_dictionary
from akde.runner import name_key


def simplex_l2_fit(means, variances, target, subset=None):
    """Best simplex combination in L2 against the TRUE density (oracle)."""
    if subset is not None:
        means, variances = means[subset], variances[subset]
    G = gaussops.overlap_diag(means, variances, means, variances)
    c = gaussops.overlap_diag(means, variances, target.means, target.variances) @ target.weights
    r = solvers.simplex_qp(2.0 * G, -2.0 * c)
    keep = r.x > 0
    fit = mixture_density(r.x[keep], means[keep], variances[keep])
    return metrics.ise(fit, target), int(keep.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="f5,f9,f13")
    ap.add_argument("--sizes", default="100,500,1000,5000")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-loc", type=int, default=150)
    ap.add_argument("--n-scale", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", default="results/diagnostics")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]

    rows = []
    for tname in targets:
        target = get_target(tname)
        for N in sizes:
            for rep in range(args.reps):
                rng = np.random.default_rng([args.seed, name_key(tname), N, rep])
                X = target.sample(N, rng)
                cfg = AKDEConfig(dictionary=shared_dictionary(args.n_loc, args.n_scale))
                est = AKDE(cfg)
                fit = est.fit(X, np.random.default_rng([args.seed, rep, 7]))
                S = np.asarray(fit.support)

                atoms = build_dictionary(X, cfg.dictionary,
                                         rng=np.random.default_rng([args.seed, rep, 7]))
                meas = build_measurements(
                    X, "independent", n_loc=cfg.meas_n_loc, n_scale=cfg.meas_n_scale,
                    scale_lo_mult=cfg.meas_scale_lo_mult,
                    scale_hi_mult=cfg.meas_scale_hi_mult,
                    rng=np.random.default_rng([args.seed, rep, 7]))
                D = gaussops.design_matrix(meas.means, meas.variances,
                                           atoms.means, atoms.variances)
                DS = D[:, S]
                sv = np.linalg.svd(DS, compute_uv=False)

                ise_oracle_full, k_full = simplex_l2_fit(atoms.means, atoms.variances, target)
                ise_oracle_S, _ = simplex_l2_fit(atoms.means, atoms.variances, target, S)

                rows.append({
                    "target": tname, "label": target.label, "N": N, "rep": rep,
                    "ise_estimator": metrics.ise(fit, target),
                    "ise_oracle_support": ise_oracle_S,
                    "ise_oracle_dictionary": ise_oracle_full,
                    "n_support": int(S.size),
                    "n_oracle_support": k_full,
                    "n_dictionary": atoms.size,
                    "n_measurements": meas.size,
                    "cond_DS": float(sv[0] / max(sv[-1], 1e-300)),
                    "eff_rank_DS": float(np.exp(-np.sum(
                        (sv ** 2 / (sv ** 2).sum()) * np.log(sv ** 2 / (sv ** 2).sum() + 1e-300)))),
                    "cov_rank": fit.diagnostics.get("cov_rank_numeric"),
                    "cov_rank_ceiling": min(meas.size, N - 1),
                    "cov_effective_rank": fit.diagnostics.get("cov_effective_rank"),
                    "omega_cond": fit.diagnostics.get("omega_cond"),
                    "selected_rho": fit.diagnostics.get("rho"),
                    "selected_lam_ratio": fit.diagnostics.get("lam_ratio"),
                })
                print(f"{tname} N={N} rep={rep}: est={rows[-1]['ise_estimator']:.3e} "
                      f"oracleS={ise_oracle_S:.3e} oracleDict={ise_oracle_full:.3e} "
                      f"|S|={S.size} cond(D_S)={rows[-1]['cond_DS']:.2e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "diagnostics_raw.csv", index=False)

    summ = df.groupby(["target", "N"]).agg(
        ise_estimator=("ise_estimator", "median"),
        ise_oracle_support=("ise_oracle_support", "median"),
        ise_oracle_dictionary=("ise_oracle_dictionary", "median"),
        n_support=("n_support", "median"),
        cond_DS=("cond_DS", "median"),
        cov_rank=("cov_rank", "median"),
        cov_rank_ceiling=("cov_rank_ceiling", "median"),
    ).reset_index()
    # how much of the gap is the criterion rather than the support?
    summ["loss_mismatch_factor"] = summ["ise_estimator"] / summ["ise_oracle_support"]
    summ["support_loss_factor"] = summ["ise_oracle_support"] / summ["ise_oracle_dictionary"]
    summ.to_csv(out / "diagnostics_summary.csv", index=False)

    print("\n" + summ.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {out}/diagnostics_summary.csv")
    print("\nloss_mismatch_factor = ISE(moment refit) / ISE(L2 refit on the SAME support).")
    print("A large value means the fitting criterion, not support selection, is binding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
