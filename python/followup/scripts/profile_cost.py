"""Where does the time go?  Phase-level profile of V2 AKDE and the finalist.

Profile first, then optimise: this script attributes cost to sample/kernel
evaluation, D construction, covariance assembly, weighting factorisation,
the lambda-path solves, the stage-2 refit, and validation, separately for the
cross-validation phase and the final fit.  It also reports the cold
end-to-end cost and, separately, what a fold-keyed cache would save.

No estimator behaviour is changed here; this only measures.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from akde import gaussops, solvers                           # noqa: E402
from akde.densities import get_target                        # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary  # noqa: E402
from akde.estimators.base import TuningConfig, kfold_indices  # noqa: E402
from akde.metrics import j_score                             # noqa: E402
from followup.akde_followup.seeds import data_rng            # noqa: E402
from followup.akde_followup.variants import (                # noqa: E402
    FollowupAKDE, STAGE_A, build_measurement_bank)

OUT = ROOT / "followup" / "results" / "cost"


def phase_profile(cfg, X, n_folds=3, seed=0):
    """Hand-instrumented replay of fit(): the same work, timed by phase."""
    t = defaultdict(float)
    est = FollowupAKDE(cfg, TuningConfig(n_folds=n_folds))
    cands = est.candidates(X)
    rng = np.random.default_rng(seed)
    folds = kfold_indices(X.shape[0], n_folds, rng, True)
    domain = est._domain(X)
    n_lam_solves = 0
    n_stage2 = 0

    def one(Xa, score_on=None):
        nonlocal n_lam_solves, n_stage2
        t0 = perf_counter()
        atoms = build_dictionary(Xa, DictionaryConfig(cfg.dict_n_loc, cfg.dict_n_scale,
                                                      "grid", cfg.dict_lo_mult,
                                                      cfg.dict_hi_mult))
        bank = build_measurement_bank(Xa, cfg.measurement, dictionary=atoms)
        t["1_dictionary_and_bank"] += perf_counter() - t0
        t0 = perf_counter()
        D = gaussops.design_matrix(bank.means, bank.variances,
                                   atoms.means, atoms.variances)
        t["2_design_matrix_D"] += perf_counter() - t0
        t0 = perf_counter()
        K = FollowupAKDE.eval_kernels(Xa, bank)
        t["3_kernel_evaluation"] += perf_counter() - t0
        t0 = perf_counter()
        y = K.mean(axis=0)
        Kc = K - y
        C = (Kc.T @ Kc) / (Xa.shape[0] * (Xa.shape[0] - 1.0))
        t["4_covariance_assembly"] += perf_counter() - t0

        by_rho = defaultdict(list)
        for i, c in enumerate(cands):
            by_rho[float(c["rho"])].append(i)
        fits = [None] * len(cands)
        for rho, idxs in by_rho.items():
            t0 = perf_counter()
            ap, info = FollowupAKDE.omega_apply(C, cfg.weighting, rho)
            t["5_weighting_factorisation"] += perf_counter() - t0
            t0 = perf_counter()
            H1 = D.T @ ap(D)
            H1 = 0.5 * (H1 + H1.T)
            g = D.T @ ap(y)
            t["6_weighted_hessian"] += perf_counter() - t0
            lam_max = solvers.lambda_max_nonneg(g)
            lams = np.array([cands[i]["lam_ratio"] for i in idxs]) * lam_max
            t0 = perf_counter()
            path = solvers.nonneg_lasso_path(H1, g, lams)
            t["7_lambda_path_stage1"] += perf_counter() - t0
            n_lam_solves += len(lams)
            for slot, res in zip(idxs, path):
                t0 = perf_counter()
                u = res.x
                umax = float(u.max()) if u.size else 0.0
                S = np.flatnonzero(u > cfg.tau * umax) if umax > 0 else np.array([0])
                if S.size == 0:
                    S = np.array([0])
                DS = D[:, S]
                H2 = DS.T @ ap(DS)
                H2 = 0.5 * (H2 + H2.T)
                c2 = -DS.T @ ap(y)
                st = u[S] / u[S].sum() if u[S].sum() > 0 else None
                r2 = solvers.simplex_qp(H2, c2, x0=st)
                t["8_stage2_refit"] += perf_counter() - t0
                n_stage2 += 1
                w = np.zeros(atoms.size)
                w[S] = r2.x
                keep = w > 0
                from akde.model import mixture_density
                fits[slot] = mixture_density(w[keep], atoms.means[keep],
                                             atoms.variances[keep])
        if score_on is not None:
            t0 = perf_counter()
            for f in fits:
                j_score(f, score_on, domain=domain)
            t["9_validation_scoring"] += perf_counter() - t0
        return fits

    t_all = perf_counter()
    for tr, va in folds:
        one(X[tr], score_on=X[va])
    t["A_cv_total"] = perf_counter() - t_all
    t_all = perf_counter()
    one(X)
    t["B_final_path_total"] = perf_counter() - t_all
    t["_n_candidates"] = len(cands)
    t["_n_stage1_solves"] = n_lam_solves
    t["_n_stage2_solves"] = n_stage2
    return dict(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="f5,f9")
    ap.add_argument("--sizes", default="500,5000")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {"v2_akde": STAGE_A["A1_v2_moment_m300"],
                "finalist": STAGE_A["A6_moment_matched_m300"]}
    rows = []
    for tname in a.targets.split(","):
        for N in [int(s) for s in a.sizes.split(",")]:
            X = get_target(tname).sample(N, data_rng("development", tname, N, 0))
            for vname, cfg in variants.items():
                p = phase_profile(cfg, X)
                p.update(target=tname, N=N, variant=vname)
                rows.append(p)
                phases = {k: v for k, v in p.items()
                          if k[0].isdigit()}
                tot = sum(phases.values())
                print("\n%s  %s N=%d   instrumented total %.2f s "
                      "(CV %.2f + final %.2f); %d candidates, %d stage-1 solves, "
                      "%d stage-2 solves"
                      % (vname, tname, N, tot, p["A_cv_total"], p["B_final_path_total"],
                         p["_n_candidates"], p["_n_stage1_solves"], p["_n_stage2_solves"]))
                for k in sorted(phases):
                    print("    %-28s %7.3f s  %5.1f%%" % (k, phases[k],
                                                          100 * phases[k] / max(tot, 1e-12)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase_profile.csv", index=False)
    phase_cols = [c for c in df.columns if c[0].isdigit()]
    share = df.groupby("variant")[phase_cols].sum()
    share = share.div(share.sum(axis=1), axis=0)
    share.to_csv(OUT / "phase_share.csv")
    print("\n== share of instrumented time by phase, pooled over the probe grid ==")
    print((100 * share).round(1).T.to_string())
    print("\nwrote", OUT / "phase_profile.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
