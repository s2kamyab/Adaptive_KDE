"""Oracle controls: separate sampling noise from criterion mismatch.

For each development cell, on the pinned 60x5 dictionary, this computes a
ladder of fits that differ in exactly one ingredient at a time:

  R_full*            minimum exact risk on the full dictionary simplex
                     (what the dictionary can do at all)
  moment(y0, C0)     full-simplex moment fit with POPULATION moments and
                     POPULATION covariance weighting -> no sampling error at
                     all; any excess over R_full* is the criterion
  moment(y0, C_hat)  population moments, estimated weighting -> isolates the
                     effect of estimating Omega
  moment(y,  C0)     empirical moments, population weighting
  moment(y,  C_hat)  the deployable full-simplex moment fit (= variant A2)

and, alongside them, the L2 ladder:

  l2(a)              full-simplex fit to the POPULATION L2 risk = R_full*
  l2(s)              full-simplex fit to the empirical integrated-square loss
                     (= variant A4)

Everything here needs f0 and is therefore an oracle control.  None of it is
returned to any estimator.

It also records the ambiguity probe: feasible simplex directions with small
weighted moment response but large v'Gv, together with the ISE change they
actually achieve at the largest feasible step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401  (pins BLAS)
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from akde import gaussops, solvers                           # noqa: E402
from akde.densities import get_target                        # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary  # noqa: E402
from followup.akde_followup import oracle                    # noqa: E402
from followup.akde_followup.manifest import write_manifest   # noqa: E402
from followup.akde_followup.seeds import data_rng            # noqa: E402
from followup.akde_followup.variants import (                # noqa: E402
    BANK_MATCHED, BASE_BANK, FollowupAKDE, build_measurement_bank)

OUT = ROOT / "followup" / "results" / "oracle_controls"
TARGETS = ("f0", "f2", "f5", "f9")
SIZES = (500, 5000)
DICT = dict(n_loc=60, n_scale=5, lo=0.05, hi=4.0)


def simplex_moment_fit(D, y, apply_omega):
    H = D.T @ apply_omega(D)
    H = 0.5 * (H + H.T)
    c = -D.T @ apply_omega(y)
    res = solvers.simplex_qp(H, c)
    return res.x, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rho", type=float, default=1e-3,
                    help="ridge factor held fixed across the ladder")
    ap.add_argument("--bank", choices=["base", "matched"], default="base")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = BASE_BANK if a.bank == "base" else BANK_MATCHED

    rows, ambig = [], []
    for tname in TARGETS:
        tgt = get_target(tname)
        for N in SIZES:
            for rep in range(a.reps):
                X = tgt.sample(N, data_rng("development", tname, N, rep))
                atoms = build_dictionary(X, DictionaryConfig(
                    DICT["n_loc"], DICT["n_scale"], "grid", DICT["lo"], DICT["hi"]))
                bank = build_measurement_bank(X, spec, dictionary=atoms)
                D = gaussops.design_matrix(bank.means, bank.variances,
                                           atoms.means, atoms.variances)
                K = FollowupAKDE.eval_kernels(X, bank)
                y = K.mean(axis=0)
                Kc = K - y
                C_hat = (Kc.T @ Kc) / (N * (N - 1.0))
                y0 = oracle.population_moments(bank, tgt)
                C0 = oracle.population_covariance(bank, tgt, N)

                G, avec, c0 = oracle.exact_risk_parts(atoms, tgt)
                R_full, w_full, res_full = oracle.min_risk_on(G, avec, c0)

                om_hat, _ = FollowupAKDE.omega_apply(C_hat, "ridge", a.rho)
                om_0, _ = FollowupAKDE.omega_apply(C0, "ridge", a.rho)

                rec = {"target": tname, "N": N, "rep": rep, "rho": a.rho,
                       "bank": a.bank, "m": int(bank.size), "M": int(atoms.size),
                       "R_full_star": R_full, "kkt_R_full": res_full.kkt_residual,
                       "c0_int_f0_sq": c0}
                for label, yy, om in (("y0_C0", y0, om_0),
                                      ("y0_Chat", y0, om_hat),
                                      ("y_C0", y, om_0),
                                      ("y_Chat", y, om_hat)):
                    w, res = simplex_moment_fit(D, yy, om)
                    rec["ise_moment_" + label] = oracle.risk(w, G, avec, c0)
                    rec["supp_moment_" + label] = int((w > 1e-12).sum())
                    rec["kkt_moment_" + label] = res.kkt_residual
                    # weighted moment residual achieved, for reference
                    r = D @ w - yy
                    rec["wmr_" + label] = float(r @ om(r))

                # L2 ladder
                s = FollowupAKDE.eval_kernels(X, atoms).mean(axis=0)
                res_l2 = solvers.simplex_qp(2.0 * G, -2.0 * s)
                rec["ise_l2_empirical"] = oracle.risk(res_l2.x, G, avec, c0)
                rec["supp_l2_empirical"] = int((res_l2.x > 1e-12).sum())
                rec["kkt_l2_empirical"] = res_l2.kkt_residual

                # conditioning
                sp = oracle.spectra(C_hat, D, om_hat)
                rec.update({("spec_" + k): v for k, v in sp.items()
                            if not isinstance(v, list)})
                rec["cov_eigs_top3"] = str([float("%.4g" % v)
                                            for v in sp["cov_eigs_top10"][:3]])
                rows.append(rec)

                if rep == 0:
                    for d in oracle.ambiguity_probe(D, om_hat, G, w_full, n_dir=4):
                        d.update(target=tname, N=N, rho=a.rho, bank=a.bank)
                        ambig.append(d)
                print("  %s N=%-5d rep=%d  R_full*=%.4g  moment(y0,C0)=%.4g  "
                      "moment(y,Chat)=%.4g  l2(s)=%.4g"
                      % (tname, N, rep, R_full, rec["ise_moment_y0_C0"],
                         rec["ise_moment_y_Chat"], rec["ise_l2_empirical"]), flush=True)

    df = pd.DataFrame(rows)
    tag = "%s_rho%g" % (a.bank, a.rho)
    df.to_csv(OUT / ("ladder_%s.csv" % tag), index=False)
    pd.DataFrame(ambig).to_csv(OUT / ("ambiguity_%s.csv" % tag), index=False)
    write_manifest(OUT / ("manifest_%s.json" % tag), stage="oracle_controls",
                   config={"reps": a.reps, "rho": a.rho, "bank": a.bank,
                           "targets": list(TARGETS), "sizes": list(SIZES),
                           "dict": DICT},
                   outputs={"ladder": OUT / ("ladder_%s.csv" % tag)})

    cols = ["R_full_star", "ise_moment_y0_C0", "ise_moment_y0_Chat",
            "ise_moment_y_C0", "ise_moment_y_Chat", "ise_l2_empirical"]
    print("\n== median exact ISE of each rung (%s bank, rho=%g) ==" % (a.bank, a.rho))
    print(df.groupby(["target", "N"])[cols].median().to_string(
        float_format=lambda v: "%.4g" % v))
    print("\nwrote", OUT / ("ladder_%s.csv" % tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
