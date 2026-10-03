"""`FollowupAKDE` configured as V2 must equal the frozen V2 estimator exactly.

This is the guard that lets the follow-up claim "variant A1 *is* the V2
estimator on a pinned dictionary".  It compares, on identical data and with
identical hyperparameters:

    followup.akde_v2_original.akde.estimators.akde.AKDE   (frozen snapshot)
    akde.estimators.akde.AKDE                             (live package)
    followup.akde_followup.variants.FollowupAKDE          (stagnation_stop=False)

on every candidate of the path, not just the selected one, and on the final
cross-validated fit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akde.densities import get_target                                   # noqa: E402
from akde.estimators.akde import AKDE, AKDEConfig                       # noqa: E402
from akde.estimators.base import TuningConfig                           # noqa: E402
from akde.registry import shared_dictionary                             # noqa: E402
from akde import metrics                                                # noqa: E402
from followup.akde_v2_original.akde.estimators.akde import (            # noqa: E402
    AKDE as AKDEFrozen, AKDEConfig as AKDEConfigFrozen)
from followup.akde_followup.variants import (                           # noqa: E402
    FollowupAKDE, FollowupConfig, MeasurementSpec)

FAILS = []


def check(name, ok, detail=""):
    print("[%s] %s   %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def v2_config(nl, ns):
    return dict(dictionary=shared_dictionary(nl, ns), dict_sizes=None,
                meas_mode="independent", meas_n_loc=100, meas_n_scale=3,
                meas_scale_lo_mult=0.25, meas_scale_hi_mult=2.0,
                weighting="ridge", rho_grid=(1e-6, 1e-3, 1e-1, 1.0),
                n_lambda=10, lam_max_ratio=0.7, lam_min_ratio=3e-4,
                tau=1e-3, max_support=400)


def followup_config(nl, ns):
    return FollowupConfig(
        dict_n_loc=nl, dict_n_scale=ns, dict_lo_mult=0.05, dict_hi_mult=4.0,
        measurement=MeasurementSpec(100, 3, 0.25, 2.0, "independent", "v2_base"),
        weighting="ridge", rho_grid=(1e-6, 1e-3, 1e-1, 1.0),
        selection="l1", refit="moment",
        n_lambda=10, lam_max_ratio=0.7, lam_min_ratio=3e-4,
        tau=1e-3, max_support=400,
        stagnation_stop=False, label="A1_v2_identity_check")


def weights_on_grid(fit, grid):
    return fit.pdf(grid)


for tname, N, (nl, ns) in [("f9", 300, (60, 5)), ("f5", 400, (60, 5)),
                           ("f2", 300, (100, 6))]:
    tgt = get_target(tname)
    X = tgt.sample(N, np.random.default_rng(17))
    grid = np.linspace(-6, 6, 1501)

    live = AKDE(AKDEConfig(**v2_config(nl, ns)), tuning=TuningConfig(n_folds=3))
    froz = AKDEFrozen(AKDEConfigFrozen(**v2_config(nl, ns)), tuning=TuningConfig(n_folds=3))
    foll = FollowupAKDE(followup_config(nl, ns), tuning=TuningConfig(n_folds=3))

    cands_live = live.candidates(X)
    cands_foll = foll.candidates(X)
    # same set of (rho, lam_ratio) pairs, possibly in a different order
    key = lambda c: (round(float(c["rho"]), 15), round(float(c["lam_ratio"]), 15))
    check("%s N=%d: candidate sets identical" % (tname, N),
          sorted(map(key, cands_live)) == sorted(map(key, cands_foll)),
          "%d vs %d" % (len(cands_live), len(cands_foll)))

    p_live = live.fit_path(X, cands_live, np.random.default_rng(0))
    p_froz = froz.fit_path(X, cands_live, np.random.default_rng(0))
    p_foll = foll.fit_path(X, cands_foll, np.random.default_rng(0))
    by_foll = {key(c): f for c, f in zip(cands_foll, p_foll)}

    worst_lf, worst_ff = 0.0, 0.0
    for c, fl, fz in zip(cands_live, p_live, p_froz):
        ff = by_foll[key(c)]
        worst_ff = max(worst_ff, float(np.max(np.abs(
            weights_on_grid(fl, grid) - weights_on_grid(fz, grid)))))
        worst_lf = max(worst_lf, float(np.max(np.abs(
            weights_on_grid(fl, grid) - weights_on_grid(ff, grid)))))
    check("%s N=%d: live == frozen V2 on every path candidate" % (tname, N),
          worst_ff == 0.0, "max |pdf diff| %.3e over %d candidates"
          % (worst_ff, len(cands_live)))
    check("%s N=%d: FollowupAKDE(stagnation_stop=False) == V2 on every candidate"
          % (tname, N), worst_lf < 1e-12,
          "max |pdf diff| %.3e" % worst_lf)

    f1 = live.fit(X, np.random.default_rng(5))
    f2 = foll.fit(X, np.random.default_rng(5))
    d = float(np.max(np.abs(f1.pdf(grid) - f2.pdf(grid))))
    check("%s N=%d: cross-validated fits identical" % (tname, N), d < 1e-12,
          "max |pdf diff| %.3e; ISE %.8g vs %.8g"
          % (d, metrics.ise(f1, tgt), metrics.ise(f2, tgt)))

print("\n" + ("ALL IDENTITY CHECKS PASSED" if not FAILS else "FAILURES: %s" % FAILS))
sys.exit(1 if FAILS else 0)
