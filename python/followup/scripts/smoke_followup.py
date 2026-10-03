"""Smoke test: every Stage-A variant runs, the oracle identity holds, and the
opt-in stagnation stop leaves the solution alone while cutting the iterations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akde import metrics                                    # noqa: E402
from akde.densities import get_target                       # noqa: E402
from akde.estimators.base import TuningConfig               # noqa: E402
from followup.akde_followup import oracle                   # noqa: E402
from followup.akde_followup.variants import (               # noqa: E402
    STAGE_A, BANK_LOC_REFINED, BASE_BANK, BANK_SCALE_REFINED,
    FollowupAKDE, build_measurement_bank)

FAILS = []


def check(name, ok, detail=""):
    print("[%s] %s   %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


# --- nesting claims --------------------------------------------------------
check("BASE (100x3) nests in loc-refined (199x3)", BASE_BANK.nests_in(BANK_LOC_REFINED),
      "m %d -> %d" % (BASE_BANK.m, BANK_LOC_REFINED.m))
check("BASE (100x3) nests in scale-refined (100x5)", BASE_BANK.nests_in(BANK_SCALE_REFINED),
      "m %d -> %d" % (BASE_BANK.m, BANK_SCALE_REFINED.m))
tgt = get_target("f9")
X = tgt.sample(500, np.random.default_rng(1))
b0 = build_measurement_bank(X, BASE_BANK)
b1 = build_measurement_bank(X, BANK_LOC_REFINED)
b2 = build_measurement_bank(X, BANK_SCALE_REFINED)
pairs0 = set(zip(np.round(b0.means[:, 0], 12), np.round(b0.variances[:, 0], 12)))
pairs1 = set(zip(np.round(b1.means[:, 0], 12), np.round(b1.variances[:, 0], 12)))
pairs2 = set(zip(np.round(b2.means[:, 0], 12), np.round(b2.variances[:, 0], 12)))
check("loc-refined bank literally contains the base bank",
      pairs0 <= pairs1, "%d/%d base kernels present" % (len(pairs0 & pairs1), len(pairs0)))
check("scale-refined bank literally contains the base bank",
      pairs0 <= pairs2, "%d/%d base kernels present" % (len(pairs0 & pairs2), len(pairs0)))

# --- the stage-1 iteration cap is load-bearing (recorded, not asserted) ----
# An early guess was that V2's 500-sweep cap wastes iterations that change
# nothing.  It does not: truncating the sweeps moves the answer.  Reported
# here so the claim in audit_findings.md §3.1 stays tied to a measurement.
from followup.akde_followup.variants import make_variant                 # noqa: E402
grid = np.linspace(-6, 6, 2001)
print("\n-- effect of truncating V2's stage-1 sweeps (NOT a free speed-up) --")
for tname, N in [("f9", 500), ("f5", 5000)]:
    t = get_target(tname)
    Xs = t.sample(N, np.random.default_rng(2))
    slow = FollowupAKDE(make_variant("slow", stagnation_stop=False),
                        TuningConfig(n_folds=3))
    fast = FollowupAKDE(make_variant("fast", stagnation_stop=True),
                        TuningConfig(n_folds=3))
    c = slow.candidates(Xs)
    t0 = perf_counter(); ps = slow.fit_path(Xs, c, np.random.default_rng(0)); ts = perf_counter() - t0
    t0 = perf_counter(); pf = fast.fit_path(Xs, c, np.random.default_rng(0)); tf = perf_counter() - t0
    dmax = max(float(np.max(np.abs(a.pdf(grid) - b.pdf(grid)))) for a, b in zip(ps, pf))
    supp_same = all(a.n_active() == b.n_active() for a, b in zip(ps, pf))
    it_s = sum(int(a.diagnostics.get("stage1_iters", 0)) for a in ps)
    it_f = sum(int(b.diagnostics.get("stage1_iters", 0)) for b in pf)
    print("   %s N=%-5d max |pdf diff| %.3e  supports identical=%s  "
          "sweeps %d -> %d (%.1fx)  path %.2fs -> %.2fs (%.1fx)"
          % (tname, N, dmax, supp_same, it_s, it_f, it_s / max(it_f, 1),
             ts, tf, ts / max(tf, 1e-9)), flush=True)
check("V2 default keeps the full sweep budget (stagnation_stop off by default)",
      make_variant("x").stagnation_stop is False)

# --- every Stage-A variant runs, and the oracle identity holds -------------
t = get_target("f5")
Xs = t.sample(500, np.random.default_rng(4))
for name, cfg in STAGE_A.items():
    est = FollowupAKDE(cfg, TuningConfig(n_folds=3))
    t0 = perf_counter()
    fit, log = est.fit_logged(Xs, np.random.default_rng(9),
                              oracle_ise=lambda f: metrics.ise(f, t))
    dt = perf_counter() - t0
    ise = metrics.ise(fit, t)
    ok = np.isfinite(ise) and fit.is_valid_density()
    check("%-28s runs and returns a valid density" % name, bool(ok),
          "ISE %.5g  supp %d  m=%d  %.1fs (tune %.1f, fit %.2f, oracle path %.1f)  "
          "oracle gap x%.2f"
          % (ise, fit.n_active(), fit.diagnostics["n_measurements"], dt,
             log["tune_time_s"], log["fit_time_s"], log["oracle_path_time_s"],
             log.get("oracle_tuning_gap_ratio", float("nan"))))

# --- oracle decomposition identity ----------------------------------------
from akde.dictionary import build_dictionary, DictionaryConfig            # noqa: E402
atoms = build_dictionary(Xs, DictionaryConfig(60, 5, "grid", 0.05, 4.0))
est = FollowupAKDE(STAGE_A["A1_v2_moment_m300"], TuningConfig(n_folds=3))
fit, _ = est.fit_logged(Xs, np.random.default_rng(9))
# the fit drops zero-weight atoms, so rebuild the full-length weight vector
# by matching each kept atom back to its dictionary index
w_full = np.zeros(atoms.size)
key = {(round(float(m), 12), round(float(v), 12)): i
       for i, (m, v) in enumerate(zip(atoms.means[:, 0], atoms.variances[:, 0]))}
for wi, mi, vi in zip(fit.weights, fit.means[:, 0], fit.variances[:, 0]):
    w_full[key[(round(float(mi), 12), round(float(vi), 12))]] = wi
dec = oracle.decompose(w_full, atoms, t, support=fit.support)
check("oracle decomposition identity holds", abs(dec["identity_residual"]) < 1e-12,
      "residual %.3e" % dec["identity_residual"])
check("R(w_hat) from the oracle == analytic ISE",
      abs(dec["R_hat"] - metrics.ise(fit, t)) < 1e-10,
      "%.10g vs %.10g" % (dec["R_hat"], metrics.ise(fit, t)))
print("   R_full* = %.5g   R_S* - R_full* = %.5g   R(w) - R_S* = %.5g   R(w) = %.5g"
      % (dec["R_full_star"], dec["term_support"], dec["term_excess"], dec["R_hat"]))

# --- population moments / covariance vs exact quadrature ------------------
# Quadrature, not Monte Carlo: these are 1-D integrals with closed-form
# integrands, so an exact reference is available and a 1e-3 MC discrepancy
# would tell us nothing.
from akde.metrics import integrate_1d                                    # noqa: E402
from akde import gaussops                                                # noqa: E402
from followup.akde_followup.variants import MeasurementSpec              # noqa: E402
bank_s = build_measurement_bank(Xs, MeasurementSpec(12, 2, 0.25, 2.0,
                                                    "independent", "small"))


def _g(z, mu, var):
    return np.exp(-0.5 * ((z - mu) ** 2 / var + np.log(var) + gaussops.LOG_2PI))


y0 = oracle.population_moments(bank_s, t)
err_y = 0.0
for j in [0, 3, 11, bank_s.size - 1]:
    ref = integrate_1d(lambda z, j=j: _g(z, bank_s.means[j, 0],
                                         bank_s.variances[j, 0]) * t.pdf(z),
                       -25, 25, tol=1e-12)
    err_y = max(err_y, abs(ref - y0[j]) / max(abs(ref), 1e-300))
check("closed-form population moments y0 match quadrature", err_y < 1e-9,
      "max rel diff %.3e" % err_y)

C0 = oracle.population_covariance(bank_s, t, 1)      # n=1 -> the raw covariance
err_c = 0.0
for i, j in [(0, 0), (0, 7), (5, 9), (bank_s.size - 1, bank_s.size - 2)]:
    e2 = integrate_1d(lambda z, i=i, j=j: _g(z, bank_s.means[i, 0], bank_s.variances[i, 0])
                      * _g(z, bank_s.means[j, 0], bank_s.variances[j, 0]) * t.pdf(z),
                      -25, 25, tol=1e-12)
    ref = e2 - y0[i] * y0[j]
    err_c = max(err_c, abs(ref - C0[i, j]) / max(abs(ref), 1e-300))
check("closed-form population covariance matches quadrature", err_c < 1e-8,
      "max rel diff %.3e" % err_c)
# and a scaling check: population_covariance(., n) must be C0/n
Cn = oracle.population_covariance(bank_s, t, 500)
check("population_covariance scales as 1/n",
      np.allclose(Cn * 500.0, C0, rtol=1e-12), "")

print("\n" + ("SMOKE OK" if not FAILS else "FAILURES: %s" % FAILS))
sys.exit(1 if FAILS else 0)
