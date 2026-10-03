"""Stage C — locked confirmation.

15 Marron-Wand targets x N in {200, 1000, 5000} x R independent datasets,
six methods: original V2 AKDE, the frozen finalist, density-valid SPADES,
Sheather-Jones KDE, Abramson adaptive KDE, and Gaussian-mixture EM.

Everything about this stage is fixed by protocol_followup_locked.yaml before
the first confirmation seed is drawn: the grid, the methods, the finalist, the
tuning grid, the outcomes, the seeds, and R.  No retuning on confirmation
results.  Truth is used only for scoring.

Seeds come from the ``confirmation`` namespace, disjoint from both the
development namespace and the V2 benchmark.

    python followup/scripts/run_stage_c.py --reps 10 --jobs 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import followup                                              # noqa: E402,F401
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from akde import metrics                                     # noqa: E402
from akde.densities import get_target                        # noqa: E402
from followup.akde_followup.comparators import build_comparator  # noqa: E402
from followup.akde_followup.manifest import write_manifest, sha256_file  # noqa: E402
from followup.akde_followup.runner import pin_threads        # noqa: E402
from followup.akde_followup.seeds import data_rng, fit_rng, test_rng  # noqa: E402
from followup.akde_followup.variants import STAGE_A, STAGE_B, resolve_config  # noqa: E402

RES = ROOT / "followup" / "results"
OUT = RES / "stage_c"
TARGETS = tuple("f%d" % i for i in range(15))
SIZES = (200, 1000, 5000)
METHODS = ("v2_akde", "finalist", "spades_simplex", "kde_sj", "abramson", "gmm_em")


def _finalist_cfg():
    spec = json.load(open(RES / "finalist.json", encoding="utf-8"))
    pool = dict(STAGE_A)
    pool.update(STAGE_B)
    cfg = pool[spec["variant"]]
    return spec, cfg


def _cell(target, N, reps, spec, fcfg, save_fits, n_test):
    tgt = get_target(target)
    rows = []
    fitdir = OUT / "fits"
    fitdir.mkdir(parents=True, exist_ok=True)
    for rep in range(reps):
        X = tgt.sample(N, data_rng("confirmation", target, N, rep))
        X_test = tgt.sample(n_test, test_rng("confirmation", target, N, rep))
        for mi, key in enumerate(METHODS):
            base = {"stage": "stage_c", "namespace": "confirmation",
                    "target": target, "target_label": tgt.label, "N": N,
                    "rep": rep, "variant": key}
            try:
                est = build_comparator(key, finalist=resolve_config(fcfg, N)
                                       if key == "finalist" else None)
                rng = fit_rng("confirmation", target, N, rep, mi)
                t0 = perf_counter()
                if key == "finalist":
                    fit, log = est.fit_logged(X, rng, oracle_ise=None)
                else:
                    fit = est.fit(X, rng)
                    log = {}
                wall = perf_counter() - t0
                row = dict(base)
                row["method_label"] = getattr(est, "name", key)
                row["wall_time_s"] = wall
                row["ise"] = float(metrics.ise(fit, tgt))
                row["total_mass"] = float(fit.total_mass())
                row["mass_error"] = float(metrics.l1_mass_error(fit))
                row["is_valid_density"] = bool(fit.is_valid_density())
                row["n_active"] = int(fit.n_active(tol=1e-8))
                nll, floored = metrics.held_out_nll(fit, X_test)
                row["nll"] = nll
                row["nll_floored"] = floored
                for k in ("n_selected", "n_dictionary", "n_measurements",
                          "fallback_used", "support_capped", "stage1_kkt",
                          "stage1_converged", "stage1_iters", "stage1_solver",
                          "stage1_stop", "stage2_kkt", "setup_time_s",
                          "stage1_time_s", "stage2_time_s", "tune_time_s",
                          "fit_time_s", "total_time_s", "rho", "lam_ratio",
                          "weighting", "cov_rank_numeric", "cov_effective_rank",
                          "omega_cond", "selection", "refit", "selector",
                          "sel_k", "em_converged", "mass_below_meas_scale",
                          "mean_atom_sd_over_h"):
                    if k in fit.diagnostics:
                        row[k] = fit.diagnostics[k]
                sel = fit.diagnostics.get("selected")
                if isinstance(sel, dict):
                    for k, v in sel.items():
                        if isinstance(v, (int, float, bool, str)):
                            row["sel_" + k] = v
                for k in ("rho_at_grid_edge", "lam_at_grid_edge", "n_candidates",
                          "fold_errors", "all_candidates_failed"):
                    if k in log:
                        row[k] = log[k]
                row["error"] = ""
                rows.append(row)
                if save_fits:
                    np.savez_compressed(
                        fitdir / ("%s_N%d_r%d_%s.npz" % (target, N, rep, key)),
                        weights=fit.weights, means=fit.means,
                        variances=fit.variances, ise=row["ise"])
            except Exception as exc:
                row = dict(base)
                row["error"] = "%s: %s" % (type(exc).__name__, exc)
                row["traceback_text"] = traceback.format_exc(limit=4)
                row["ise"] = np.nan
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pin_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    spec, fcfg = _finalist_cfg()
    targets = tuple(a.targets.split(","))
    sizes = tuple(int(s) for s in a.sizes.split(","))
    print("Stage C (LOCKED): %d targets x %d sizes x %d reps x %d methods = %d fits"
          % (len(targets), len(sizes), a.reps, len(METHODS),
             len(targets) * len(sizes) * a.reps * len(METHODS)), flush=True)
    print("finalist: %s (%s)" % (spec["variant"], spec.get("frozen_utc", "?")), flush=True)

    todo = []
    for t in targets:
        for n in sizes:
            p = OUT / ("%s_N%d.csv" % (t, n))
            if p.exists() and not a.force:
                continue
            todo.append((t, n))
    print("%d cells to run, %d cached" % (len(todo), len(targets) * len(sizes) - len(todo)),
          flush=True)

    if todo:
        todo.sort(key=lambda tn: -tn[1])
        from joblib import Parallel, delayed

        def _run(t, n):
            rows = _cell(t, n, a.reps, spec, fcfg, True, a.n_test)
            df = pd.DataFrame(rows)
            p = OUT / ("%s_N%d.csv" % (t, n))
            tmp = p.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, p)
            return str(p)

        Parallel(n_jobs=a.jobs, backend="loky", verbose=5,
                 inner_max_num_threads=1)(delayed(_run)(t, n) for t, n in todo)

    frames = [pd.read_csv(OUT / ("%s_N%d.csv" % (t, n)), low_memory=False)
              for t in targets for n in sizes
              if (OUT / ("%s_N%d.csv" % (t, n))).exists()]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "all_rows.csv", index=False)
    write_manifest(OUT / "manifest.json", stage="stage_c",
                   config={"targets": list(targets), "sizes": list(sizes),
                           "reps": a.reps, "methods": list(METHODS),
                           "finalist": spec, "n_test": a.n_test},
                   protocol_hash=sha256_file(ROOT.parents[1] /
                                             "protocol_followup_locked.yaml")
                   if (ROOT.parents[1] / "protocol_followup_locked.yaml").exists() else None,
                   outputs={"rows": OUT / "all_rows.csv"})
    err = df["error"].fillna("").astype(str)
    print("\n%d rows, %d errors" % (len(df), int((err != "").sum())))
    ok = df[err == ""]
    print(ok.pivot_table(index="variant", columns="N", values="ise",
                         aggfunc="median").to_string(float_format=lambda v: "%.5g" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
