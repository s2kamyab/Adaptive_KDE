"""Per-fit, resumable runner for the improved-AKDE study.

Unit = (stage, namespace, target, N, rep, method).  Writes
    results/improved/<stage>/rows/<key>.json   metrics + manifest
    results/improved/<stage>/fits/<key>.npz    the fitted mixture
atomically; a unit is skipped on re-run only when the stored manifest (code
hash, config hash, dataset sha256, protocol hash) equals the current one.

Two target families and their seed/fold policies:
  * Marron-Wand targets, namespaces 'development' / 'confirmation' from
    followup.akde_followup.seeds: fold stream = method index (Stage-C policy;
    index 0 in development = shared folds as Stage A used).
  * hr family, namespaces 'bf_dev' / 'bf_main' from .hr_family: fold stream
    100 shared by every method (the bandwidth-flexibility policy).
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from akde import gaussops, metrics
from akde.densities import get_target as mw_target
from akde.estimators.base import TuningConfig
from followup.akde_followup import seeds as mw_seeds
from followup.akde_followup.comparators import LOCKED_TUNING
from followup.akde_followup.manifest import canonical_json, sha256_bytes, sha256_file
from followup.akde_followup.variants import FollowupAKDE, STAGE_A

from . import PY_ROOT, hr_family as hr
from .estimator import ImprovedAKDE, VARIANTS

RESULTS = PY_ROOT / "followup" / "results" / "improved"
__all__ = ["RESULTS", "run_unit", "run_units", "collect", "code_hash", "method_config"]


def code_hash() -> str:
    files = {}
    for root in (PY_ROOT / "akde", PY_ROOT / "followup" / "akde_followup",
                 PY_ROOT / "followup" / "improved"):
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" not in p.parts:
                files[str(p.relative_to(PY_ROOT)).replace("\\", "/")] = sha256_file(p)
    return sha256_bytes(canonical_json(files))


def method_config(method: str) -> dict:
    if method == "I0_finalist":
        return {"method": method, "class": "FollowupAKDE", "config": STAGE_A["A6_moment_matched_m300"].as_dict(),
                "n_folds": LOCKED_TUNING["n_folds"]}
    return {"method": method, "class": "ImprovedAKDE", "config": VARIANTS[method].as_dict()}


def build(method: str):
    if method == "I0_finalist":
        return FollowupAKDE(STAGE_A["A6_moment_matched_m300"], TuningConfig(n_folds=LOCKED_TUNING["n_folds"]))
    return ImprovedAKDE(VARIANTS[method])


def _dataset(namespace, target, N, rep, n_test):
    if namespace in hr.NAMESPACES:
        tgt = hr.make_target(hr.r_of(target))
        X = tgt.sample(N, hr.data_rng(namespace, target, N, rep))
        Xt = tgt.sample(n_test, hr.test_rng(namespace, target, N, rep))
        return tgt, X, Xt
    tgt = mw_target(target)
    X = tgt.sample(N, mw_seeds.data_rng(namespace, target, N, rep))
    Xt = tgt.sample(n_test, mw_seeds.test_rng(namespace, target, N, rep))
    return tgt, X, Xt


def _fold_rng(namespace, target, N, rep, method_index):
    if namespace in hr.NAMESPACES:
        return hr.fold_rng(namespace, target, N, rep)
    return mw_seeds.fit_rng(namespace, target, N, rep, method_index)


def _regional(fit, tgt, region):
    from scipy.special import ndtr
    a, b = region
    fw, fm, fv = fit.weights, fit.means[:, 0], fit.variances[:, 0]
    gw, gm, gv = tgt.weights, tgt.means[:, 0], tgt.variances[:, 0]

    def q(w1, m1, v1, w2, m2, v2):
        s = v1[:, None] + v2[None, :]
        d = m1[:, None] - m2[None, :]
        const = np.exp(-0.5 * (d * d / s + np.log(s) + gaussops.LOG_2PI))
        mean = (m1[:, None] * v2[None, :] + m2[None, :] * v1[:, None]) / s
        sd = np.sqrt(v1[:, None] * v2[None, :] / s)
        frac = ndtr((b - mean) / sd) - ndtr((a - mean) / sd)
        return float(w1 @ ((const * frac) @ w2))
    return max(0.0, q(fw, fm, fv, fw, fm, fv) - 2 * q(fw, fm, fv, gw, gm, gv) + q(gw, gm, gv, gw, gm, gv))


def run_unit(stage, namespace, target, N, rep, method, method_index, *, protocol_hash,
             n_test=2000, force=False, ch=None):
    out_root = RESULTS / stage
    rows_dir, fits_dir = out_root / "rows", out_root / "fits"
    rows_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    key = "%s_N%d_r%d_%s" % (target, N, rep, method)
    row_path, fit_path = rows_dir / (key + ".json"), fits_dir / (key + ".npz")
    tgt, X, Xt = _dataset(namespace, target, N, rep, n_test)
    dhash = hashlib.sha256(np.ascontiguousarray(X, np.float64).tobytes()).hexdigest()
    ch = ch or code_hash()
    mcfg = method_config(method)
    manifest = {"code_sha256": ch, "config_sha256": sha256_bytes(canonical_json(mcfg)),
                "dataset_sha256": dhash, "protocol_sha256": protocol_hash}
    if row_path.exists() and fit_path.exists() and not force:
        try:
            old = json.load(open(row_path, encoding="utf-8"))
            if old.get("manifest") == manifest and not old["row"].get("error"):
                return {"key": key, "status": "cached"}
        except Exception:
            pass
    row = {"stage": stage, "namespace": namespace, "target": target, "N": int(N), "rep": int(rep),
           "method": method, "method_index": method_index, "dataset_sha256": dhash}
    if namespace in hr.NAMESPACES:
        row["r"] = hr.r_of(target)
    try:
        est = build(method)
        rng = _fold_rng(namespace, target, N, rep, method_index)
        t0 = perf_counter()
        if isinstance(est, FollowupAKDE):
            fit, log = est.fit_logged(X, rng, oracle_ise=None)
        else:
            fit, log = est.fit_logged(X, rng)
        wall = perf_counter() - t0
        row["wall_time_s"] = wall
        row["ise"] = float(metrics.ise(fit, tgt))
        if namespace in hr.NAMESPACES:
            a, b = hr.narrow_region(hr.r_of(target))
            row["ise_narrow"] = _regional(fit, tgt, (a, b))
        row["total_mass"] = float(fit.total_mass())
        row["mass_error"] = float(metrics.l1_mass_error(fit))
        row["min_weight"] = float(fit.weights.min())
        row["is_valid_density"] = bool(fit.is_valid_density())
        row["n_components"] = int(fit.weights.size)
        row["n_active_rel1e-8"] = int(fit.n_active(tol=1e-8))
        row["minimal_storage_bytes"] = 24 * int(fit.weights.size)
        nll, fl = metrics.held_out_nll(fit, Xt)
        row["nll"], row["nll_floored"] = nll, fl
        for k in ("n_selected", "n_dictionary", "fallback_used", "support_capped", "stage1_kkt",
                  "stage1_converged", "stage1_iters", "stage1_objective", "stage2_kkt",
                  "setup_time_s", "stage1_time_s", "stage2_time_s", "tune_time_s", "fit_time_s",
                  "total_time_s", "rho", "lam_ratio", "weighting", "mode", "solver", "penalty",
                  "h_ref", "dict_n_loc", "dict_n_scale", "merge_n_merged", "merge_loss_before_merge",
                  "merge_loss_after_merge", "merge_loss_after_refit", "merge_refit_kkt",
                  "merge_time_s", "n_before_merge", "stage1_solver", "stage1_stop",
                  "cov_effective_rank", "omega_cond"):
            if k in fit.diagnostics:
                row[k] = fit.diagnostics[k]
        for k in ("scale_weights", "dict_sds"):
            if k in fit.diagnostics:
                row[k + "_json"] = json.dumps([float(x) for x in fit.diagnostics[k]])
        for k in ("n_candidates", "n_folds", "fold_errors", "all_candidates_failed", "best_cv_score",
                  "best_cv_se", "sel_lam_ratio", "sel_rho", "sel_anchor", "lam_at_grid_edge",
                  "lam_edge_side", "anchor_at_grid_edge", "rho_at_grid_edge"):
            if k in log:
                row[k] = log[k]
        row["error"] = ""
        np.savez_compressed(fit_path.with_suffix(".tmp.npz"), weights=fit.weights, means=fit.means,
                            variances=fit.variances,
                            support=np.asarray(fit.support if fit.support is not None else [], dtype=int),
                            ise=row["ise"])
        os.replace(fit_path.with_suffix(".tmp.npz"), fit_path)
        row["npz_file_bytes"] = int(fit_path.stat().st_size)
        status = "ok"
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        row["traceback_text"] = traceback.format_exc(limit=6)
        row["ise"] = np.nan
        status = "error"
    tmp = row_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"row": row, "manifest": manifest, "method_config": mcfg}, indent=1,
                              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    os.replace(tmp, row_path)
    return {"key": key, "status": status}


def run_units(stage, units, *, protocol_hash, n_test=2000, n_jobs=1, force=False):
    """units: list of (namespace, target, N, rep, method, method_index)."""
    ch = code_hash()
    if n_jobs == 1:
        return [run_unit(stage, *u, protocol_hash=protocol_hash, n_test=n_test, force=force, ch=ch)
                for u in units]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=5, inner_max_num_threads=1)(
        delayed(run_unit)(stage, *u, protocol_hash=protocol_hash, n_test=n_test, force=force, ch=ch)
        for u in units)


def collect(stage) -> pd.DataFrame:
    rows = []
    for p in sorted((RESULTS / stage / "rows").glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        r = dict(d["row"])
        r.update({"code_sha256": d["manifest"]["code_sha256"], "protocol_sha256": d["manifest"]["protocol_sha256"]})
        rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["target", "N", "rep", "method"]).reset_index(drop=True)
        tmp = RESULTS / stage / "all_rows.csv.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, RESULTS / stage / "all_rows.csv")
    return df
