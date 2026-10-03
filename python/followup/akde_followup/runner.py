"""Shared runner for the follow-up stages.

One dataset per (namespace, target, N, rep); every variant sees the identical
realisation and the identical CV folds, so all comparisons are paired.

Outputs per stage, under ``followup/results/<stage>/``:

    <target>_N<n>.csv        one row per (rep, variant): metrics, selection,
                             solver diagnostics, conditioning, timings
    <target>_N<n>_oracle.csv oracle solves, kept in a separate file so no
                             oracle column can be mistaken for an estimator
                             output
    fits/<...>.npz           fitted weights / means / variances / support
    manifest_<target>_N<n>.json

Shards are skipped if already present, so a run is resumable.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

_PY = Path(__file__).resolve().parents[2]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from akde import metrics                                    # noqa: E402
from akde.densities import get_target                       # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary  # noqa: E402
from akde.estimators.base import TuningConfig               # noqa: E402

from . import oracle as oracle_mod                          # noqa: E402
from .manifest import write_manifest, config_hash           # noqa: E402
from .seeds import data_rng, fit_rng, test_rng              # noqa: E402
from .variants import FollowupAKDE, FollowupConfig, resolve_config  # noqa: E402

__all__ = ["StageConfig", "run_stage", "reference_fit"]

RESULTS = _PY / "followup" / "results"


def pin_threads(n: int = 1) -> None:
    """Assert (do not merely request) that BLAS is single-threaded.

    ``followup/__init__.py`` sets the variables before numpy is imported,
    which is the only point at which they take effect.  Calling this later can
    only *check*, so that is what it does: a warning here means the process
    was started in a way that bypassed the package import.
    """
    bad = [v for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
           if os.environ.get(v) != str(n)]
    if bad:
        print("WARNING: BLAS thread pinning not in effect for %s; "
              "expect oversubscription" % ", ".join(bad), flush=True)


@dataclass
class StageConfig:
    name: str
    namespace: str
    targets: tuple[str, ...]
    sizes: tuple[int, ...]
    n_reps: int
    variants: dict = field(default_factory=dict)
    n_folds: int = 3
    n_test: int = 2000
    oracle: bool = True
    save_fits: bool = True
    protocol_hash: str | None = None

    def key(self) -> dict:
        return {
            "name": self.name, "namespace": self.namespace,
            "targets": list(self.targets), "sizes": list(self.sizes),
            "n_reps": self.n_reps, "n_folds": self.n_folds,
            "n_test": self.n_test, "oracle": self.oracle,
            "variants": {k: v.as_dict() for k, v in self.variants.items()},
        }


def _full_weight_vector(fit, atoms) -> np.ndarray:
    """Re-expand a fit's dropped-zero weights onto the full dictionary index."""
    w = np.zeros(atoms.size)
    key = {(round(float(m), 12), round(float(v), 12)): i
           for i, (m, v) in enumerate(zip(atoms.means[:, 0], atoms.variances[:, 0]))}
    for wi, mi, vi in zip(fit.weights, fit.means[:, 0], fit.variances[:, 0]):
        w[key[(round(float(mi), 12), round(float(vi), 12))]] = wi
    return w


def _run_and_write(cfg: StageConfig, target: str, N: int, out_root: Path,
                   verbose: bool) -> str:
    """Run one (target, N) cell and write its shard atomically.  Resumable."""
    shard = out_root / ("%s_N%d.csv" % (target, N))
    oshard = out_root / ("%s_N%d_oracle.csv" % (target, N))
    rows, orows = _run_cell(cfg, target, N, out_root, verbose)
    df = pd.DataFrame(rows)
    tmp = shard.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, shard)
    if orows:
        pd.DataFrame(orows).to_csv(oshard, index=False)
    write_manifest(
        out_root / ("manifest_%s_N%d.json" % (target, N)),
        stage=cfg.name, config=cfg.key(), protocol_hash=cfg.protocol_hash,
        outputs={"rows": shard, **({"oracle": oshard} if orows else {})},
        extra={"target": target, "N": N, "n_rows": len(df)})
    return str(shard)


def run_stage(cfg: StageConfig, force: bool = False, verbose: bool = True,
              n_jobs: int = 1) -> pd.DataFrame:
    pin_threads(1)
    out_root = RESULTS / cfg.name
    (out_root / "fits").mkdir(parents=True, exist_ok=True)
    with open(out_root / "stage_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.key(), fh, indent=2)

    todo = []
    for target in cfg.targets:
        for N in cfg.sizes:
            shard = out_root / ("%s_N%d.csv" % (target, N))
            if shard.exists() and not force:
                if verbose:
                    print("skip (cached) %s" % shard.name, flush=True)
            else:
                todo.append((target, N))

    if todo:
        todo.sort(key=lambda tn: -tn[1])      # biggest cells first
        if n_jobs == 1:
            for t, n in todo:
                _run_and_write(cfg, t, n, out_root, verbose)
        else:
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs, backend="loky", verbose=5,
                     inner_max_num_threads=1)(
                delayed(_run_and_write)(cfg, t, n, out_root, False) for t, n in todo)

    frames = []
    for target in cfg.targets:
        for N in cfg.sizes:
            shard = out_root / ("%s_N%d.csv" % (target, N))
            if shard.exists():
                frames.append(pd.read_csv(shard, low_memory=False))
    allrows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not allrows.empty:
        allrows.to_csv(out_root / "all_rows.csv", index=False)
    return allrows


def _run_cell(cfg: StageConfig, target: str, N: int, out_root: Path, verbose: bool):
    tgt = get_target(target)
    rows, orows = [], []
    vnames = list(cfg.variants)
    for rep in range(cfg.n_reps):
        X = tgt.sample(N, data_rng(cfg.namespace, target, N, rep))
        X_test = tgt.sample(cfg.n_test, test_rng(cfg.namespace, target, N, rep))
        for vi, vname in enumerate(vnames):
            vcfg: FollowupConfig = resolve_config(cfg.variants[vname], N)
            base = {"stage": cfg.name, "namespace": cfg.namespace,
                    "target": target, "target_label": tgt.label, "N": N,
                    "rep": rep, "variant": vname, "family": vcfg.family,
                    "measurement_bank": vcfg.measurement.label}
            try:
                est = FollowupAKDE(vcfg, TuningConfig(n_folds=cfg.n_folds))
                # identical folds for every variant on this dataset: the fold
                # split is drawn from a per-dataset stream, not a per-variant one
                fold_rng = fit_rng(cfg.namespace, target, N, rep, 0)
                t0 = perf_counter()
                oracle_fn = (lambda f: metrics.ise(f, tgt)) if cfg.oracle else None
                fit, log = est.fit_logged(X, fold_rng, oracle_ise=oracle_fn)
                wall = perf_counter() - t0

                row = dict(base)
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
                          "stage1_time_s", "stage2_time_s", "rho", "lam_ratio",
                          "weighting", "cov_eig_max", "cov_eig_min",
                          "cov_rank_numeric", "cov_effective_rank", "omega_cond",
                          "selection", "refit", "mass_below_meas_scale",
                          "mean_atom_sd_over_h"):
                    if k in fit.diagnostics:
                        row[k] = fit.diagnostics[k]
                for k in ("n_candidates", "n_folds", "fold_errors",
                          "all_candidates_failed", "n_nan_scores", "best_cv_score",
                          "sel_rho", "sel_lam_ratio", "rho_at_grid_edge",
                          "lam_at_grid_edge", "rho_edge_side", "lam_edge_side",
                          "tune_time_s", "fit_time_s", "oracle_path_time_s"):
                    row[k] = log.get(k)
                row["error"] = ""
                rows.append(row)

                if cfg.save_fits:
                    np.savez_compressed(
                        out_root / "fits" / ("%s_N%d_r%d_%s.npz" % (target, N, rep, vname)),
                        weights=fit.weights, means=fit.means, variances=fit.variances,
                        support=np.asarray(fit.support if fit.support is not None else []),
                        ise=row["ise"])

                if cfg.oracle:
                    orows.append(_oracle_row(base, fit, vcfg, X, tgt, log))
            except Exception as exc:
                row = dict(base)
                row["error"] = "%s: %s" % (type(exc).__name__, exc)
                row["traceback"] = traceback.format_exc(limit=4)
                row["ise"] = np.nan
                rows.append(row)
                if verbose:
                    print("  ERROR %s rep=%d %s: %s" % (target, rep, vname, exc), flush=True)
        if verbose:
            done = [r for r in rows if r["rep"] == rep and not r["error"]]
            print("  %s N=%d rep=%d: %d/%d variants, median ISE %.4g"
                  % (target, N, rep, len(done), len(vnames),
                     float(np.median([r["ise"] for r in done])) if done else np.nan),
                  flush=True)
    return rows, orows


def _oracle_row(base, fit, vcfg: FollowupConfig, X, tgt, log) -> dict:
    """Oracle decomposition for one fit.  Isolated: never returned to the estimator."""
    atoms = build_dictionary(X, DictionaryConfig(vcfg.dict_n_loc, vcfg.dict_n_scale,
                                                 "grid", vcfg.dict_lo_mult,
                                                 vcfg.dict_hi_mult))
    w = _full_weight_vector(fit, atoms)
    support = np.asarray(fit.support) if fit.support is not None else np.flatnonzero(w > 0)
    dec = oracle_mod.decompose(w, atoms, tgt, support=support)
    rec = dict(base)
    rec.update({k: v for k, v in dec.items() if not isinstance(v, np.ndarray)})
    rec["oracle_best_ise"] = log.get("oracle_best_ise")
    rec["oracle_selected_ise"] = log.get("oracle_selected_ise")
    rec["oracle_tuning_gap_ratio"] = log.get("oracle_tuning_gap_ratio")
    rec["oracle_path_time_s"] = log.get("oracle_path_time_s")
    return rec


def reference_fit(est_name: str, X, tgt, rng, n_folds: int = 5):
    """Fit one registry baseline (for the figure references), returning (fit, row)."""
    from akde.registry import build
    est = build(est_name, n_loc=150, n_scale=10)
    est.tuning.n_folds = n_folds
    t0 = perf_counter()
    fit = est.fit(X, rng)
    wall = perf_counter() - t0
    row = {"estimator": est_name, "wall_time_s": wall,
           "ise": float(metrics.ise(fit, tgt)),
           "n_active": int(fit.n_active(tol=1e-8)),
           "is_valid_density": bool(fit.is_valid_density())}
    return fit, row
