"""Replication protocol and parallel experiment runner.

Protocol, following the evaluation recommendations:

* One **independently generated dataset** per replicate.  Every estimator
  receives the identical realisation, which is what makes the paired
  bootstrap in ``bootstrap.py`` valid and much tighter than an unpaired
  comparison.
* Seeds derive deterministically from ``(seed, target, N, rep)`` through
  ``SeedSequence``, so a replicate is reproducible on its own and results do
  not depend on execution order or on how the work was parallelised.
* Data generation and estimator-internal randomness use **separate** streams,
  so changing an estimator cannot perturb the data another estimator sees.
* Held-out negative log likelihood uses a fresh test sample from the truth.
  That is legitimate test data; it is never available during fitting or
  tuning.  Only the true density used for *scoring* is off-limits to the
  estimator, and it is structurally unreachable -- ``fit`` takes only ``X``.

Runs are sharded by ``(target, N)`` and skipped if already complete, so a long
run can be interrupted and resumed.
"""

from __future__ import annotations

import json
import os
import traceback
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import metrics
from .densities import get_target
from .registry import build

__all__ = ["RunConfig", "run", "run_one_replicate", "SCALAR_DIAGNOSTICS", "name_key"]


#: Diagnostics copied into the results table as columns.
SCALAR_DIAGNOSTICS = (
    "n_selected", "n_dictionary", "n_measurements", "fallback_used",
    "support_capped", "stage1_kkt", "stage2_kkt", "stage1_converged",
    "setup_time_s", "stage1_time_s", "stage2_time_s",
    "tune_time_s", "fit_time_s", "total_time_s",
    "rho", "lam_ratio", "weighting", "cov_eig_max", "cov_eig_min",
    "cov_rank_numeric", "cov_effective_rank", "omega_cond", "trunc_kept",
    "kernel_mult", "nu", "budget_frac", "bw_mult", "thinning_method",
    "selector", "em_converged", "min_component_sd", "lam_clipped",
    "lam_min_used", "lam_max_used", "density_floor", "pilot_rule",
)


@dataclass
class RunConfig:
    name: str = "run"
    targets: tuple[str, ...] = tuple(f"f{i}" for i in range(15))
    sample_sizes: tuple[int, ...] = (50, 100, 200, 500, 1000, 5000)
    n_reps: int = 30
    estimators: tuple[str, ...] = ()
    seed: int = 20260915
    out_dir: str = "results"
    n_jobs: int = -1
    dict_n_loc: int = 150
    dict_n_scale: int = 10
    n_test: int = 2000
    compute_iae: bool = True
    compute_hellinger: bool = True
    n_folds: int = 5

    def shard_path(self, target: str, n: int) -> Path:
        return Path(self.out_dir) / self.name / f"{target}_N{n}.csv"


def name_key(name: str) -> int:
    """Stable integer key for a target name.

    ``hash()`` on a str is salted per interpreter process (PYTHONHASHSEED is
    random by default), so using it here would make the dataset for a given
    ``(seed, target, N, rep)`` depend on which worker process happened to run
    the shard and on when the run was started.  That silently breaks the
    reproducibility guarantee documented at the top of this module, and it
    means a resumed run does not continue the same experiment.  crc32 is
    stable across processes, machines and Python versions.
    """
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def _child_seed(master: int, target: str, n: int, rep: int, stream: int) -> np.random.Generator:
    key = [int(master), name_key(target), int(n), int(rep), int(stream)]
    return np.random.default_rng(np.random.SeedSequence(key))


def run_one_replicate(cfg: RunConfig, target_name: str, n: int, rep: int) -> list[dict]:
    """Fit every estimator on one shared dataset and score them all."""
    target = get_target(target_name)
    d = target.dim

    rng_data = _child_seed(cfg.seed, target_name, n, rep, 0)
    X = target.sample(n, rng_data)
    X_test = target.sample(cfg.n_test, _child_seed(cfg.seed, target_name, n, rep, 1))

    rows: list[dict] = []
    for ei, est_name in enumerate(cfg.estimators):
        base = {
            "run": cfg.name, "target": target_name, "target_label": target.label,
            "dim": d, "N": n, "rep": rep, "estimator": est_name,
        }
        try:
            est = build(est_name, n_loc=cfg.dict_n_loc, n_scale=cfg.dict_n_scale)
            est.tuning.n_folds = cfg.n_folds
            rng_fit = _child_seed(cfg.seed, target_name, n, rep, 100 + ei)

            t0 = perf_counter()
            fit = est.fit(X, rng_fit)
            wall = perf_counter() - t0

            row = dict(base)
            row["method_label"] = getattr(est, "name", est_name)
            row["wall_time_s"] = wall

            row["ise"] = metrics.ise(fit, target)
            row["total_mass"] = fit.total_mass()
            row["mass_error"] = metrics.l1_mass_error(fit)
            row["neg_weight_mass"] = fit.negative_weight_mass()
            row["is_valid_density"] = fit.is_valid_density()
            row["n_active"] = fit.n_active(tol=1e-8)
            row["n_active_tol"] = 1e-8

            if d == 1:
                row["neg_mass"] = metrics.negative_mass(fit, target)
                if cfg.compute_iae:
                    row["iae"] = metrics.iae(fit, target)
                if cfg.compute_hellinger:
                    row["hellinger"] = metrics.hellinger(fit, target)

            nll, floored = metrics.held_out_nll(fit, X_test)
            row["nll"] = nll
            row["nll_floored"] = floored

            for k in SCALAR_DIAGNOSTICS:
                if k in fit.diagnostics:
                    v = fit.diagnostics[k]
                    if isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_)):
                        row[k] = v
            sel = fit.diagnostics.get("selected")
            if isinstance(sel, dict):
                for k, v in sel.items():
                    if isinstance(v, (int, float, bool, str)):
                        row[f"sel_{k}"] = v

            row["error"] = ""
            rows.append(row)
        except Exception as exc:  # a failure is a result, not a crash
            row = dict(base)
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=3)
            row["ise"] = np.nan
            rows.append(row)
    return rows


def _run_shard(cfg: RunConfig, target: str, n: int) -> str:
    """Run one ``(target, N)`` cell and write it to disk immediately.

    The worker writes its own shard rather than returning a frame for the
    parent to write at the end.  A full run takes hours; if results were only
    persisted after every task finished, an interruption would lose all of it
    and ``--force``-free resumption would be pointless.  Writing to a
    temporary file and renaming makes a shard appear atomically, so a
    half-written file is never mistaken for a completed cell.
    """
    rows: list[dict] = []
    for rep in range(cfg.n_reps):
        rows.extend(run_one_replicate(cfg, target, n, rep))
    df = pd.DataFrame(rows)
    path = cfg.shard_path(target, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return str(path)


def run(cfg: RunConfig, verbose: int = 10, force: bool = False) -> pd.DataFrame:
    """Execute the whole grid, one shard per ``(target, N)``, and concatenate.

    Shards already on disk are reused unless ``force=True``, which makes long
    runs interruptible.
    """
    out_root = Path(cfg.out_dir) / cfg.name
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    todo = []
    existing = []
    for target in cfg.targets:
        for n in cfg.sample_sizes:
            p = cfg.shard_path(target, n)
            if p.exists() and not force:
                existing.append(p)
            else:
                todo.append((target, n))

    if todo:
        # Largest cells first: with a long tail of slow shards, scheduling the
        # expensive ones early keeps every worker busy to the end instead of
        # leaving one core finishing N=5000 while 19 idle.
        todo.sort(key=lambda tn: -tn[1])
        # inner_max_num_threads=1: keep each worker single-threaded in BLAS.
        # See the note at the top of experiments/run_benchmark.py -- this
        # prevents thread oversubscription and makes replicates reproducible.
        Parallel(n_jobs=cfg.n_jobs, verbose=verbose, backend="loky",
                 inner_max_num_threads=1)(
            delayed(_run_shard)(cfg, t, n) for t, n in todo
        )

    frames = []
    for target in cfg.targets:
        for n in cfg.sample_sizes:
            p = cfg.shard_path(target, n)
            if p.exists():
                frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_root / "all_results.csv", index=False)
    return out
