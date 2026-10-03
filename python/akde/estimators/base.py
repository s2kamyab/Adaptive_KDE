"""Estimator interface and the shared tuning protocol.

Every estimator tunes itself using **training data only**, by K-fold
cross-validation of the held-out integrated-square (J) score.  The true
synthetic density is never consulted during fitting or tuning; it appears only
in the final scoring step.  This is the single most important protocol
constraint in the evaluation recommendations, so it is enforced structurally:
``fit`` receives nothing but ``X``.

Hyperparameter candidates are expressed in *relative* terms wherever the
natural scale is data-dependent (``lam`` as a fraction of the ``lam`` at which
the solution collapses to zero, ``rho`` as a fraction of the mean covariance
eigenvalue).  Each fold then resolves them against its own training split.
This is what makes the penalty comparable across methods and across weighting
matrices whose scales differ by factors of N -- rather than sharing one
numerical constant between two differently scaled objectives.

Timing is reported split into tuning and final fitting, as requested, since
the two have very different meanings for a practitioner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from ..metrics import j_score
from ..model import FittedDensity

__all__ = ["Estimator", "TuningConfig", "kfold_indices"]


@dataclass
class TuningConfig:
    n_folds: int = 5
    shuffle: bool = True
    min_val_size: int = 5


def kfold_indices(n: int, n_folds: int, rng: np.random.Generator,
                  shuffle: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plain K-fold split.

    Cross-validation rather than one large holdout: at N = 50 a 30% validation
    set would throw away most of the information the estimator needs, and the
    protocol notes this explicitly.
    """
    idx = np.arange(n)
    if shuffle:
        idx = rng.permutation(n)
    n_folds = max(2, min(n_folds, n))
    folds = np.array_split(idx, n_folds)
    out = []
    for k in range(n_folds):
        val = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        if val.size == 0 or train.size == 0:
            continue
        out.append((train, val))
    return out


class Estimator(ABC):
    """Base class.  Subclasses implement :meth:`candidates` and :meth:`fit_path`."""

    name: str = "estimator"
    #: Human-readable note recorded in the results, e.g. what a baseline
    #: actually computes when that is easy to get wrong.
    provenance: str = ""

    def __init__(self, tuning: TuningConfig | None = None) -> None:
        self.tuning = tuning or TuningConfig()

    @abstractmethod
    def candidates(self, X: np.ndarray) -> list[dict]:
        """Hyperparameter settings to select among, in relative terms."""

    @abstractmethod
    def fit_path(self, X: np.ndarray, cands: list[dict],
                 rng: np.random.Generator) -> list[FittedDensity]:
        """Fit every candidate on ``X``; return one density per candidate."""

    # ------------------------------------------------------------------

    def _domain(self, X: np.ndarray) -> tuple[float, float]:
        X = np.asarray(X, dtype=float)
        lo, hi = float(np.min(X)), float(np.max(X))
        span = max(hi - lo, 1e-6)
        return lo - span, hi + span

    def fit(self, X: np.ndarray, rng: np.random.Generator | None = None) -> FittedDensity:
        rng = rng or np.random.default_rng(0)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]

        cands = self.candidates(X)
        diagnostics: dict = {"n_candidates": len(cands)}

        t0 = perf_counter()
        if len(cands) > 1:
            folds = kfold_indices(X.shape[0], self.tuning.n_folds, rng, self.tuning.shuffle)
            scores = np.full((len(folds), len(cands)), np.nan)
            domain = self._domain(X)
            for fi, (tr, va) in enumerate(folds):
                try:
                    fits = self.fit_path(X[tr], cands, rng)
                except Exception:
                    continue
                for ci, f in enumerate(fits):
                    if f is None:
                        continue
                    try:
                        scores[fi, ci] = j_score(f, X[va], domain=domain)
                    except Exception:
                        scores[fi, ci] = np.nan
            with np.errstate(invalid="ignore"):
                mean_scores = np.nanmean(scores, axis=0)
            if np.all(np.isnan(mean_scores)):
                best_i = 0
            else:
                best_i = int(np.nanargmin(mean_scores))
            diagnostics["cv_scores"] = mean_scores.tolist()
            diagnostics["cv_n_folds"] = len(folds)
        else:
            best_i = 0
        tune_time = perf_counter() - t0

        best = cands[best_i]
        t1 = perf_counter()
        fit = self.fit_path(X, [best], rng)[0]
        fit_time = perf_counter() - t1

        fit.diagnostics.setdefault("estimator", self.name)
        fit.diagnostics.update(
            tune_time_s=tune_time,
            fit_time_s=fit_time,
            total_time_s=tune_time + fit_time,
            selected=dict(best),
            provenance=self.provenance,
            **{k: v for k, v in diagnostics.items() if k != "cv_scores"},
        )
        return fit
