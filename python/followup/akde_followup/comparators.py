"""The Stage-C comparator set, built under one tuning policy.

Every dictionary-based method gets the *same pinned* 60x5 = 300-atom
dictionary — the control that V2 documented but did not run (audit_findings.md
Q1) — and every method gets the same 3-fold CV budget and, where applicable,
the same 6-point lambda path.  That is why no V2 comparator row is reused:
the policy differs, so the rows are not interchangeable.

`build_comparator` returns an object with `.fit(X, rng) -> FittedDensity` and
`.name`, so the runner treats follow-up variants and registry baselines alike.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PY = Path(__file__).resolve().parents[2]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from akde.dictionary import DictionaryConfig                # noqa: E402
from akde.estimators.akde import AKDE, AKDEConfig           # noqa: E402
from akde.estimators.base import TuningConfig               # noqa: E402
from akde.estimators.emgmm import EMConfig, EMGaussianMixture  # noqa: E402
from akde.estimators.kde import AbramsonConfig, AbramsonKDE, FixedKDE  # noqa: E402
from akde.estimators.spades import SPADESConfig, SPADESSimplex  # noqa: E402

from .variants import FollowupAKDE, FollowupConfig          # noqa: E402

__all__ = ["PINNED_DICT", "LOCKED_TUNING", "build_comparator", "COMPARATORS"]

#: The one dictionary every dictionary-based method uses in the follow-up.
PINNED_DICT = dict(n_loc=60, n_scale=5, location_rule="grid",
                   scale_lo_mult=0.05, scale_hi_mult=4.0)

LOCKED_TUNING = dict(n_folds=3, n_lambda=6, lam_max_ratio=0.7,
                     lam_min_ratio=3e-4, tau=1e-3,
                     rho_grid=(1e-6, 1e-3, 1e-1, 1.0))


@dataclass
class Comparator:
    key: str
    label: str
    note: str
    make: object

    def build(self):
        return self.make()


def _v2_akde():
    """The V2 estimator, on the pinned dictionary and the follow-up CV budget.

    This is V2's *algorithm* (identical objective, identical solver, identical
    tau, identical measurement bank), evaluated under the new policy.  Two
    deliberate differences from the V2 benchmark rows, both required by the
    audit: the dictionary size is pinned instead of cross-validated, and the
    tuning budget is 3 folds and a 6-point lambda path.  Because of these the
    V2 rows are not reused as comparator rows.
    """
    cfg = AKDEConfig(dictionary=DictionaryConfig(**PINNED_DICT), dict_sizes=None,
                     meas_mode="independent", meas_n_loc=100, meas_n_scale=3,
                     meas_scale_lo_mult=0.25, meas_scale_hi_mult=2.0,
                     weighting="ridge", rho_grid=LOCKED_TUNING["rho_grid"],
                     n_lambda=LOCKED_TUNING["n_lambda"],
                     lam_max_ratio=LOCKED_TUNING["lam_max_ratio"],
                     lam_min_ratio=LOCKED_TUNING["lam_min_ratio"],
                     tau=LOCKED_TUNING["tau"], max_support=400)
    est = AKDE(cfg, tuning=TuningConfig(n_folds=LOCKED_TUNING["n_folds"]),
               name="V2 AKDE")
    return est


def _spades_simplex():
    cfg = SPADESConfig(dictionary=DictionaryConfig(**PINNED_DICT),
                       n_lambda=LOCKED_TUNING["n_lambda"],
                       lam_max_ratio=LOCKED_TUNING["lam_max_ratio"],
                       lam_min_ratio=LOCKED_TUNING["lam_min_ratio"],
                       tau=LOCKED_TUNING["tau"], max_support=400)
    return SPADESSimplex(cfg, tuning=TuningConfig(n_folds=LOCKED_TUNING["n_folds"]))


def _kde_sj():
    return FixedKDE("sj", tuning=TuningConfig(n_folds=LOCKED_TUNING["n_folds"]))


def _abramson():
    return AbramsonKDE(AbramsonConfig(),
                       tuning=TuningConfig(n_folds=LOCKED_TUNING["n_folds"]))


def _gmm_em():
    return EMGaussianMixture(EMConfig(),
                             tuning=TuningConfig(n_folds=LOCKED_TUNING["n_folds"]))


COMPARATORS = {
    "v2_akde": Comparator("v2_akde", "Original V2 AKDE",
                          "ridge-weighted moment, two-stage, m=300, pinned 300-atom dict",
                          _v2_akde),
    "spades_simplex": Comparator("spades_simplex", "SPADES-simplex (density-valid)",
                                 "non-negative L1 selection on the integrated-square "
                                 "loss + simplex refit; same pinned dictionary",
                                 _spades_simplex),
    "kde_sj": Comparator("kde_sj", "KDE [Sheather-Jones]",
                         "solve-the-equation plug-in bandwidth", _kde_sj),
    "abramson": Comparator("abramson", "Abramson adaptive KDE",
                           "square-root law, pilot-normalised", _abramson),
    "gmm_em": Comparator("gmm_em", "Gaussian mixture (EM)",
                         "component count by the same validation criterion", _gmm_em),
}


def build_comparator(key: str, finalist: FollowupConfig | None = None):
    if key == "finalist":
        if finalist is None:
            raise ValueError("the finalist configuration must be supplied")
        return FollowupAKDE(finalist, TuningConfig(n_folds=LOCKED_TUNING["n_folds"]))
    return COMPARATORS[key].build()
