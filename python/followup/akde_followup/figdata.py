"""Reconstruct everything the figures need from saved fits + seeds.

The estimators are never refitted here: each panel's curves come from the
``.npz`` written by the runner at fit time.  Datasets are regenerated from
their seed (deterministic), which is what makes the residual panels of
Figure C possible without storing every design matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PY = Path(__file__).resolve().parents[2]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from akde import gaussops                                    # noqa: E402
from akde.densities import get_target                        # noqa: E402
from akde.dictionary import DictionaryConfig, build_dictionary  # noqa: E402
from akde.model import mixture_density                       # noqa: E402

from . import oracle                                         # noqa: E402
from .seeds import data_rng                                  # noqa: E402
from .variants import FollowupAKDE, build_measurement_bank   # noqa: E402

RESULTS = _PY / "followup" / "results"

__all__ = ["load_fit", "fit_pdf", "dataset", "residual_panel", "pinned_dictionary"]


def pinned_dictionary(X, n_loc=60, n_scale=5, lo=0.05, hi=4.0):
    return build_dictionary(X, DictionaryConfig(n_loc, n_scale, "grid", lo, hi))


def dataset(namespace: str, target: str, N: int, rep: int) -> np.ndarray:
    return get_target(target).sample(N, data_rng(namespace, target, N, rep))


def load_fit(stage: str, target: str, N: int, rep: int, variant: str):
    """Return the saved mixture, or None if that fit is missing."""
    p = RESULTS / stage / "fits" / ("%s_N%d_r%d_%s.npz" % (target, N, rep, variant))
    if not p.exists():
        return None
    z = np.load(p)
    return {"weights": z["weights"], "means": z["means"],
            "variances": z["variances"], "ise": float(z["ise"]),
            "path": str(p)}


def fit_pdf(fit: dict, x: np.ndarray) -> np.ndarray:
    return gaussops.mixture_pdf_diag(np.asarray(x, float)[:, None],
                                     fit["weights"], fit["means"], fit["variances"])


def _full_weights(fit, atoms):
    w = np.zeros(atoms.size)
    key = {(round(float(m), 12), round(float(v), 12)): i
           for i, (m, v) in enumerate(zip(atoms.means[:, 0], atoms.variances[:, 0]))}
    missing = 0
    for wi, mi, vi in zip(fit["weights"], fit["means"][:, 0], fit["variances"][:, 0]):
        k = (round(float(mi), 12), round(float(vi), 12))
        if k in key:
            w[key[k]] = wi
        else:
            missing += 1
    return w, missing


def residual_panel(stage: str, target: str, N: int, rep: int, variant: str,
                   spec, namespace: str = "development") -> dict | None:
    """Measurement residuals for one fit, training vs population.

    Returns centres, scales, the empirical residual ``D u - y``, the
    population residual ``D u - y0``, and the standardised versions
    ``(D u - y)_j / sqrt(C_hat_jj)`` — the scale on which the estimator
    actually "sees" a discrepancy.
    """
    fit = load_fit(stage, target, N, rep, variant)
    if fit is None:
        return None
    tgt = get_target(target)
    X = dataset(namespace, target, N, rep)
    atoms = pinned_dictionary(X)
    w, missing = _full_weights(fit, atoms)
    bank = build_measurement_bank(X, spec, dictionary=atoms)
    D = gaussops.design_matrix(bank.means, bank.variances,
                               atoms.means, atoms.variances)
    K = FollowupAKDE.eval_kernels(X, bank)
    y = K.mean(axis=0)
    Kc = K - y
    C = (Kc.T @ Kc) / (N * (N - 1.0))
    y0 = oracle.population_moments(bank, tgt)
    Du = D @ w
    sd = np.sqrt(np.maximum(np.diag(C), 0.0))
    # Standardising by an estimated sampling sd is only meaningful where that
    # estimate is itself based on something.  A narrow probe centred where the
    # sample has (almost) no points has a sampling variance estimated from a
    # handful of effective observations, and the ratio explodes.  The
    # effective count for probe j is Kish's
    #     n_eff_j = (sum_n k_j(x_n))^2 / sum_n k_j(x_n)^2,
    # which is N for a probe that weights every observation equally and 1 for
    # one dominated by a single point.  Probes with n_eff < n_eff_min are
    # excluded and COUNTED, never clipped or silently plotted.
    ssum = K.sum(axis=0)
    ssq = (K ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(ssq > 0, ssum ** 2 / ssq, 0.0)
    n_eff_min = 5.0
    usable = (n_eff >= n_eff_min) & (sd > 0)
    std = np.full(sd.shape, np.nan)
    std[usable] = (Du - y)[usable] / sd[usable]
    return {
        "centres": bank.means[:, 0], "scales": np.sqrt(bank.variances[:, 0]),
        "emp_residual": Du - y, "pop_residual": Du - y0,
        "std_emp_residual": std, "sampling_sd": sd, "n_eff": n_eff,
        "std_usable": usable.astype(float),
        "n_probes_unresolved": int((~usable).sum()),
        "n_eff_min": n_eff_min,
        "sd_floor": float(np.min(sd[usable])) if usable.any() else 0.0,
        "y": y, "y0": y0, "Du": Du, "atoms_missing_from_dict": missing,
        "m": int(bank.size), "ise": fit["ise"], "n_support": int(fit["weights"].size),
    }
