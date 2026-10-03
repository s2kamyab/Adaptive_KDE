"""Experimental framework for Adaptive KDE via GMM and compressive kernel selection.

Companion code for *Adaptive Kernel Density Estimation using the Generalized
Method of Moments and Compressive Kernel Selection* (Kamyab, Nasihatkon &
Boostani).

The framework is built around the requirements in
``Experimental_comparison_recommendations.md``:

* the revised two-stage estimator (non-negative selection, simplex refit),
  which is a different estimator from the one behind the historical figures;
* a comparison set that keeps the direct predecessors rather than replacing
  them with more recent but less comparable methods;
* integrated squared error computed analytically from Gaussian overlaps;
* hyperparameters chosen from training data only, by held-out
  integrated-square score;
* covariance estimated from the training sample, never from fresh draws of
  the true density;
* bootstrap confidence intervals on paired differences over independently
  generated replicate datasets.
"""

from . import (
    bandwidths, bootstrap, densities, dictionary, gaussops, metrics, model,
    registry, runner, solvers,
)
from .densities import MARRON_WAND_MAPPING, get_target, list_targets
from .model import FittedDensity
from .registry import ABLATIONS, ALL, ESSENTIAL, NOT_IMPLEMENTED, build
from .runner import RunConfig, run

__version__ = "1.0.0"

__all__ = [
    "bandwidths", "bootstrap", "densities", "dictionary", "gaussops",
    "metrics", "model", "registry", "runner", "solvers",
    "MARRON_WAND_MAPPING", "get_target", "list_targets", "FittedDensity",
    "build", "ESSENTIAL", "ABLATIONS", "ALL", "NOT_IMPLEMENTED",
    "RunConfig", "run",
]
