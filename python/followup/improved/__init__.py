"""Improved AKDE: the recommended repairs, implemented as one configurable
estimator next to (never inside) the confirmed finalist.

Repairs (each a switch, so every combination is an ablation):

  weighting  'gram'      Omega = (G + eps I)^-1, the metric the error is scored in
                         (Proposition 3: with matched readings this IS the
                         integrated-square loss)
             'identity'  Omega = I
             'ridge'     Omega = (C_hat + rho s I)^-1, the confirmed finalist
  solver     'fista'     residual-space FISTA run to a relative KKT tolerance
             'cd_v2'     the deployed 500-sweep coordinate descent
  mode       'two_stage' L1 selection + simplex refit (as the finalist)
             'simplex_l2' no selection stage: the L2 loss minimised over the
                         simplex on the full dictionary (fully corrective
                         Frank-Wolfe); sparsity comes from the simplex
  penalty    'uniform' | 'colnorm'  per-atom penalty lambda_i proportional to the
                         atom's L2 norm (narrow atoms must earn their place)
  lambda     6-point path to 3e-4 (finalist) or 8-point path to 3e-5 (extended)
  select     'min' | '1se' (sparsest lambda within one SE of the best CV score)
  folds      3 (finalist) or 5
  anchor     'silverman' scales geomspace(0.05, 4, 5) x h_Silverman (finalist)
             'cv'        base bandwidth chosen by validation from a 9-point grid,
                         then relative scales (0.25, 0.5, 1, 2, 4) x h_cv
  merge      post-fit greedy moment-matched merging of near-duplicate atoms,
             accepted only while the TRAINING L2 loss rises by < merge_tol,
             then a simplex refit of the weights

Importing this package pins BLAS to one thread via ``followup``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[2]
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import followup  # noqa: E402,F401

__all__ = ["PY_ROOT"]
