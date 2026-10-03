"""FROZEN configuration for the bounded post-confirmation numerical checks.

Frozen 2026-09-28, before any of the Task-4 checks were run. These are
*diagnostics on an already-completed confirmation*, not a new confirmation.
Nothing here may be re-tuned to obtain a preferred outcome; in particular the
rank-threshold list is a sensitivity sweep and every value is reported.

The diagnostic subset is deterministic: the first two existing DEVELOPMENT
replications of four targets at two sample sizes = 16 datasets. Development
seeds are used because these are diagnostics; the confirmation seeds are not
touched.
"""
from __future__ import annotations

TARGETS = ("f0", "f2", "f5", "f9")
#: Expected Marron-Wand identities, asserted at run time against
#: akde.densities.MARRON_WAND_MAPPING so a silent renumbering cannot pass.
TARGET_LABELS = {"f0": (1, "Gaussian"), "f2": (3, "Strongly skewed"),
                 "f5": (6, "Bimodal"), "f9": (10, "Claw")}
SIZES = (500, 5000)
REPS = (0, 1)
NAMESPACE = "development"

#: Both measurement designs, always compared on the same data and dictionary.
BANKS = ("base", "matched")

#: Pinned dictionary for every check (the Stage-A/B/C dictionary).
DICT = dict(n_loc=60, n_scale=5, lo=0.05, hi=4.0)

#: Ridge factor held fixed across the centre/oracle checks. 1e-3 is the value
#: the original centre computation used; 1.0 is added as a sensitivity point.
RHO_MAIN = 1e-3
RHO_SENSITIVITY = (1e-6, 1e-3, 1e-1, 1.0)

#: Relative singular-value cutoffs for the pseudoinverse sensitivity sweep.
#: ALL of these are reported; none is selected post hoc.
RANK_THRESHOLDS = (1e-8, 1e-10, 1e-12, 1e-14)

#: Stage-1 iteration budgets for the solver-budget sensitivity.
STAGE1_BUDGETS = (500, 2000, 10000)

#: Practical-equivalence criterion, declared BEFORE running the solver
#: comparison. Two solvers are "practically equivalent on this subset" only if
#: BOTH hold on every instance:
PRACTICAL_EQUIVALENCE = {
    "max_rel_ise_change": 0.02,        # |ISE_a - ISE_b| / ISE_b <= 2%
    "max_fitted_density_l2": 1e-3,     # ||f_a - f_b||_2 <= 1e-3
}

#: Higher-precision cross-check settings (mpmath).
HP_DIGITS = 50
HP_DICT = dict(n_loc=12, n_scale=3, lo=0.05, hi=4.0)   # M = 36, tractable
HP_N = 200

#: Analytic-ISE resolution model. ISE is computed as ff - 2fg + gg with each
#: term O(int f0^2); the double-precision resolution of that difference is
#: modelled as CANCELLATION_FACTOR * eps * int f0^2 and is VALIDATED against
#: mpmath in check_centres.py rather than assumed.
CANCELLATION_FACTOR = 1e4

#: Number of worst-residual stress instances pulled from the existing logs.
#: These are explicitly NOT a representative sample.
N_STRESS = 8

CONFIG_VERSION = "frozen-2026-09-28-a"
