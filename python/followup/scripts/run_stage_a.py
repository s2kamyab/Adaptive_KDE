"""Stage A — focused diagnosis.

4 targets x 2 sample sizes x 5 development replicates = 40 datasets,
8 variants = 320 final fits (plus 3-fold CV subfits and separately logged
oracle solves).  One literal dictionary per cell: 60x5 = 300 atoms, pinned,
never cross-validated — the mechanism controls require the compared variants
to sit on the same atom set.

    python followup/scripts/run_stage_a.py [--force] [--reps 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from followup.akde_followup.runner import StageConfig, run_stage, pin_threads  # noqa: E402
from followup.akde_followup.variants import STAGE_A                            # noqa: E402

TARGETS = ("f0", "f2", "f5", "f9")     # MW1 Gaussian, MW3 strongly skewed,
SIZES = (500, 5000)                    # MW6 bimodal, MW10 claw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    pin_threads(1)

    cfg = StageConfig(name="stage_a", namespace="development",
                      targets=TARGETS, sizes=SIZES, n_reps=a.reps,
                      variants=STAGE_A, n_folds=a.folds, oracle=True)
    print("Stage A: %d targets x %d sizes x %d reps x %d variants = %d final fits"
          % (len(TARGETS), len(SIZES), a.reps, len(STAGE_A),
             len(TARGETS) * len(SIZES) * a.reps * len(STAGE_A)), flush=True)
    df = run_stage(cfg, force=a.force, n_jobs=a.jobs)
    # an empty error cell round-trips through CSV as NaN, not ""
    err = df["error"].fillna("").astype(str)
    print("\n%d rows, %d errors" % (len(df), int((err != "").sum())))
    ok = df[err == ""]
    print("\nmedian ISE by variant and N:")
    print(ok.pivot_table(index="variant", columns="N", values="ise",
                         aggfunc="median").to_string(float_format=lambda v: "%.5g" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
