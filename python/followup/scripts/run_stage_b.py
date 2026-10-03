"""Stage B — one targeted refinement round.

The SAME 40 development datasets and the SAME folds as Stage A, with at most
four additional variants chosen from Stage-A evidence: 4 x 40 = 160 more final
fits.  Same pinned 60x5 dictionary.  Rationale for each variant is in
followup/akde_followup/variants.py and diagnosis_report.md.

    python followup/scripts/run_stage_b.py [--force] [--reps 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from followup.akde_followup.runner import StageConfig, run_stage, pin_threads  # noqa: E402
from followup.akde_followup.variants import STAGE_B                            # noqa: E402

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

    cfg = StageConfig(name="stage_b", namespace="development",
                      targets=TARGETS, sizes=SIZES, n_reps=a.reps,
                      variants=STAGE_B, n_folds=a.folds, oracle=True)
    print("Stage B: %d targets x %d sizes x %d reps x %d variants = %d final fits"
          % (len(TARGETS), len(SIZES), a.reps, len(STAGE_B),
             len(TARGETS) * len(SIZES) * a.reps * len(STAGE_B)), flush=True)
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
