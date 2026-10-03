"""Main experiment driver.

Usage
-----
    python experiments/run_benchmark.py --tier pilot
    python experiments/run_benchmark.py --tier full --jobs 20
    python experiments/run_benchmark.py --suite ablation --tier full
    python experiments/run_benchmark.py --suite multivariate --tier full

Tiers trade replication against wall time.  ``full`` is the paper-grade
setting from the evaluation protocol: sample sizes 50-5000, 30 independently
generated datasets per condition.  ``pilot`` is the same design at reduced
replication, for checking that a configuration behaves before committing to
the long run.

Runs are sharded by (target, N) and skipped if the shard already exists, so an
interrupted run resumes where it stopped.  Pass --force to recompute.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Pin BLAS to a single thread BEFORE numpy is imported.
#
# Two reasons, both learned the hard way:
#  * Oversubscription. Each joblib worker would otherwise start its own pool of
#    BLAS threads (20 workers x 22 threads = 440 threads on this machine),
#    which thrashes and inflates peak memory enough to get the run killed.
#  * Reproducibility. Multi-threaded matrix products sum in a
#    scheduling-dependent order. The moment design matrix here is numerically
#    singular (condition numbers around 1e17), so that last-bit difference can
#    change which atoms are selected and move the reported ISE by a factor of
#    three on identical data with identical seeds. Single-threaded BLAS makes a
#    replicate bit-reproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akde import registry
from akde.densities import list_targets
from akde.runner import RunConfig, run

TIERS = {
    "smoke": dict(sample_sizes=(100, 500), n_reps=3),
    "pilot": dict(sample_sizes=(100, 500, 1000), n_reps=10),
    "full": dict(sample_sizes=(50, 100, 200, 500, 1000, 5000), n_reps=30),
}

SUITES = {
    # the recommended first round: five essential rows plus kernel thinning
    "main": dict(targets=tuple(f"f{i}" for i in range(15)),
                 estimators=tuple(registry.ESSENTIAL)),
    # core ablations: weighting scheme, scales, sparsity, refit, measurement design
    "ablation": dict(targets=tuple(f"f{i}" for i in range(15)),
                     estimators=tuple(registry.ABLATIONS)),
    # outliers and rare narrow modes, kept separate rather than averaged in
    "stress": dict(targets=tuple(list_targets("stress")),
                   estimators=tuple(registry.ESSENTIAL)),
    # 2D and 5D full-rank targets with unequal scales
    "multivariate": dict(targets=tuple(list_targets("multivariate")),
                         estimators=("akde", "spades_simplex", "song", "kde_sj",
                                     "abramson", "gmm_em", "thin_kt")),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="main", choices=sorted(SUITES))
    ap.add_argument("--tier", default="pilot", choices=sorted(TIERS))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="", help="override the run name")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--n-loc", type=int, default=150)
    ap.add_argument("--n-scale", type=int, default=10)
    ap.add_argument("--reps", type=int, default=0, help="override replicate count")
    ap.add_argument("--sizes", default="", help="override sample sizes, e.g. 100,500")
    ap.add_argument("--targets", default="", help="override target list")
    ap.add_argument("--estimators", default="", help="override estimator list")
    ap.add_argument("--force", action="store_true", help="recompute existing shards")
    ap.add_argument("--no-iae", action="store_true",
                    help="skip IAE/Hellinger quadrature (faster)")
    args = ap.parse_args()

    spec = dict(SUITES[args.suite])
    tier = dict(TIERS[args.tier])
    if args.reps:
        tier["n_reps"] = args.reps
    if args.sizes:
        tier["sample_sizes"] = tuple(int(s) for s in args.sizes.split(","))
    if args.targets:
        spec["targets"] = tuple(args.targets.split(","))
    if args.estimators:
        spec["estimators"] = tuple(args.estimators.split(","))

    # Multivariate targets have no 1-D quadrature path for IAE/Hellinger.
    multivariate = args.suite == "multivariate"

    cfg = RunConfig(
        name=args.name or f"{args.suite}_{args.tier}",
        seed=args.seed,
        out_dir=args.out,
        n_jobs=args.jobs,
        dict_n_loc=args.n_loc,
        dict_n_scale=args.n_scale,
        compute_iae=not (args.no_iae or multivariate),
        compute_hellinger=not (args.no_iae or multivariate),
        **spec,
        **tier,
    )

    n_cells = len(cfg.targets) * len(cfg.sample_sizes)
    print(f"suite={args.suite} tier={args.tier}")
    print(f"  targets     : {len(cfg.targets)}  {list(cfg.targets)}")
    print(f"  sample sizes: {list(cfg.sample_sizes)}")
    print(f"  replicates  : {cfg.n_reps}")
    print(f"  estimators  : {len(cfg.estimators)}  {list(cfg.estimators)}")
    print(f"  total fits  : {n_cells * cfg.n_reps * len(cfg.estimators):,}")
    print(f"  output      : {Path(cfg.out_dir) / cfg.name}\n")

    df = run(cfg, force=args.force)
    if df.empty:
        print("no results produced")
        return 1

    ok = df["error"].isna() | (df["error"].astype(str).str.len() == 0)
    print(f"\ncompleted: {len(df):,} rows, {int((~ok).sum())} failures")
    if (~ok).any():
        print("\nfailure counts by estimator:")
        print(df.loc[~ok, "estimator"].value_counts().to_string())
    print(f"\nwritten to {Path(cfg.out_dir) / cfg.name / 'all_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
