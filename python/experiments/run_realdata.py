"""Real-data application: held-out density estimation on real univariate marginals.

On real data there is no true density, so accuracy is measured by the same
held-out integrated-square (J) score used for tuning -- evaluated on a *test*
split never seen by fitting or tuning -- and, for validated probability
densities only, by held-out negative log likelihood.

    J(f) = int f(x)^2 dx - (2 / n_test) sum_j f(x_j^test)

This is an unbiased estimate of ISE up to the unknown constant ``int f0^2``,
which is the same for every estimator on a given split.  Differences in J are
therefore differences in ISE, and paired differences across splits are exactly
comparable.

Replication is by repeated stratified train/test splits.  The bootstrap
resamples splits, keeping estimators paired within a split.

Datasets are marginals of scikit-learn's bundled datasets (no download, fully
reproducible).  They were chosen for shapes that matter here -- strong skew,
bimodality, heavy right tails, and a near-discrete variable -- rather than for
being easy.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter

# Single-threaded BLAS for reproducibility; see run_benchmark.py for why.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from akde import gaussops, registry
from akde.metrics import held_out_nll, integrate_1d, j_score
from akde.model import FittedDensity
from akde.runner import name_key

#: (dataset, column, human label). Loaded lazily so a missing optional dataset
#: does not break the whole script.
DATASETS = [
    ("breast_cancer", "mean area", "Breast cancer: mean area (skewed, heavy right tail)"),
    ("breast_cancer", "mean texture", "Breast cancer: mean texture (near-Gaussian)"),
    ("breast_cancer", "worst concave points", "Breast cancer: worst concave points (bimodal)"),
    ("wine", "proline", "Wine: proline (skewed, multimodal)"),
    ("wine", "flavanoids", "Wine: flavanoids (multimodal)"),
    ("diabetes", "bmi", "Diabetes: BMI (mildly skewed)"),
    ("california", "MedInc", "California housing: median income (skewed, capped)"),
    ("california", "HouseAge", "California housing: house age (near-discrete, capped)"),
]


def load_column(name: str, column: str) -> np.ndarray | None:
    from sklearn import datasets as skd

    try:
        if name == "breast_cancer":
            d = skd.load_breast_cancer(as_frame=True)
        elif name == "wine":
            d = skd.load_wine(as_frame=True)
        elif name == "diabetes":
            d = skd.load_diabetes(as_frame=True, scaled=False)
        elif name == "california":
            d = skd.fetch_california_housing(as_frame=True)
        else:
            return None
    except Exception:
        return None
    frame = d.frame if hasattr(d, "frame") else d.data
    if column not in frame.columns:
        return None
    x = frame[column].to_numpy(dtype=float)
    return x[np.isfinite(x)]


def standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre and scale using TRAINING statistics only.

    Test data must not influence the preprocessing; the Jacobian of the
    transform is returned so scores can be reported on the original scale.
    """
    mu = float(train.mean())
    sd = float(train.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, sd


def evaluate(fit: FittedDensity, x_test: np.ndarray, domain, sd: float) -> dict:
    """J-score and NLL on the standardised scale, plus the original-scale J.

    Under x = sd * z + mu the density transforms as f_X(x) = f_Z(z) / sd, so
    int f_X^2 dx = (int f_Z^2 dz) / sd and the J-score scales by 1/sd.
    """
    j_std = j_score(fit, x_test[:, None], domain=domain)
    nll_std, floored = held_out_nll(fit, x_test[:, None])
    return {
        "j_score_std": j_std,
        "j_score_orig": j_std / sd,
        "nll_std": nll_std,
        "nll_orig": nll_std + float(np.log(sd)) if np.isfinite(nll_std) else np.nan,
        "nll_floored": floored,
        "is_valid_density": fit.is_valid_density(),
        "n_active": fit.n_active(),
        "total_mass": fit.total_mass(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=30, help="random train/test splits")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--max-n", type=int, default=2000,
                    help="subsample large datasets to this training size")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", default="results/realdata")
    ap.add_argument("--estimators", default=",".join(registry.ESSENTIAL))
    ap.add_argument("--n-loc", type=int, default=150)
    ap.add_argument("--n-scale", type=int, default=10)
    args = ap.parse_args()

    estimators = [e for e in args.estimators.split(",") if e]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds, col, label in DATASETS:
        x_all = load_column(ds, col)
        if x_all is None:
            print(f"skip {ds}:{col} (not available)")
            continue
        print(f"\n{label}  (n={x_all.size})")

        for split in range(args.splits):
            rng = np.random.default_rng([args.seed, name_key(ds + col), split])
            n = x_all.size
            perm = rng.permutation(n)
            n_test = max(20, int(round(args.test_frac * n)))
            test_idx, train_idx = perm[:n_test], perm[n_test:]
            if train_idx.size > args.max_n:
                train_idx = train_idx[: args.max_n]

            tr_raw, te_raw = x_all[train_idx], x_all[test_idx]
            tr, te, sd = standardise(tr_raw, te_raw)
            span = float(max(tr.max(), te.max()) - min(tr.min(), te.min()))
            domain = (float(min(tr.min(), te.min()) - span),
                      float(max(tr.max(), te.max()) + span))

            for est_name in estimators:
                rec = {"dataset": ds, "column": col, "label": label,
                       "split": split, "estimator": est_name,
                       "n_train": int(tr.size), "n_test": int(te.size)}
                try:
                    est = registry.build(est_name, n_loc=args.n_loc, n_scale=args.n_scale)
                    t0 = perf_counter()
                    fit = est.fit(tr[:, None], np.random.default_rng([args.seed, split, 7]))
                    rec["wall_time_s"] = perf_counter() - t0
                    rec.update(evaluate(fit, te, domain, sd))
                    rec["tune_time_s"] = fit.diagnostics.get("tune_time_s", np.nan)
                    rec["fit_time_s"] = fit.diagnostics.get("fit_time_s", np.nan)
                    rec["error"] = ""
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["j_score_std"] = np.nan
                    traceback.print_exc(limit=2)
                rows.append(rec)
            if (split + 1) % 10 == 0:
                print(f"   {split + 1}/{args.splits} splits done")

    df = pd.DataFrame(rows)
    path = out_dir / "realdata_results.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {len(df):,} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
