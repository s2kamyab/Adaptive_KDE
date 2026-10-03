"""Task 6.5 - serialized model size and density-evaluation latency.

Uses the SAVED Stage-C fitted models; nothing is refitted. Every method is
queried with identical points, identical batching, identical warmup and the
same number of timing repeats on the same hardware in the same process.

Timing repeats are TIMING REPEATS. They are not independent statistical
replications and are never aggregated as if they were: the statistical unit
stays the fitted model (one per replicate), and repeats only reduce timer
noise for a fixed model.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from akde import gaussops                          # noqa: E402
from akde.densities import get_target              # noqa: E402
from followup.verify import check_config as CFG    # noqa: E402

from followup.paths import RAW_ROOT, DIAG_ROOT   # noqa: E402
RES = RAW_ROOT
OUT = DIAG_ROOT

METHODS = ("v2_akde", "finalist", "spades_simplex", "kde_sj", "abramson", "gmm_em")
N_QUERY = 10000          # identical query grid for every method
WARMUP = 2
REPEATS = 7              # timing repeats, NOT statistical replications
BATCH = 2000


def load_fit(target, N, rep, variant):
    p = RES / "stage_c" / "fits" / ("%s_N%d_r%d_%s.npz" % (target, N, rep, variant))
    if not p.exists():
        return None, None
    z = np.load(p)
    return {"weights": z["weights"], "means": z["means"],
            "variances": z["variances"]}, p


def query_latency(fit, xq, batch=BATCH, warmup=WARMUP, repeats=REPEATS):
    w, m, v = fit["weights"], fit["means"], fit["variances"]
    for _ in range(warmup):
        gaussops.mixture_pdf_diag(xq[:batch], w, m, v)
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        for s in range(0, xq.shape[0], batch):
            gaussops.mixture_pdf_diag(xq[s:s + batch], w, m, v)
        times.append(perf_counter() - t0)
    times = np.array(times)
    return {"median_s": float(np.median(times)), "min_s": float(times.min()),
            "max_s": float(times.max()),
            "us_per_query_median": float(1e6 * np.median(times) / xq.shape[0])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    targets = list(CFG.TARGETS)
    sizes = (200, 1000, 5000)
    reps = (0, 1)
    for target in targets:
        tgt = get_target(target)
        lo, hi = tgt.plot_range(pad=4.0)
        xq = np.linspace(lo, hi, N_QUERY)[:, None]
        for N in sizes:
            for rep in reps:
                for meth in METHODS:
                    fit, p = load_fit(target, N, rep, meth)
                    if fit is None:
                        rows.append({"target": target, "N": N, "rep": rep,
                                     "method": meth, "status": "MISSING"})
                        continue
                    K = int(fit["weights"].size)
                    d = int(fit["means"].shape[1])
                    # canonical parameter count: weight + mean(d) + variance(d)
                    params = K * (1 + 2 * d)
                    lat = query_latency(fit, xq)
                    rows.append({
                        "target": target, "N": N, "rep": rep, "method": meth,
                        "status": "ok",
                        "n_components": K, "n_parameters_float64": params,
                        "canonical_model_bytes": params * 8,
                        "npz_file_bytes": int(p.stat().st_size),
                        "n_query_points": N_QUERY, "batch": BATCH,
                        "timing_repeats": REPEATS, "warmup_calls": WARMUP,
                        **{("query_" + k): v for k, v in lat.items()},
                    })
        print("  cost: %s done" % target, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "model_size_and_query_cost.csv", index=False)
    ok = df[df.status == "ok"]
    pd.set_option("display.width", 220)
    print("\n== canonical serialized model size (float64 parameters) ==")
    print(ok.pivot_table(index="method", columns="N",
                         values="canonical_model_bytes", aggfunc="median")
          .to_string(float_format=lambda v: "%.0f" % v))
    print("\n== density-evaluation latency, microseconds per query point (median "
          "over %d timing repeats, then median over models) ==" % REPEATS)
    print(ok.pivot_table(index="method", columns="N",
                         values="query_us_per_query_median", aggfunc="median")
          .to_string(float_format=lambda v: "%.4f" % v))
    print("\n== number of mixture components ==")
    print(ok.pivot_table(index="method", columns="N", values="n_components",
                         aggfunc="median").to_string(float_format=lambda v: "%.0f" % v))

    with open(OUT / "cost_measurement_config.json", "w", encoding="utf-8") as fh:
        json.dump({
            "source": "saved Stage-C fitted models; nothing refitted",
            "n_query_points": N_QUERY, "batch": BATCH, "warmup_calls": WARMUP,
            "timing_repeats": REPEATS,
            "timing_repeats_note": (
                "REPEATS are timing repeats for one fixed fitted model. They "
                "reduce timer noise only. The statistical unit remains the "
                "fitted model (one per replicate); repeats are never treated as "
                "independent statistical replications."),
            "identical_queries": True,
            "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version.split()[0],
            "blas_threads": "1 (pinned in followup/__init__.py before numpy import)",
            "fit_cost_reported_separately": (
                "cold end-to-end fit time is the wall_time_s column of "
                "stage_c/all_rows.csv and is reported in its own table"),
        }, fh, indent=2)
    print("\nwrote", OUT / "model_size_and_query_cost.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
