"""Query latency and size of the saved confirmation / hr models (no refitting).

    python followup/improved/scripts/measure_cost.py

Same procedure as the earlier studies: 10,000 identical points, batches of
2,000, 2 warm-ups, 7 timing repeats, median within model.  For the
Marron-Wand targets the query points are the target's plot range (as in
measure_cost.py of the package); for hr the fixed [-5, 6] window.
Writes results/improved/cost/query_cost.csv.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from akde import gaussops  # noqa: E402
from akde.densities import get_target  # noqa: E402
from followup.improved.runner import RESULTS  # noqa: E402

N_QUERY, BATCH, WARMUP, REPEATS = 10000, 2000, 2, 7


def latency(w, m, v, xq):
    for _ in range(WARMUP):
        gaussops.mixture_pdf_diag(xq[:BATCH], w, m, v)
    t = []
    for _ in range(REPEATS):
        t0 = perf_counter()
        for s in range(0, N_QUERY, BATCH):
            gaussops.mixture_pdf_diag(xq[s:s + BATCH], w, m, v)
        t.append(perf_counter() - t0)
    return float(1e6 * np.median(t) / N_QUERY)


def main():
    out = RESULTS / "cost"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for stage in ("confirm", "hr"):
        p = RESULTS / stage / "all_rows.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df = df[df["error"].fillna("").astype(str) == ""]
        for r in df.itertuples():
            if stage == "hr":
                xq = np.linspace(-5.0, 6.0, N_QUERY)[:, None]
            else:
                lo, hi = get_target(r.target).plot_range(pad=4.0)
                xq = np.linspace(lo, hi, N_QUERY)[:, None]
            fp = RESULTS / stage / "fits" / ("%s_N%d_r%d_%s.npz" % (r.target, r.N, r.rep, r.method))
            with np.load(fp) as z:
                w, m, v = z["weights"], z["means"], z["variances"]
            rows.append({"stage": stage, "target": r.target, "N": int(r.N), "rep": int(r.rep),
                         "method": r.method, "n_components": int(w.size),
                         "minimal_storage_bytes": 24 * int(w.size),
                         "npz_file_bytes": int(fp.stat().st_size),
                         "query_us_per_point_median": latency(w, m, v, xq)})
        print("cost:", stage, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "query_cost.csv", index=False)
    (out / "cost_config.json").write_text(json.dumps({
        "n_query_points": N_QUERY, "batch": BATCH, "warmup": WARMUP, "timing_repeats": REPEATS,
        "per_model": "median over timing repeats; repeats are not replications",
        "platform": platform.platform(), "processor": platform.processor()}, indent=2), encoding="utf-8")
    print(d.pivot_table(index=["stage", "method"], columns="N", values="query_us_per_point_median",
                        aggfunc="mean").to_string(float_format=lambda v: "%.3f" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
