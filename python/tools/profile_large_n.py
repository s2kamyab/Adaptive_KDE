"""Time one full replicate at the largest sample size before committing to a long run."""
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde.runner import RunConfig, run_one_replicate

for N in (1000, 5000):
    cfg = RunConfig(name="timing", targets=("f9",), sample_sizes=(N,), n_reps=1,
                    estimators=("akde", "spades", "spades_simplex", "song", "kde_sj",
                                "kde_lscv", "abramson", "kde_botev", "gmm_em", "thin_kt"))
    t0 = perf_counter()
    rows = run_one_replicate(cfg, "f9", N, 0)
    total = perf_counter() - t0
    print(f"\nN={N}: one replicate, all 10 estimators = {total:.1f}s")
    for r in sorted(rows, key=lambda r: -(r.get("wall_time_s") or 0)):
        print("   {:16s} {:7.2f}s  ISE={:.3e}  err={}".format(
            r["estimator"], r.get("wall_time_s") or 0, r.get("ise", float("nan")),
            (r.get("error") or "")[:60]))
    sys.stdout.flush()

