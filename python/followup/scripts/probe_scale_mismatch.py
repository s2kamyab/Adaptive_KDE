"""Where does V2 AKDE put its mass, relative to what the measurement bank sees?

The dictionary's finest atoms have sd 0.05 x h_Silverman; the default
measurement bank's finest probe has sd 0.25 x h.  An atom five times narrower
than the narrowest probe is nearly invisible to the moment criterion while
contributing enormously to int fhat^2 and hence to ISE.  This probe measures
whether that gap is actually exploited by the fitted solution.

Diagnostic only; it consults the truth for scoring, never for fitting.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akde import gaussops, metrics                          # noqa: E402
from akde.bandwidths import silverman                       # noqa: E402
from akde.densities import get_target                       # noqa: E402
from akde.registry import build                             # noqa: E402

OUT = ROOT / "followup" / "results" / "audit_tables"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for tname, N in [("f0", 500), ("f0", 5000), ("f2", 5000),
                 ("f5", 500), ("f5", 5000), ("f9", 500), ("f9", 5000)]:
    tgt = get_target(tname)
    X = tgt.sample(N, np.random.default_rng(1))
    h = float(silverman(X[:, 0]))
    est = build("akde", n_loc=150, n_scale=10)
    est.tuning.n_folds = 3           # diagnostic only; V2 used 5
    fit = est.fit(X, np.random.default_rng(3))
    sd = np.sqrt(fit.variances[:, 0])
    w = fit.weights
    sel = fit.diagnostics["selected"]
    rec = {
        "target": tname, "N": N, "h_silverman": h,
        "ise": float(metrics.ise(fit, tgt)),
        "M_selected": int(np.prod(sel["dict"])), "rho": sel["rho"],
        "lam_ratio": sel["lam_ratio"], "support": int(w.size),
        "atom_sd_min_over_h": float(sd.min() / h),
        "atom_sd_wmean_over_h": float(w @ sd / h),
        "mass_below_0.25h": float(w[sd < 0.25 * h].sum()),
        "mass_below_0.10h": float(w[sd < 0.10 * h].sum()),
        "int_fhat2": float(gaussops.self_overlap_quad(w, fit.means, fit.variances)),
        "int_f0_2": float(tgt.self_overlap()),
    }
    rec["int_fhat2_over_int_f0_2"] = rec["int_fhat2"] / rec["int_f0_2"]
    rows.append(rec)
    print("%s N=%-5d ISE %.5g  M=%d  supp=%d  sd_wmean/h=%.3f  "
          "mass<0.25h=%.3f  int fhat^2 / int f0^2 = %.3f"
          % (tname, N, rec["ise"], rec["M_selected"], rec["support"],
             rec["atom_sd_wmean_over_h"], rec["mass_below_0.25h"],
             rec["int_fhat2_over_int_f0_2"]), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "scale_mismatch_probe.csv", index=False)
print("\nwrote", OUT / "scale_mismatch_probe.csv")
