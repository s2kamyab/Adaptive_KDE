"""Is the near-singular design my configuration's fault, or intrinsic?

Two candidate causes:
  (a) the measurement bank is too coarse to resolve the dictionary's finest
      scales -- fixable by widening the measurement scale range;
  (b) the dictionary packs 10 scales across two decades at every location, so
      neighbouring atoms are indistinguishable through ANY smoothing kernel
      -- fixable only by using fewer, better separated scales.

Vary both and report the conditioning of D and the achievable ISE.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import gaussops, metrics
from akde.densities import get_target
from akde.dictionary import DictionaryConfig
from akde.estimators.akde import AKDE, AKDEConfig

target = get_target(sys.argv[1] if len(sys.argv) > 1 else "f9")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
X = target.sample(N, np.random.default_rng(1))
print(f"target={target.label}  N={N}\n")

hdr = "{:>6s} {:>10s} {:>11s} {:>11s} {:>11s} {:>6s}"
print(hdr.format("dictq", "meas", "cond(D)", "best ISE", "CV ISE", "|S|"))
print("-" * 62)

for n_scale in (3, 5, 10):
    n_loc = 1500 // n_scale          # hold the dictionary size fixed at 1500
    for (lo, hi, ms) in ((0.25, 2.0, 3), (0.05, 4.0, 5), (0.05, 1.0, 5)):
        cfg = AKDEConfig(
            dictionary=DictionaryConfig(n_loc=n_loc, n_scale=n_scale,
                                        scale_lo_mult=0.05, scale_hi_mult=4.0),
            meas_mode="independent", meas_n_loc=100, meas_n_scale=ms,
            meas_scale_lo_mult=lo, meas_scale_hi_mult=hi,
        )
        est = AKDE(cfg)
        try:
            cands = est.candidates(X)
            fits = est.fit_path(X, cands, np.random.default_rng(7))
            ises = np.array([metrics.ise(f, target) for f in fits])
            i = int(np.argmin(ises))

            from akde.dictionary import build_dictionary, build_measurements
            atoms = build_dictionary(X, cfg.dictionary, rng=np.random.default_rng(7))
            meas = build_measurements(X, "independent", n_loc=100, n_scale=ms,
                                      scale_lo_mult=lo, scale_hi_mult=hi,
                                      rng=np.random.default_rng(7))
            D = gaussops.design_matrix(meas.means, meas.variances,
                                       atoms.means, atoms.variances)
            sv = np.linalg.svd(D, compute_uv=False)
            cond = sv[0] / max(sv[-1], 1e-300)

            cvfit = est.fit(X, np.random.default_rng(7))
            print(hdr.format(str(n_scale), f"{lo}-{hi}x{ms}", f"{cond:.2e}",
                             f"{ises[i]:.4e}", f"{metrics.ise(cvfit, target):.4e}",
                             str(fits[i].diagnostics["n_selected"])))
        except Exception as e:
            print(f"  q={n_scale} meas={lo}-{hi}x{ms} FAILED {type(e).__name__}: {e}")
    print()

