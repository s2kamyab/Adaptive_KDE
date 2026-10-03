"""Compare what cross-validation picks against what actually minimises ISE."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akde import metrics, registry
from akde.densities import get_target
from akde.estimators.base import kfold_indices

target_name = sys.argv[1] if len(sys.argv) > 1 else "f9"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
method = sys.argv[3] if len(sys.argv) > 3 else "akde"

target = get_target(target_name)
X = target.sample(N, np.random.default_rng(1))

est = registry.build(method, n_loc=150, n_scale=10)
cands = est.candidates(X)
rng = np.random.default_rng(7)

# cross-validated J, exactly as base.fit does it
folds = kfold_indices(X.shape[0], 5, np.random.default_rng(7))
scores = np.full((len(folds), len(cands)), np.nan)
for fi, (tr, va) in enumerate(folds):
    fits = est.fit_path(X[tr], cands, np.random.default_rng(7))
    for ci, f in enumerate(fits):
        scores[fi, ci] = metrics.j_score(f, X[va])
cv = np.nanmean(scores, axis=0)

# what each candidate actually achieves on the full data
full = est.fit_path(X, cands, np.random.default_rng(7))
ise = np.array([metrics.ise(f, target) for f in full])
supp = np.array([f.diagnostics["n_selected"] for f in full])

print("{:>8s} {:>10s} {:>6s} {:>13s} {:>12s}".format("rho", "lam_ratio", "supp", "CV J", "true ISE"))
for c, s, cvv, iv in zip(cands, supp, cv, ise):
    mark = ""
    if cvv == np.nanmin(cv):
        mark += "  <-- CV picks"
    if iv == ise.min():
        mark += "  <-- best ISE"
    print("{:8.0e} {:10.2e} {:6d} {:13.5f} {:12.4e}{}".format(
        c["rho"], c["lam_ratio"], s, cvv, iv, mark))

i_cv, i_best = int(np.nanargmin(cv)), int(np.argmin(ise))
print("\nCV chose ISE = {:.4e};  best available = {:.4e};  ratio {:.1f}x".format(
    ise[i_cv], ise[i_best], ise[i_cv] / ise[i_best]))

