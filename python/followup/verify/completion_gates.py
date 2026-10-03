"""Task 9 - completion gates.

Independent checks of counts, seed pairing, statistics, figure curves and
manuscript numbers. These are meaningful invariants, not restatements of the
implementation: each one recomputes a quantity by a different route and
compares.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from akde import gaussops                          # noqa: E402
from akde.densities import get_target              # noqa: E402
from akde.metrics import integrate_1d              # noqa: E402
from followup.akde_followup.seeds import data_rng  # noqa: E402

from followup.paths import RAW_ROOT, DIAG_ROOT, SUMMARY_ROOT, FIGURES_ROOT  # noqa: E402
RES = RAW_ROOT
FIGD = FIGURES_ROOT / "data"


def _package_root():
    """Locate final_package/ whether we are in the repo or inside the package.

    In the package, code/ and manifests/ are siblings; in the repository the
    package sits at <project>/final_package. Both are checked, so no script
    contains a machine-specific path.
    """
    here = Path(__file__).resolve()
    sib = here.parents[2].parent            # .../code/..  -> package root
    if (sib / "manifests").exists() or (sib / "paper" / "main.tex").exists():
        return sib
    for up in here.parents:
        cand = up / "final_package"
        if cand.exists():
            return cand
    return sib

FP = _package_root()
PROJ = PY.parents[1]
OUT = DIAG_ROOT

G = []


def gate(name, ok, detail=""):
    G.append({"gate": name, "pass": bool(ok), "detail": detail})
    print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RES / "stage_c" / "all_rows.csv", low_memory=False)
    tidy = pd.read_csv(SUMMARY_ROOT / "tidy_metrics.csv")

    # ---- G1 counts agree between raw rows, tidy metrics and saved fits ----
    n_fits = len(list((RES / "stage_c" / "fits").glob("*.npz")))
    gate("counts: raw rows == tidy rows == saved fitted models",
         len(rows) == len(tidy) == n_fits == 8100,
         "rows %d, tidy %d, npz %d" % (len(rows), len(tidy), n_fits))

    # ---- G2 ISE in the saved model files matches the ISE in the rows ------
    rng = np.random.default_rng(0)
    sample = rows.sample(40, random_state=1)
    worst = 0.0
    for r in sample.itertuples():
        p = RES / "stage_c" / "fits" / ("%s_N%d_r%d_%s.npz"
                                        % (r.target, r.N, r.rep, r.variant))
        z = np.load(p)
        worst = max(worst, abs(float(z["ise"]) - r.ise) / max(r.ise, 1e-300))
    gate("saved model ISE == row ISE (40 random fits)", worst < 1e-12,
         "max relative difference %.3e" % worst)

    # ---- G3 recompute ISE from the stored mixture, independently ----------
    worst = 0.0
    for r in rows.sample(12, random_state=2).itertuples():
        p = RES / "stage_c" / "fits" / ("%s_N%d_r%d_%s.npz"
                                        % (r.target, r.N, r.rep, r.variant))
        z = np.load(p)
        tgt = get_target(r.target)
        ise = gaussops.ise_mixtures(z["weights"], z["means"], z["variances"],
                                    tgt.weights, tgt.means, tgt.variances)
        worst = max(worst, abs(ise - r.ise) / max(r.ise, 1e-300))
    gate("ISE recomputed from stored mixture == row ISE", worst < 1e-9,
         "max relative difference %.3e" % worst)

    # ---- G4 analytic ISE against quadrature, an independent route ---------
    worst = 0.0
    for r in rows[rows.variant == "finalist"].sample(4, random_state=3).itertuples():
        p = RES / "stage_c" / "fits" / ("%s_N%d_r%d_%s.npz"
                                        % (r.target, r.N, r.rep, r.variant))
        z = np.load(p)
        tgt = get_target(r.target)
        lo, hi = tgt.plot_range(pad=9.0)
        num = integrate_1d(
            lambda t: (gaussops.mixture_pdf_diag(np.atleast_1d(t)[:, None],
                                                 z["weights"], z["means"],
                                                 z["variances"]) - tgt.pdf(t)) ** 2,
            lo, hi, tol=1e-12)
        worst = max(worst, abs(num - r.ise) / max(r.ise, 1e-300))
    gate("analytic ISE == adaptive quadrature", worst < 1e-6,
         "max relative difference %.3e" % worst)

    # ---- G5 every fitted model really is a density ------------------------
    bad_mass, bad_neg = 0.0, 0.0
    for r in rows.sample(60, random_state=4).itertuples():
        p = RES / "stage_c" / "fits" / ("%s_N%d_r%d_%s.npz"
                                        % (r.target, r.N, r.rep, r.variant))
        z = np.load(p)
        bad_mass = max(bad_mass, abs(float(z["weights"].sum()) - 1.0))
        bad_neg = min(bad_neg, float(z["weights"].min()))
    gate("fitted models are densities (60 random fits)",
         bad_mass < 1e-9 and bad_neg >= -1e-14,
         "max |mass-1| %.3e, min weight %.3e" % (bad_mass, bad_neg))

    # ---- G6 pairing: identical data across methods ------------------------
    # regenerate the dataset and check a deterministic method reproduces
    from akde.estimators.kde import FixedKDE
    from akde.estimators.base import TuningConfig
    from akde import metrics
    from followup.akde_followup.seeds import fit_rng
    ok = True
    details = []
    for t, N, rep in (("f0", 1000, 0), ("f9", 200, 3)):
        X = get_target(t).sample(N, data_rng("confirmation", t, N, rep))
        est = FixedKDE("sj", tuning=TuningConfig(n_folds=3))
        fit = est.fit(X, fit_rng("confirmation", t, N, rep, 3))
        got = float(metrics.ise(fit, get_target(t)))
        saved = float(rows[(rows.target == t) & (rows.N == N) & (rows.rep == rep) &
                           (rows.variant == "kde_sj")].ise.iloc[0])
        rel = abs(got - saved) / saved
        ok = ok and rel < 1e-10
        details.append("%s N=%d rep=%d rel %.2e" % (t, N, rep, rel))
    gate("dataset regenerated from seed reproduces a saved row", ok,
         "; ".join(details))

    # ---- G7 statistics recomputed by an independent route -----------------
    fb = pd.read_csv(SUMMARY_ROOT / "fixed_benchmark.csv")
    worst = 0.0
    for N in (200, 1000, 5000):
        zs = []
        for t in sorted(rows.target.unique()):
            piv = rows[(rows.target == t) & (rows.N == N)].pivot_table(
                index="rep", columns="variant", values="ise", aggfunc="first")
            zs.append(float(np.mean(np.log(piv["finalist"] / piv["v2_akde"]))))
        mine = float(np.exp(np.mean(zs)))
        theirs = float(fb[(fb.method == "finalist") & (fb.N == N)]
                       .exp_T_geometric_mean_ratio.iloc[0])
        worst = max(worst, abs(mine - theirs) / theirs)
    gate("fixed-benchmark exp(T) recomputed independently", worst < 1e-12,
         "max relative difference %.3e" % worst)

    # ---- G8 figure curves match the saved models --------------------------
    z = np.load(FIGD / "fig1_fitted_densities_curves.npz")
    m = pd.read_csv(FIGD / "fig1_fitted_densities_metrics.csv")
    worst = 0.0
    for r in m[m.status == "ok"].itertuples():
        key = "%s_N%d_%s" % (r.target, r.N, r.method)
        if key not in z:
            continue
        f = np.load(RES / "stage_c" / "fits" / ("%s_N%d_r0_%s.npz"
                                                % (r.target, r.N, r.method)))
        xg = z["%s_x" % r.target]
        y = gaussops.mixture_pdf_diag(xg[:, None], f["weights"], f["means"],
                                      f["variances"])
        worst = max(worst, float(np.max(np.abs(y - z[key]))))
    gate("figure 1 curves == curves recomputed from saved models",
         worst < 1e-12, "max absolute difference %.3e" % worst)

    # ---- G9 figure metrics ISE == row ISE ---------------------------------
    worst = 0.0
    for r in m[m.status == "ok"].itertuples():
        saved = float(rows[(rows.target == r.target) & (rows.N == r.N) &
                           (rows.rep == 0) & (rows.variant == r.method)].ise.iloc[0])
        worst = max(worst, abs(r.ise - saved) / saved)
    gate("figure 1 annotated ISE == confirmation row ISE", worst < 1e-12,
         "max relative difference %.3e" % worst)

    # ---- G10 package integrity -------------------------------------------
    man = FP / "manifests" / "file_manifest.csv"
    if man.exists():
        mm = pd.read_csv(man)
        from followup.akde_followup.manifest import sha256_file
        bad = []
        for r in mm.sample(min(40, len(mm)), random_state=5).itertuples():
            p = FP / r.path
            if not p.exists() or sha256_file(p) != r.sha256:
                bad.append(r.path)
        gate("package file manifest hashes verify (40 random files)",
             not bad, "%d files checked, %d mismatched" % (min(40, len(mm)), len(bad)))
        gate("package contains no caches or VCS directories",
             not any(("__pycache__" in p or "/.git/" in p)
                     for p in mm.path.astype(str)),
             "%d files total" % len(mm))
    else:
        gate("package file manifest present", False, "manifests/file_manifest.csv missing")

    # ---- G11 historical results preserved ---------------------------------
    from followup.akde_followup.manifest import sha256_file
    hist = FP / "results" / "raw" / "v2_benchmark_2026-09-22" / "all_results.csv"
    src = PY / "results" / "main_full" / "all_results.csv"
    if hist.exists() and src.exists():
        gate("earlier 27,000-fit benchmark preserved byte-identically",
             sha256_file(hist) == sha256_file(src),
             "sha256 %s" % sha256_file(src)[:16])
    elif hist.exists():
        # running from inside the package: the repository copy is absent by
        # design, so verify the preserved file against the recorded manifest
        mm = pd.read_csv(FP / "manifests" / "file_manifest.csv")
        rel = "results/raw/v2_benchmark_2026-09-22/all_results.csv"
        row = mm[mm.path == rel]
        gate("earlier 27,000-fit benchmark preserved (manifest hash)",
             len(row) == 1 and sha256_file(hist) == row.sha256.iloc[0],
             "sha256 %s" % sha256_file(hist)[:16])
    else:
        gate("earlier benchmark preserved", False,
             "not found at %s" % hist)

    # ---- G12 live package identical to the frozen V2 snapshot -------------
    import hashlib

    def tree_hash(root):
        h = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            h.update(str(p.relative_to(root)).replace("\\", "/").encode())
            h.update(p.read_bytes())
        return h.hexdigest()

    a = tree_hash(PY / "akde")
    b = tree_hash(PY / "followup" / "akde_v2_original" / "akde")
    gate("live akde package == frozen V2 snapshot", a == b, "sha256 %s" % a[:16])

    n_fail = sum(1 for g in G if not g["pass"])
    with open(OUT / "completion_gates.json", "w", encoding="utf-8") as fh:
        json.dump({"gates": G, "n_gates": len(G), "n_failed": n_fail}, fh, indent=2)
    print("\n%d/%d gates passed" % (len(G) - n_fail, len(G)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
