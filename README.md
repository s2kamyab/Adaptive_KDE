# Adaptive Kernel Density Estimation — experimental framework

Companion code for *Adaptive Kernel Density Estimation using the Generalized Method of
Moments and Compressive Kernel Selection* (Kamyab, Nasihatkon & Boostani), manuscript
`../AKDE_paper_draft/Adaptive_KDE_GMM_revised_nonnegative_constraint.tex`.

This directory contains two things:

| Path | What it is |
|---|---|
| `Scripts/` | The original MATLAB prototype. Kept for reference. Superseded — see [Differences from the MATLAB prototype](#differences-from-the-matlab-prototype). |
| `python/` | The experimental framework used for the revised manuscript. This is what you run. |

The framework was designed against `Experimental_comparison_recommendations.md`.
Every protocol requirement in that document is either implemented, or listed under
[What is deliberately not included](#what-is-deliberately-not-included) with a reason.

> **Security note.** An earlier revision of this file contained a pasted Anthropic API
> key. It has been removed. If that key was real, revoke it at
> <https://console.anthropic.com/settings/keys> — anything committed to a file should be
> treated as compromised. Do not paste credentials into repository files.

---

## 1. Install

Python 3.10+. From this directory:

```bash
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r python/requirements.txt    # Windows
# source ../.venv/bin/activate && pip install -r python/requirements.txt   # Linux/macOS
```

`numba` is optional but **strongly recommended**: it JIT-compiles the coordinate-descent
sweeps and is worth roughly two orders of magnitude on the dictionary methods. Without it
everything still runs, just far more slowly.

Verify the numerical core before anything else:

```bash
cd python
../../.venv/Scripts/python tests/test_core.py
```

This checks the closed-form Gaussian overlap against quadrature, the Marron–Wand
parameters against their published moments, the analytic ISE against numerical
integration, every solver against its KKT conditions, and the bandwidth selectors against
the known AMISE optimum. All 16 checks should pass.

A fast end-to-end sanity check of every estimator on one dataset:

```bash
../../.venv/Scripts/python experiments/smoke.py --target f9 --n 200
```

---

## 2. Run the bootstrapped experiments

### 2.0 Ordered command list — run these in this order

Everything is written to `results/<run-name>/`, sharded by `(density, N)`. **Runs are
resumable**: a shard already on disk is skipped, so you can stop with `Ctrl-C` and re-run
the same command to continue. Add `--force` only if you want to recompute.

Open a terminal in `Adaptive_KDE/python` and set `PY` once:

```powershell
# Windows PowerShell
cd C:\dev\Adaptive_KDE_project\Adaptive_KDE\python
$PY = "C:\dev\Adaptive_KDE_project\.venv\Scripts\python.exe"
```

```bash
# Linux / macOS
cd Adaptive_KDE/python
PY=../../.venv/bin/python
```

> **Choose `--jobs` by memory, not by core count.** Kernel thinning needs an `N x N`
> kernel matrix, so at `N=5000` each worker holds ~250 MB. On a 32 GB machine use
> `--jobs 10` for anything that includes `N=5000`, and up to `--jobs 20` otherwise.
> Too many workers will get the run killed by the OS.

| # | Step | Command | Approx. time (22 cores) |
|---|---|---|---|
| 0 | Verify the numerics | `& $PY tests/test_core.py` | ~2 min |
| 1 | Check the pipeline | `& $PY experiments/run_benchmark.py --suite main --tier smoke --jobs 10 --name main_smoke` | ~10 min |
| 2 | **Main benchmark** | `& $PY experiments/run_benchmark.py --suite main --tier full --jobs 10 --name main_full` | ~6–8 h |
| 3 | Ablations | `& $PY experiments/run_benchmark.py --suite ablation --tier full --jobs 10 --name ablation_full` | ~6–8 h |
| 4 | Stress tests | `& $PY experiments/run_benchmark.py --suite stress --tier full --jobs 10 --name stress_full` | ~1.5 h |
| 5 | Multivariate | `& $PY experiments/run_benchmark.py --suite multivariate --tier full --jobs 10 --name multivariate_full` | ~1.5 h |
| 6 | Real data | `& $PY experiments/run_realdata.py --splits 30 --out results/realdata` | ~1 h |
| 7 | Diagnostics | `& $PY experiments/run_diagnostics.py --targets f5,f9,f13 --out results/diagnostics` | ~30 min |
| 8 | **Bootstrap analysis** | `& $PY experiments/analyze.py --run results/main_full --boot 10000` | ~5 min |
| 9 | Analysis, ablations | `& $PY experiments/analyze.py --run results/ablation_full --reference akde_ridge --boot 10000` | ~5 min |
| 10 | Figures | `& $PY experiments/make_figures.py --run results/main_full --fitted` | ~15 min |

If you would rather start smaller, replace `--tier full` with `--tier pilot` in steps 2–5
(3 sample sizes, 10 replicates instead of 6 and 30) — about 1/8 the time, and enough to
see whether the conclusions hold before committing to the long run.

**One command for everything** (runs steps 2–10 in order, skipping what already exists):

```powershell
& $PY run_all.py --tier full --jobs 10
```

Results you will want to look at afterwards:

- `results/main_full/analysis/summary.md` — readable summary
- `results/main_full/analysis/paired_vs_akde.csv` — the bootstrap intervals
- `results/main_full/analysis/tables.tex` — tables to `\input` into the manuscript
- `results/main_full/figures/` — plots

### 2.1 Tiers

| Tier | Sample sizes | Replicates | Purpose |
|---|---|---|---|
| `smoke` | 100, 500 | 3 | pipeline check |
| `pilot` | 100, 500, 1000 | 10 | configuration check |
| `full` | 50, 100, 200, 500, 1000, 5000 | 30 | the protocol's setting |

### 2.2 Suites

| Suite | Contents |
|---|---|
| `main` | 15 Marron–Wand densities, the essential comparison set + kernel thinning |
| `ablation` | weighting scheme, single vs multiscale, sparsity on/off, refit on/off, measurement design |
| `stress` | outlier contamination and rare narrow modes, reported separately (never averaged in) |
| `multivariate` | 2-D and 5-D full-rank targets with unequal scales and overlapping modes |

### 2.3 Real-data application

No true density exists, so accuracy is the held-out integrated-square (J) score on a test
split, plus held-out NLL for validated densities only. Replication is by repeated
train/test splits; the bootstrap resamples splits with estimators kept paired.

Uses marginals of scikit-learn's bundled datasets (breast cancer, wine, diabetes,
California housing) chosen for skew, bimodality, heavy tails and near-discreteness.

### 2.4 What the analysis step produces

Written to `results/<run>/analysis/`:

| File | Contents |
|---|---|
| `accuracy_ise.csv` | mean and median ISE per (density, N, estimator) with BCa intervals |
| `paired_vs_akde.csv` | paired differences and error ratios, Holm-adjusted across densities |
| `paired_median_vs_akde.csv` | the same on medians, for catastrophic-failure robustness |
| `winloss.csv` | significant wins / losses / ties per baseline |
| `cost.csv` | fit time, tuning time, wall time, active representation size |
| `diagnostics.csv` | density validity, negative mass, solver KKT, covariance rank, fallback rate |
| `tables.tex` | booktabs tables ready to `\input` into the manuscript |
| `summary.md` | readable summary |

---

## 3. How the bootstrap works

The unit of replication is the **independently generated dataset**, and that is what gets
resampled. Multiple optimiser restarts on one dataset are not independent replications and
are never counted as such.

Within a replicate, **every estimator sees the identical data realisation**. The reported
quantity is therefore the *paired* difference, whose bootstrap distribution is far tighter
than the difference of two marginal distributions. Resampling is by replicate index, so
pairing survives resampling exactly.

- **Intervals**: BCa (bias-corrected and accelerated), 10 000 resamples by default. BCa
  matters here because ISE distributions are strongly right-skewed.
- **Both mean and median** are reported. The mean is the conventional MISE summary; the
  median is what survives a handful of catastrophic fits. The historical `f5` entry
  (AKDE 3.1 vs SPADES 0.24) is exactly the case where these diverge.
- **Error ratios**: the paired geometric mean of `ISE(reference) / ISE(baseline)`, which is
  scale-free and therefore comparable across densities whose errors span decades.
- **Multiplicity**: Holm–Bonferroni across the 15-density family, within each
  `(baseline, N)` group.

---

## 4. Protocol

| Requirement | How it is enforced |
|---|---|
| No true density in fitting or tuning | Structural: `Estimator.fit(X)` receives only the sample. The truth is reachable only by the scoring code, after fitting. |
| Covariance from training data only | `C_hat` is computed from the same `x_n` the estimator is given (manuscript eq. `covariance-regularized`). The MATLAB prototype drew 50 *fresh* datasets from the true density — that is privileged information and is not reproduced. |
| Tuning on training data only | K-fold CV of the held-out integrated-square score `J(f) = ∫f² − (2/n)Σf(x_val)`, inside `fit`. Cross-validation rather than one large holdout, because at N=50 a 30% holdout discards most of the signal. |
| Comparable regularisation across methods | `λ` is parameterised as a fraction of the `λ` at which the solution collapses to zero, computed per method and per fold. A shared numeric constant (the historical `γ=7`) does **not** give two differently scaled objectives equivalent regularisation. |
| ISE, not grid MSE | Analytic from Gaussian overlaps for every mixture-valued estimator (which is all of them except EM in d>1). Quadrature elsewhere records its domain and convergence. |
| Same data, same splits for all estimators | Seeds derive deterministically from `(seed, target, N, rep)`; data and estimator randomness use separate streams. |
| Nonzero-coefficient counts need a tolerance | `n_active(tol=1e-8, relative=True)`, tolerance recorded in the results. |
| Report failures | Solver KKT residuals, convergence flags, zero-solution fallback rate, support-cap triggers, covariance rank and condition number are all columns in the output. |

---

## 5. What is compared

Following the recommended prioritisation — the five essential rows plus kernel thinning.

| Key | Method | Note |
|---|---|---|
| `akde` | **Proposed**: non-negative penalised selection → simplex refit | ridge-weighted; `akde_identity/diagonal/trunc_pinv` are the weighting ablations |
| `spades` | SPADES (Bunea et al. 2010), original signed objective | signed and unnormalised, scored as such — **not** divided by its coefficient sum |
| `spades_simplex` | Density-valid SPADES variant from the manuscript | isolates the fitting loss from the density constraint |
| `song` | Kernel-embedding mixture fitting (Song et al., ICML 2008) | fixed-prototype convex QP on the simplex; closed-form `Q`, `q`; no sparsity term |
| `kde_sj` | Gaussian KDE, Sheather–Jones solve-the-equation | follows R's `bw.SJ(method="ste")` |
| `kde_lscv` | Gaussian KDE, least-squares CV | exact closed-form leave-one-out risk |
| `kde_botev` | Gaussian KDE at the Botev–Grotowski–Kroese improved-SJ bandwidth | **this is what their released `kde.m` computes** — a fixed-bandwidth estimator, *not* the adaptive linear-diffusion estimator of their Section 5 |
| `abramson` | Abramson (1982) square-root-law adaptive KDE | density floor and bandwidth caps explicit and reported |
| `gmm_em` | Gaussian mixture by EM | component count by the same CV J-score; covariance floor, 5 restarts |
| `thin_kt` | Kernel thinning + KDE | kernel halving + KT-swap implemented directly (not the `goodpoints` package), then an equal-weight KDE at a CV bandwidth |

All dictionary methods share **one** dictionary, so the comparison is about the fitting
criterion rather than the representation. Penalties are tuned independently per method.

### What is deliberately not included

Recorded in `akde/registry.py::NOT_IMPLEMENTED` so their absence is a scope decision
rather than an oversight:

- **PSD probability model** (Rudi & Ciliberto, NeurIPS 2021) — the next priority. Needs
  their specific regularised fitting and Nyström construction; an approximation would not
  be a fair comparison, and its model size is not interchangeable with mixture support size.
- **Low-Rank Thinning** (Carrell et al., ICML 2025) — the proceedings software link points
  to Thinformer, which targets attention approximation, so the exact routine for a KDE
  experiment is unsettled.
- **DGMM** (Zhang et al., 2025 preprint) — code promised on publication; singular components
  may lack a full-dimensional density, so it needs a compatible metric and its own
  experiment. **The `akde_diagonal` ablation is an ablation of *this* estimator's weighting
  matrix and must not be labelled as Zhang et al.'s method.**
- **Neural spline flows** (Durkan et al., NeurIPS 2019) — most informative beyond
  small-sample univariate data.

---

## 6. Test densities

The 15 densities in `Scripts/marronwand.m` were checked component-by-component and **are**
the Marron–Wand (1992) suite in published order. The manuscript should cite
Marron & Wand, *Ann. Statist.* 20(2), 712–736 and give this mapping:

| Label | MW # | Name | | Label | MW # | Name |
|---|---|---|---|---|---|---|
| `f0` | 1 | Gaussian | | `f8` | 9 | Trimodal |
| `f1` | 2 | Skewed unimodal | | `f9` | 10 | Claw |
| `f2` | 3 | Strongly skewed | | `f10` | 11 | Double claw |
| `f3` | 4 | Kurtotic unimodal | | `f11` | 12 | Asymmetric claw |
| `f4` | 5 | Outlier | | `f12` | 13 | Asymmetric double claw |
| `f5` | 6 | Bimodal | | `f13` | 14 | Smooth comb |
| `f6` | 7 | Separated bimodal | | `f14` | 15 | Discrete comb |
| `f7` | 8 | Skewed bimodal | | | | |

Note this bears on the historical `f5` anomaly (AKDE 3.1 vs SPADES 0.24): `f5` is MW6
**Bimodal**, a smooth, well-separated, entirely unexceptional density. A catastrophic
error there is not explained by the target being hard, which makes the old implementation
the more likely cause.

Also available: `--suite stress` (outlier contamination, rare narrow modes) and
`--suite multivariate` (2-D and 5-D full-rank targets).

---

## 7. Differences from the MATLAB prototype

`Scripts/` implements the *earlier* estimator, and the manuscript already states its
figures have not been recomputed. Beyond that, four issues in `demo.m` and `Proposed.m`
materially affect the published numbers:

1. **Oracle covariance.** `demo.m` estimates `Sig` from **50 freshly drawn datasets from
   the true density**. No practitioner has that. The Python framework estimates the
   covariance from the training sample only.
2. **Post-hoc normalisation.** `Proposed.m` ends with `w = w / sum(w)` applied to a signed
   solution — which does not make it a density, and changes sign behaviour unpredictably
   if the sum is near zero. The revised estimator is a density by construction.
3. **Grid MSE, not ISE.** `immse` on the sorted sample points is a sample-weighted
   pointwise error, not an integral. ISE is now analytic.
4. **Shared penalty.** `γ=7` was used for both AKDE and SPADES. Their objectives differ in
   scale by a factor of order `N`, so this does not give them comparable regularisation.

The MATLAB `demo.m` also calls `generateSamples`, `computeDensities` and `plotDensities`,
which are **not present** in `Scripts/` — it cannot run as shipped.

---

## 8. Layout

```
python/
  akde/
    gaussops.py     closed-form Gaussian overlaps -> D, G, analytic ISE
    densities.py    Marron-Wand suite, multivariate and stress targets
    dictionary.py   multiscale dictionary and measurement-kernel banks
    solvers.py      non-negative / signed lasso paths, simplex QP (numba-accelerated)
    bandwidths.py   Silverman, LSCV, Sheather-Jones, Botev ISJ
    metrics.py      ISE, IAE, Hellinger, J-score, NLL, mass diagnostics
    model.py        fitted-density container and validity checks
    estimators/     akde, spades, kde, song, emgmm, thinning
    registry.py     named configurations; NOT_IMPLEMENTED with reasons
    runner.py       replication protocol, parallel + resumable
    bootstrap.py    BCa and paired bootstrap, Holm adjustment
    analysis.py     tables (CSV + LaTeX)
  experiments/      smoke, run_benchmark, run_realdata, analyze, make_figures
  tests/            test_core.py
  run_all.py
```
