# Runbook: the repaired estimator (AKDE-R) study

Code: `Adaptive_KDE/python/followup/improved/` (this directory). Nothing in
`akde/` or `followup/akde_followup/` was modified; the repaired estimator is
`followup/improved/estimator.py` and reuses the package's solvers, Gaussian
algebra, seeds and comparators. The confirmed finalist (AKDE-M) and all
comparator rows are the saved Stage-C fits; only the new estimator was run on
those datasets, so every comparison is paired on data.

## 1. Questions

Do the repairs recommended after the earlier studies, namely
(a) metric-consistent weighting `Omega = (G + eps I)^-1` instead of the
inverse reading covariance, (b) a residual-space FISTA first stage run to its
KKT tolerance, (c) minimising the integrated-square loss over the simplex on
the full dictionary with no selection stage (fully corrective Frank-Wolfe),
(d) a per-atom scale-aware penalty, (e) an extended lambda path with the
one-SE rule and five folds, (f) CV-anchored dictionary scales, and (g) a
post-fit moment-matched merge of near-duplicate atoms,
make AKDE more accurate and/or more compact; and does the repaired
estimator still fall behind SPADES-simplex, the Sheather-Jones and Abramson
kernel estimators and an EM mixture? All three are hypotheses.

## 2. Directory and artifact map

```
followup/improved/
  __init__.py  estimator.py  merge.py  hr_family.py  runner.py
  protocol_improved.yaml|md|json, protocol_improved_lock.json   locked before the dev run
  finalist_improved.json                                        frozen by select_finalist.py
  scripts/ lock_protocol.py select_finalist.py run_stage.py measure_cost.py
           analyse.py make_figures.py make_tables.py integrate_paper.py verify.py
  figures/ improved_*.pdf|svg|png, data/, captions.md
  tables/  improved_*.tex (+ .csv twins), improved_numbers.tex
  logs/    run logs
  runbook.md  experiment_log.md  results_analysis.md  manuscript_changes.md  claims_to_evidence.csv
followup/results/improved/
  dev/      rows/*.json fits/*.npz all_rows.csv dev_ablation.csv run_summary.json
  confirm/  rows/ fits/ all_rows.csv run_summary.json        (1350 datasets x 1-2 methods)
  hr/       rows/ fits/ all_rows.csv run_summary.json        (160 datasets x 1-2 methods)
  cost/     query_cost.csv cost_config.json
  analysis/ tidy_confirm.csv per_cell_summary.csv benchmark_means.csv pooled_mean_sd.csv
            fixed_benchmark.csv per_target_ratios.csv reliability.csv hr_per_cell.csv hr_ratios.csv
  verification.json
```

Comparator inputs (read only): `followup/results/stage_c/all_rows.csv` and
`fits/`, `followup/results/verification/model_size_and_query_cost.csv`,
`final_package/experiments/bandwidth_flexibility/results/main/all_rows.csv`.

## 3. Environment

```powershell
cd C:\dev\Adaptive_KDE_project\Adaptive_KDE\python
$PY = "C:\dev\Adaptive_KDE_project\.venv\Scripts\python.exe"    # Python 3.10, numpy 2.2.6, scipy 1.15.3, numba 0.67
```

## 4. Commands, in order (all were run exactly as written)

```powershell
& $PY followup\improved\scripts\lock_protocol.py                 # lock BEFORE the dev run
& $PY followup\improved\scripts\run_stage.py --stage dev --jobs 12
& $PY followup\improved\scripts\select_finalist.py               # applies the locked rule
& $PY followup\improved\scripts\run_stage.py --stage confirm --jobs 12
& $PY followup\improved\scripts\run_stage.py --stage hr --jobs 12
& $PY followup\improved\scripts\measure_cost.py
& $PY followup\improved\scripts\analyse.py
& $PY followup\improved\scripts\make_figures.py                  # or --plot-only to redraw from figures\data
& $PY followup\improved\scripts\make_tables.py
& $PY followup\improved\scripts\integrate_paper.py               # edits final_package\paper\main (2).tex, copies assets
& $PY followup\improved\scripts\verify.py
```

Resume: every stage is resumable per fit (manifest = code hash + config hash
+ dataset sha256 + protocol hash); re-run the same command. `--force`
recomputes.

## 5. Expected counts

| stage | datasets | methods | fits | result |
|---|---|---|---|---|
| dev | 48 (4 MW targets x 2 N x 5 + 2 hr targets x 2 N x 2) | 11 (I0 finalist + I1..I10) | 528 | 528 rows, 0 errors |
| confirm | 1350 (15 MW targets x 3 N x 30) | 2 (AKDE-R, AKDE-R core) | 2700 | 2700 rows, 0 errors |
| hr | 160 (4 r x 2 N x 20) | 2 | 320 | 320 rows, 0 errors |
| cost | all 3020 saved confirm + hr models | - | - | `results/improved/cost/query_cost.csv` |

Comparator rows are not re-run: 8100 Stage-C rows and 960 bandwidth-flexibility
rows are read from their saved files.

## 6. Observed runtime and hardware

Observed on 2026-10-02, Windows 11, 16 physical / 22 logical cores, Python 3.10,
BLAS single-threaded, numba JIT kernels:

| step | wall time | workers |
|---|---|---|
| lock protocol | < 10 s | 1 |
| dev ablation (528 fits) | about 20 min in total: interrupted twice by the tool's 10-minute background limit and resumed from the per-fit cache; the final detached segment fitted 308 fits in 757 s | 12 |
| select finalist | < 1 min | 1 |
| confirmation (2700 fits) | 6.5 min | 10 |
| hr (320 fits, concurrent with confirmation) | 2.5 min | 4 |
| query cost (3020 models) | about 10 min | 1 |
| analysis (B = 10,000) | about 1 min | 1 |
| figures / tables / integrate / verify | about 2 min | 1 |

The dev ablation is dominated by the FISTA two-stage variants at N = 5000
(I3 ridge: 85 s per fit, I10: 134 s per fit); the finalist AKDE-R costs
0.85 / 1.9 / 5.2 s per fit at N = 200 / 1000 / 5000 and its core 0.04 / 0.08 / 0.23 s.

## 7. Seeds

Marron-Wand targets: `followup/akde_followup/seeds.py` namespaces
`development` (id 11, master 20260927; dev stage, fold stream 0 shared by all
variants) and `confirmation` (id 23, master 20260928; fold stream = method
index 6 / 7, the Stage-C policy). hr family: `hr_family.py` namespaces
`bf_dev` (43, 20261001) and `bf_main` (41, 20260930), fold stream 100 shared.
Datasets are identical to the saved comparator rows by construction (same
keys); `verify.py` checks the dataset sha256 against the saved rows where
stored.

## 8. Failure handling and solver diagnostics

Per fit: `error` / traceback, density validity, mass error, stage-1 KKT,
iterations, objective and convergence flag (FISTA to 1e-8 or Frank-Wolfe
gap), stage-2 KKT (< 1e-7), selected lambda / rho / anchor with grid-edge
flags, merge counts and training-loss change, large-error flag (ISE > 5 x own
cell median). A fit is never called converged because the solver returned.

## 9. Manuscript mapping

| manuscript element | file (copied to `final_package/paper/`) | built by | from |
|---|---|---|---|
| Table `tab:imp-dev` | `improved_dev_ablation.tex` | `make_tables.py` | `results/improved/dev/dev_ablation.csv` |
| Table `tab:imp-bench` | `improved_benchmark_mean_sd.tex` | `make_tables.py` | `results/improved/analysis/pooled_mean_sd.csv` |
| Table `tab:imp-ratio` | `improved_fixed_benchmark.tex` | `make_tables.py` | `analysis/fixed_benchmark.csv` |
| Table `tab:imp-cost` | `improved_cost.tex` | `make_tables.py` | `analysis/tidy_confirm.csv`, `cost/query_cost.csv` |
| Table `tab:imp-hr` | `improved_hr.tex` | `make_tables.py` | `analysis/hr_per_cell.csv` |
| all numbers in the section | `improved_numbers.tex` (+ `.csv`) | `make_tables.py` | `analysis/*`, `dev/dev_ablation.csv`, `cost/*` |
| Figure `fig:imp-acc` | `improved_accuracy_vs_N.pdf` | `make_figures.py` | `analysis/benchmark_means.csv`, `fixed_benchmark.csv`, `per_target_ratios.csv` |
| Figure `fig:imp-fits` | `improved_fitted_densities.pdf` | `make_figures.py` | `confirm/fits/*_r0_*.npz`, `followup/results/stage_c/fits/*_r0_*.npz` |
| Figure `fig:imp-dev` | `improved_dev_ablation.pdf` | `make_figures.py` | `dev/dev_ablation.csv` |
| Figure `fig:imp-cost` | `improved_compactness.pdf` | `make_figures.py` | `analysis/per_cell_summary.csv`, `cost/query_cost.csv`, `followup/results/verification/model_size_and_query_cost.csv` |
| Figure `fig:imp-hr` | `improved_hr_error_vs_r.pdf` | `make_figures.py` | `analysis/hr_per_cell.csv`, `hr_ratios.csv` |
| plot data | `figures/data/*_curves.npz`, `*_metrics.csv` | `make_figures.py` | - |

`claims_to_evidence.csv` maps every numerical claim of the section to its
metric, aggregation rule and artefact.

## 10. Limitations

* **Anchor grid too narrow at the wide end.** The validated base bandwidth
  is the top of its 9-point grid (4 h_Silverman) in 57% of confirmation fits.
  A wider grid may improve AKDE-R further; it was not widened because the
  protocol was locked. Needs a fresh confirmation.
* **Still behind the kernel estimators.** exp(T) vs Sheather-Jones is 1.28 to
  1.31 with intervals excluding one; vs Abramson 1.28 to 1.63; vs the EM
  mixture 1.54 to 2.13. Compactness, not accuracy, remains the case for AKDE-R.
* **The gain is from the dictionary, not the criterion.** The core
  reproduces SPADES-simplex to solver precision; the improvement is the
  CV-anchored scales, which any dictionary method can adopt.
* **Large-error fits** remain at a low rate: 7 of 1350 (0.5%, worst 10.4x
  cell median), all at N = 200, none at a grid edge.
* **Confirmed follow-up, not an independent replication.** The repairs were
  chosen with the earlier results in hand; development and confirmation seeds
  are disjoint and the selection rule was locked, but the confirmation seeds
  had already served one confirmation.
* **Fold policy differs between families.** Marron-Wand confirmation uses
  Stage-C's per-method fold streams (new indices 6 and 7); the hr family
  uses the shared fold stream of the bandwidth-flexibility experiment. Both
  match the comparator rows they are paired with.
* **No compiled PDF.** No TeX engine on this machine; structural checks only
  (`verify.py`). Build with `latexmk -pdf "main (2).tex"` in
  `final_package/paper`.
* **Timing.** The hr run overlapped the confirmation run; fit-time columns
  may be inflated by contention. Query latencies were measured afterwards in
  one quiet process.
