"""Fill runbook placeholders from run summaries and complete the experiment log."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parents[1] / "followup" / "results" / "improved"


def summ(stage):
    p = RES / stage / "run_summary.json"
    return json.load(open(p)) if p.exists() else {}


dev, con, hr = summ("dev"), summ("confirm"), summ("hr")
COUNTS = """| stage | datasets | methods | fits | result |
|---|---|---|---|---|
| dev | 48 (4 MW targets x 2 N x 5 + 2 hr targets x 2 N x 2) | 11 (I0 finalist + I1..I10) | 528 | %d rows, %d errors |
| confirm | 1350 (15 MW targets x 3 N x 30) | 2 (AKDE-R, AKDE-R core) | 2700 | %d rows, %d errors |
| hr | 160 (4 r x 2 N x 20) | 2 | 320 | %d rows, %d errors |
| cost | all 3020 saved confirm + hr models | - | - | `results/improved/cost/query_cost.csv` |

Comparator rows are not re-run: 8100 Stage-C rows and 960 bandwidth-flexibility
rows are read from their saved files.""" % (dev.get("n_rows", 0), dev.get("n_errors", 0), con.get("n_rows", 0),
                                              con.get("n_errors", 0), hr.get("n_rows", 0), hr.get("n_errors", 0))

RUNTIME = """Observed on 2026-10-02, Windows 11, 16 physical / 22 logical cores, Python 3.10,
BLAS single-threaded, numba JIT kernels:

| step | wall time | workers |
|---|---|---|
| lock protocol | < 10 s | 1 |
| dev ablation (528 fits) | about 20 min in total: interrupted twice by the tool's 10-minute background limit and resumed from the per-fit cache; the final detached segment fitted 308 fits in %.0f s | 12 |
| select finalist | < 1 min | 1 |
| confirmation (2700 fits) | %.1f min | 10 |
| hr (320 fits, concurrent with confirmation) | %.1f min | 4 |
| query cost (3020 models) | about 10 min | 1 |
| analysis (B = 10,000) | about 1 min | 1 |
| figures / tables / integrate / verify | about 2 min | 1 |

The dev ablation is dominated by the FISTA two-stage variants at N = 5000
(I3 ridge: 85 s per fit, I10: 134 s per fit); the finalist AKDE-R costs
0.85 / 1.9 / 5.2 s per fit at N = 200 / 1000 / 5000 and its core 0.04 / 0.08 / 0.23 s.""" % (
    dev.get("elapsed_s_this_invocation", dev.get("elapsed_s", 0)), con.get("elapsed_s", 0) / 60, hr.get("elapsed_s", 0) / 60)

MAPPING = """| manuscript element | file (copied to `final_package/paper/`) | built by | from |
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
metric, aggregation rule and artefact."""

LIMITS = """* **Anchor grid too narrow at the wide end.** The validated base bandwidth
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
  one quiet process."""

rb = HERE / "runbook.md"
s = rb.read_text(encoding="utf-8")
for k, v in (("COUNTS_PLACEHOLDER", COUNTS), ("RUNTIME_PLACEHOLDER", RUNTIME), ("MAPPING_PLACEHOLDER", MAPPING), ("LIMITS_PLACEHOLDER", LIMITS)):
    s = s.replace(k, v)
rb.write_text(s, encoding="utf-8")

log = HERE / "experiment_log.md"
L = log.read_text(encoding="utf-8")
if "## 6. Post-processing" not in L:
    L += """
* Completed: confirmation 2700 rows, 0 errors, %.1f min on 10 workers; hr 320
  rows, 0 errors, %.1f min on 4 workers (concurrent).

## 6. Post-processing

* `measure_cost.py` (3020 models), `analyse.py` (B = 10,000, seed 20261002),
  `make_tables.py`, `make_figures.py` (then `--plot-only`), `integrate_paper.py`,
  `verify.py` (all checks pass; `results/improved/verification.json`).
* Repairs during post-processing (no refitting): macro names in
  `make_tables.py` made digit-free with method tags (`\\ir...`); a
  dev-ablation figure tick overlap fixed; five sentences of the section text
  corrected after checking against the rows, recorded in
  `scripts/_fix_section_text.py` (large-error fits are all at N = 200 with
  no grid-edge anchor, not "small bandwidth"; hr ratios vs the single-scale
  ablation are within 2.6x not 2x; the merge cuts components by half or
  more; endpoint labelling of hr ratios; the anchor-grid limitation added).
* Observation recorded: the selected base bandwidth is the top of its grid
  in 57%% of confirmation fits (`sel_anchor = 4.0`), a limitation stated in
  the manuscript; the grid was not changed because the protocol was locked.

## 7. Manuscript

* `integrate_paper.py` inserted Section "A repaired estimator" before
  "Analysis of estimation error" in `final_package/paper/main (2).tex`, added
  abstract sentences, contribution 7, a limitation cross-reference and a
  conclusion paragraph, and copied the tables, macros and figures next to the
  manuscript (flat paths). Pre-edit snapshot in `snapshot/`.
""" % (con.get("elapsed_s", 0) / 60, hr.get("elapsed_s", 0) / 60)
    log.write_text(L, encoding="utf-8")
print("docs finalised")
