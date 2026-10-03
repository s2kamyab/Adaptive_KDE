# Experiment log: the repaired estimator (AKDE-R)

## 1. Starting point (2026-10-02)

* Source tree `Adaptive_KDE/python` verified byte-identical to
  `final_package/code` (only extra untracked helper files differ). Saved
  Stage-A/B/C rows and fits present under `followup/results/`.
* Recommendations to implement (from the two earlier studies): metric-
  consistent weighting; converged stage-1 solver; full-simplex L2 fit with no
  selection stage; scale-aware penalty; extended lambda path with 1-SE rule
  and 5 folds; CV-anchored dictionary scales; post-fit merge.
* Nothing in `akde/` or `followup/akde_followup/` is modified. New package:
  `followup/improved/`.

## 2. Implementation and smoke test

* `estimator.py`: `ImprovedAKDE` with the switches above; `merge.py`;
  `hr_family.py` (the heterogeneity family and its seeds, identical keys to
  the bandwidth-flexibility experiment); `runner.py` (per-fit resume,
  manifests). One shape bug in `build_dictionary_abs` fixed before any run.
* Smoke test on one development dataset (f9 claw, N = 500):
  I1 (Gram weighting + FISTA) reproduced SPADES-simplex exactly (ISE 0.01760,
  56 atoms), the numerical confirmation of Proposition 3; I4 (full-simplex L2)
  gave the same answer; the CV-anchored variants halved the error.
  The column-norm penalty collapsed to one atom on that dataset (recorded;
  nothing changed).

## 3. Protocol lock

* `protocol_improved.yaml` locked 2026-10-02 before the dev run (sha256
  `54b63c22…`), including the finalist selection rule: primary = equal-weight
  geometric-mean paired ISE ratio vs AKDE-M over the 8 Marron-Wand dev cells;
  gates = hr ratio <= 1, stage-1 and stage-2 at tolerance in >= 90% of fits,
  no errors, all valid; tie-break within 10% of the best = fewer tuned
  hyperparameters, then lower fit time; confirm the finalist plus I4 as the
  parameter-free ablation if the finalist is not I4.

## 4. Development ablation

* 528 fits (11 variants x 48 datasets), 0 errors, 757 s on 12 workers for the
  last 308 fits (the run was interrupted twice by the tool's 10-minute
  background limit and resumed from the per-fit cache; 220 fits were reused).
  Final launch was a detached process to avoid a third interruption.
* Results (`results/improved/dev/dev_ablation.csv`), ratio vs AKDE-M:
  I1 Gram+FISTA 0.84, I2 identity 0.85, I3 ridge+FISTA 0.89, I4 simplex-L2
  0.84, I5 Gram+colnorm 0.33, I6 ext-grid+1SE+5f 1.03, I7 simplex+CV-anchor
  0.34 [0.27, 0.41], I8 simplex+merge 0.91, I9 simplex+anchor+merge 0.49,
  I10 Gram+colnorm+anchor 0.28 (fails the KKT gate: stage 1 at tolerance in
  81%; 134 s per fit).
* Finalist by the locked rule: **I7_simplex_cvanchor** (within 10% of the
  best gated ratio together with I5; both have one tuned hyperparameter; I7
  is 9x cheaper). Ablation: I4_simplex_l2. Frozen in
  `finalist_improved.json`.
* Observations recorded, not acted on: the merge step reduces components
  (65 -> 16) but costs accuracy; the extended-grid/1-SE variant is worse than
  the finalist; the colnorm penalty is strong but expensive and
  budget-sensitive.

## 5. Confirmation and hr runs

* Launched detached 2026-10-02: confirmation (15 targets x 3 N x 30 reps x
  2 methods = 2700 fits, fold streams 6 and 7) and hr (4 r x 2 N x 20 x 2 =
  320 fits, fold stream 100). Completion recorded in
  `results/improved/{confirm,hr}/run_summary.json`.

* Completed: confirmation 2700 rows, 0 errors, 6.5 min on 10 workers; hr 320
  rows, 0 errors, 2.5 min on 4 workers (concurrent).

## 6. Post-processing

* `measure_cost.py` (3020 models), `analyse.py` (B = 10,000, seed 20261002),
  `make_tables.py`, `make_figures.py` (then `--plot-only`), `integrate_paper.py`,
  `verify.py` (all checks pass; `results/improved/verification.json`).
* Repairs during post-processing (no refitting): macro names in
  `make_tables.py` made digit-free with method tags (`\ir...`); a
  dev-ablation figure tick overlap fixed; five sentences of the section text
  corrected after checking against the rows, recorded in
  `scripts/_fix_section_text.py` (large-error fits are all at N = 200 with
  no grid-edge anchor, not "small bandwidth"; hr ratios vs the single-scale
  ablation are within 2.6x not 2x; the merge cuts components by half or
  more; endpoint labelling of hr ratios; the anchor-grid limitation added).
* Observation recorded: the selected base bandwidth is the top of its grid
  in 57% of confirmation fits (`sel_anchor = 4.0`), a limitation stated in
  the manuscript; the grid was not changed because the protocol was locked.

## 7. Manuscript

* `integrate_paper.py` inserted Section "A repaired estimator" before
  "Analysis of estimation error" in `final_package/paper/main (2).tex`, added
  abstract sentences, contribution 7, a limitation cross-reference and a
  conclusion paragraph, and copied the tables, macros and figures next to the
  manuscript (flat paths). Pre-edit snapshot in `snapshot/`.
