# Results analysis: the repaired estimator (AKDE-R)

Every number below is read from the files in brackets; nothing is typed by
hand. Protocol locked before the development run
(`protocol_improved_lock.json`); finalist frozen by the locked rule
(`finalist_improved.json`).

## Short answer

The repairs work, and most of the gain comes from one of them. Fitting the
integrated-square loss directly over the simplex on a dictionary whose scales
are anchored to a validated bandwidth (AKDE-R: no first stage, no λ, no ρ, no
τ, one tuned hyperparameter) cuts the paired geometric-mean ISE to 0.26, 0.40
and 0.51 of AKDE-M's at N = 200, 1000, 5000 on the 15-target confirmation,
beats SPADES-simplex on the original dictionary by 0.32 to 0.58, and narrows
the gap to the Sheather–Jones KDE to about 1.3x (from 2.6 to 4.9x). It stays
behind every conventional comparator. Compactness is still the reason to use
it, and it is now more compact too (42 vs 56 components at N = 5000).

## 1. Development ablation [`results/improved/dev/dev_ablation.csv`]

Equal-weight geometric-mean paired ISE ratio vs AKDE-M over the 8
Marron–Wand development cells (hr gate in parentheses):

| variant | ratio [95% CI] | hr | at tol (s1/s2) | K | s/fit |
|---|---|---|---|---|---|
| I1 Gram weighting + FISTA | 0.84 [0.67, 0.96] (0.94) | 96% / 100% | 60 | 9.5 |
| I2 identity + FISTA | 0.85 [0.67, 0.97] (0.97) | 75% / 98% | 48 | 23 |
| I3 ridge-covariance + FISTA (finalist's Ω, converged) | 0.89 [0.72, 1.00] (1.01) | 48% / 96% | 54 | 85 |
| I4 full-simplex L2, Frank–Wolfe (no stage 1) | 0.84 [0.67, 0.95] (0.94) | – / 100% | 66 | 0.1 |
| I5 Gram + scale-aware penalty | 0.33 [0.24, 0.43] (0.46) | 98% / 100% | 15 | 22 |
| I6 Gram + extended λ grid + 1-SE + 5 folds | 1.03 [0.83, 1.18] (1.11) | 100% / 100% | 42 | 16 |
| **I7 full-simplex + CV-anchored scales** | **0.34 [0.27, 0.41] (0.36)** | – / 100% | 32 | 2.4 |
| I8 full-simplex + merge | 0.91 [0.72, 1.07] (0.86) | – / 100% | 16 | 1.2 |
| I9 full-simplex + anchor + merge | 0.49 [0.38, 0.60] (0.57) | – / 100% | 18 | 15 |
| I10 Gram + colnorm + anchor | 0.28 [0.21, 0.34] (0.24) | 81% / 100% (fails gate) | 12 | 134 |

Readings:
* Replacing the ridge-covariance weighting by `G⁻¹` (I1) or by the identity
  (I2), or dropping the first stage entirely (I4), each recovers about 15%:
  the covariance weighting was costing that much, and the solver that
  reached tolerance with the old Ω (I3) only recovers 11%.
* I4 ≡ SPADES-simplex (ratio 1.000 to SPADES in confirmation): Proposition 3
  made numerical.
* Anchoring the scales to a validated bandwidth (I7) is the big repair
  (0.34). The scale-aware penalty (I5, 0.33) does similar work at 9x the
  cost; both together (I10, 0.28) is best but fails the convergence gate and
  costs 134 s per fit.
* The extended grid / 1-SE rule (I6) does not help; the merge (I8, I9) costs
  accuracy. Neither is in the finalist.
* Selection: I7 and I5 are within 10% of the best gated ratio; both have one
  tuned hyperparameter; I7 is 9x cheaper, so I7 is the finalist and I4 its
  parameter-free core (ablation).

## 2. Confirmation [`results/improved/analysis/fixed_benchmark.csv`, `pooled_mean_sd.csv`, `per_target_ratios.csv`]

exp(T) paired ratios, AKDE-R in the numerator (targets favouring AKDE-R of 15):

| reference | N = 200 | N = 1000 | N = 5000 |
|---|---|---|---|
| AKDE-M | 0.258 [0.243, 0.273] (15) | 0.396 [0.380, 0.412] (15) | 0.505 [0.473, 0.536] (14) |
| SPADES-simplex | 0.316 [0.299, 0.335] (15) | 0.425 [0.409, 0.441] (14) | 0.580 [0.568, 0.592] (14) |
| Sheather–Jones KDE | 1.275 [1.210, 1.347] (3) | 1.309 [1.250, 1.375] (4) | 1.310 [1.269, 1.353] (4) |
| Abramson KDE | 1.283 (2) | 1.405 (2) | 1.625 (1) |
| EM mixture | 1.537 (4) | 1.743 (4) | 2.128 (5) |

Core (I4) vs AKDE-M: 0.815, 0.932, 0.874; core vs SPADES: 1.000, 1.000, 1.002.

Benchmark mean ± pooled SD over 450 fits: AKDE-R 0.0245 ± 0.030, 0.0076 ±
0.0064, 0.0030 ± 0.0031 versus AKDE-M 0.0727 ± 0.066, 0.0148 ± 0.0061, 0.0116
± 0.047 and SJ 0.0198 ± 0.018, 0.0066 ± 0.0058, 0.0026 ± 0.0027.

## 3. Reliability [`analysis/reliability.csv`]

* 2700/2700 fits valid densities, 0 errors, stage-2 KKT < 1e-7 in every fit
  (max 1.8e-15); no first stage exists.
* Large-error fits (> 5x own cell median): AKDE-R 7 (0.52%, worst 10.4x, all
  at N = 200, none at a grid edge); core 0; AKDE-M 13 (worst 169x); SPADES 0;
  SJ 0; EM 37.
* Selected base bandwidth at the top of the grid (4h) in 57% of fits: the
  anchor grid is too narrow at the wide end (limitation; not changed because
  locked).

## 4. Compactness and cost [`analysis/tidy_confirm.csv`, `cost/query_cost.csv`]

At N = 5000 (mean over 450 fits): AKDE-R 42 ± 15 components, 0.64 µs per
query, 5.2 s to fit (nine-candidate bandwidth search included); core 75 ± 8
components, 2.0 µs, 0.23 s; AKDE-M 56 ± 12 components, 5.2 s. Kernel
estimators: 5000 components, about 150 µs per query.

## 5. Heterogeneity family [`analysis/hr_ratios.csv`, `hr_per_cell.csv`]

AKDE-R vs AKDE-M: 0.22, 0.19, 0.28, 0.52 (N = 500) and 0.18, 0.36, 0.35, 0.64
(N = 5000) for r = 1, 2, 4, 8. Vs the single-scale ablation: 1.89, 0.71, 1.06,
0.97 and 2.53, 1.15, 1.05, 0.41: within a factor 2.6 everywhere, better in 3
of 8. Vs Sheather–Jones: 2.72, 2.48, 1.54, 1.04 and 2.89, 1.78, 0.97, 0.92.
The anchored multiscale dictionary is no longer worse than one validated
bandwidth on this family, but it is not clearly better either.

## 6. What is and is not established

Established: the three numerical pathologies are gone (no non-converging
stage, no non-metric weighting, no λ grid edge); a 2 to 4x accuracy gain over
AKDE-M and 1.7 to 3x over SPADES on the same centres; fewer components; the
Proposition-3 identity on 1350 datasets; the gain is attributable to the
validated dictionary scales.

Not established: superiority over kernel or mixture estimators (still behind
in every N); any asymptotic claim; transfer to other dictionary
constructions; that a wider anchor grid would help (likely, untested).
