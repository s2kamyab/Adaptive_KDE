# Replication count for the Stage-C confirmation

Decided 2026-09-28T02:31:24+00:00, before any confirmation seed was drawn.

Primary outcome: z = log(ISE_finalist / ISE_v2_akde), paired within replicate.

Precision target: 95% half-width <= log(1.25) = 0.2231 for the mean of z, per (target, N) cell.


## SD(z) estimates

| source | cells | median SD | 75th pct | 90th pct | max |
|---|---|---|---|---|---|
| Stage A+B development (A6_moment_matched_m300 vs V2 AKDE), R=5 | 8 | 0.391 | 0.732 | 1.446 | 2.684 |
| V2 archive (spades_simplex vs akde), R=30, N in {200,1000,5000} | 45 | 0.580 | 0.754 | 0.892 | 2.134 |

The development estimate uses only 5 replicates per cell, so each SD is itself very uncertain (a chi distribution with 4 degrees of freedom: the 95% interval for a single such SD spans roughly 0.6x to 2.2x the truth). The archive estimate is from 30 replicates and a *different* pair of methods, so it describes the general scale of paired log-ratio variability in this benchmark rather than this specific contrast. Both are reported; neither is treated as exact.


## Achievable half-width

| R | half-width at median SD (dev) | at 90th pct (dev) | at median SD (archive) | at 90th pct (archive) | cells meeting target (dev) | cells meeting target (archive) |
|---|---|---|---|---|---|---|
| 10 | 0.280 | 1.034 | 0.415 | 0.638 | 38% | 0% |
| 20 | 0.183 | 0.677 | 0.272 | 0.417 | 62% | 31% |
| 30 | 0.146 | 0.540 | 0.217 | 0.333 | 62% | 56% |

## Decision

**R = 30 (the cap).** No candidate in {10, 20, 30} meets the log(1.25) half-width target in 80% of cells under both SD estimates, so the budgeted cap is kept and the shortfall is reported as a limitation: per-cell intervals for the mean log ratio will be wider than log(1.25) in a substantial fraction of cells, and per-cell conclusions are correspondingly weak. The benchmark-wide summary, which averages one number per target over 15 targets, is much better resolved than any single cell.


R is fixed here and not revisited. Replicates are not added until significance is reached, and the run is not stopped early because one method wins. No sequential procedure is specified, so none is used.
