# Improved-AKDE study: protocol (locked)

prospective development-then-confirmation protocol; NOT preregistered: the repairs were chosen from the findings of the earlier studies.

**Question.** do the recommended repairs (metric-consistent weighting, converged solver, full-simplex L2 fit, scale-aware penalty, extended lambda grid + 1-SE rule, CV-anchored scales, post-fit merge) make AKDE more accurate and/or more compact, and does the repaired estimator still fall behind the conventional comparators?

**Development.** namespace development, targets ['f0', 'f2', 'f5', 'f9'], N [500, 5000], R=5 (+ hr dev ['hr1', 'hr8'] x [500, 5000] x 2), variants: I0_finalist, I1_gram_fista, I2_identity_fista, I3_ridge_fista, I4_simplex_l2, I5_gram_colnorm, I6_gram_ext1se5f, I7_simplex_cvanchor, I8_simplex_merge, I9_simplex_cvanchor_merge, I10_gram_colnorm_cvanchor.

**Selection rule (fixed before the development run).** Primary: equal-weight-per-cell geometric mean, over the 8 Marron-Wand dev cells, of the paired ISE ratio variant / I0_finalist (lower is better). Gates: geometric-mean ratio vs I0 on the 4 hr dev cells <= 1.0; stage-1 AND stage-2 KKT at tolerance in >= 90% of dev fits (stage-1 n/a for simplex_l2); no execution error; all fits valid densities. Tie-break: among variants whose primary ratio is within 10% of the best, prefer the one with fewer tuned hyperparameters, then lower mean wall time. the finalist; plus I4_simplex_l2 if the finalist is not I4 (the parameter-free core), so the contribution of anchor/merge is confirmed too.

**Confirmation.** 15 Marron-Wand targets x N in {200, 1000, 5000} x R=30 on the confirmation namespace (datasets identical to Stage C, comparator rows reused, fold stream = new method index), plus the hr family (4 r x 2 N x 20) on bf_main with comparator rows reused.

**Analysis.** {"per_cell": "mean +- SD (ddof=1), median", "benchmark": "equal-weight-per-target mean ISE; fixed-benchmark exp(T) paired ratio vs AKDE-M and vs SPADES-simplex / KDE-SJ / Abramson / GMM", "bootstrap": {"B": 10000, "seed": 20261002, "unit": "replicate id within (target, N) cell", "targets_never_resampled": true}, "tables": "mean +- SD; bootstrap intervals only in labelled tables/figures"}
