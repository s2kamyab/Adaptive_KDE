## improved_accuracy_vs_N

Accuracy of the repaired estimator on the 15-target confirmation benchmark (30 datasets per target and N; comparator rows are the saved Stage-C fits on the identical datasets). Left: equal-weight-per-target benchmark mean ISE with 95% bootstrap intervals. Middle: fixed-benchmark paired geometric-mean ratios exp(T) of AKDE-R against AKDE-M, SPADES-simplex and the Sheather-Jones KDE; values below one favour AKDE-R. Right: per-target paired ratios against AKDE-M. Bootstrap: 10,000 resamples of replicate ids within each (target, N) cell; the target index is never resampled.

## improved_fitted_densities

Fitted densities on confirmation replicate 0 (the first locked seed; no selection after inspection) for four Marron-Wand targets at N = 200 and 5000. Within a panel every method is fitted to the identical observations. Annotations give ISE and the number of mixture components K.

## improved_dev_ablation

Development ablation on the development seeds (4 Marron-Wand targets x N in {500, 5000} x 5 datasets, plus 2 hr targets x 2 N x 2 datasets). Left: equal-weight-per-cell geometric-mean paired ISE ratio of each variant against the confirmed finalist AKDE-M, with 95% bootstrap intervals; the finalist was chosen by the rule fixed in the locked protocol. Middle and right: mean component count and mean fit time. Development results; not confirmation.

## improved_compactness

Compactness and cost on the confirmation benchmark. Left: every (target, N) cell's validation-selected operating point (mean ISE vs mean number of components over 30 datasets); not a Pareto frontier. Middle: query latency per point (mean over timed models of the median over 7 timing repeats; comparator latencies from the package's saved measurement). Right: mean ISE vs mean end-to-end fit time including cross-validation.

## improved_hr_error_vs_r

The repaired estimator on the scale-heterogeneity family f_r (20 datasets per cell; comparator rows are the saved bandwidth-flexibility fits on the identical datasets). Left and middle: total and narrow-region ISE, mean +- SD across datasets. Right: paired geometric-mean ratios of AKDE-R against AKDE-M, the single-scale ablation and the Sheather-Jones KDE with 95% bootstrap intervals.

