# Analysis: main_smoke

- successful fits: 900  (failures: 0)
- densities: 15, sample sizes: [np.int64(100), np.int64(500)], replicates per cell: 3
- bootstrap resamples: 400 (BCa, resampling replicate datasets)

## Overall accuracy (ISE), pooled over densities

| estimator | N | median ISE | mean ISE | valid density |
|---|---|---|---|---|
| abramson | 100 | 0.02164 | 0.03327 | 100% |
| abramson | 500 | 0.006148 | 0.009017 | 100% |
| akde | 100 | 0.2345 | 0.3679 | 100% |
| akde | 500 | 0.06758 | 0.08531 | 100% |
| gmm_em | 100 | 0.02874 | 0.03311 | 100% |
| gmm_em | 500 | 0.006807 | 0.008971 | 100% |
| kde_botev | 100 | 0.0176 | 0.03065 | 100% |
| kde_botev | 500 | 0.006369 | 0.009432 | 100% |
| kde_lscv | 100 | 0.02555 | 0.03419 | 100% |
| kde_lscv | 500 | 0.00655 | 0.009362 | 100% |
| kde_sj | 100 | 0.01708 | 0.0298 | 100% |
| kde_sj | 500 | 0.006211 | 0.01145 | 100% |
| song | 100 | 0.05161 | 0.0893 | 100% |
| song | 500 | 0.02542 | 0.03814 | 100% |
| spades | 100 | 0.1506 | 0.197 | 0% |
| spades | 500 | 0.04369 | 0.05019 | 0% |
| spades_simplex | 100 | 0.1625 | 0.2017 | 100% |
| spades_simplex | 500 | 0.04234 | 0.04605 | 100% |
| thin_kt | 100 | 0.02222 | 0.03377 | 100% |
| thin_kt | 500 | 0.006158 | 0.01108 | 100% |

## Paired comparison against `akde` (Holm-adjusted, alpha=0.05)

A 'win' means `akde` had significantly lower ISE on that density; comparisons are paired on identical datasets.

| baseline | N | wins | losses | ties | median error ratio |
|---|---|---|---|---|---|
| abramson | 100 | 0 | 15 | 0 | 11.8 |
| gmm_em | 100 | 0 | 15 | 0 | 11.3 |
| kde_botev | 100 | 0 | 15 | 0 | 11.7 |
| kde_lscv | 100 | 0 | 15 | 0 | 11.2 |
| kde_sj | 100 | 0 | 15 | 0 | 12.6 |
| song | 100 | 0 | 15 | 0 | 4.29 |
| spades | 100 | 0 | 9 | 6 | 1.34 |
| spades_simplex | 100 | 1 | 8 | 6 | 1.21 |
| thin_kt | 100 | 0 | 15 | 0 | 10.7 |
| abramson | 500 | 0 | 15 | 0 | 12.3 |
| gmm_em | 500 | 0 | 15 | 0 | 8.71 |
| kde_botev | 500 | 0 | 15 | 0 | 8.24 |
| kde_lscv | 500 | 0 | 14 | 1 | 8.9 |
| kde_sj | 500 | 0 | 14 | 1 | 7.43 |
| song | 500 | 0 | 12 | 3 | 2.36 |
| spades | 500 | 1 | 5 | 9 | 1.33 |
| spades_simplex | 500 | 0 | 6 | 9 | 1.4 |
| thin_kt | 500 | 0 | 14 | 1 | 7.54 |

'median error ratio' is the paired geometric mean of ISE(akde) / ISE(baseline): below 1 favours akde.

## Validity and solver diagnostics

| estimator      |   N |   frac_valid_density |   max_mass_error |   mean_negative_mass |   fallback_rate |   max_stage1_kkt |   max_stage2_kkt |   median_cov_rank |   median_omega_cond |
|:---------------|----:|---------------------:|-----------------:|---------------------:|----------------:|-----------------:|-----------------:|------------------:|--------------------:|
| akde           | 100 |                    1 |        3.331e-16 |            0         |               0 |        0.001106  |        0.002959  |                59 |              426    |
| spades         | 100 |                    0 |        0.756     |            0.0001083 |             nan |        3.704e-05 |      nan         |               nan |              nan    |
| spades_simplex | 100 |                    1 |        4.441e-16 |            0         |               0 |        4.99e-05  |        2.526e-15 |               nan |              nan    |
| song           | 100 |                    1 |        2.22e-16  |            0         |             nan |      nan         |        3.684e-06 |               nan |              nan    |
| kde_sj         | 100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       | 100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       | 100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      | 100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         | 100 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        | 100 |                    1 |        0         |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| akde           | 500 |                    1 |        7.772e-16 |            0         |               0 |        0.001039  |        0.1322    |                79 |               57.09 |
| spades         | 500 |                    0 |        0.2414    |            4.768e-05 |             nan |        9.773e-05 |      nan         |               nan |              nan    |
| spades_simplex | 500 |                    1 |        4.441e-16 |            0         |               0 |        9.247e-05 |        2.477e-15 |               nan |              nan    |
| song           | 500 |                    1 |        2.22e-16  |            0         |             nan |      nan         |        2.755e-06 |               nan |              nan    |
| kde_sj         | 500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       | 500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       | 500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      | 500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         | 500 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        | 500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
