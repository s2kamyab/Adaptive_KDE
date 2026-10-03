# Analysis: interim_main

- successful fits: 13,500  (failures: 0)
- densities: 15, sample sizes: [np.int64(500), np.int64(1000), np.int64(5000)], replicates per cell: 30
- bootstrap resamples: 10,000 (BCa, resampling replicate datasets)

## Overall accuracy (ISE), pooled over densities

| estimator | N | median ISE | mean ISE | valid density |
|---|---|---|---|---|
| abramson | 500 | 0.006331 | 0.009698 | 100% |
| abramson | 1000 | 0.004343 | 0.005876 | 100% |
| abramson | 5000 | 0.001566 | 0.001739 | 100% |
| akde | 500 | 0.06779 | 0.3011 | 100% |
| akde | 1000 | 0.04722 | 0.187 | 100% |
| akde | 5000 | 0.04869 | 0.05716 | 100% |
| gmm_em | 500 | 0.006199 | 0.009817 | 100% |
| gmm_em | 1000 | 0.004031 | 0.005565 | 100% |
| gmm_em | 5000 | 0.00164 | 0.002145 | 100% |
| kde_botev | 500 | 0.006315 | 0.009268 | 100% |
| kde_botev | 1000 | 0.004607 | 0.00552 | 100% |
| kde_botev | 5000 | 0.001745 | 0.001737 | 100% |
| kde_lscv | 500 | 0.006849 | 0.009649 | 100% |
| kde_lscv | 1000 | 0.004612 | 0.00578 | 100% |
| kde_lscv | 5000 | 0.001782 | 0.001746 | 100% |
| kde_sj | 500 | 0.006216 | 0.01138 | 100% |
| kde_sj | 1000 | 0.004796 | 0.006864 | 100% |
| kde_sj | 5000 | 0.001825 | 0.002657 | 100% |
| song | 500 | 0.02709 | 0.03844 | 100% |
| song | 1000 | 0.02835 | 0.03305 | 100% |
| song | 5000 | 0.02118 | 0.02246 | 100% |
| spades | 500 | 0.04373 | 0.05319 | 0% |
| spades | 1000 | 0.02623 | 0.03019 | 0% |
| spades | 5000 | 0.006615 | 0.007533 | 0% |
| spades_simplex | 500 | 0.04243 | 0.04859 | 100% |
| spades_simplex | 1000 | 0.02435 | 0.0267 | 100% |
| spades_simplex | 5000 | 0.005765 | 0.006196 | 100% |
| thin_kt | 500 | 0.006655 | 0.01113 | 100% |
| thin_kt | 1000 | 0.004791 | 0.00689 | 100% |
| thin_kt | 5000 | 0.001866 | 0.002158 | 100% |

## Paired comparison against `akde` (Holm-adjusted, alpha=0.05)

A 'win' means `akde` had significantly lower ISE on that density; comparisons are paired on identical datasets.

| baseline | N | wins | losses | ties | median error ratio |
|---|---|---|---|---|---|
| abramson | 500 | 0 | 15 | 0 | 14.4 |
| gmm_em | 500 | 0 | 15 | 0 | 15.7 |
| kde_botev | 500 | 0 | 15 | 0 | 9.38 |
| kde_lscv | 500 | 0 | 15 | 0 | 8.51 |
| kde_sj | 500 | 0 | 15 | 0 | 9.58 |
| song | 500 | 0 | 15 | 0 | 2.43 |
| spades | 500 | 0 | 14 | 1 | 1.62 |
| spades_simplex | 500 | 0 | 14 | 1 | 1.77 |
| thin_kt | 500 | 0 | 15 | 0 | 8.28 |
| abramson | 1000 | 0 | 15 | 0 | 16.1 |
| gmm_em | 1000 | 0 | 15 | 0 | 15 |
| kde_botev | 1000 | 0 | 15 | 0 | 14.2 |
| kde_lscv | 1000 | 0 | 15 | 0 | 13.8 |
| kde_sj | 1000 | 0 | 15 | 0 | 14.3 |
| song | 1000 | 1 | 14 | 0 | 2.03 |
| spades | 1000 | 0 | 14 | 1 | 2.11 |
| spades_simplex | 1000 | 0 | 15 | 0 | 2.16 |
| thin_kt | 1000 | 0 | 15 | 0 | 13.8 |
| abramson | 5000 | 0 | 15 | 0 | 33 |
| gmm_em | 5000 | 0 | 15 | 0 | 37.8 |
| kde_botev | 5000 | 0 | 15 | 0 | 28.6 |
| kde_lscv | 5000 | 0 | 15 | 0 | 29.4 |
| kde_sj | 5000 | 0 | 15 | 0 | 24 |
| song | 5000 | 1 | 14 | 0 | 2.26 |
| spades | 5000 | 1 | 14 | 0 | 7.46 |
| spades_simplex | 5000 | 0 | 14 | 1 | 8.71 |
| thin_kt | 5000 | 0 | 15 | 0 | 26.5 |

'median error ratio' is the paired geometric mean of ISE(akde) / ISE(baseline): below 1 favours akde.

## Validity and solver diagnostics

| estimator      |    N |   frac_valid_density |   max_mass_error |   mean_negative_mass |   fallback_rate |   max_stage1_kkt |   max_stage2_kkt |   median_cov_rank |   median_omega_cond |
|:---------------|-----:|---------------------:|-----------------:|---------------------:|----------------:|-----------------:|-----------------:|------------------:|--------------------:|
| akde           | 1000 |                    1 |        1.332e-15 |            0         |               0 |        0.001906  |        0.09879   |                85 |               54.98 |
| spades         | 1000 |                    0 |        0.2199    |            6.378e-05 |             nan |        0.000244  |      nan         |               nan |              nan    |
| spades_simplex | 1000 |                    1 |        6.661e-16 |            0         |               0 |        0.0002121 |        3.95e-15  |               nan |              nan    |
| song           | 1000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |        0.0006111 |               nan |              nan    |
| kde_sj         | 1000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       | 1000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       | 1000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      | 1000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         | 1000 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        | 1000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| akde           |  500 |                    1 |        8.882e-16 |            0         |               0 |        0.001846  |        0.1636    |                79 |               58.59 |
| spades         |  500 |                    0 |        0.2969    |            8.229e-05 |             nan |        0.0002169 |      nan         |               nan |              nan    |
| spades_simplex |  500 |                    1 |        7.772e-16 |            0         |               0 |        0.0001838 |        3.034e-15 |               nan |              nan    |
| song           |  500 |                    1 |        3.331e-16 |            0         |             nan |      nan         |        0.0001207 |               nan |              nan    |
| kde_sj         |  500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       |  500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       |  500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      |  500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         |  500 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        |  500 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| akde           | 5000 |                    1 |        9.326e-15 |            0         |               0 |        0.0001938 |        0.1892    |               152 |               45.7  |
| spades         | 5000 |                    0 |        0.1393    |            6.976e-05 |             nan |        0.0001309 |      nan         |               nan |              nan    |
| spades_simplex | 5000 |                    1 |        4.441e-16 |            0         |               0 |        8.705e-05 |        4.314e-09 |               nan |              nan    |
| song           | 5000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |        0.005     |               nan |              nan    |
| kde_sj         | 5000 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       | 5000 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       | 5000 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      | 5000 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         | 5000 |                    1 |        4.441e-16 |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        | 5000 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
