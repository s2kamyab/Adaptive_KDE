# Analysis: main_full

- successful fits: 27,000  (failures: 0)
- densities: 15, sample sizes: [np.int64(50), np.int64(100), np.int64(200), np.int64(500), np.int64(1000), np.int64(5000)], replicates per cell: 30
- bootstrap resamples: 10,000 (BCa, resampling replicate datasets)

## Overall accuracy (ISE), pooled over densities

| estimator | N | median ISE | mean ISE | valid density |
|---|---|---|---|---|
| abramson | 50 | 0.03388 | 0.04776 | 100% |
| abramson | 100 | 0.02301 | 0.02922 | 100% |
| abramson | 200 | 0.01395 | 0.01821 | 100% |
| abramson | 500 | 0.006331 | 0.009698 | 100% |
| abramson | 1000 | 0.004343 | 0.005876 | 100% |
| abramson | 5000 | 0.001566 | 0.001739 | 100% |
| akde | 50 | 0.3343 | 0.7296 | 100% |
| akde | 100 | 0.2176 | 0.4408 | 100% |
| akde | 200 | 0.1355 | 0.2138 | 100% |
| akde | 500 | 0.06779 | 0.3011 | 100% |
| akde | 1000 | 0.04722 | 0.187 | 100% |
| akde | 5000 | 0.04869 | 0.05716 | 100% |
| gmm_em | 50 | 0.03877 | 0.0588 | 100% |
| gmm_em | 100 | 0.02723 | 0.03513 | 100% |
| gmm_em | 200 | 0.01354 | 0.02026 | 100% |
| gmm_em | 500 | 0.006199 | 0.009817 | 100% |
| gmm_em | 1000 | 0.004031 | 0.005565 | 100% |
| gmm_em | 5000 | 0.00164 | 0.002145 | 100% |
| kde_botev | 50 | 0.02986 | 0.04784 | 100% |
| kde_botev | 100 | 0.01914 | 0.03111 | 100% |
| kde_botev | 200 | 0.01238 | 0.01777 | 100% |
| kde_botev | 500 | 0.006315 | 0.009268 | 100% |
| kde_botev | 1000 | 0.004607 | 0.00552 | 100% |
| kde_botev | 5000 | 0.001745 | 0.001737 | 100% |
| kde_lscv | 50 | 0.03258 | 0.04762 | 100% |
| kde_lscv | 100 | 0.02123 | 0.03092 | 100% |
| kde_lscv | 200 | 0.01404 | 0.01801 | 100% |
| kde_lscv | 500 | 0.006849 | 0.009649 | 100% |
| kde_lscv | 1000 | 0.004612 | 0.00578 | 100% |
| kde_lscv | 5000 | 0.001782 | 0.001746 | 100% |
| kde_sj | 50 | 0.0294 | 0.04258 | 100% |
| kde_sj | 100 | 0.01909 | 0.02911 | 100% |
| kde_sj | 200 | 0.01263 | 0.01892 | 100% |
| kde_sj | 500 | 0.006216 | 0.01138 | 100% |
| kde_sj | 1000 | 0.004796 | 0.006864 | 100% |
| kde_sj | 5000 | 0.001825 | 0.002657 | 100% |
| song | 50 | 0.06765 | 0.1208 | 100% |
| song | 100 | 0.04797 | 0.08665 | 100% |
| song | 200 | 0.03548 | 0.06368 | 100% |
| song | 500 | 0.02709 | 0.03844 | 100% |
| song | 1000 | 0.02835 | 0.03305 | 100% |
| song | 5000 | 0.02118 | 0.02246 | 100% |
| spades | 50 | 0.2375 | 0.3367 | 0% |
| spades | 100 | 0.1501 | 0.1905 | 0% |
| spades | 200 | 0.08916 | 0.113 | 0% |
| spades | 500 | 0.04373 | 0.05319 | 0% |
| spades | 1000 | 0.02623 | 0.03019 | 0% |
| spades | 5000 | 0.006615 | 0.007533 | 0% |
| spades_simplex | 50 | 0.263 | 0.3714 | 100% |
| spades_simplex | 100 | 0.1573 | 0.1914 | 100% |
| spades_simplex | 200 | 0.08932 | 0.1087 | 100% |
| spades_simplex | 500 | 0.04243 | 0.04859 | 100% |
| spades_simplex | 1000 | 0.02435 | 0.0267 | 100% |
| spades_simplex | 5000 | 0.005765 | 0.006196 | 100% |
| thin_kt | 50 | 0.03316 | 0.04816 | 100% |
| thin_kt | 100 | 0.02255 | 0.03183 | 100% |
| thin_kt | 200 | 0.01363 | 0.01938 | 100% |
| thin_kt | 500 | 0.006655 | 0.01113 | 100% |
| thin_kt | 1000 | 0.004791 | 0.00689 | 100% |
| thin_kt | 5000 | 0.001866 | 0.002158 | 100% |

## Paired comparison against `akde` (Holm-adjusted, alpha=0.05)

A 'win' means `akde` had significantly lower ISE on that density; comparisons are paired on identical datasets.

| baseline | N | wins | losses | ties | median error ratio |
|---|---|---|---|---|---|
| abramson | 50 | 0 | 15 | 0 | 14.8 |
| gmm_em | 50 | 0 | 15 | 0 | 11.8 |
| kde_botev | 50 | 0 | 15 | 0 | 19.4 |
| kde_lscv | 50 | 0 | 15 | 0 | 17 |
| kde_sj | 50 | 0 | 15 | 0 | 20.1 |
| song | 50 | 0 | 15 | 0 | 5.27 |
| spades | 50 | 0 | 15 | 0 | 1.43 |
| spades_simplex | 50 | 0 | 12 | 3 | 1.37 |
| thin_kt | 50 | 0 | 15 | 0 | 17.9 |
| abramson | 100 | 0 | 15 | 0 | 11.7 |
| gmm_em | 100 | 0 | 15 | 0 | 12.4 |
| kde_botev | 100 | 0 | 15 | 0 | 13.5 |
| kde_lscv | 100 | 0 | 15 | 0 | 12.1 |
| kde_sj | 100 | 0 | 15 | 0 | 14.2 |
| song | 100 | 0 | 15 | 0 | 4.75 |
| spades | 100 | 0 | 13 | 2 | 1.56 |
| spades_simplex | 100 | 0 | 14 | 1 | 1.48 |
| thin_kt | 100 | 0 | 15 | 0 | 12.4 |
| abramson | 200 | 0 | 15 | 0 | 11.5 |
| gmm_em | 200 | 0 | 15 | 0 | 14 |
| kde_botev | 200 | 0 | 15 | 0 | 12.3 |
| kde_lscv | 200 | 0 | 15 | 0 | 11.7 |
| kde_sj | 200 | 0 | 15 | 0 | 12.8 |
| song | 200 | 1 | 14 | 0 | 3.79 |
| spades | 200 | 1 | 14 | 0 | 1.43 |
| spades_simplex | 200 | 0 | 14 | 1 | 1.43 |
| thin_kt | 200 | 0 | 15 | 0 | 12.5 |
| abramson | 500 | 0 | 15 | 0 | 14.4 |
| gmm_em | 500 | 0 | 15 | 0 | 15.7 |
| kde_botev | 500 | 0 | 15 | 0 | 9.38 |
| kde_lscv | 500 | 0 | 15 | 0 | 8.51 |
| kde_sj | 500 | 0 | 15 | 0 | 9.58 |
| song | 500 | 0 | 15 | 0 | 2.43 |
| spades | 500 | 0 | 14 | 1 | 1.62 |
| spades_simplex | 500 | 0 | 15 | 0 | 1.77 |
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
| akde           |   50 |                    1 |        4.441e-16 |            0         |               0 |        0.001239  |        0.005836  |                42 |               83.01 |
| spades         |   50 |                    0 |        0.9757    |            2.157e-05 |             nan |        1.483e-05 |      nan         |               nan |              nan    |
| spades_simplex |   50 |                    1 |        4.441e-16 |            0         |               0 |        7.972e-05 |        8.993e-15 |               nan |              nan    |
| song           |   50 |                    1 |        3.331e-16 |            0         |             nan |      nan         |        5.772e-06 |               nan |              nan    |
| kde_sj         |   50 |                    1 |        0         |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       |   50 |                    1 |        0         |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       |   50 |                    1 |        0         |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      |   50 |                    1 |        0         |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         |   50 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        |   50 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| akde           |  100 |                    1 |        7.772e-16 |            0         |               0 |        0.001549  |        0.01393   |                59 |              493.6  |
| spades         |  100 |                    0 |        0.9687    |            8.187e-05 |             nan |        0.0001083 |      nan         |               nan |              nan    |
| spades_simplex |  100 |                    1 |        6.661e-16 |            0         |               0 |        0.0001274 |        4.358e-15 |               nan |              nan    |
| song           |  100 |                    1 |        4.441e-16 |            0         |             nan |      nan         |        4.35e-06  |               nan |              nan    |
| kde_sj         |  100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       |  100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       |  100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      |  100 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         |  100 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        |  100 |                    1 |        0         |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| akde           |  200 |                    1 |        6.661e-16 |            0         |               0 |        0.001514  |        0.03555   |                68 |               72.11 |
| spades         |  200 |                    0 |        0.5735    |            0.0001603 |             nan |        0.0001501 |      nan         |               nan |              nan    |
| spades_simplex |  200 |                    1 |        6.661e-16 |            0         |               0 |        0.0001552 |        7.092e-15 |               nan |              nan    |
| song           |  200 |                    1 |        4.441e-16 |            0         |             nan |      nan         |        5.393e-06 |               nan |              nan    |
| kde_sj         |  200 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_lscv       |  200 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| abramson       |  200 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| kde_botev      |  200 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| gmm_em         |  200 |                    1 |        2.22e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
| thin_kt        |  200 |                    1 |        1.11e-16  |            0         |             nan |      nan         |      nan         |               nan |              nan    |
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
