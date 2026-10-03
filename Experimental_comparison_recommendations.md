# Experimental comparison recommendations

For *Adaptive Kernel Density Estimation using the Generalized Method of Moments and Compressive Kernel Selection* — Shima Kamyab, Behrooz Nasihatkon, and Reza Boostani.

Literature checked: 15 September 2026. These are proposed experiments and manuscript checks, not experiments performed or new results.

## Recommended comparison set

My recommendation is to retain SPADES, add a kernel-embedding mixture estimator, and establish strong conventional KDE references before investing in more distant neural methods. Add Kernel Thinning plus KDE as the first recent compression baseline. Prioritize direct relevance and reliable publication records over publication year alone. The review combines established foundational work with 2021–2025 developments; it does not claim that every recent paper is highly cited or rank papers by an audited citation count.

| Priority | Comparator | What it tests in your paper | Implementation decision |
|---|---|---|---|
| Essential | SPADES — Bunea et al., Annals of Statistics, 2010 | Whether covariance-weighted moments improve over sparse integrated-square fitting | Use the same location–bandwidth dictionary; tune the penalty separately for each method. Check the original method's weighting convention. |
| Essential | Kernel-embedding mixture fitting — Song et al., ICML 2008 | Whether your weighting scheme adds value beyond convex kernel moment matching | Reproduce the fixed-prototype formulation; report any common-dictionary adaptation explicitly. |
| Essential | Gaussian KDE with Sheather–Jones bandwidth; add cross-validated bandwidth | Whether adaptation beats a well-tuned ordinary estimator | Use a documented selector and training-only preprocessing. |
| Essential | Abramson adaptive KDE | Whether selecting discrete scales improves on established local bandwidth adaptation | Tune pilot and global scale; state density floors and bandwidth caps. |
| Essential | Botev–Grotowski–Kroese diffusion-based estimator | Whether your approach competes with a strong low-dimensional smoothing method | Distinguish improved plug-in Gaussian KDE from the full adaptive diffusion estimator. |
| High: recent | Kernel Thinning + KDE — Dwivedi and Mackey, JMLR 2024 | Accuracy and query cost at matched retained-kernel budgets | Thin training data, then construct a normalized KDE. Use the same target support-size budgets as your estimator. |
| High: kernel alternative | PSD density model — Rudi and Ciliberto, NeurIPS 2021 | Accuracy versus a nonnegative kernel probability model | Compare memory, fitting time, and prediction cost; PSD rank is not directly interchangeable with mixture support size. |
| Useful conventional control | Gaussian mixture fitted by EM | Whether a compact likelihood-fitted mixture is competitive | Select component count on training/validation data; use covariance floors and multiple initializations. |
| Optional: recent extension | Low-Rank Thinning — Carrell et al., ICML 2025 | Further distribution-compression comparisons | Identify the precise thinning routine used and add a smoothing stage. It is not a ready-made adaptive KDE. |
| Optional: separate experiment | DGMM — Zhang et al., 2025 preprint | Moment weighting for compatible mixture-parameter estimation | Respect its given component count and rank assumptions. Public repository currently promises code on publication rather than providing the implementation. |
| Optional: broader density benchmark | Neural spline flow — Durkan et al., NeurIPS 2019 | Performance against an explicit neural density model | More useful after adding multivariate data; use a functioning scalar spline for a univariate experiment. |

If resources are limited, first run the five essential rows plus Kernel Thinning + KDE. Add PSD estimation next. Do not replace direct predecessors with recent but less comparable methods.

## Source and implementation links

- [SPADES paper](https://doi.org/10.1214/09-AOS790).
- [Song et al.: kernel moment matching paper](https://icml.cc/Conferences/2008/papers/377.pdf).
- [Sheather–Jones paper](https://doi.org/10.1111/j.2517-6161.1991.tb01857.x); [R's official bandwidth-selector documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/bandwidth.html), including `bw.SJ` and `bw.ucv`.
- [Abramson paper](https://doi.org/10.1214/aos/1176345986).
- [Botev et al. paper](https://doi.org/10.1214/10-AOS799); [recent diffKDE development, Pelz et al. 2023](https://doi.org/10.5194/gmd-16-6609-2023). Check which estimator each software implementation actually computes.
- [Kernel Thinning, JMLR 2024 version](https://arxiv.org/abs/2105.05842); [authors' Goodpoints software](https://github.com/microsoft/goodpoints).
- [Low-Rank Thinning, ICML 2025](https://proceedings.mlr.press/v267/carrell25a.html). Its proceedings software link points to [Thinformer](https://github.com/microsoft/thinformer), which focuses on attention approximation. This is not evidence of a ready-to-run KDE benchmark.
- [PSD probability models, NeurIPS 2021](https://arxiv.org/abs/2106.16116).
- [DGMM preprint](https://arxiv.org/abs/2507.20459); [authors' repository](https://github.com/liu-lzhang/dgmm). The retrieved README states that code and data will be released upon publication.
- [Neural spline flows](https://arxiv.org/abs/1906.04032).

Kremer et al.'s [ICML 2023 Kernel Method of Moments](https://proceedings.mlr.press/v202/kremer23a.html) belongs in the discussion, but is not a direct density-estimation baseline without a further problem-specific construction.

## Resolve these issues before rerunning

These observations follow from the uploaded manuscript's equations; the experiment code was not supplied.

1. **Valid probability density and sparsity.** The current constraints only implement absolute values. They permit negative coefficients and do not enforce unit mass. Adding both nonnegativity and unit sum makes the uniform L1 norm exactly one, so simply appending those constraints removes its intended penalization effect. One concrete research option is to use nonnegative penalized fitting to select a support, then refit on that support under simplex constraints. Another is a budget-controlled simplex method. Both change the estimator, must be documented, and require fresh results. Do not silently repair existing results by clipping curves.

2. **Covariance estimation without oracle access.** For fixed measurement functions, estimate the covariance of the sample mean from the available training data:

   \[
   \widehat C=\frac{1}{N(N-1)}\sum_{n=1}^{N}(k(x_n)-\bar k)(k(x_n)-\bar k)^T.
   \]

   If measurement locations/bandwidths are chosen using the same sample, this is a plug-in estimate that does not capture all feature-selection uncertainty; disclose that or use sample splitting. Bootstrapping observed training points is feasible, but drawing fresh observations from the true simulation distribution solely to estimate C gives privileged information. Identify whether the existing experiments did this.

3. **Rank deficiency.** For m features, this empirical covariance has rank at most min(m, N−1). Increasing bootstrap repetitions does not remove the span limitation imposed by the original observations. A null direction of an empirical covariance is not automatically a deterministic identity of the population distribution. The proposed exact residual constraints in the current Remarks section therefore need justification and may be infeasible under misspecification. Prefer a clearly specified, validated regularization scheme in new experiments.

4. **Objective and penalty scaling.** Inverting Cov(sample mean) rather than Cov(single feature vector) multiplies the quadratic term by N. Scale or retune the sparsity penalty accordingly. The existing common value gamma = 7 does not establish fair regularization across AKDE and SPADES.

5. **Identification and theoretical wording.** A statement that m ≥ M is always required is too strong when sparse recovery is considered: identification can use additional structure, with suitable restricted conditions. There is also no distribution-independent threshold at moment order 6–8 beyond which raw moments fail. Both claims remain in the original methods discussion and need revision or supporting assumptions. The new literature review avoids them.

6. **Scope of novelty.** The most defensible candidate contribution is the tested combination of localized multiscale moments, stable covariance weighting, and sparse representation. Convex kernel moment matching and Gaussian integral identities are established. The supplied equations alone do not establish a new consistency, efficiency, or sparse-recovery theorem.

## Proposed evaluation protocol

**Data.** Recover and explicitly publish the parameters of all 15 existing densities. Do not assume their f0–f14 labels correspond to a standard benchmark without checking. If they are the Marron–Wand suite, cite [Marron and Wand, 1992](https://doi.org/10.1214/aos/1176348653) and give the mapping. Add two- and five-dimensional full-rank examples with unequal scales, overlapping modes, and skewness. Keep a separate stress test for outliers and rare narrow modes. Investigate the existing f5 error, where the table reports AKDE = 3.1 and SPADES = 0.24, rather than averaging it away.

**Replication.** Suggested sample sizes: 50, 100, 200, 500, 1,000, and 5,000. Start with 30 independently generated datasets per condition, then increase replication if confidence intervals remain too wide for the intended conclusion. Give every estimator the same data realization and validation split. Multiple optimizer initializations on one dataset are not independent statistical replications. Separate variability from dataset sampling and optimization.

**Tuning.** Choose centers, bandwidth ranges, covariance regularization, sparsity penalties, component counts, and training stopping rules using training/validation data only. For small samples, cross-validation avoids wasting a large holdout set. A held-out integrated-square score can be used even for signed estimates:

\[
\widehat R(f)=\int f(x)^2\,dx-\frac{2}{n_v}\sum_{j=1}^{n_v}f(x_j^{\mathrm{val}}).
\]

Do not use the true synthetic density to choose hyperparameters in the main comparison. A separately labeled oracle diagnostic is acceptable for studying dictionary approximation error. Report tuning ranges and computation budgets, rather than claiming equal effort from an equal number of parameters.

**Primary accuracy.** Use integrated squared error (ISE), not an unspecified grid MSE. For Gaussian-mixture truth and Gaussian dictionary estimates, obtain ISE analytically from pairwise Gaussian overlaps. For other estimators, document numerical integration domains and convergence checks. If using an equally spaced grid, include the grid spacing in the integral approximation. Report per-density results and confidence intervals for paired differences; also show median errors when catastrophic failures dominate means.

**Additional measures.** Report held-out negative log likelihood only for valid normalized densities, together with any numerical floor. Include integrated absolute error or Hellinger distance for valid densities, total mass and negative-mass diagnostics, fitting time including covariance construction, tuning time separately, peak memory, query latency, and active representation size. A count of nonzero coefficients must use a stated tolerance. Plot ISE against representation size and end-to-end runtime.

**Core ablations.** Compare identity, diagonal, ridge-regularized full, and truncated-pseudoinverse weighting with the same dictionary and validation protocol. Compare one bandwidth versus multiple bandwidths, no sparsity versus support selection, and matched versus independently designed measurement/basis families. Vary the number of moments relative to N and dictionary size. Report covariance condition numbers or effective rank and solver failure rates. Call the diagonal experiment an ablation of your estimator, not “Zhang et al. DGMM.”

**DGMM-specific extension.** Its singular Gaussian components may lack a density with respect to full-dimensional Lebesgue measure. Use compatible distribution/parameter metrics for that experiment. If adding isotropic noise to make ordinary ISE or likelihood meaningful, apply it consistently and identify this as a modified benchmark. Do not compare singular and nonsingular models through incompatible density evaluations.

## What changed in the LaTeX manuscript

The placeholder Related Works section is replaced by an approximately 1,700-word review in eight subsections. Thirteen references were added, Kremer et al.'s venue was corrected to ICML 2023, the SPADES citation was completed, and the existing DGMM reference was retained as a preprint. Two unsuitable references were removed: the student project report and the case-control variable-selection paper previously cited for EM. There are now 26 bibliography entries, including retained background references.

Two introductory claims identifying the proposed estimator with maximum likelihood were corrected, and the existing paragraph mischaracterizing Kernel Method of Moments as Gaussian-mixture estimation was replaced. Original optimization equations and reported numerical results were retained. The methodological checks above are unresolved author actions, not implemented corrections.

All citation keys resolve. A temporary validation copy compiled with placeholders for the 30 figure files that were not attached; the delivered source retains the original figure filenames. Place the original images beside the revised source in the LaTeX project to compile the complete paper. The references use the manuscript's existing thebibliography format, so no separate .bib file is required.
