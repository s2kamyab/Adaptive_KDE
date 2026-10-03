# Diagnostic tools

Single-purpose scripts for investigating *why* the estimator behaves as it does.
They print to the terminal and write nothing, so they are safe to run at any time.
None of them is part of the reported comparison — several deliberately use the true
density and are labelled below as **oracle**.

Run from this directory:

```powershell
$PY = "C:\dev\Adaptive_KDE_project\.venv\Scripts\python.exe"
& $PY sweep_dictionary_size_multi.py
```

| Script | Question it answers |
|---|---|
| `profile_estimators.py` | What does each estimator cost, and what ISE does it reach, on one draw? Run this after any change that could affect speed. |
| `profile_large_n.py` | Is one replicate at N=1000 / N=5000 affordable? Use it to size a run before launching hours of compute. |
| `inspect_lambda_path.py` | How do support size, J-score and true ISE vary along the λ path? Shows directly whether the penalty is doing anything. |
| `compare_cv_vs_ise.py` | Does cross-validation pick a good λ? Prints the CV choice beside the ISE-optimal choice, so tuning failure is distinguishable from estimator failure. |
| `sweep_weighting.py` | **Oracle.** Compares identity / diagonal / ridge / truncated-pseudoinverse weighting and matched vs independent measurement banks, against the oracle dictionary bound. |
| `sweep_measurement_bank.py` | How much does accuracy improve as the measurement bank grows? Establishes where `m` stops mattering. |
| `fit_exact_moments.py` | **Oracle.** Fits the noise-free moment vector. Separates "the estimator is chasing sampling noise" from "these moment conditions do not identify the density". |
| `sweep_conditioning.py` | Is the near-singular design caused by a coarse measurement bank or by packing too many scales per location? |
| `sweep_dictionary_size.py` | Cost and accuracy against dictionary size, one density. |
| `sweep_dictionary_size_multi.py` | The same across three densities and three draws — use this one before changing a default. |

For the oracle *bound* used in the manuscript (best achievable simplex fit on a given
dictionary) and the support-versus-criterion error decomposition, use the packaged
experiment instead, which writes CSVs:

```powershell
& $PY ../experiments/run_diagnostics.py --targets f5,f9,f13 --out ../results/diagnostics
```

## What these tools established

Recorded here because each conclusion is reproducible with one command above.

1. **The dictionary is not the limitation.** `sweep_weighting.py` on MW10 at N=200 gives an
   oracle bound of ISE 1.4e-5 with 140 atoms, against 1e-1 for the fitted estimator.
2. **The moment design matrix is numerically singular.** `run_diagnostics.py` reports
   cond(D_S) ≈ 1e17 at N = 200, 1000 and 5000 alike.
3. **The fitting criterion costs 3–10× beyond support selection.** Refitting the *same*
   selected support L²-optimally instead of moment-optimally takes ISE from 1.08e-1 to
   1.68e-2 at N=200.
4. **Exact moments do not fix it.** `fit_exact_moments.py` still plateaus around 2e-2,
   so part of the gap is identification, not sampling noise.
5. **Dictionary size matters more than expected, and in the opposite direction.**
   `sweep_dictionary_size.py` on MW10: M=600 reaches ISE 1.7e-2 at N=5000 where M=1500
   reaches 2.2e-1. An over-complete dictionary packs near-duplicate atoms whose moment
   signatures coincide, which is what destroys the conditioning.
6. **Multi-threaded BLAS changed the answer.** Because the design is near-singular, a
   last-bit difference in summation order altered atom selection and moved ISE by 3×.
   All experiment entry points now pin BLAS to one thread; `tests/test_reproducible.py`
   checks it.
