#!/usr/bin/env bash
# Paper-grade experiment chain, run sequentially so suites do not compete for
# cores. Every stage is resumable: re-running skips shards already on disk.
#
#   bash run_paper.sh [jobs]
#
# Scoping (see EXPERIMENT_LOG.md):
#   main         all six sample sizes -- N=5000/1000/500 were already computed
#                before the rescope, so finishing the cheap sizes is ~1.7 h and
#                discarding the expensive ones would throw away finished work.
#   ablation     N in {100, 1000} only. It is a mechanism study (weighting,
#                scales, sparsity, refit, measurement design, dictionary size)
#                and two well-separated sample sizes answer it; at 11 AKDE
#                variants x 512 s/replicate the full size grid is ~28 h alone.
#   stress /
#   multivariate N <= 1000.
#
# Unlike the first version, a killed or failing RUN stage aborts the chain
# instead of silently walking on to the next suite.
set -u
JOBS="${1:-14}"
PY="C:/dev/Adaptive_KDE_project/.venv/Scripts/python.exe"
LOG="results/run_paper_full.log"
SIZES_SMALL="50,100,200,500,1000"
mkdir -p results

stage () {
  local label="$1"; shift
  echo "" | tee -a "$LOG"
  echo "======== BEGIN ${label}  $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a "$LOG"
  local t0=$(date +%s)
  "$PY" -u "$@" >>"$LOG" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "======== END   ${label}  rc=${rc}  ${dt}s ========" | tee -a "$LOG"
  # 130 = SIGINT, 137 = SIGKILL, 143 = SIGTERM, 127 = command vanished.
  if [ "$rc" -ge 126 ]; then
    echo "ABORTING CHAIN: ${label} was killed (rc=${rc})" | tee -a "$LOG"
    exit "$rc"
  fi
  return 0
}

stage "RUN main"          experiments/run_benchmark.py --suite main --tier full \
                            --jobs "$JOBS" --name main_full
stage "RUN ablation"      experiments/run_benchmark.py --suite ablation --tier full \
                            --sizes 100,1000 --jobs "$JOBS" --name ablation_full
stage "RUN stress"        experiments/run_benchmark.py --suite stress --tier full \
                            --sizes "$SIZES_SMALL" --jobs "$JOBS" --name stress_full
stage "RUN multivariate"  experiments/run_benchmark.py --suite multivariate --tier full \
                            --sizes "$SIZES_SMALL" --jobs "$JOBS" --name multivariate_full
stage "RUN realdata"      experiments/run_realdata.py --splits 30 --out results/realdata
stage "RUN diagnostics"   experiments/run_diagnostics.py --targets f5,f9,f13 \
                            --out results/diagnostics

stage "ANALYSE main"          experiments/analyze.py --run results/main_full \
                                --reference akde       --boot 10000
stage "ANALYSE ablation"      experiments/analyze.py --run results/ablation_full \
                                --reference akde_ridge --boot 10000
stage "ANALYSE stress"        experiments/analyze.py --run results/stress_full \
                                --reference akde       --boot 10000
stage "ANALYSE multivariate"  experiments/analyze.py --run results/multivariate_full \
                                --reference akde       --boot 10000

stage "FIGURES main"      experiments/make_figures.py --run results/main_full --fitted

echo "" | tee -a "$LOG"
echo "ALL STAGES FINISHED $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
