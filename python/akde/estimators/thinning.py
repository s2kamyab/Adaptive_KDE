"""Distribution compression followed by kernel smoothing.

The recent-compression comparator.  Thinning alone returns a discrete summary,
so -- as the manuscript notes -- an explicit smoothing rule has to be supplied
before it is a density estimator.  Here the retained points carry equal weight
in a Gaussian KDE whose bandwidth is cross-validated on the *retained* set,
which keeps the whole pipeline training-only.

Two compression routines are implemented, and which one produced a given
number is always recorded, because "we used kernel thinning" is not a
reproducible statement on its own:

``kt``
    Kernel thinning (Dwivedi & Mackey, JMLR 2024): recursive **kernel
    halving** followed by **KT-swap** refinement, with a Gaussian kernel.
    Halving splits the current point set into two balanced halves using the
    self-balancing Hilbert walk of their Algorithm 1; swap then greedily
    exchanges points to reduce the MMD to the full sample.  This is a direct
    implementation, not a wrapper around the authors' ``goodpoints`` package,
    so it should be described as "kernel thinning as specified in Algorithm
    1-2" rather than as their reference software.

``herding``
    Greedy kernel herding, i.e. repeatedly add the point that most reduces the
    MMD to the empirical measure.  A well-defined, easily audited compression
    baseline at an exactly prescribed budget.

Budgets are given as target support sizes so the accuracy-versus-representation
-size curve can be compared against the proposed estimator at matched sizes,
which is the comparison the recommendations ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..bandwidths import silverman, select_bandwidth
from ..metrics import j_score
from ..model import FittedDensity, mixture_density
from .base import Estimator, TuningConfig, kfold_indices

__all__ = ["ThinnedKDE", "ThinningConfig", "kernel_halving", "kt_swap", "kernel_herding"]


def _gauss_kernel(A: np.ndarray, B: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Gaussian kernel matrix, filled in row blocks.

    Written into a preallocated (n, m) array rather than built from an
    (n, m, d) difference tensor.  Thinning needs the full kernel matrix by
    construction, so at N = 5000 the result is already 200 MB; the naive
    expression transiently holds three arrays that size, which is what
    exhausted memory when many workers ran in parallel.
    """
    n, m = A.shape[0], B.shape[0]
    out = np.empty((n, m), dtype=float)
    block = max(1, int(4_000_000 // max(m, 1)))
    inv = 1.0 / bw
    for start in range(0, n, block):
        stop = min(start + block, n)
        diff = (A[start:stop, None, :] - B[None, :, :]) * inv[None, None, :]
        np.exp(-0.5 * np.einsum("ijk,ijk->ij", diff, diff), out=out[start:stop])
    return out


def kernel_halving(X: np.ndarray, bw: np.ndarray, rng: np.random.Generator,
                   K: np.ndarray | None = None) -> np.ndarray:
    """One round of self-balancing kernel halving (Dwivedi & Mackey, Alg. 1).

    Points are considered in random pairs; a running "swap" vector accumulates
    the kernel discrepancy contributed by the choices made so far, and each
    pair is resolved by a randomised rule whose acceptance probability is set
    so the accumulated discrepancy stays sub-Gaussian.  Returns the indices of
    the retained half (indices into ``X``).

    ``K`` may be supplied to avoid rebuilding the kernel matrix on every round
    of a recursive thinning schedule.
    """
    n = X.shape[0]
    idx = rng.permutation(n)
    if n % 2 == 1:
        idx = idx[:-1]
    pairs = idx.reshape(-1, 2)
    m = pairs.shape[0]

    if K is None:
        K = _gauss_kernel(X, X, bw)
    keep = np.empty(m, dtype=np.int64)
    # sigma controls the sub-Gaussian growth of the running discrepancy
    sigma = 0.0
    # running difference between the retained set and the full set, tracked
    # through kernel evaluations only
    acc = np.zeros(n)

    for t in range(m):
        i, j = int(pairs[t, 0]), int(pairs[t, 1])
        diff_vec = K[i] - K[j]
        # discrepancy contributed if i is kept rather than j
        a = float(acc @ diff_vec)
        b = float(np.sqrt(max(diff_vec @ diff_vec, 1e-300)))
        if b <= 0:
            prob = 0.5
        else:
            sigma = float(np.sqrt(max(sigma ** 2 + b ** 2 * (1.0 - min(1.0, abs(a) / (b * max(sigma, b)))), 0.0)))
            thresh = max(sigma * b, 1e-300)
            prob = 0.5 * (1.0 - np.clip(a / thresh, -1.0, 1.0))
        if rng.random() < prob:
            keep[t] = i
            acc += 0.5 * diff_vec
        else:
            keep[t] = j
            acc -= 0.5 * diff_vec
    return keep


def _mmd_sq(idx: np.ndarray, K: np.ndarray, k_bar: np.ndarray, k_mean: float) -> float:
    """Squared MMD between the uniform measure on ``idx`` and the full sample."""
    sub = K[np.ix_(idx, idx)]
    m = idx.size
    return float(sub.sum() / (m * m) - 2.0 * k_bar[idx].mean() + k_mean)


def kt_swap(idx: np.ndarray, X: np.ndarray, bw: np.ndarray, K: np.ndarray,
            max_rounds: int = 3) -> np.ndarray:
    """KT-swap refinement: greedily exchange retained points to reduce MMD.

    The swap decisions are the same greedy ones as a direct implementation
    that recomputes ``_mmd_sq`` for every candidate, but the MMD of a
    one-element exchange is obtained by an incremental update instead.

    Writing ``cs[x] = sum_{l in S} K[x, l]`` and ``SS = sum_{i in S} cs[i]``,
    replacing retained point ``c`` (at position ``pos``) by an outside point
    ``j`` gives, with ``|S|`` unchanged,

        SS(j) = SS - 2 cs[c] + K[c, c] + 2 (cs[j] - K[j, c]) + K[j, j],

    so every candidate costs O(1) and a whole position is scored in one
    vectorised pass over the outside set.  A pass over all positions is
    therefore O(m n), which is what the original docstring claimed; the
    previous version rebuilt the m x m submatrix per candidate and was
    O(m^2 n) times an m x m sum, i.e. O(m^3 n).  That cost is the only
    reason swap had to be switched off above a few hundred points, which
    silently made this estimator *halving only* at the larger sample sizes
    while it stayed *halving + swap* at the smaller ones -- two different
    algorithms reported in one row.
    """
    n = X.shape[0]
    k_bar = K.mean(axis=1)
    k_mean = float(K.mean())
    idx = np.array(sorted(set(idx.tolist())), dtype=np.int64)
    m = idx.size
    if m == 0 or m >= n:
        return idx

    inside = np.zeros(n, dtype=bool)
    inside[idx] = True
    diagK = np.diag(K).copy()
    all_j = np.arange(n)

    for _ in range(max_rounds):
        improved = False
        # Recomputed once per round rather than carried across rounds, so the
        # incremental updates below cannot accumulate drift.
        # chunked so the (n, m) slice is never materialised: at N = 5000 that
        # temporary is 100 MB on top of the 200 MB kernel matrix, and there are
        # ten of these workers.
        cs = np.zeros(n)
        for s in range(0, m, 512):
            cs += K[:, idx[s:s + 512]].sum(axis=1)
        SS = float(cs[idx].sum())
        kbar_sum = float(k_bar[idx].sum())
        best = SS / (m * m) - 2.0 * kbar_sum / m + k_mean

        for pos in range(m):
            cur = int(idx[pos])
            out = all_j[~inside]
            if out.size == 0:
                break
            ss_base = SS - 2.0 * cs[cur] + diagK[cur]
            ss_new = ss_base + 2.0 * (cs[out] - K[out, cur]) + diagK[out]
            kb_new = kbar_sum - k_bar[cur] + k_bar[out]
            vals = ss_new / (m * m) - 2.0 * kb_new / m + k_mean

            b = int(np.argmin(vals))
            if vals[b] < best - 1e-15:
                new = int(out[b])
                cs += K[:, new] - K[:, cur]
                SS = float(ss_new[b])
                kbar_sum = float(kb_new[b])
                best = float(vals[b])
                inside[cur] = False
                inside[new] = True
                idx[pos] = new
                improved = True
        if not improved:
            break
    return idx


def kernel_herding(X: np.ndarray, budget: int, bw: np.ndarray) -> np.ndarray:
    """Greedy MMD minimisation: add the point that most reduces MMD each step."""
    n = X.shape[0]
    K = _gauss_kernel(X, X, bw)
    k_bar = K.mean(axis=1)
    chosen: list[int] = []
    running = np.zeros(n)
    for t in range(min(budget, n)):
        # objective for adding point i:  (t/(t+1)) * running_i + K_ii/(t+1) - k_bar_i
        score = k_bar - (running * t + np.diag(K) / 2.0) / (t + 1.0)
        score[chosen] = -np.inf
        i = int(np.argmax(score))
        chosen.append(i)
        running = (running * t + K[i]) / (t + 1.0)
    return np.array(chosen, dtype=np.int64)


@dataclass
class ThinningConfig:
    method: str = "kt"                # 'kt' | 'herding'
    #: target retained fractions; 'kt' rounds to the nearest power-of-two halving
    budgets: tuple[float, ...] = (0.5, 0.25, 0.125)
    bw_mults: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    kernel_mult: float = 1.0          # thinning kernel bandwidth, x Silverman
    swap_rounds: int = 2
    #: Swap is genuinely O(m n) per pass now (see :func:`kt_swap`), so it runs
    #: at every sample size.  It used to be gated at 600, which meant the
    #: estimator was "halving + swap" for N <= 500 and "halving only" for
    #: N >= 1000 -- an algorithm change across the very axis the results are
    #: reported against.  Kernel thinning is halving *and* swap; running both
    #: everywhere is also the faithful reading of Dwivedi & Mackey Alg. 1-2.
    max_n_for_swap: int = 1_000_000


class ThinnedKDE(Estimator):
    """Compress the sample, then build a normalised KDE on the retained points."""

    def __init__(self, config: ThinningConfig | None = None,
                 tuning: TuningConfig | None = None) -> None:
        super().__init__(tuning)
        self.config = config or ThinningConfig()
        self.name = f"Thin[{self.config.method}]+KDE"
        self.provenance = (
            "Kernel thinning (kernel halving + KT-swap, Dwivedi & Mackey 2024, "
            "Algorithms 1-2), implemented directly rather than via the authors' "
            "goodpoints package; Gaussian thinning kernel; retained points smoothed "
            "by an equal-weight Gaussian KDE with cross-validated bandwidth."
            if self.config.method == "kt" else
            "Greedy kernel herding compression; retained points smoothed by an "
            "equal-weight Gaussian KDE with cross-validated bandwidth."
        )

    def candidates(self, X: np.ndarray) -> list[dict]:
        return [{"budget": b, "bw_mult": m}
                for b in self.config.budgets
                for m in self.config.bw_mults]

    def _compress(self, X, budget_frac, rng) -> tuple[np.ndarray, float]:
        cfg = self.config
        n, d = X.shape
        bw = np.array([silverman(X[:, k]) for k in range(d)]) * cfg.kernel_mult
        bw = np.where(np.isfinite(bw) & (bw > 0), bw, 1.0)

        t0 = perf_counter()
        target = max(1, int(round(budget_frac * n)))
        if cfg.method == "herding":
            idx = kernel_herding(X, target, bw)
            return idx, perf_counter() - t0

        # Kernel thinning: repeated halving down to the budget, then swap.
        # The kernel matrix is built once over all N and sub-indexed per round;
        # it is O(N^2) either way, but rebuilding it on every halving round both
        # wasted time and doubled peak memory.
        K_full = _gauss_kernel(X, X, bw)
        idx = np.arange(n)
        while idx.size > target * 1.5 and idx.size > 2:
            sub = kernel_halving(X[idx], bw, rng, K=K_full[np.ix_(idx, idx)])
            idx = idx[sub]
        if idx.size > target:
            idx = idx[:target]
        if cfg.swap_rounds > 0 and n <= cfg.max_n_for_swap and idx.size > 0:
            idx = kt_swap(idx, X, bw, K_full, max_rounds=cfg.swap_rounds)
        return idx, perf_counter() - t0

    def fit_path(self, X, cands, rng) -> list[FittedDensity]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        d = X.shape[1]

        out = []
        cache: dict[float, tuple[np.ndarray, float]] = {}
        for c in cands:
            b = float(c["budget"])
            if b not in cache:
                cache[b] = self._compress(X, b, rng)
            idx, comp_time = cache[b]
            pts = X[idx]
            m = pts.shape[0]

            t1 = perf_counter()
            base = np.array([
                select_bandwidth(pts[:, k], "sj") if m >= 5 else silverman(X[:, k])
                for k in range(d)
            ])
            h = base * float(c["bw_mult"])
            h = np.where(np.isfinite(h) & (h > 0), h, silverman(X[:, 0]))
            bw_time = perf_counter() - t1

            out.append(
                mixture_density(
                    np.full(m, 1.0 / m), pts, np.tile((h ** 2)[None, :], (m, 1)),
                    support=idx,
                    n_selected=m,
                    n_dictionary=int(X.shape[0]),
                    budget_frac=b,
                    bw_mult=float(c["bw_mult"]),
                    thinning_method=self.config.method,
                    setup_time_s=comp_time,
                    stage1_time_s=bw_time,
                    stage2_time_s=0.0,
                )
            )
        return out
