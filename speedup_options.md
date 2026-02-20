# Speed-up Options for WWHR and WTTT

## Current Performance Profile

| System | Refit iters/day | Walk-forward (1677 days) | Main bottleneck |
|--------|----------------|--------------------------|-----------------|
| **WWHR** | 20 | ~30-45 min | Newton-Raphson: sequential player loop × 20 iters |
| **WTTT** | 2 | ~2 min | Forward-backward sweeps (already fast) |

WWHR is the primary target. WTTT is already fast enough that most optimizations yield marginal gains.

### Where the time goes

**WWHR:** Each Newton-Raphson iteration loops over all ~32k players sequentially. For each player, it computes a gradient and Hessian (looping over all that player's games), then solves a tridiagonal system (Thomas algorithm). This happens 20 times per walk-forward day × 1677 days. The sigmoid computation in the inner game loop accounts for ~40-50% of runtime; the tridiagonal solve ~20-30%.

**WTTT:** Forward and backward sweeps are sequential across ~5,730 day-batches. Within each batch, ~126 games are processed with Gaussian operations (multiply, divide, truncated moments involving `sqrt`, `erf`). With only 2 refit iterations, the total work is modest.

---

## Tier 1: Numba/NumPy improvements (no GPU)

### 1. Jacobi parallelism for WWHR

**Expected speedup: 5-8x on 8 cores**

Currently `run_iteration_weighted` updates players sequentially (Gauss-Seidel style), reading opponents' ratings as they change in-place. Switch to Jacobi iteration: snapshot `pd_r` at the start of each iteration, compute all player updates against the snapshot in parallel via `prange`, then apply updates.

```python
@njit(cache=True, fastmath=True, parallel=True)
def run_iteration_jacobi_weighted(num_players, ..., pd_r, pd_r_snapshot, ...):
    pd_r_snapshot[:] = pd_r[:]
    max_change = 0.0
    for player_id in prange(num_players):
        change = update_single_player_weighted_jacobi(
            player_id, ..., pd_r_snapshot, pd_r,  # read snapshot, write live
        )
        if change > max_change:
            max_change = change
    return max_change
```

Trade-off: Jacobi converges slightly slower than Gauss-Seidel (~1.3-1.5x more iterations), but with 8 cores the net speedup is ~4-6x. Active-set optimization still works — partition active players across threads.

### 2. Pre-allocate per-player work arrays

**Expected speedup: ~1.3x**

`update_single_player_weighted` allocates `gradient`, `hess_diag`, `hess_off`, `delta` on every call — 4 allocations × 32k players × 20 iterations = 2.56M allocations per walk-forward day. Pre-allocate a pool sized to the max player-day count and pass work buffers in. For the Jacobi parallel version, allocate `num_threads` sets of work arrays.

### 3. Warm-start WWHR from WTTT ratings

**Expected speedup: 2-3x fewer iterations (refit_max_iterations: 20 → ~7)**

Since WTTT converges in 1-2 iterations and produces similar ratings, use WTTT's converged ratings to initialise WWHR's `pd_r`. The starting point would already be close to the optimum, reducing Newton-Raphson iterations substantially. This requires mapping WTTT's Gaussian beliefs to log-gamma scale — straightforward since both are on Elo-like scales.

### 4. Tune active-set reactivation

**Expected speedup: ~1.5x in late iterations**

Currently when a player's rating changes, all historical opponents are reactivated. More aggressive pruning:
- Only reactivate opponents from recent player-days (e.g., last 365 days)
- Skip opponents from games where the coupling is weak (large time gap → large Wiener variance → small influence)

This could reduce the active-set size by 50%+ in late iterations without affecting convergence quality.

### 5. L-BFGS global optimizer

**Expected speedup: 2-3x fewer iterations. Higher effort.**

The current approach is block coordinate descent (one player at a time). A global L-BFGS optimizer over the full `pd_r` vector captures cross-player coupling and converges in fewer iterations. The gradient is cheap (same game loop), and L-BFGS only needs gradient + function value, no Hessian. The tridiagonal Hessian could serve as a preconditioner for even faster convergence.

---

## Tier 2: GPU / PyTorch

### 6. PyTorch port of WWHR core loop

**Expected speedup: 20-50x vs single-core Numba**

The WWHR Newton step maps naturally to GPU tensor operations:

**Vectorised gradient/Hessian computation (replaces the entire nested loop):**
```python
r_diff = pd_r[game_player_pd] - pd_r[game_opp_pd]   # (n_game_refs,)
p_win = torch.sigmoid(r_diff)                         # GPU sigmoid
grad_contrib = weights * (scores - p_win)
hess_contrib = -weights * p_win * (1 - p_win)

gradient = torch.zeros(total_pd, device='cuda')
gradient.scatter_add_(0, game_to_pd_idx, grad_contrib)
hess_diag = torch.zeros(total_pd, device='cuda')
hess_diag.scatter_add_(0, game_to_pd_idx, hess_contrib)
```

This replaces the per-player, per-day, per-game nested loop with one vectorised pass over all ~1.4M game references.

**Batched tridiagonal solve:** Use cuSPARSE's `gtsv2StridedBatch` (via CuPy interop) or a custom CUDA Thomas algorithm kernel. Each player's tridiagonal system is independent — 32k systems solved in parallel.

**Wiener/virtual priors:** Vectorised with simple indexing into `pd_days` arrays.

### 7. PyTorch port of WTTT batch processing

**Expected speedup: 2-5x per batch (limited by sequential batch dependency)**

Within each batch, game processing can be vectorised:
```python
p1_prior = gaussian_mul_batch(fwd[game_p1], bwd[game_p1])
p2_prior = gaussian_mul_batch(fwd[game_p2], bwd[game_p2])
lik1, lik2 = compute_game_likelihoods_batch(p1_prior, p2_prior,
                                             scores, game_beta_eff[batch_games])
```

The fundamental limiter is that forward/backward sweeps are sequential across ~5,730 batches, each with only ~126 games on average. This barely saturates GPU hardware. Multi-day batching (grouping into 7-day windows) would increase parallelism to ~900 games per step, but with only 2 refit iterations the absolute gain is small.

### 8. Mixed precision (float32)

**Expected speedup: ~2x memory bandwidth, ~2x throughput on GPU**

Use float32 for intermediate gradient/Hessian computation, accumulating in float64 only where needed (e.g., tridiagonal solve). Halves memory footprint and doubles GPU throughput. For WWHR's convergence threshold of 1e-6, float32 precision (7 decimal digits) is sufficient for all but the final iterations.

---

## Tier 3: Algorithmic changes

### 9. Sliding window for WWHR

**Expected speedup: ~5x problem size reduction**

Instead of optimising over the full 5,730-day history each refit, maintain a sliding window of recent history (e.g., 365 days). Older ratings are frozen. This:
- Reduces the number of player-days by ~85%
- Reduces game references proportionally
- Still captures recent form accurately
- Trade-off: slightly worse for players returning after long absences

### 10. Stochastic block coordinate descent

**Expected speedup: ~2x per iteration (with Anderson acceleration)**

Instead of updating all players each iteration, randomly sample a subset. Anderson acceleration stabilises convergence. This is essentially stochastic Newton — each iteration is cheaper, though more iterations may be needed.

---

## Recommendation Summary

| # | Optimisation | Effort | Speedup | Applies to |
|---|-------------|--------|---------|------------|
| 1 | Jacobi parallel (`prange`) | Medium | 5-8x | WWHR |
| 2 | Pre-allocate work arrays | Low | 1.3x | WWHR |
| 3 | Warm-start from WTTT | Low | 2-3x fewer iters | WWHR |
| 4 | Tune active-set reactivation | Low | 1.5x | WWHR |
| 5 | L-BFGS global optimizer | High | 2-3x fewer iters | WWHR |
| 6 | PyTorch GPU (scatter_add + batched tridiag) | High | 20-50x | WWHR |
| 7 | PyTorch vectorised batch games | Medium | 2-5x per batch | WTTT |
| 8 | Mixed precision (float32) | Low | ~2x on GPU | Both |
| 9 | Sliding window (365 days) | Medium | ~5x | WWHR |
| 10 | Stochastic block coordinate descent | Medium | ~2x | WWHR |

**Quick wins (WWHR, no GPU):** Combining #1 + #2 + #3 could reduce walk-forward time from ~30 min to ~3-5 min.

**With sliding window:** Adding #9 could push WWHR under 1 minute.

**With GPU:** #6 alone could bring WWHR to under 1 minute on a single GPU; combining with #1-#4 makes the walk-forward near-instant.

**WTTT:** Already at ~2 min. The sequential batch dependency is the fundamental limiter. GPU port (#7) helps the initial fit but barely affects walk-forward refits.
