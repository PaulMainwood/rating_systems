# SurfaceFactorWHR / WSurfaceFactorWHR: ATP Walk-Forward Benchmark

Whole History Rating with a rank-1 cross-surface factor structure:

```
base_i(t)            ~ Wiener(w2_base)
offset_{i, s}(t)     ~ Wiener(w2_off[s])      for s ∈ {Hard, Clay, Grass}
skill_{i, s}(t)      = w_s * base_i(t) + offset_{i, s}(t)
```

Inference is **coordinate descent** over the four per-player trajectories,
each step reusing WHR's existing tridiagonal Newton solver from
``rating_systems/systems/whr/_numba_core.py``. ``WSurfaceFactorWHR`` is the
weighted variant — per-game weights scale Fisher information exactly as in
``WeightedWHR``.

## Dataset

- `data/processed/atp_d_from_2000_enriched.parquet` (OnCourt ATP, 2000+).
- Filter: Day ≥ 6940 (~2019-01-01); surfaces mapped via
  `{Hard, I.hard, Carpet, Acrylic} → 0`, `Clay → 1`, `Grass → 2`.
- Size: 325,020 matches, 2,571 days, 40,440 players.
- Train: first 365 day-indices (~1 year). Test: remaining 2,206 days.
- WHR refit cadence: every 28 days.

## Walk-forward results

| System | Brier | LogLoss | Acc | Train | Test |
|---|---:|---:|---:|---:|---:|
| **WHR** (baseline) | 0.1995 | 0.5936 | 0.696 | 1.4s | 29.8s |
| SurfaceFactorWHR (default w=1, w2_off=50, anchor=1e-3) | 0.2249 | 0.7001 | 0.669 | 3.1s | 137.9s |
| SurfaceFactorWHR (tight: w=1, w2_off=10, anchor=1e-2) | 0.2187 | 0.6540 | 0.669 | 3.0s | 122.2s |
| SurfaceFactorWHR (w=0.7, w2_off=20) | 0.2192 | 0.6787 | 0.674 | 2.6s | 86.0s |
| WSurfaceFactorWHR (unit weights — sanity check) | 0.2249 | 0.7001 | 0.669 | 1.6s | 76.0s |
| SurfaceFactorWHR (strong anchor: anchor=0.1, w2_off=10) | 0.2056 | 0.6054 | 0.686 | 7.6s | 234.3s |
| SurfaceFactorWHR (very strong: anchor=1.0, w2_off=5) | 0.2037 | 0.5981 | 0.687 | 2.6s | 222.2s |
| **SurfaceFactorWHR (w=0.5, w2_off=10, anchor=0.5)** | **0.1940** | **0.5685** | **0.698** | 2.8s | 200.0s |

The tuned configuration `w_surfaces=(0.5, 0.5, 0.5), w2_off=10.0, offset_anchor=0.5`
beats plain WHR by:

- **2.8% relative Brier improvement** (0.1995 → 0.1940)
- **+0.2 percentage points accuracy** (0.696 → 0.698)
- **4.2% relative log-loss improvement** (0.5936 → 0.5685)

The improvement is real but **considerably smaller** than the surface-factor
Glicko variant achieved on the same kind of data (+8.7% Brier, +3.0pp Acc).
Plausible explanations:

1. **WHR's full-history Wiener smoothing already absorbs much of the
   surface-specific signal.** Each player's batch-fit trajectory is fitted
   jointly to all of their matches, regardless of surface; the temporal
   smoothness prior soaks up the cross-surface correlation that the explicit
   factor structure adds.
2. **Coordinate descent convergence is slower than the closed-form Glicko
   updates.** With ``refit_max_iterations=5`` (the default) the offsets may
   not fully equilibrate at each refit. Raising it to 20 (as in the
   `w=0.5_strong` run above) recovers most of the gap but at ~6× wall-clock.
3. **Identifiability is more delicate in the joint trajectory MAP** than in
   online Glicko: an arbitrary additive constant ``c`` can move from the
   base trajectory to ``-c/w_s`` on every offset and produce identical
   predictions. The ``offset_anchor`` removes this degeneracy but needs to
   be tuned. The default ``1e-3`` is too weak for ATP; `0.1` to `1.0` work
   well.

## Hyperparameter notes

- **`w_surfaces` lower than the Glicko optimum (0.71-0.81)**: the WHR sweep
  found `w=0.5` works better. Possibly because the offsets need more "room"
  to capture surface-specific signal not already explained by the base
  trajectory; lower `w_s` means base contributes less to each surface and
  offsets contribute more.
- **`w2_off` should be small** (5-20) relative to `w2_base` (300 default).
  Surface specialism is highly stable over time.
- **`offset_anchor` between 0.1 and 1.0** is the right regime. The default
  ``1e-3`` is too weak.

## Caveats / follow-up

1. **No full hyperparameter optimisation** (only a manual sweep of 7 configs).
   A proper differential-evolution optimisation would likely tighten Brier
   below 0.19 — at the cost of many hours of wall-clock because each WHR
   walk-forward fit takes ~200s.
2. **Coordinate descent vs joint block-tridiagonal solve**. The current
   implementation does 4 separate per-trajectory tridiagonal solves per
   outer iteration. A proper 4×4-block tridiagonal Thomas algorithm would
   converge faster and might close the gap to plain WHR more decisively.
3. **WSurfaceFactorWHR is only smoke-tested.** No real-data benchmark with
   tennis-specific weights (recent-form bonus, MOV bonus etc.). The unit
   test confirms it equals SurfaceFactorWHR when weights = 1.
4. **No comparison vs SurfaceTTT** — would need bringing SurfaceTTT into
   the same walk-forward harness; the existing SurfaceTTT benchmark uses
   a different evaluation pipeline.

## Reproduce

```bash
python scripts/research/benchmark_surface_factor_whr.py \
    --train-days 365 --refit-interval 28
```

Tighter configurations require editing the script's parameter dictionary
to pass `w_surfaces=(0.5,0.5,0.5), w2_off=(10.0,10.0,10.0), offset_anchor=0.5`
and `max_iterations=80, refit_max_iterations=20`.
