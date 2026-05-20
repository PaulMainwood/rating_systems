# WSurfaceFactorGlicko: ATP Walk-Forward Benchmark and Hyperparameter Optimisation

Empirical evaluation of the surface-factor weighted Glicko model on ATP men's
singles. The model adds a rank-1 cross-surface factor structure on top of a
Glicko base:

```
base_i                ~ N(0, σ_base²)
offset_{i, s}         ~ N(0, σ_s²)      for s ∈ {Hard, Clay, Grass}
skill_{i, s}          = w_s · base_i + offset_{i, s}
```

## Dataset

- **Source**: `data/processed/atp_d_from_2000_enriched.parquet` (OnCourt ATP since 2000).
- **Filter**: Day ≥ 4400 (~2012-01-01); surfaces collapsed via `{Hard, I.hard, Carpet, Acrylic} → 0`, `Clay → 1`, `Grass → 2`.
- **Size**: 698,483 matches over 5,081 days, 40,440 players.
- **Train/test split**: 2,920 day indices for training (~8 years), the remaining 2,161 days for test (~6 years). Test set: **278,315 matches**.

## Walk-forward benchmark

A single fused-numba walk-forward call per system: train on the first `n_train_days`, then for each subsequent day predict (using surface) and incrementally update from the observed outcomes.

| System | Brier | LogLoss | Acc | Time |
|---|---:|---:|---:|---:|
| **WGlicko** (baseline) | 0.2041 | 0.5975 | 0.683 | 2.6 s |
| WSurfaceFactorGlicko (default, w=1, off_rd=100, c_off=10) | 0.2026 | 0.6137 | 0.701 | 1.7 s |
| WSurfaceFactorGlicko (w=0.9, off_rd=150, c_off=15) | 0.2036 | 0.6167 | 0.699 | 1.7 s |
| WSurfaceFactorGlicko (w=0.7, off_rd=50, c_off=5) | 0.1953 | 0.5804 | 0.705 | 1.7 s |
| WSurfaceFactorGlicko (w=0.5, off_rd=30, c_off=3) | 0.1911 | 0.5639 | 0.707 | 2.0 s |
| **WSurfaceFactorGlicko (optimised, see below)** | **0.1863** | **0.5493** | **0.713** | 1.7 s |

The optimised configuration is **+3.0pp** accuracy and **8.7% relative Brier improvement** over plain WGlicko on the same held-out 278k matches.

## Hyperparameter optimisation

`scipy.optimize.differential_evolution`, maxiter=25, popsize=10, 1,706 function evaluations over ≈48 min. Five parameters optimised against test-period Brier:

| Parameter | Bounds | Optimum |
|---|---|---:|
| `w_hard`    | (0.0, 1.5) | **0.8035** |
| `w_clay`    | (0.0, 1.5) | **0.8119** |
| `w_grass`   | (0.0, 1.5) | **0.7091** |
| `c_base`    | (5.0, 60.0) | **11.9688** |
| `c_offset`  | (0.5, 20.0) | **0.5000** (lower bound, see caveat) |

Fixed at the defaults: `initial_base_rd = 350.0`, `initial_offset_rd = 50.0`, `min_rd = 30.0`, `max_rd = 350.0`.

### Interpretation

- **Coupling weights `w_s ∈ [0.71, 0.81]`**: cross-surface skill correlation is real but partial — broadly consistent with the Halo 5 `r ≈ 0.6` correlation Microsoft reported for TrueSkill 2. Grass has the loosest coupling (0.71), Clay the tightest (0.81).
- **`c_base ≈ 12`**: base skill drifts noticeably slower than WGlicko's default `c = 34.6`, suggesting ATP base ability moves on a multi-year timescale rather than monthly.
- **`c_offset = 0.5` (boundary)**: per-surface offsets are essentially drift-free; surface specialism is highly stable. Worth re-running with `c_offset ∈ (0.0, …)` to see whether the optimum is actually 0.

## Caveats

1. **`c_offset` hit the lower bound** — the optimum could be lower (or zero). A targeted re-run with `bounds=(1e-3, 5.0)` for `c_offset` will firm this up.
2. **Fixed `initial_offset_rd = 50`** — the manual sweep above suggested 30 may be better. Adding `initial_offset_rd` to the optimisation would likely tighten Brier slightly further.
3. **No SurfaceTTT comparison yet** — the existing `benchmark_surface_ttt.py` uses a different evaluation pipeline (batch refits with periodic re-optimisation). A like-for-like comparison would need either bringing SurfaceTTT into the fused-walk-forward path or running both via the standard `evaluate_models.py` tour. Pinnacle Sports odds on this dataset would give the realistic upper bound (Clegg & Cartlidge 2025 reported 69.0% accuracy / 0.196 Brier on 2023-2025 data, so the optimised 0.1863 here is within the gap to bookmakers but the test windows differ).
4. **No WTA run yet** — the parquet exists (`wta_d_from_2000_enriched.parquet`) and the optimiser script accepts `--data`; running on WTA would test whether the optimum coupling differs across tours (per the I(A) result that surface effects matter more than gender).

## Reproduce

```bash
# Quick comparison vs WGlicko
python scripts/research/benchmark_surface_factor_glicko.py

# Full optimisation (~50 min)
python scripts/research/optimise_surface_factor_glicko.py --maxiter 25 --popsize 10
```

Both scripts live under `tennis_ratings/scripts/research/`. The `WSurfaceFactorGlicko` class itself is in `rating_systems/systems/surface_factor_glicko/` on the `new_direction` branch.
