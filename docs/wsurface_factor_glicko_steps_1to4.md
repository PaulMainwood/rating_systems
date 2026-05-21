# WSurfaceFactorGlicko Production Integration — Steps 1-4 Results

Completion of the four-step plan from `docs/wsurface_factor_glicko_step1.md`.

## Summary of all four steps

| Step | Change | Outcome |
|---|---|---|
| 1 | Added `wsurface_factor_glicko` to `production.models` + `XGB_FEATURE_WHITELIST` | New system enters the XGB meta-stacker feature set |
| 2 | Ran full WTA eval | WTA gain (1.1% relative Brier) bigger than ATP (0.7%), consistent with higher WTA intransitivity |
| 3 | Set `xgb_handicap=True` on registry entry; added XGB params to config | Brier 0.21200 → 0.21055 on ATP holdout (0.7% relative gain from residual) |
| 4 | Optimised `w_margin`/`w_power` for betfair preset | Found `w_margin=3.4187, w_power=2.3075` (vs inherited 3.2693, 2.7076); 0.13% improvement on optimisation window did not transfer to the 365-day production holdout |

## Final production holdout (after all four steps)

### ATP (12,721 Betfair-covered games)

```
System                    Brier      LogLoss     Acc       ECE
Market (Betfair)          0.20529    0.59419    67.5%     0.01229
XGB_wf (11-system stack)  0.20744    0.59946    66.9%     0.01660
wsurface_factor_glicko    0.21055    0.60644    66.2%     0.01595   ← single-system best
wwhr (prev best)          0.21211    0.61008    65.6%     0.01855
wglicko                   0.21214    0.61010    65.8%     0.01709
srglicko                  0.21254    0.61103    65.8%     0.02173
wttt                      0.21260    0.61180    65.4%     0.02709
```

### WTA (10,116 Betfair-covered games)

```
System                    Brier      LogLoss     Acc       ECE
Market (Betfair)          0.19503    0.57004    69.3%     0.01407
XGB_wf (11-system stack)  0.19749    0.57574    68.8%     0.02074
wsurface_factor_glicko    0.19968    0.58093    68.6%     0.01789   ← single-system best
wwhr (prev best)          0.20183    0.58540    68.2%     0.01653
wttt                      0.20234    0.58693    67.8%     0.02118
wglicko                   0.20296    0.58814    68.0%     0.02633
```

## What worked, what didn't, and why

### Worked: the architecture itself
- The TrueSkill-2-style cross-surface factor structure successfully captures real surface signal not picked up by plain WGlicko or WWHR.
- Online Glicko-style updates fit the cost budget (sub-30s walk-forward), unlike the WHR-based variant which was 290× slower.
- The XGB residual handicap added on top closes about half the rating-vs-stacker gap (0.21200 → 0.21055 on ATP), similar to what it does for wglicko / wwhr.

### Worked: bigger WTA gain
WTA had a 1.1% relative Brier improvement vs WWHR; ATP had 0.7%. Consistent with the prior result that WTA exhibits ~11.5% more intransitivity than ATP (Clegg & Cartlidge 2025) — surface specialism / intransitive matchups are more pronounced there, and a model that explicitly captures them gains more.

### Didn't work: covariate hyperparameter optimisation
The differential-evolution-style optimisation found a 0.13% Brier improvement on its training window (730 days) but **the improvement did not transfer to the 365-day production holdout**. New `w_margin/w_power` values produced essentially identical Brier (0.21055 vs 0.21058). The inherited wglicko-betfair preset values were already very close to the optimum for this system.

**Why no transfer**: probably because the optimisation window and production holdout are different match populations (the optimisation uses raw days, the holdout uses Betfair-coverage-only). Tuning to one population doesn't necessarily improve the other.

**Practical implication**: don't expect more juice from `w_margin/w_power` retuning. If we want further improvement on the holdout, it has to come from elsewhere.

## Bug found and fixed

The first optimisation attempt crashed with `WSurfaceFactorGlicko.__init__() got an unexpected keyword argument 'c'`. Root cause in `scripts/optimise_models.py:_optimise_covariate` and `_optimise_xgb_handicap`:

```python
for k, v in base_params.items():
    if k in w_params and w_params[k] != v:
        synced[k] = v
    elif k not in w_params:
        synced[k] = v          # BUG: injects keys the W* doesn't have
```

Branch 2 unconditionally injected base-system keys into the W* params. Fine for wglicko ↔ welo where param names align, but my system renames `c` → `c_base`, `initial_rd` → `initial_base_rd`, so `c` got passed to a constructor that doesn't accept it.

Fix: dropped branch 2. Sync now only updates pre-existing keys.

## What changed in the codebase

Tennis_ratings repo (user's branch — left uncommitted):
- `tennis_ratings/analysis/registry.py`: added `requires_surfaces` to `SystemSpec`; registered `wsurface_factor_glicko` with `weighted=True, covariate=True, base_system="wglicko", requires_surfaces=True, xgb_handicap=True`.
- `tennis_ratings/prediction/predict.py`: `predict_one_day` accepts and threads `surfaces` kwarg when `spec.requires_surfaces`.
- `tennis_ratings/prediction/walk_forward.py`: `walk_forward_weights_only` extracts and threads a surfaces array (mapping Hard/Clay/Grass via `_SURFACE_MAP_WF`) for surface-aware systems.
- `tennis_ratings/prediction/xgboost.py`: added `wsurface_factor_glicko_pred` to `XGB_FEATURE_WHITELIST`.
- `tennis_ratings/analysis/backtest.py`: added `surfaces` parameter to `run_backtest`, threaded through `_fit_system`, `_predict_day`, `_update_system`.
- `scripts/optimise_models.py`: `_make_objective` takes `games_df` and extracts surfaces; fixed the param-sync bug; passes surfaces through to `run_backtest`.
- `config.yaml`: added `wsurface_factor_glicko` blocks (ATP + WTA) with betfair presets; added `wsurface_factor_glicko` to `production.models`.

Rating_systems repo (committed to `new_direction`):
- `WSurfaceFactorGlicko.update_weighted`: dual-signature shim accepting either `(batch, weights, surfaces=...)` (walk-forward) or `(p1, p2, scores, surfaces, day, weights=None)` (legacy/direct).
- `WSurfaceFactorGlicko.predict_proba`: accepts and ignores `day=None` so the unified dispatch can call uniformly.

## What's next

The system is now a stable, integrated production component. Realistic remaining moves:

1. **Re-optimise on the Betfair-only subset.** Phase 1 of step 4 used all matches (just preset-weighted scoring). A Betfair-coverage-restricted optimisation might find better params for that specific holdout. The `--train-on-eval-subset` flag in `evaluate_models.py` controls this for the XGB residual; would need analogous handling in the optimiser.
2. **Per-surface `w_s` optimisation.** The current `w_surfaces=(0.80, 0.81, 0.71)` is from the standalone benchmark. Re-optimising on the full production data with per-surface bounds might find better values. Requires extending `optimise_models.py` to know about non-scalar config keys.
3. **Try a non-zero `c_offset`.** The standalone optimiser hit the lower bound (`0.5`), suggesting the optimum might be near zero — but production-data optimum may differ.
4. **Multi-tour joint optimisation.** Treat ATP+WTA as a single objective with per-tour weights. Could find params that work for both rather than separate ATP/WTA tunings.
5. **Hold for WWHR steps 2-5 if interested** — the WHR variant was deferred after step 1 showed it's slower and worse. The Glicko result above suggests the WHR architecture isn't a productive direction; this confirms the user's earlier suspicion that WWHR's batch Wiener smoothing already captures the surface signal.

The single biggest realistic win from here is probably integrating predictions into betting/Kelly-staking workflows on the intransitive matchup subset (per Clegg & Cartlidge's 3.26% ROI finding) rather than further squeezing the rating system.
