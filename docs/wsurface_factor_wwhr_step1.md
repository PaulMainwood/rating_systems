# WSurfaceFactorWWHR — Step 1 (Default-Params Evaluation)

**Status:** completed (negative result). The default-params surface-factor variant
of WWHR underperforms plain WWHR on the production evaluation pipeline.

## Setup

- ATP, full 15-year walk-forward (`wf_days = 5475`), final 365 days held out.
- `--score-preset betfair` (the production default), so display rows are
  evaluated on the 12,721-match Betfair-covered subset.
- Two systems compared end-to-end: `wwhr` (production, with XGB residual
  handicap applied during walk-forward) and the new `wsurface_factor_wwhr`
  (pure surface-factor weighted WHR — no XGB residual).

## Result

```
HOLDOUT EVALUATION — ATP — last 365 days (12,721 Betfair-covered games)
─────────────────────────────────────────────────────────────────────
System                    Brier      LogLoss     Acc       ECE
Market (Betfair)          0.20529    0.59419    67.5%     0.01229
wwhr                      0.21211    0.61008    65.6%     0.01855
wsurface_factor_wwhr      0.21745    0.62240    64.4%     0.02729   ← worse
```

The factor model trails plain WWHR by:
- Brier: +0.00534 absolute (2.5% relative worse)
- LogLoss: +0.01232 (2.0% relative worse)
- Accuracy: −1.2 pp
- ECE: +0.009 (markedly worse calibration)

Walk-forward time: **10,636 s (2 hr 57 min)** for the factor variant, vs
**1,390 s (23 min)** for plain WWHR — **7.6× slower** per evaluation.

## Why it underperformed

Three contributing factors:

1. **Covariate-param mismatch.** `wwhr` ran with the optimised
   `betfair`-preset values (`w_margin=1.109, w_power=1.92`) — those are
   conservative weights tuned for the Betfair-covered subset.
   `wsurface_factor_wwhr` ran with the all-preset values
   (`w_margin=2.585, w_power=1.97`) because no preset block was added
   yet. The much stronger weighting biased the new system toward weighting
   high-MOV matches more heavily, which hurts on the calmer Betfair
   subset.
2. **Default factor params from a different test window.** The defaults
   (`w_surfaces=0.5, w2_off=10, offset_anchor=0.5`) were chosen from the
   2019+ ATP slice benchmark (`docs/wsurface_factor_whr_benchmarks.md`).
   The full 15-year history has different surface dynamics and these
   defaults are not optimal here.
3. **WHR's batch Wiener smoothing already captures part of the surface
   signal.** The standalone benchmark only ever got a 2.8% relative
   Brier improvement vs plain WHR even with hand-tuned params — versus
   8.7% for the analogous Glicko system. The information ceiling for
   WHR-style surface factoring on this kind of data appears to be small.

## Cost / feasibility implications for optimisation

At 3 hours per evaluation, a `differential_evolution`-style optimisation of
five hyperparameters (~100 evaluations) would take ~12.5 days of wall-clock.
Not viable as-is. Three knobs can reduce the per-eval cost:

| Change | Speedup | Risk |
|---|---|---|
| `refit_max_iterations = 2` (match WWHR) | ~2.5× | Less converged per-day fit |
| `refit_interval = 7` (weekly refits) | ~7× | Stale ratings between refits |
| Both | ~17× → 10 min/eval | Both risks |

Even at 10 min/eval, a 100-call DE would take ~17 hours — feasible overnight.

## Recommended next move

Two viable continuations of the original plan, plus one pivot:

### A. Continue as planned, with cheaper per-eval params
- Add a `presets.betfair` block to `wsurface_factor_wwhr` matching `wwhr`'s
  betfair preset (`w_margin=1.109, w_power=1.92, w2_base=22.31`).
- Set `refit_max_iterations=2, refit_interval=7` in the config to make
  the walk-forward ~17× faster.
- Re-run step 1 — expected ~10 min instead of 3 hr.
- Then proceed to step 3 (covariate optimisation) and step 4 (surface-factor
  optimisation).
- **Expected upside:** modest — the standalone benchmark capped at +2.8%
  Brier even with tuned params. Best case here might be matching plain
  WWHR or beating it by ~1%.

### B. Pivot to wsurface_factor_wglicko (the Glicko variant)
- The Glicko-based system `WSurfaceFactorGlicko` already exists on this
  branch.
- Standalone benchmark showed **+8.7% relative Brier improvement** over
  WGlicko — much larger than the WHR-side gain.
- Glicko is fully online: walk-forward time should be ~2-3 min total,
  not 3 hours.
- Wire it into the registry as `wsurface_factor_wglicko` (same one-line
  pattern as the WHR variant), add to the eval pipeline, optimise.
- **Expected upside:** if the standalone benchmark holds, this could be
  the first single system to approach Market on the Betfair subset
  (current gap: 0.207 → 0.205 Brier; an 8% relative improvement of WWHR's
  0.212 → 0.195 would beat Market by 0.010 Brier).

### C. Investigate the architecture
- The 7.6× slowdown without quality improvement suggests the factor
  parameterisation may not match how WHR's Newton-Raphson is set up.
- Specifically, coordinate-descent over 4 trajectories may be much less
  efficient than the joint block-tridiagonal solve. A proper 4×4-block
  Thomas solver would be 1 outer iteration vs ~5.
- **Highest engineering cost, uncertain payoff.** Probably worth deferring.

## My recommendation

**Option B** — pivot to `wsurface_factor_wglicko`. The standalone evidence
is much stronger for the Glicko variant, walk-forward is far cheaper, and
the integration work is identical (we already built the WHR integration
this session). If that also fails to beat WWHR, the conclusion is "surface
factor structure doesn't add much over the existing W* family"; if it
works, it's a real new system worth productionising.

Hold step 2-5 of the original WWHR plan in reserve as Option A.
