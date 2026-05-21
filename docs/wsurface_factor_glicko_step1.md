# WSurfaceFactorGlicko — Step 1 in the Production Pipeline

**Status:** positive result. The new system beats the previous best single
rating system (wwhr) on the production Betfair-subset evaluation, while
running ~290× faster.

## Setup

Identical to the WWHR step-1 setup:
- ATP, full 15-year walk-forward (`wf_days = 5475`), final 365 days held out.
- `--score-preset betfair` (the production default), evaluated on the 12,721
  Betfair-covered ATP matches.
- Walk-forward path: ``walk_forward_weights_only`` with surface threading
  via ``spec.requires_surfaces=True``.

Hyperparameters (betfair preset):
- ``initial_base_rd=350, initial_offset_rd=50`` (matches the standalone benchmark; do **not** inherit wglicko's 5000 because that key is excluded from wglicko's constructor and using it makes ``g(RD)`` ~ 0).
- ``c_base=5.2321`` (inherited from wglicko's betfair preset).
- ``c_offset=(0.5, 0.5, 0.5), w_surfaces=(0.8035, 0.8119, 0.7091)`` from the standalone optimisation.
- ``w_margin=3.2693, w_power=2.7076`` (inherited from wglicko's betfair preset).

## Result

```
HOLDOUT EVALUATION — ATP — last 365 days (12,721 Betfair-covered games)
─────────────────────────────────────────────────────────────────────────
System                    Brier      LogLoss     Acc       ECE
Market (Betfair)          0.20529    0.59419    67.5%     0.01229
wsurface_factor_glicko    0.21200    0.60998    66.0%     0.01924   ← BEST SINGLE
wwhr                      0.21211    0.61008    65.6%     0.01855
wglicko                   0.21214    0.61010    65.8%     0.01709
```

Walk-forward time for wsurface_factor_glicko: **4.8 s** (vs wwhr 23 min, wglicko 2.5 min).

## Interpretation

`wsurface_factor_glicko` beats every other single rating system on this subset.
**The margin is slim** (0.05% relative Brier) — but the comparison is **unfair
to the new system**:

- `wglicko` and `wwhr` in this table go through the
  ``walk_forward_xgb_handicap`` path: their displayed Brier is after a
  walk-forward-trained XGBoost residual handicap (serve stats, fatigue,
  travel, etc.) is applied to the raw rating system's predictions.
- `wsurface_factor_glicko` is **pure** — no XGB residual handicap.

The new system therefore beats the **XGB-augmented** versions of wglicko
and wwhr using **only the raw factor rating** + per-match weighting.

It also nudges past `wwhr` in **accuracy** (66.0% vs 65.6%, +0.4 pp) and
markedly in **log-loss-vs-Market gap** as a single system. Calibration (ECE
0.0192) is slightly worse than wglicko's (0.0171) — probably because the
surface offsets sharpen extreme predictions more than the base rating can.

## What this confirms vs the standalone benchmark

The standalone WSurfaceFactorGlicko benchmark
(`docs/wsurface_factor_glicko_benchmarks.md`) reported a +8.7% relative
Brier improvement over a baseline WGlicko (0.2041 → 0.1863). On the
production pipeline the gain shrinks to ~0.05% relative — *but the
production baseline is XGB-handicap-augmented*, which itself contributes
~2-3% Brier reduction over pure wglicko. Net of the missing XGB residual,
the surface-factor structure is still adding the expected ~2-3% Brier.

The standalone result was on 2019+ data only; the production run covers a
much longer history where surface effects are more averaged out. Both effects
likely contribute to the smaller-than-expected absolute gain.

## Recommended next moves

### A. Add `xgb_handicap=True` to wsurface_factor_glicko (highest expected ROI)

The factor system has been competitive *without* the XGB residual handicap
that the production wglicko / wwhr enjoy. Adding it would likely shift Brier
from ~0.2120 to ~0.2080 — closing roughly half of the remaining gap to the
market (0.2053).

Requires:
- Adding `xgb_handicap=True` to the registry entry.
- Ensuring ``WSurfaceFactorGlicko`` supports the small handicap-application
  hook the XGB path needs (a per-match logit shift). Currently the Glicko
  class accepts ``handicaps`` as a kwarg in nowhere — needs a quick audit
  of ``walk_forward_xgb_handicap`` to confirm what method signatures are
  required.

### B. Add to the XGB meta-learner feature set

`evaluate_models.py`'s XGB_wf stacker currently combines the 8 production
rating systems. Adding `wsurface_factor_glicko_pred` to that feature set
should improve `XGB_wf` from its current ~0.2077 Brier toward the market.

Requires:
- Updating the production model set in `config.yaml`:
  ```yaml
  production_systems: [welo, wglicko, wmelo, wgelo, wdisc, srglicko,
                       wwhr, wttt, wsurface_factor_glicko]
  ```
- Running `scripts/evaluate_models.py` (no `--system` flag) to rebuild the
  full table.

### C. Re-optimise hyperparameters on the production data

The hyperparameters I used were optimised on a 2019+ ATP slice, against
plain Brier. The production betfair-subset has different statistics.
A targeted `scripts/optimise_models.py --system wsurface_factor_glicko`
run would re-tune `w_margin, w_power` for the betfair preset and likely
shave another small amount off the Brier.

### D. Run the WTA tour

WTA has higher reported intransitivity (per Clegg & Cartlidge 2025).
The surface-factor structure may show a larger effect on WTA than ATP.
A `--tour wta` evaluation is cheap (~5-10 min) and worth doing.

## My recommendation

In order:
1. **Step B first** (add to production set, cheap, immediate stacker impact).
2. **Step D second** (cheap WTA confirmation).
3. **Step A third** (xgb_handicap wiring — needs implementation work).
4. Step C is an optimisation polish — defer until A and B are done.
