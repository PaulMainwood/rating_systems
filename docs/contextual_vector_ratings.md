# Extending Rating Systems to Contextual Vector Representations

This document explores how to extend traditional scalar rating systems (Elo, Glicko, WHR, TTT) to multi-dimensional vector representations that can incorporate contextual factors like surface type, fatigue, travel distance, and match importance.

## Table of Contents

1. [The Fundamental Challenge](#the-fundamental-challenge)
2. [Theoretical Approaches](#theoretical-approaches)
3. [Contextual Elo](#contextual-elo)
4. [Contextual Glicko](#contextual-glicko)
5. [Contextual WHR](#contextual-whr)
6. [Contextual TTT](#contextual-ttt)
7. [Attention Mechanisms](#attention-mechanisms)
8. [Implementation Roadmap](#implementation-roadmap)

---

## The Fundamental Challenge

Current rating systems produce a single scalar representing "overall ability." But tennis ability is clearly multi-dimensional:

- **Surface-specific skills**: Movement on clay vs hard court footwork vs grass
- **Physical attributes**: Stamina, recovery, travel tolerance
- **Mental attributes**: Performance under pressure, consistency
- **Playing style matchups**: Some styles counter others

**The core challenge**: Match outcomes provide only 1 bit of information (win/loss), but we want to infer a d-dimensional vector. This is fundamentally an identifiability problem that requires either strong priors, regularization, or massive amounts of data.

### The Identifiability Problem

If player A beats player B on clay, we observe:
```
(A_base + A_clay) - (B_base + B_clay) > 0  (probably)
```

We **cannot** determine from this single match:
- Is A's base rating high?
- Is A's clay adjustment high?
- Is B's base rating low?
- Is B's clay adjustment low?

**How identification works**: We need matches across **different surfaces** against **different opponents** to disentangle base ability from surface effects.

---

## Theoretical Approaches

### Approach 1: Additive Contextual Decomposition

Decompose the effective rating as a dot product:
```
R_effective(player, context) = r · w(context)
```

Where:
- **r** is the player's ability vector (d dimensions)
- **w(context)** is a context-dependent weighting vector

### Approach 2: Multivariate Gaussian (Glicko/TTT style)

Each player has:
- Mean vector **μ** representing skill in each dimension
- Covariance matrix **Σ** representing uncertainty and correlations

### Approach 3: Temporal Trajectories (WHR/TTT style)

Each dimension has its own time-varying trajectory with potentially different drift rates:
```
r_base(t+Δt) | r_base(t) ~ N(r_base(t), w²_base·Δt)
r_clay(t+Δt) | r_clay(t) ~ N(r_clay(t), w²_clay·Δt)
```

### Approach 4: Attention-Based

Learn context-dependent weightings via attention mechanisms:
```
attention = softmax(query · keys / √d)
R_effective = attention · skill_values
```

---

## Contextual Elo

### Structure

Each player has a 4-dimensional rating vector:
```
r = [r_base, r_clay_adj, r_hard_adj, r_grass_adj]
```

For a match on surface s, the effective rating is:
```
R_effective = r_base + r_s_adj
```

### Win Probability

Same as standard Elo:
```
E(A beats B) = 1 / (1 + 10^((R_eff_B - R_eff_A) / scale))
```

### Update Rule

The gradient-derived update:
```
r_A := r_A + K · (S - E) · w(context)
r_B := r_B + K · (E - S) · w(context)
```

Where `w = [1, is_clay, is_hard, is_grass]` is the context vector.

**Key insight**: The update is proportional to the context weights. A clay match updates the clay adjustment; other surface adjustments are unchanged.

### Regularization Strategies

**Different K-factors per dimension**:
```
K_base = 40      # learns quickly
K_surface = 15   # learns slowly, needs more evidence
```

**Decay toward zero**:
```
r_surface_adj *= (1 - λ)  # after each match
```

**Sum-to-zero constraint**:
```
r_clay + r_hard + r_grass = 0  # enforced after each update
```

### Implementation

```python
@dataclass
class ContextualEloRating:
    base: float = 1500.0
    surface_adj: Dict[str, float] = field(
        default_factory=lambda: {'clay': 0.0, 'hard': 0.0, 'grass': 0.0}
    )

    def effective(self, surface: str) -> float:
        return self.base + self.surface_adj.get(surface, 0.0)


class ContextualElo:
    def __init__(self, k_base=40.0, k_surface=15.0, scale=400.0, decay=0.001):
        self.k_base = k_base
        self.k_surface = k_surface
        self.scale = scale
        self.decay = decay
        self.ratings = {}

    def update(self, player_a, player_b, score_a, surface):
        expected = self.expected_score(player_a, player_b, surface)
        surprise = score_a - expected

        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)

        # Update base ratings
        rating_a.base += self.k_base * surprise
        rating_b.base -= self.k_base * surprise

        # Update surface adjustments
        if surface in rating_a.surface_adj:
            rating_a.surface_adj[surface] += self.k_surface * surprise
            rating_b.surface_adj[surface] -= self.k_surface * surprise

        # Apply decay
        for s in rating_a.surface_adj:
            rating_a.surface_adj[s] *= (1 - self.decay)
            rating_b.surface_adj[s] *= (1 - self.decay)
```

### Hyperparameters

| Parameter | Description | Suggested Range |
|-----------|-------------|-----------------|
| `k_base` | Learning rate for base rating | 30-60 |
| `k_surface` | Learning rate for surface adjustments | 5-25 |
| `scale` | Logistic scale parameter | 250-450 |
| `decay` | Per-match decay of surface adjustments | 0.0001-0.01 |

---

## Contextual Glicko

Glicko is more natural to extend because it already has a Bayesian interpretation. The rating deviation (RD) represents uncertainty.

### Structure

Each player has a vector of (rating, RD) pairs:
```
Base:  (R_base, RD_base)
Clay:  (R_clay_adj, RD_clay)
Hard:  (R_hard_adj, RD_hard)
Grass: (R_grass_adj, RD_grass)
```

### Effective Rating and RD

For a match on surface s:
```
R_effective = R_base + R_s_adj
RD_effective = sqrt(RD_base² + RD_s²)
```

The effective RD is the sum of variances (for independent components).

### The Bayesian Update: Credit Assignment

When we observe a match outcome, we learn about **R_total = R_base + R_adj**. The update to each component is proportional to its prior variance.

**Kalman filter form**:
```
weight_base = RD_base² / (RD_base² + RD_adj²)
weight_adj = RD_adj² / (RD_base² + RD_adj²)

ΔR_base = weight_base * ΔR_total
ΔR_adj = weight_adj * ΔR_total
```

**Key insight**: If we're very uncertain about the surface adjustment (high RD_surface) but certain about base (low RD_base), a surprising result mostly updates the surface adjustment.

### RD Dynamics

**Increase over time** (uncertainty grows):
```
new_RD² = RD² + c²·days_elapsed
```

Different drift rates per dimension:
- `c_base`: Moderate (overall form fluctuates)
- `c_surface`: Smaller (surface skills are stable)

**Decrease after matches** (we learn):
```
new_RD² = 1 / (1/RD² + info_gained)
```

Info gained is split between dimensions proportionally to their prior variances.

### Implementation

```python
@dataclass
class RatingWithRD:
    R: float
    RD: float

@dataclass
class ContextualGlickoRating:
    base: RatingWithRD = field(default_factory=lambda: RatingWithRD(1500.0, 350.0))
    surface_adj: Dict[str, RatingWithRD] = field(default_factory=lambda: {
        'clay': RatingWithRD(0.0, 150.0),
        'hard': RatingWithRD(0.0, 150.0),
        'grass': RatingWithRD(0.0, 150.0)
    })

    def effective(self, surface: str) -> tuple[float, float]:
        adj = self.surface_adj.get(surface, RatingWithRD(0.0, 150.0))
        R_eff = self.base.R + adj.R
        RD_eff = sqrt(self.base.RD**2 + adj.RD**2)
        return R_eff, RD_eff
```

### Key Differences from Contextual Elo

| Aspect | Contextual Elo | Contextual Glicko |
|--------|----------------|-------------------|
| Uncertainty | Implicit (K-factor) | Explicit (RD per dimension) |
| Update allocation | Fixed weights | Adaptive (based on RD) |
| New players | Fixed high K | High RD → large updates |
| Time decay | Optional regularization | Built-in RD increase |

---

## Contextual WHR

WHR (Whole History Rating) models skill as a continuous function of time with a Wiener process prior.

### Current WHR Structure

```
r(t+Δt) | r(t) ~ N(r(t), w²·Δt)
```

**Key insight**: WHR solves for the entire history jointly. Unlike Elo/Glicko which only filter (past affects present), WHR also smooths (future affects past).

### Contextual WHR Structure

Each player has multiple skill trajectories:
```
r_base(t):  base skill over time
r_clay(t):  clay adjustment over time
r_hard(t):  hard adjustment over time
r_grass(t): grass adjustment over time
```

Each with its own drift rate:
```
r_base(t+Δt) | r_base(t) ~ N(r_base(t), w²_base·Δt)
r_clay(t+Δt) | r_clay(t) ~ N(r_clay(t), w²_clay·Δt)
```

**Design choice**: `w²_surface < w²_base` (surface skills are more stable than overall form).

### The Coupling Problem

The dimensions are coupled through the likelihood. Each match outcome depends on the sum of base and surface skill:
```
P(A beats B on clay) = σ((r_base_A + r_clay_A) - (r_base_B + r_clay_B))
```

Even with independent priors, the posterior is not factorized.

### The Hessian Structure

For scalar WHR, the Hessian is tridiagonal (O(n) solve). For contextual WHR with d dimensions:

- **Block-tridiagonal** with d×d blocks
- Complexity: O(d³·n) using block Thomas algorithm
- For d=4, this is ~64× more work than scalar, but tractable

### Identifiability and Regularization

**Sum-to-zero constraint**:
```
r_clay(t) + r_hard(t) + r_grass(t) = 0  ∀t
```

**Regularization toward zero**:
```
Loss += λ · Σ_t Σ_s r_s(t)²
```

**Informative prior**:
```
r_surface(0) ~ N(0, σ²_init)  (anchors adjustments at career start)
```

---

## Contextual TTT

TTT (TrueSkill Through Time) uses message passing instead of Newton-Raphson optimization.

### Current TTT Structure

Each player at time t has:
- μ(t): skill mean
- σ²(t): skill variance

### Contextual TTT Structure

Each player at time t has a multivariate Gaussian:
```
skill(t) ~ N(μ(t), Σ(t))
```

Where:
- μ(t) ∈ ℝ^d is the mean vector
- Σ(t) ∈ ℝ^(d×d) is the covariance matrix

### Temporal Dynamics

```
μ(t+Δt) | μ(t) ~ N(μ(t), Γ·Δt)
```

Where Γ = diag(γ²_base, γ²_clay, γ²_hard, γ²_grass) allows different drift rates.

### Message Passing Update

For a match outcome, the update involves:

1. **Project to 1D**: The performance difference δ = w·μ_A - w·μ_B
2. **Truncate**: Condition on δ > 0 (if A won)
3. **Back-project**: Update full multivariate beliefs

The back-projection uses Kalman filter formulas:
```
m_A_new = m_A + (Σ_A · w) · Δm / v_δ
Σ_A_new = Σ_A + (Σ_A · w · wᵀ · Σ_A) · Δv / v_δ²
```

**Key insight**: The term `Σ_A · w` points in the "direction of maximum uncertainty along w". If we're uncertain about clay (high Σ[clay,clay]), the update shifts clay more.

### Full vs Diagonal Covariance

**Full covariance** (d×d matrix):
- Captures correlations between dimensions
- Learns: "if base skill improves, clay skill probably improves too"
- O(d³) per update

**Diagonal covariance** (d values):
- Assumes dimensions are independent
- Simpler, O(d) per update
- Loses correlation structure

For d=4, full covariance is feasible.

### Implementation Sketch

```python
@dataclass
class MultivariateGaussian:
    mean: np.ndarray      # d-dimensional
    cov: np.ndarray       # d×d matrix

    def project(self, w: np.ndarray) -> tuple[float, float]:
        """Project onto direction w, return (mean, variance)."""
        m = w @ self.mean
        v = w @ self.cov @ w
        return m, v


class ContextualTTT:
    def update_match(self, belief_a, belief_b, winner, surface):
        w = self._context_vector(surface)

        # Project to 1D
        m_a, v_a = belief_a.project(w)
        m_b, v_b = belief_b.project(w)
        m_delta = m_a - m_b
        v_delta = v_a + v_b + self.beta**2

        # Truncated moments (conditioning on outcome)
        if winner == 'a':
            m_delta_new, v_delta_new = truncated_normal_moments(m_delta, v_delta, lower=0)
        else:
            m_delta_new, v_delta_new = truncated_normal_moments(-m_delta, v_delta, lower=0)
            m_delta_new = -m_delta_new

        # Back-project updates
        delta_m = m_delta_new - m_delta
        delta_v = v_delta_new - v_delta

        # Kalman-style update for player A
        k_a = belief_a.cov @ w / v_delta
        new_mean_a = belief_a.mean + k_a * delta_m
        new_cov_a = belief_a.cov + np.outer(k_a, k_a) * delta_v * v_delta

        return MultivariateGaussian(new_mean_a, new_cov_a), ...
```

---

## Attention Mechanisms

Attention provides a fundamentally different approach to context-dependent ratings.

### The Transformer Analogy

- **Query (Q)**: What information am I looking for? (current match context)
- **Key (K)**: What information do I have? (skill aspect descriptors)
- **Value (V)**: The actual information (skill levels)

### Attention Over Skill Dimensions

Instead of fixed context vectors w = [1, 1, 0, 0], learn the weighting:

```python
Query:   q = embed(surface, fatigue, importance, ...)
Keys:    k_i = learned embedding for dimension i
Values:  v_i = μ_i(t)  (skill level in dimension i)

Effective_skill = Σ_i softmax(q · k_i / √d) · v_i
```

**What this learns**:
- Which skill dimensions matter for which contexts
- Non-linear, context-dependent combinations
- Potentially matchup-specific effects

### Temporal Attention (Replacing Wiener Process)

The Wiener process assumes exponential decay of correlation. Attention can learn arbitrary relevance patterns:

```python
Query:   q_t = f(current_context, t)
Keys:    k_s = g(past_context_s, s)  for all s < t
Values:  v_s = skill_estimate(s)

Skill(t) = Σ_{s<t} softmax(q_t · k_s / √d) · v_s + innovation
```

This allows:
- Non-exponential decay (Grand Slam wins have lasting impact)
- Context-dependent memory (attend to past clay matches for current clay match)
- Automatic recency weighting

### Multi-Head Attention

Use multiple attention heads for different aspects:
```
Head 1: Surface attention
Head 2: Physical condition attention
Head 3: Mental/pressure attention

Effective = Σ_h W_h · Head_h(skills, context)
```

### Self-Attention Across Players (Matchup Effects)

Model matchup-specific effects via player embeddings:

```python
matchup_effect(A, B) = e_Aᵀ · M · e_B
```

Where **e** captures "playing style". This handles sparse matchup data by borrowing strength from similar players.

### Hybrid Approach

Combine WHR/TTT backbone with attention:
```
Base skill:  μ_base(t) evolves via Wiener process (proven to work)
Adjustment:  μ_adj(t, context) = Attention(context, history)
Effective:   μ_eff(t) = μ_base(t) + μ_adj(t, context)
```

---

## Implementation Roadmap

### Phase 1: Contextual Elo

1. Implement vector Elo with d=4: [base, clay_adj, hard_adj, grass_adj]
2. Add surface information to data pipeline
3. Use dimension-specific K-factors and L2 regularization
4. Validate against scalar Elo baseline (Brier score)

**Success criteria**: Improvement over scalar Elo, learned adjustments match known specialists (Nadal high on clay).

### Phase 2: Contextual Glicko

1. Extend to track RD per dimension
2. Implement Bayesian credit assignment (update proportional to prior variance)
3. Add dimension-specific RD drift rates
4. Compare to Phase 1

**Success criteria**: Better calibration than contextual Elo, sensible uncertainty estimates.

### Phase 3: Contextual WHR

1. Implement block-tridiagonal Newton-Raphson
2. Add dimension-specific drift rates (w²)
3. Add regularization/constraints for identifiability
4. Analyze how surface adjustments evolve over careers

**Success criteria**: Better historical estimates, insight into career trajectories.

### Phase 4: Contextual TTT

1. Implement multivariate Gaussian message passing
2. Choose full vs diagonal covariance
3. Forward-backward algorithm for smoothing
4. Compare convergence and accuracy to WHR

**Success criteria**: Faster than WHR with comparable accuracy, meaningful covariance structure.

### Phase 5: Attention Extensions (Optional)

1. Add temporal attention (replace Wiener process)
2. Add dimension attention (replace fixed context vectors)
3. Requires more data and careful regularization
4. Only pursue if Phase 1-4 show promise

---

## Computational Complexity Summary

| Model | Parameters per player | Update cost | Training |
|-------|----------------------|-------------|----------|
| Scalar Elo | 1 | O(1) | Online |
| Contextual Elo | d | O(d) | Online |
| Scalar WHR | n | O(n) | Batch (Newton) |
| Contextual WHR | d·n | O(d³·n) | Batch (Block Newton) |
| Scalar TTT | 2n | O(n) | Message passing |
| Contextual TTT | d·n + d²·n | O(d³·n) | Message passing |
| + Attention | + attention params | + O(d·h) | Gradient descent |

For d=4 and n~1000 matches per player, all are tractable.

---

## Key Theoretical Insights

### Why Attention Might Be Natural

The core insight of attention: **not all information is equally relevant, and relevance is context-dependent**.

For tennis ratings:
- Not all past matches are equally informative (attention over history)
- Not all skill dimensions matter equally for each context (attention over aspects)
- Not all opponents reveal the same information (matchup attention)

### The Bias-Variance Tradeoff

- **Scalar Elo**: High bias (ignores context), low variance (one parameter)
- **Vector Elo**: Lower bias, higher variance (more parameters)
- **Attention**: Lowest bias (highly flexible), highest variance (needs regularization)

### Connection to Gaussian Processes

There's a deep connection between attention and GPs:
- GP: Kernel function measures similarity between inputs
- Attention: Softmax of dot product measures similarity

Vector TTT with learned covariance is essentially a GP over time. Attention-based temporal modeling is a learned, data-dependent kernel.

---

## What We Might Discover

**Surface effects**:
- Magnitude of surface adjustments
- Which players have extreme adjustments
- Whether adjustments are stable or time-varying

**Temporal patterns**:
- How fast does skill really drift?
- Are there "peak periods"?
- Does recent form matter more for some contexts?

**Correlations**:
- Base ↔ surface correlations
- Surface ↔ surface correlations (clay vs grass?)
- Whether correlations vary by player

**Attention patterns**:
- Which contexts activate which skill dimensions
- Whether patterns are player-specific or universal

---

## References

- Glickman, M. E. (1999). Parameter estimation in large dynamic paired comparison experiments. Applied Statistics.
- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: A Bayesian skill rating system. NIPS.
- Dangauthier, P., Herbrich, R., Minka, T., & Graepel, T. (2007). TrueSkill Through Time. NIPS.
- Coulom, R. (2008). Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength. CG.
- Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.
