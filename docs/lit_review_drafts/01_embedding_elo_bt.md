# Agent 1: Embedding-Based & Multi-Dimensional Extensions of Paired-Comparison Rating Systems

**IMPORTANT honesty disclaimer (from agent):** WebFetch was blocked in this session, so the agent could NOT directly read full PDFs. Every citation marked [ABSTRACT/SECONDARY] is based on WebSearch result snippets, abstracts, or summary articles — not the primary paper. Mathematical content is consistent across secondary sources but unverified.

---

## 1. Executive Summary

Three broad lines of work extend classical scalar rating systems:

1. **mElo / vector-Elo / disk-decomposition line** (Balduzzi et al. 2018 onwards): k-dimensional vector rating plus an antisymmetric low-rank matrix to capture intransitivity via Hodge/Schur decomposition. Most theoretically grounded; closest to the user's contextual-vector idea.
2. **Neural / embedding-parameterised Bradley-Terry line** (Causeur & Husson 2005; NBTR 2023; GPM/Beyond-BT 2025): latent scalar replaced with a vector or function of features from a neural network.
3. **Graph- and side-information-augmented line** (538's surface-Elo, glicko-boost, Ingram 2019, GElo 2023, Tennis GNN 2025): keep Elo/BT parametric form but inject covariates or learned graph embeddings.

State of the field for tennis: mature multidimensional BT models from statistics exist, mElo is theoretically clean but not widely adopted in sports, and 2023-2025 sees rapid growth in graph-based embeddings and LLM-style neural BT. **Gap**: no work combining *time-varying Bayesian* ratings (Glicko/WHR style) with *learned multi-dimensional embeddings* in *non-stationary* sports with margin-of-victory + match covariates.

---

## 2. Foundational Papers (1990s-2015)

- **Bradley & Terry (1952)**, Zermelo (1929): classical BT model.
- **Elo (1978)**: online logistic regression interpretation; SGD on logistic loss with learning rate K.
- **Glickman (1995, 1999, 2001)**: Glicko, Glicko-2 — explicit ratings-deviation parameter.
- **Causeur & Husson (2005)**, *J. Statist. Plann. Inference* 135(2): 245-259 — first explicit multidimensional BT in statistics literature. Each item gets two scores; comparison function includes interactions. Framed as **multidimensional scaling inside a logistic model**. MLE; asymptotic confidence ellipses. Applied to consumer cornflakes data. https://www.sciencedirect.com/science/article/abs/pii/S0378375804002393
- **Herbrich, Minka & Graepel (2007)**: TrueSkill (factor graph + EP).
- **Coulom (2008)**: Whole-History Rating; Bayesian batch MAP with Newton's method on tridiagonal Hessian.
- **Jiang, Lim, Yao & Ye (2008/2011)**: "Statistical ranking and combinatorial Hodge theory" — mathematical foundation for mElo. Decomposes the comparison-graph edge-flow into: gradient flow (transitive ranking, recoverable by least squares), harmonic flow (globally cyclic, locally acyclic), curl flow (locally cyclic, 3-cycles). https://web.stanford.edu/~yyye/hodgeRank2011.pdf
- **Stephenson (2012)** Kaggle FIDE chess winner; Glicko-Boost (Glickman) runner-up. https://www.glicko.net/glicko/glicko-boost.pdf
- **Rendle (2010)**: Factorization Machines. **Rendle et al. (2012) BPR** (arXiv:1205.2618): Bayesian Personalised Ranking — `sigmoid(score(u,i_pos) - score(u,i_neg))` with low-rank matrix factorisation. **Mathematically a multi-dimensional BT where embeddings are factor vectors and score is their dot product.**
- **Barkan & Koenigstein (2016)** item2vec (arXiv:1603.04259): skip-gram with negative sampling for items. Imported into sports analytics by NBA2Vec, batter/pitcher2vec.

---

## 3. The mElo / Vector-Elo Line (critical section)

### 3.1 Balduzzi, Tuyls, Perolat & Graepel (2018) — Re-evaluating Evaluation
NeurIPS 2018, arXiv:1806.02643. https://proceedings.neurips.cc/paper/2018/hash/cdf1035c34ec380218a8cc9a43d438f9-Abstract.html

**Core idea (reconstructed from snippets):**
- Win-probability logit between agents i and j decomposed via combinatorial Hodge theory into a **transitive component** (classical Elo skill diff `r_i - r_j`) PLUS a **cyclic component** captured by an antisymmetric low-rank matrix.
- Each agent gets scalar Elo `r_i` AND a k-dim vector `c_i`. Logit ≈ `r_i - r_j + c_i^T Ω c_j` where Ω is skew-symmetric. Skew-symmetric matrices have even rank; "mElo2k" sets rank = 2k.
- Updates are SGD on logistic loss; antisymmetry preserved by parameterisation.
- Design properties: P1 invariance to redundant agents/tasks; P2 continuity; P3 interpretability.
- Empirical: Hearthstone, AlphaStar, 10,5-Blotto, Go (7 AlphaGo variants + Zen). mElo2 achieved `||P - P_hat||_F = 0.35`, log-loss 1.27 on Go.
- Companion contribution: **Nash averaging** for evaluation across tasks.

**Code**: R port https://github.com/dclaz/mELO; docs https://dclaz.github.io/mELO/

**Limitations**: designed for game-theoretic intransitivity (RPS-style). In tennis (≈11% extra intransitivity in WTA vs ATP per Hamilton et al. 2024), the added dimensions can be hard to identify without huge data. No built-in mechanism for time-variation or surface adjustment.

### 3.2 Du, Yan, Chen, Wang & Zhang (2021) — α-Rank from a Few Entries
ICML 2021, PMLR 139: 2870-2879. http://proceedings.mlr.press/v139/du21e/du21e.pdf
Low-rank matrix completion to recover the meta-game payoff matrix; nuclear-norm minimisation of the rank-2k approximation of the antisymmetric payoff matrix.

### 3.3 Omidshafiei et al. (2019) — α-Rank
*Scientific Reports*, arXiv:1903.01373. Evaluation framework for non-transitive games. Markov-Conley chains replace Nash equilibrium; stationary distribution of evolutionary dynamic gives ranking. Polynomial time. Benchmark target for mElo-style methods.

### 3.4 Bertrand et al. (2023) — "Disk Decomposition"
**Could not access primary paper.** Search snippets describe it as alternative naming of Elo_2k as low-rank approximation of the logits.

### 3.5 Hamilton, Kalenkova & Roughan (2024/2025) — Elo in the Presence of Intransitivity
arXiv:2412.14427; PLOS ONE 2025. Shows even mild intransitivity makes Elo's converged ratings depend on the *matchup schedule*. Connects Elo's converged rating to **Hodge rank** of expected payoff matrix. Reports **WTA has ~11.5% more intransitivity than ATP**. https://arxiv.org/pdf/2412.14427

### 3.6 Strang, Abbott & Thomas (2021) — The Network HHD
*SIAM Review*, arXiv:2011.01825. Rigorous Helmholtz-Hodge Decomposition for tournament networks; interpretable intransitivity measure.

### 3.7 Okahara, Nakagawa & Sugasawa (2026) — Bayesian Intransitive BT via Hodge
arXiv:2601.07158. Embeds Hodge decomposition into Bayesian BT. Pair gets gradient (transitive) + curl (cycle) components with **global-local shrinkage priors on the curl**, so model collapses to classical BT when no intransitivity present. Gibbs sampler. Most recent principled Bayesian treatment.

---

## 4. Neural Bradley-Terry

### 4.1 Causeur & Husson (2005)
See §2.

### 4.2 Neural Bradley-Terry Rating (NBTR) — arXiv:2307.13709
Item features `f(x)` → shared-weight network `φ` → scalar scores → BT functional `σ(φ(f(x_i)) - φ(f(x_j)))`. Trained end-to-end on pairwise outcomes. Optional "advantage adjuster" for asymmetric comparisons. **Killer feature**: generalises to unseen items via features.

**Limitations**: requires meaningful item features. Tennis features (recent form, opponent histories, surface) are noisy.

### 4.3 BT-based Reward Models in RLHF (2024-2025)
Modern LLM alignment treats reward modelling as BT over (prompt, response_A, response_B). Reward is a neural network. References: arXiv:2411.04991 (Rethinking BT in RLHF), arXiv:2601.14727 (Recent Advances Survey).

### 4.4 Zhang et al. (2024/2025) — Beyond Bradley-Terry: GPM
ICML 2025, arXiv:2410.02197. Code: https://github.com/general-preference/general-preference-model

**Core idea**: Each response embedded as vector `v_i`; preference probability `v_i^T A v_j` where A is **learned skew-symmetric matrix**. **Structurally identical to mElo with k≥1** except embeddings learned end-to-end from text rather than from outcomes. Near-perfect accuracy on synthetic cyclic preference data where vanilla BT performs at chance.

**This is the most directly sport-relevant transfer: GPM is mElo for language models.**

### 4.5 Multidimensional BT with MCMC
"Individual Differences Multidimensional BT Using Reversible Jump MCMC" — *Behaviormetrika* 37(2). RJMCMC over embedding dimensionality.

---

## 5. Embedding-Based and Side-Information Extensions

### 5.1 FiveThirtyEight surface-adjusted Elo (Morris, Bialik, Boice ~2016)
Hard-court formula: `0.71 × overall_Elo + 0.29 × surface_Elo`. Two parallel Elos with convex combination at prediction. Kovalchik (2016) found it best automated tennis predictor at the time.

### 5.2 Ingram (2019) — Point-based Bayesian Hierarchical Model
*JQAS* 15(4): 313-325. Serve and return skills as Gaussian random walks over time + surface-specific perturbations + tournament intercepts. 68.8% accuracy / 0.592 log-loss on 2014 ATP. Essentially **multi-dimensional Glicko with structured covariates** rather than learned embedding. https://martiningram.github.io/papers/bayes_point_based.pdf

### 5.3 Kovalchik & Reid; McHale & Morton (2011)
McHale & Morton applied BT with surface-specific adjustments to tennis. Kovalchik developed Elo extensions with margin-of-victory.

### 5.4 Maystre, Kristof & Grossglauser (2019) — Pairwise Comparisons with Flexible Time-Dynamics
KDD'19, arXiv:1903.07746. Code: https://github.com/lucasmaystre/kickscore. Static BT/Elo parameters replaced with **Gaussian processes over time** with domain-specific kernels. Closest existing analogue to "Glicko with side information" — the kernel IS the prior.

### 5.5 Wang (2023) — GElo: Graph Embedding Augmented Skill Rating
arXiv:2304.08257. Undirected weighted skill-gap graph from match histories; embeddings via DeepWalk/node2vec; vanilla Elo updates adjusted by cosine similarity between player embeddings and player activeness. Pragmatic, not fully end-to-end.

### 5.6 Tennis GNN Paper (2025) — Intransitive Player Dominance
arXiv:2510.20454. Temporal directed graph of tennis matches; MagNet spectral GCN for directed graphs. 65.7% acc / 0.215 Brier on high-intransitivity subsets; **3.26% ROI with Kelly staking over 1903 bets** vs Pinnacle. Claim: bookmakers under-price intransitive matchups. **Most directly relevant paper to a tennis researcher in 2025.**

### 5.7 Dynamic Graph-Based Forecasts of Bookmakers' Odds (2025)
arXiv:2508.15956. Dynamic graph models vs odds. Limited detail in snippets.

### 5.8 NBA2Vec, batter/pitcher2vec, Player2Vec language modelling
arXiv:2302.13386, arXiv:2404.04234. Apply word2vec/skip-gram to sports tracking data for descriptive embeddings; **none treat embeddings as paired-comparison ratings in BT/Elo sense**.

### 5.9 Bradley-Terry Stochastic Block Model (2025)
arXiv:2511.03467. BT inside SBM so items cluster into tiers. ATP 2000-2022 men's data: top-100 partitions into 3-4 tiers. Gibbs sampler. Mid-point between scalar BT and full vector embeddings.

### 5.10 Generalised BT with Covariates
arXiv:2507.22472: `logit = β_i - β_j + Z^T γ` with covariates Z and fixed regression coefficient γ. Asymptotic normality of MLE. arXiv:2503.18256 considers covariate-shift inference.

---

## 6. Gaps / Open Problems for a Tennis Researcher

1. **Time-varying multi-dimensional ratings under a Bayesian prior**: mElo has no time dynamics; WHR/Glicko-2 have time but scalar; Maystre's kickscore has GP time but scalar params. **Vector-valued GP per player** with kernel structure capturing surface correlation looks unexplored.

2. **Surface as a learned axis rather than fixed indicator**: 538/Ingram/McHale-Morton treat surface as discrete pre-specified. **Learn low-rank latent court characteristic directly from data** — surface "embedding" data-driven rather than imposed. Conceptually mElo where one player-vector dim dots with a learned surface vector.

3. **End-to-end neural BT with WHR-style smoothing**: GPM/NBTR assume i.i.d. comparisons; sports outcomes are massively non-stationary. **Neural BT with smoothness regulariser on each player's embedding trajectory** — not published to my knowledge.

4. **Out-of-sample test of mElo vs scalar Elo on tennis**: Balduzzi paper light on CV; tennis offers clean tournament-by-tournament temporal split. The user's project is well-placed.

5. **Combining margin-of-victory with embeddings**: Kovalchik's Weighted Elo / 538's MOV adjustments live on scalar side. **Does the cyclic component carry MOV information?** Open.

6. **Cold start via player features**: NBTR generalises to unseen items via features. **Apply to junior players entering pro tour** with height/ITF results/country.

7. **Glicko/Glicko-2 with feature-based priors**: no work I found puts an embedding prior on Glicko player's initial rating distribution. **Low-hanging fruit.**

8. **Rigorous "is mElo overfitting?" study for k > 2**: dclaz R package is research-grade. Tennis offers a clean temporal split.

---

## 7. References

- **Balduzzi, Tuyls, Perolat & Graepel (2018).** Re-evaluating Evaluation. NeurIPS 2018. arXiv:1806.02643. [ABSTRACT/SECONDARY]
- **Barkan & Koenigstein (2016).** Item2vec. arXiv:1603.04259. [ABSTRACT/SECONDARY]
- **Bertrand et al. (2023).** Disk Decomposition. **COULD NOT ACCESS.**
- **Bradley & Terry (1952).** Rank analysis of incomplete block designs. *Biometrika* 39: 324-345.
- **Causeur & Husson (2005).** *J. Statist. Plann. Inference* 135(2): 245-259. [ABSTRACT/SECONDARY]
- **Coulom (2008).** WHR. https://www.remi-coulom.fr/WHR/WHR.pdf. [ABSTRACT/SECONDARY]
- **Du, Yan, Chen, Wang & Zhang (2021).** α-Rank from a Few Entries. ICML 2021. http://proceedings.mlr.press/v139/du21e/du21e.pdf. [ABSTRACT/SECONDARY]
- **Fang et al. (2026).** Recent advances in BT survey. arXiv:2601.14727. [ABSTRACT/SECONDARY]
- **Firth et al. (2024).** Many routes to the BT model. arXiv:2312.13619. [ABSTRACT/SECONDARY]
- **Glickman.** Glicko / Glicko-2 / Glicko-Boost. https://www.glicko.net/glicko/glicko-boost.pdf. [ABSTRACT/SECONDARY]
- **Hamilton, Kalenkova & Roughan (2024).** Elo in the Presence of Intransitivity. arXiv:2412.14427; PLOS ONE 2025. [ABSTRACT/SECONDARY]
- **Herbrich, Minka & Graepel (2007).** TrueSkill. NeurIPS 2006. [ABSTRACT/SECONDARY]
- **Ingram (2019).** Point-based Bayesian hierarchical tennis model. *JQAS* 15(4): 313-325. https://martiningram.github.io/papers/bayes_point_based.pdf. [ABSTRACT/SECONDARY]
- **Jiang, Lim, Yao & Ye (2011).** Statistical ranking and combinatorial Hodge theory. *Math. Program.* 127: 203-244. https://web.stanford.edu/~yyye/hodgeRank2011.pdf. [ABSTRACT/SECONDARY]
- **Maystre, Kristof & Grossglauser (2019).** Pairwise Comparisons with Flexible Time-Dynamics. KDD'19. arXiv:1903.07746. Code: https://github.com/lucasmaystre/kickscore. [ABSTRACT/SECONDARY]
- **McHale & Morton (2011).** BT-type tennis model. *Int. J. Forecasting*. [ABSTRACT/SECONDARY]
- **Okahara, Nakagawa & Sugasawa (2026).** Bayesian Intransitive BT via Hodge. arXiv:2601.07158. [ABSTRACT/SECONDARY]
- **Omidshafiei et al. (2019).** α-Rank. *Scientific Reports*. arXiv:1903.01373. [ABSTRACT/SECONDARY]
- **Rendle et al. (2012).** BPR. arXiv:1205.2618. [ABSTRACT/SECONDARY]
- **Strang, Abbott & Thomas (2021).** The Network HHD. *SIAM Review*. arXiv:2011.01825. [ABSTRACT/SECONDARY]
- **Wang (2023).** GElo. arXiv:2304.08257. [ABSTRACT/SECONDARY]
- **Neural Bradley-Terry Rating (2023).** arXiv:2307.13709. [ABSTRACT/SECONDARY]
- **Tennis GNN intransitive market inefficiency (2025).** arXiv:2510.20454. [ABSTRACT/SECONDARY]
- **BT Stochastic Block Model (2025).** arXiv:2511.03467. [ABSTRACT/SECONDARY]
- **Zhang et al. (2024/2025).** Beyond BT: GPM. ICML 2025. arXiv:2410.02197. https://github.com/general-preference/general-preference-model. [ABSTRACT/SECONDARY]
- **Yang et al. (2025).** Generalised BT with covariates. arXiv:2507.22472. [ABSTRACT/SECONDARY]
- **Rethinking BT in RLHF (2024).** arXiv:2411.04991. [ABSTRACT/SECONDARY]
- **Hamilton et al. (2025).** Impact of intransitivity on Elo. PLOS ONE. [ABSTRACT/SECONDARY]

---

## Could Not Access (despite repeated searches)
- **Bertrand et al. 2023 — Disk Decomposition** primary paper.
- **Full mathematical update equations of Balduzzi mElo** (rank-2k SGD with antisymmetric constraint preservation).
- **Full text of Causeur & Husson 2005** (paywalled).
