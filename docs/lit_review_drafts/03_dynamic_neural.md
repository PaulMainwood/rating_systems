# Agent 3: Time-Varying / Dynamic Rating Systems with Deep Learning

**Honesty note (from agent):** WebFetch returned binary FlateDecode streams for many PDFs and was denied for some URLs. Tags: **[READ FULL]** = arXiv HTML body parsed; **[READ PARTIAL]** = structured content from arXiv abstract/HTML; **[READ ABSTRACT]** = arXiv abstract only; **[SECONDARY]** = search-result snippets only.

---

## 1. Executive Summary

Classical Bayesian/optimisation-based time-varying methods — **WHR (Coulom 2008)**, **TTT (Dangauthier et al. 2007)**, **Glicko (Glickman 1999)** — remain extraordinarily strong baselines. Recent unification work (Duffield/Power/Rimella 2023; Ingram 2021; Szczeciński/Tihoń 2023) shows these are all **state-space models with Gaussian/Wiener latent dynamics solved by different approximate inference algorithms** (Newton, EP, Kalman). **Glicko is provably a 1D Extended Kalman Filter.**

The "neural" branch has *not* directly replaced these methods. Deep learning is used as:
(a) auxiliary cold-start predictors that map novice game traces to skill (QuickSkill);
(b) representation learners producing embeddings later fused into Elo (GElo);
(c) move-level rather than match-level skill estimators (CNN-LSTM Omori & Tadepalli 2024; GlickFormer 2024);
(d) human-style models conditioned on Elo as control variable rather than inferred quantity (Maia, Maia-2);
(e) GNNs attempting to model intransitivity (Clegg & Cartlidge 2025);
(f) sequence-to-outcome predictors (LSTM live-prediction style).

**Pure neural replacements of WHR/TTT are rare and generally beaten by careful Bayesian baselines on out-of-sample log-loss. Frontier = hybrid models keeping Bayesian state-space backbone but adding neural emissions or context.**

---

## 2. Classical Time-Varying Baselines (recap)

### 2.1 WHR — Coulom 2008 [READ PARTIAL]
Dynamic BT with Wiener-process prior on each player's natural-rating curve. Variance per unit time `w²` is the only hyperparameter beyond BT scale.
- **Inference**: MAP via Newton-Raphson on per-player rating curve. Tridiagonal Hessian (Wiener prior couples only adjacent anchors), `O(n_i)` per Newton step per player. Loop iterates per-player Newton, alternates over players.
- **Uncertainty**: posterior variance from inverse diagonal-block Hessian.
- **Cold-start**: prior; no observations → player at prior mean.
- **Code**: Coulom's Ruby reference; Python port https://github.com/pfmonville/whole_history_rating

### 2.2 TTT — Dangauthier et al. 2007 [READ PARTIAL]
Latent skill `s_{i,t}` per player per year; Gaussian random-walk coupling between years; Gaussian factor graph per game. **EP smoothing** (not filtering) — all past + future games inform every estimate.
- **Note**: "Matérn TTT" mentioned in prompt is **not a standard published variant** — agent could not find it. Closest is Maystre/Kristof/Grossglauser 2019 substituting Matérn-kernel GPs for the random walk. **The "Matern_ttt" label in the user's codebase may be informal/internal.**

### 2.3 Glicko / Glicko-2 — Glickman 1999, 2001 [SECONDARY]
*JRSS C* 48(3):377-394. Demonstrated formally by **Ingram (2021)** and **Szczeciński & Tihoń (2023, JQAS 19(4):295-315, arXiv:2104.14012)** to be **a scalar Extended Kalman Filter**. Posterior mean updates = Newton steps on penalised likelihood; posterior variance = inverse Hessian at mode.

### 2.4 State-space unification — Duffield, Power, Rimella 2023 [READ PARTIAL]
arXiv:2308.02414, "A State-Space Perspective on Modelling and Inference for Online Skill Rating."
Unifies Elo, Glicko, TrueSkill 2, sequential Monte Carlo, and finite-state HMMs under one factor-graph framework with modular dynamics and emissions. Empirics on chess, tennis, football.
- **Code**: https://github.com/SamDuffield/abile
- **Notable**: no neural components, **but framework cleanly admits neural emissions or transitions.**

---

## 3. RNN/LSTM-based Skill Models

### 3.1 QuickSkill — Zhang et al. CIKM 2022 [READ ABSTRACT]
arXiv:2208.07704. Sequence-based net (specific cell type **not stated in abstract**) over per-game performance features from player's first few games. Output: predicted future skill rating. Trained on two anonymised mobile-game datasets.
- **Claim**: "first framework that tackles cold-start for traditional skill rating algorithms."
- **Uncertainty**: not explicitly addressed.
- **Full PDF not retrievable.**

### 3.2 RNN view of Elo / online rating [SECONDARY]
Practitioner pieces (octosport.io blog) cast Elo as recurrent network with rating as hidden state, prediction head = BT sigmoid. **Tutorial framing, not peer-reviewed.**

### 3.3 Chess Rating from Move Sequence — Omori & Tadepalli 2024 [READ PARTIAL]
arXiv:2409.11506, CG 2024.
- **Architecture**: per-position CNN extracts board features; outputs concatenated with clock-time scalars feed bidirectional LSTM; outputs predicted Elo per ply.
- **Dataset**: >1M Lichess games across time controls.
- **MAE**: 182 Elo points on test set. No baseline numbers in abstract.
- **Cold-start tool**: estimate Elo from single game.
- **Limitations**: supervised regression to Lichess ratings → inherits their noise; cannot distinguish strong-playing-weakly from weak.

### 3.4 DeepTennis — Lerner 2019 [SECONDARY]
CS230 Stanford project (not peer-reviewed). LSTM over point-by-point sequence for live win-probability. **Not a skill-rating paper but existence proof for sequential modelling on tennis-point data.**

---

## 4. Transformer-Based Approaches

### 4.1 Maia chess — McIlroy-Young et al. KDD 2020 [READ ABSTRACT + SECONDARY]
arXiv:2006.01855. **Not a transformer — customised AlphaZero-style residual ConvNet.** Nine separate models for Elo bands 1100-1900 in 100-point increments.
- **Input**: board position. **Output**: distribution over moves predicting *human* move at that rating band.
- **Skill handling**: rating is *conditioning input* via choice of head, not inferred latent. No per-player curve.
- **Claim**: predicts human moves with much higher accuracy than Stockfish/Leela across covered Elo range.
- **Cold-start**: N/A — models population not individuals.
- **Limitations**: top of range (>1900) poorly covered; models independent and discontinuous in skill.
- **Code**: https://github.com/CSSLab/maia-chess
- **Note**: Toronto PDF fetch denied.

### 4.2 Maia-2 — Tang et al. NeurIPS 2024 [READ PARTIAL]
arXiv:2409.20553. **12-block ResNet board encoder + categorical skill-level embedding (rating binned to 100-point buckets) + skill-aware multi-head self-attention fusing skill embedding into attention queries.** Three heads: policy, value, auxiliary (captures/checks/movement).
- **Input**: board state + rating bin of both players.
- **Most attention-heavy and most recent published "rating-aware" chess model.**
- **Dataset**: 169M Lichess rapid games, 9.1B positions (Jan 2013 – Nov 2023).
- **Empirical**: **+1.9pp move-prediction accuracy over Maia-1 (53.25% vs 51.32%), perplexity 4.07 bits.** 27% of test positions show monotonic skill-conditioned move correctness.
- **Uncertainty**: implicit via softmax over moves; no Bayesian posterior on skill.
- **Cold-start**: not modelled (skill is input, not inferred).
- **Limitations**: top bucket conflates Masters; no internal search; one-shot Elo conditioning vs inferring from games.
- **Code**: https://github.com/CSSLab/maia2

### 4.3 Personalised / Individualised Maia [SECONDARY]
McIlroy-Young et al. KDD 2022. Per-player fine-tuning on top of rating-conditioned Maia. **Needed ~5000 games per player to beat base model.** Follow-up "Maia4All" / "Learning to Imitate with Less" (arXiv:2507.21488, 2025) cuts this to **~20 games**.

### 4.4 GlickFormer — Miłosz et al. IEEE BigData 2024 Cup [SECONDARY]
arXiv:2410.11078. ChessFormer backbone + factorised spatio-temporal transformer predicting **Glicko-2 difficulty of a chess puzzle**. 4.16M puzzles. **Position-difficulty rating, not player rating** — but useful as example of Glicko-2 as transformer regression target.

### 4.5 DeepMind Searchless Chess Transformer [SECONDARY]
Ruoss et al. NeurIPS 2024, arXiv:2402.04494. 270M-parameter transformer predicts Stockfish action-values from board state; reaches Lichess Blitz 2895. **Not a rating model** but shows transformers can absorb strong-policy rating distribution. Could be calibrated to Elo via BT on outputs.
- **Code**: https://github.com/google-deepmind/searchless_chess

---

## 5. GNN Approaches

### 5.1 Tennis GNN with MagNet — Clegg & Cartlidge 2025 [READ PARTIAL]
arXiv:2510.20454. **MagNet** = spectral GNN for directed graphs using magnetic Laplacian (complex Hermitian, tunable q=0.25). Chebyshev polynomial order K=2, L=2 layers (4-hop receptive field). PyTorch Geometric Signed Directed.
- **Graphs**: three surface-specific directed graphs per gender (clay/grass/hard). Edges = dominance score `D = Σ α·β·φ·g / Σ α·β·φ` with α=surface transferability, β=tournament prestige, φ=exponential time decay (learnable λ=0.38).
- **Features**: static (height, weight, birth, hand); dynamic (surface-specific in/out-degree).
- **Empirical**: **65.7% acc / Brier 0.215 overall**, vs Weighted Elo **66.5% / 0.212**, standard Elo 65.8% / 0.215, Pinnacle odds 69.0% / 0.196. **GNN competitive with Elo variants but trails bookmaker market.**
- **Cold-start**: matches with no H2H isolated in separate bin and perform worst.
- **Uncertainty**: not addressed.
- **Limitations**: poor cold-start; less interpretable than Weighted Elo; loses information collapsing histories into edge weights.

### 5.2 GElo — Wang 2023 [READ PARTIAL]
arXiv:2304.08257. Skill-gap graph (undirected weighted) → random-walk graph embeddings (graph2vec-style) → adjust Elo updates so similar players move together.
- **Cold-start**: helped via embedding similarity to neighbours.
- **Limitations**: embedding step offline/batch; not true online rating system.

### 5.3 KEGC (Knowledge-Enhanced Graph Contrastive Learning) [SECONDARY]
Match sequence graphs with edges connecting similar matches to capture skill evolution. Match-outcome prediction. Abstract-level only.

### 5.4 GATv2-TCN [SECONDARY]
arXiv:2303.16741 "Who You Play Affects How You Play." Graph Attention Network v2 + temporal-convolution over dynamic player-interaction graph. Performance prediction, not skill rating.

---

## 6. State-Space / Kalman Variants with Neural Components

### 6.1 Bradley-Terry ANN — Menke & Martinez 2008 [SECONDARY]
*Neural Computing and Applications*. BT as single-layer ANN inside state-space view of skills as latent random walks. Predates deep-learning wave.

### 6.2 KalmanNet — Revach et al. 2022 [READ ABSTRACT]
arXiv:2107.10043, IEEE Trans. Signal Processing 2022. Replaces analytic covariance recursions in Kalman update with a GRU. **Skill-rating community has not yet applied KalmanNet to player skill — clear gap.**

### 6.3 Laplace-approximation dynamic rating — Szczeciński 2023 [SECONDARY]
arXiv:2310.10386. Laplace-approximation EKF with random-walk dynamics, variance lower bound.

### 6.4 Simplified Kalman "one-fits-all" — Szczeciński & Tihoń 2023 [SECONDARY]
arXiv:2104.14012. **Elo, Glicko, TrueSkill are instances of a single approximate-Kalman family.** Important theoretical unifier; not neural.

### 6.5 Neural Bradley-Terry Rating — Fujii ICAART 2024 [READ ABSTRACT]
arXiv:2307.13709. Replaces BT skill table with neural function `features → skill`. Handles items never directly compared. **Static, not temporal** — building block.

### 6.6 Neural / Transformer Hawkes Processes [SECONDARY]
Mei & Eisner 2017, Zuo et al. ICML 2020. Continuous-time event sequences with neural emissions. **Not yet applied directly to player rating**, but natural formalism for "events at irregular times + skill evolves continuously". **Clear research gap.**

### 6.7 DATELINE — Deep Plackett-Luce with uncertainty [SECONDARY]
Hu et al. 2018, arXiv:1812.05877. Deep network outputs PL scores + uncertainty for crowdsourcing-style problems. Closest to "deep PL rating system" though not temporal.

---

## 7. Career Trajectory and Age-Curve Modelling

### 7.1 GP Priors for Dynamic Paired Comparison — Ingram 2019 [READ PARTIAL]
arXiv:1902.07378. Replaces Wiener/random-walk WHR/TTT prior with **GP prior with generic kernel** (likely Matérn in implementation). Sparse-linear-algebra Laplace approximation. Bayesian-optimisation hyperparameter tuning.
- **Empirics**: 2018 ATP season; beats Elo and Glicko on log-loss, especially with surface covariates.
- **Code**: https://github.com/martiningram/paired-comparison-gp-laplace
- **This is the closest published peer of WHR/TTT and directly relevant to the user's stack.**

### 7.2 Maystre Flexible Time-Dynamics — KDD 2019 [READ PARTIAL]
See Agent 2 draft for full coverage. Continuous-time GPs with composable kernels; linear-time approximate-Bayesian inference. Most directly comparable to Ingram (2019); these two are leading GP-prior dynamic-rating papers.

### 7.3 Junior-to-Senior Tennis Trajectory ML [SECONDARY]
Various PLOS ONE 2023 papers apply random-forest/XGBoost over junior stats to predict ATP graduation. Descriptive ML, not generative skill-curve models.

### 7.4 NBA Bayesian Aging Curves — Vaci et al. 2019 [SECONDARY]
Decomposes aging into development + decline latent factors. **Tennis literature lacks direct equivalent.**

### 7.5 Peak Chess Age — Vaci et al. Nature Sci Rep 2025 [SECONDARY]
Parametric aging curves + ML. Template for analogous tennis study.

---

## 8. Gaps and Frontier

1. **Neural-emission state-space models for ratings.** Duffield et al. (2023) framework plugs in only Gaussian/finite-state emissions. **KalmanNet-style learned-gain dynamics applied to player skill is open.**
2. **GP-prior dynamic ratings with neural covariates.** Ingram (2019) and Maystre et al. (2019) use linear covariates. **Neural feature extractor over (surface, tournament, recent form) feeding GP-prior dynamic BT** = natural next step, fits user's "covariate" SystemSpec slot.
3. **Transformer-over-match-history for player skill.** Maia-2 conditions on Elo as input. **No published work takes transformer over a player's career match history → calibrated skill curve.** Closest is QuickSkill (cold-start, not transformer).
4. **Cold-start hybrid: deep predictor seeds Bayesian rating.** QuickSkill does this for novice MMO players. **Unexplored in tennis** for junior-to-senior transitions, where fine-tuned neural over junior stats could give prior mean/variance for WHR/TTT init.
5. **GNNs for ratings have not beaten classical baselines.** Clegg & Cartlidge (2025) is strongest tennis GNN result, still trails Weighted Elo and bookmakers. Intransitivity benefit real but architecture loses too much temporal information.
6. **Neural Hawkes / point-process for rating evolution.** No paper located applies neural Hawkes to player skill — yet matches are exactly an irregular event stream with intensity tied to activity/form.
7. **Uncertainty quantification.** Neural rating papers (Maia, GlickFormer, CNN-LSTM Omori) **uniformly drop the Bayesian posterior** that classical methods provide. **Amortised VI over the WHR/TTT graph with deep observation model — absent from published literature.**

---

## 9. Honest Disclosure of What Was Read

- **Direct full-text reads (arXiv HTML):** Maia-2 (2409.20553v1), State-Space Perspective (2308.02414v3), MagNet tennis (2510.20454v1), GP priors paired comparison (1902.07378), CNN-LSTM chess (2409.11506), Graph Embedding GElo (2304.08257), Maystre flexible time-dynamics (1903.07746 — abstract only; full PDF unparseable).
- **Abstract / structured metadata:** QuickSkill (2208.07704), Maia-1 (Toronto PDF denied), Maia individual KDD 2022 (snippets only), GlickFormer (2410.11078), Searchless chess (2402.04494), KalmanNet (2107.10043), Szczeciński Kalman papers (2104.14012, 2310.10386), Neural BT (2307.13709).
- **Search-snippet / secondary only:** Glickman 1999, Maia4All (2507.21488), DeepTennis Stanford CS230, Neural/Transformer Hawkes, Vaci aging-curve papers, Bradley-Terry ANN (Menke & Martinez 2008), KEGC graph-contrastive.
- **PDF reads that failed:** Maystre 2019 PDF, Online Counter Categories Elo-RCC PDF, GElo PDF — all returned binary FlateDecode. Toronto Maia KDD 2020 PDF outright denied.

---

## References

- Clegg & Cartlidge (2025). Intransitive Player Dominance in Tennis. arXiv:2510.20454.
- Coulom (2008). WHR. https://www.remi-coulom.fr/WHR/WHR.pdf
- Dangauthier, Herbrich, Minka, Graepel (2007). TTT. NIPS 2007.
- Duffield, Power, Rimella (2023). State-Space Perspective on Online Skill Rating. arXiv:2308.02414. https://github.com/SamDuffield/abile
- Fujii (2024). Neural Bradley-Terry Rating. ICAART 2024. arXiv:2307.13709.
- Glickman (1999). *Applied Statistics* 48.
- Ingram (2019). GP Priors for Dynamic Paired Comparison. arXiv:1902.07378. https://github.com/martiningram/paired-comparison-gp-laplace
- Maystre, Kristof, Grossglauser (2019). Flexible Time-Dynamics. KDD 2019. arXiv:1903.07746.
- McIlroy-Young et al. (2020). Maia. KDD 2020. arXiv:2006.01855. https://github.com/CSSLab/maia-chess
- McIlroy-Young et al. (2022). Individual Models in Chess. KDD 2022.
- Miłosz et al. (2024). GlickFormer. arXiv:2410.11078.
- Omori & Tadepalli (2024). Chess Rating from Moves and Clock Times. CG 2024. arXiv:2409.11506.
- Revach et al. (2022). KalmanNet. arXiv:2107.10043.
- Ruoss et al. (2024). Searchless Chess. NeurIPS 2024. arXiv:2402.04494.
- Szczeciński & Tihoń (2023). Simplified Kalman. arXiv:2104.14012.
- Tang et al. (2024). Maia-2. NeurIPS 2024. arXiv:2409.20553. https://github.com/CSSLab/maia2
- Wang (2023). GElo. arXiv:2304.08257.
- Zhang et al. (2022). QuickSkill. CIKM 2022. arXiv:2208.07704.
