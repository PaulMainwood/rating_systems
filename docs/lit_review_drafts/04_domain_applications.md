# Agent 4: Deep Learning and Embeddings in Domain-Specific Rating Applications

**Honesty note (from agent):** PDF fetches mostly returned binary or were denied. Maia-2 HTML, GElo, GNN intransitivity, Heaton-Mitra MLB, Buhamra-Groll, Penn et al., Omi et al. abstract, Go strength abstract were [READ FULL] via HTML/abstract pages. Original Maia KDD 2020 PDF, KDD 2022 personalisation PDF, Behavioural Stylometry PDF, Ingram JQAS 2019 PDF, AlphaStar preprint, TrueSkill 2 PDF, Sackmann blog, OpenAI Dota 2 blog, DeepMind AlphaStar blog, AlphaZero/AlphaGo Zero Nature papers all [ABSTRACT/SECONDARY].

---

## 1. Executive Summary

Most coherent body of work = **chess Maia line** at CSSLab (Toronto) + Microsoft Research. Four recurring patterns across domains:
1. **Rating as conditioning signal** on a neural policy (Maia, Maia-2).
2. **Per-player embeddings** via stylometry or behavioural multi-task learning (McIlroy-Young 2022, Omi et al. 2025).
3. **GNNs** on win/loss graphs for intransitivity + surface effects (Clegg & Cartlidge 2025, Wang 2023).
4. **Latent skill vectors** in Bayesian hierarchical models with serve/return decomposition (Ingram 2019).

**Tennis-specific work** dominated by Bayesian hierarchical and graph methods rather than deep learning until very recently. **Field wide open for transferring Maia-style ideas.**

---

## 2. Chess: Maia and Beyond

### 2.1 Maia — McIlroy-Young et al. KDD 2020 [ABSTRACT/SECONDARY]
arXiv:2006.01855. Behaviourally-cloned variant of AlphaZero/Leela CNN, supervised to predict *next human move*. **Nine separate models, each trained on Lichess games from 100-Elo bucket (1100-1900).** Elo = partition variable for corpus, so each model implicitly represents population at one rating tier.
- **Claim**: matches human moves with substantially higher accuracy than Stockfish at comparable strength. **Maia-X peaks accuracy on games by humans rated near X** — "tunable" Elo behaviourally.
- **Code**: github.com/CSSLab/maia-chess. Lichess bots maia1/5/9. >2M games played vs the bots since release.
- **Transferability to tennis**: high but indirect. Lacks Maia-scale shot-by-shot data, but Match Charting Project could substitute. **Architectural lesson**: train rating-bucketed populations and let network learn the manifold of typical play at each tier.

### 2.2 Maia-2 — Tang, Jiao, McIlroy-Young, Kleinberg, Sen, Anderson NeurIPS 2024 [READ FULL]
arXiv:2409.20553. **One unified network with categorical skill-level embeddings (rating range → learned `d_s`-dim vector) injected via skill-aware attention in ResNet+Transformer.** Both players' skill embeddings concatenated and projected into attention queries:
```
Q*_k = Q_k + (e_a ⊕ e_o) W*
```
Network adaptively weights position features depending on the skill levels of both players.

**Why it matters**: clean instantiation of "**Elo as learned conditioning vector**". Categorical (not raw rating) → network learns non-linear skill effects without forcing monotonicity.

**Code**: github.com/CSSLab/maia2 (and github.com/lilv98/maia2-submission).

**Transferability to tennis** — **arguably the single most directly portable idea in the literature**. Analogue: server's surface-specific Elo + returner's surface-specific Elo → embeddings → conditioning of a shot/rally prediction net. Existing tennis_ratings system, which maintains per-player Elo states, could supply conditioning vectors with minimal change.

### 2.3 Personalised Maia — KDD 2022 [ABSTRACT/SECONDARY]
arXiv:2008.10086. Fine-tune Maia on single named player's games. **4-5% higher per-player accuracy than best base Maia, but only when player has ≥5000 games.**
- **Transferability**: 5000-game threshold problematic at match level (~60-80 matches/year top ATP). **But at point level**: 200-match top-100 player → 10-30k charted points, comparable scale.

### 2.4 Behavioural Stylometry — McIlroy-Young et al. 2022 [ABSTRACT/SECONDARY]
arXiv:2208.01366. **Transformer-based few-shot classifier identifies specific player from their games. 98% accuracy on identifying one of thousands of players given only 100 labelled games.** Amateur-trained models generalise to GM out-of-distribution. Learned embedding space clusters stylistically similar players.
- **Transferability to tennis**: high. Identifying "who played this rally" from MCP shot sequences → tennis-style embedding space, could augment Elo with a learned "style" coordinate.

### 2.5 Generative Stylometry at Scale — Omi et al. ICLR 2025 [READ FULL]
arXiv:2502.14998. **Per-player modelling as multi-task learning with parameter-efficient fine-tuning (PEFT). Each of 47,864 chess players (and 2000 Rocket League, 10,177 celebrities) gets explicit style vector that selectively activates shared "skill" parameters.** Style vectors are **generative** (produce actions in player's style) and **steerable**.
- **Transferability**: most modern formulation of personalisation. Treat each tour player as task; shared backbone + per-player style vectors. **Style vector dimensionality could be tiny (8-16D) and become a rating-system covariate.**

### 2.6 AlphaZero / Leela Chess Zero Rating [ABSTRACT/SECONDARY]
Silver et al. *Science* 362(6419) 2018, arXiv:1712.01815. Internal rating: periodic tournaments between checkpointed network versions → empirical win matrix → MLE Bradley-Terry → Elo. AlphaZero reached highest Elo of any computer program after 10 hours. **Rating is downstream evaluation metric, not training signal.**
- **Transferability**: limited (no self-play in tennis), but pairwise-tournament Elo is universal evaluation technique.

### 2.7 Chess2vec, ChessPos and Position Embeddings [ABSTRACT/SECONDARY]
Kapicioglu et al. arXiv:2011.01014. Vector representations for chess pieces via word2vec-analogue on bitboard tensors (8×8×12). Position-level not player-level, but establishes embedding-everything precedent.

---

## 3. Esports

### 3.1 AlphaStar (StarCraft II) Nature 2019 [ABSTRACT/SECONDARY]
Vinyals et al., *Nature* 575:350-354, 2019. **League of agents** trained with **prioritised fictitious self-play** where opponents sampled from a population. Pairwise win rates assign Elo-like ratings → inform sampling priors via PFSP. After Battle.net European release, AlphaStar Final earned **Grandmaster rank for all three races** under Blizzard's MMR (TrueSkill-style), top 0.15%.
- **Transferability**: conceptually low for tennis (no league self-play), but PFSP idea useful for synthetic-opponent generation in tactical models.

### 3.2 OpenAI Five (Dota 2) [ABSTRACT/SECONDARY]
arXiv:1912.06680. **Used TrueSkill to track agent improvement over self-play.** Report "~8.3 TrueSkill ≈ 80% win rate." TrueSkill = internal yardstick for compute scaling.
- **Transferability**: low directly; reminder that classical Bayesian rating systems remain the *evaluation* lingua franca in deep-RL self-play.

### 3.3 TrueSkill 2 (MSR 2018) [ABSTRACT/SECONDARY]
See Agent 2 draft for full coverage. **Reported 68% accuracy vs TrueSkill's 52% on historical Halo match outcome prediction.**
- **Transferability**: direct conceptual analogy. TrueSkill 2's "kills/squads/experience" → tennis "service hold %, surface, recent injury, tournament round". Bayesian covariate update is essentially what wglicko already does — **TrueSkill 2 useful reference architecture for principled covariate integration.**

### 3.4 League of Legends MMR & Riot [ABSTRACT/SECONDARY]
Undocumented in detail; per-queue Bayesian rating with hidden params. **No published research extends it with embeddings.**

### 3.5 CS:GO HLTV Rating 2.0 [ABSTRACT/SECONDARY]
Linear combination of five per-round normalised features (KAST, KPR, DPR, Impact, ADR), each in standard deviations above/below position-mean. **Not Bayesian or neural — performance metric, not skill rating.** Reverse-engineered weights (0.0073, 0.3591, −0.5329, 0.2372, 0.0032, intercept 0.1587) suggest linear regression fit.
- **Transferability**: similar per-match standardised metric for tennis (combining hold%, return points won, dominance ratio) could be useful *feature* alongside Elo.

### 3.6 Overwatch / Blizzard [ABSTRACT/SECONDARY]
Published material in *patent* form, not academic. **US patents 11,478,716 and 12,364,929**: Siamese neural network on per-player in-game stats, contextual conditioning on maps/modes/roles. Siamese network compares two players directly → relative-skill prediction.
- **Transferability**: Siamese over engineered feature vectors = natural fit for tennis match prediction. Essentially what XGBoost-on-Elo-features achieves but with stronger inductive bias.

---

## 4. Tennis-Specific Work

### 4.1 Klaassen & Magnus — IID and Point-Based Foundation [ABSTRACT/SECONDARY]
- *JASA* 96(454):500-509, 2001. Tennis points **neither independent nor identically distributed**: serving on important points harder; prior-point outcome carries info. Yet i.i.d. is useful first approximation.
- *EJOR* 148(2):257-267, 2003. All subsequent point-based tennis modelling builds on this.

### 4.2 Glickman's Broader Rating Work [ABSTRACT/SECONDARY]
No tennis-specific paper by Glickman. Closest: Glickman & Hennessy 2015 multi-competitor games (JQAS); Glickman, Hennessy & Bent 2018 beach volleyball. Tennis researchers (Kovalchik, Vaughan-Williams) have applied Glicko; Glickman himself has not.

### 4.3 Kovalchik — Most Influential Tennis Rating Researcher [ABSTRACT/SECONDARY]
- *JQAS* 12(3):127-138, 2016 — "GOAT of tennis win prediction": **FiveThirtyEight's 50/50 surface-blended Elo best of several public methods at beating bookmaker prices.**
- *IJF* 36(4):1329-1341, 2020 — Elo with margin of victory (sets/games/break-points). K factor as function of dominance.
- arXiv:2202.00583, 2022 — Statistical model of serve return impact patterns (with Albert).
- **Classical stats: no NNs, no embeddings. Sets empirical benchmark all subsequent work must beat.**

### 4.4 FiveThirtyEight + Sackmann Surface Blend [ABSTRACT/SECONDARY]
- Silver & Bialik 2016 blog: 538 tennis Elo.
- Sackmann blog 2019 — **maintains four parallel Elos (overall, hard, clay, grass), reports surface predictions as 50/50 blend of single-surface + overall.** Empirically determined.

### 4.5 Ingram Point-Based Bayesian Hierarchical [ABSTRACT/SECONDARY]
*JQAS* 15(4):313-325, 2019. Each player has latent **serve skill + return skill**, each Gaussian random walk through time, depends on surface + tournament. Hierarchical point-win probability. MCMC fit.
- **Empirical**: 2014 ATP: 68.8% acc vs 66.3% best comparator; log loss 0.592 vs 0.641.
- **Cleanest model to neuralise**: replace linear surface/time effects with neural function of (player_id_embedding, opponent_id_embedding, surface, time, score_context).

### 4.6 Recent Deep-Learning Tennis Work (2023-2026)

**Clegg & Cartlidge 2025 [READ FULL]** — GNN for intransitivity. arXiv:2510.20454. Build temporal directed graphs, apply MagNet (spectral GCN for directed graphs). **Women's tennis 11.5% more intransitive than men's.** Betting on high-intransitivity matchups: **65.7% accuracy, 3.26% ROI with Kelly staking over 1903 bets.** *Most directly relevant existing tennis paper combining GNN with market signal.*

**Buhamra & Groll 2025 [READ FULL]** — arXiv:2502.01613, Statistical Enhanced Learning for Grand Slam Tennis. **Elo as feature inside random forests / regression, not final score.** Modest improvements; mostly confirms Elo carries information XGBoost cannot recover from raw features alone.

**Penn, Michael & Bhatt 2025 [READ FULL]** — arXiv:2508.15956. Dynamic Graph Forecasts of Bookmaker Odds. **Forecasts bookmaker odds (not match outcome), using dynamic graph model.** Comparable to bookmakers, better than rankings.

**Various transformer/LSTM Wimbledon studies (2024-2025)** — predominantly small-scale supervised on shot-level data with engineered features. None learn explicit player embeddings or compete with established Elo-based methods on out-of-sample tour data.

**Note: no tennis equivalent of Maia exists.** No published paper trains a NN to predict the next shot a specific tour player will hit conditional on their rating. **Clear gap.**

---

## 5. Team Sports

### 5.1 Soccer / Football
- **FiveThirtyEight SPI** (Silver 2009, rev 2014). Offensive + defensive ratings → SPI. World Cup: 75% match-based + 25% roster-based. Uses xG, non-shot xG, red-card-adjusted goals. Not neural.
- **Opta xG**. XGBoost on ~1M historical shots with 20+ contextual features. Distance-to-goal dominates.
- **Gomez et al. 2020** Bradley-Terry-with-NMF for tennis & football. Nonnegative matrix factorisation extends BTL with latent player factors.

### 5.2 Basketball
- **RAPM / DAPM family**. Penalised least-squares estimation of per-player offensive/defensive coefficients on possession-level data. Bayesian regularisation (Sill 2010) is foundation of DARKO, BPM. Cleanest "learn per-player vector by jointly regressing all simultaneous appearances".
- **NBA2Vec / Player2Vec** arXiv:2302.13386. 8-dimensional shared player embeddings predicting per-play outcome distribution.
- **baller2vec** (Alcorn 2021) arXiv:2102.03291. Multi-entity transformer on player trajectories — spatiotemporal embeddings.

### 5.3 Baseball
- **Heaton & Mitra 2021 [READ FULL]** arXiv:2109.05280. Contrastive learning → **72-dim form vectors per player capturing recent-game impact patterns** not visible in classical sabermetrics. Reveals form clusters distinct from classical-stat clusters.
- **Statcast neural pitch classification**. MLB Baseball Savant. Not rating per se, but illustrates embedding-by-pitcher pattern.

---

## 6. Cross-Domain Lessons for a Tennis Researcher

Transferable ideas that consistently work:

1. **Rating as learned conditioning vector** (Maia-2). Bucket Elo, embed it (8-64 dim), inject via attention or feature-modulation. For tennis: surface-specific Elo of server, returner → embeddings → conditioning of shot/rally prediction net. **Single most actionable idea from chess literature.**

2. **Per-player latent style vectors via multi-task PEFT** (Omi et al. 2025). Shared backbone + low-rank per-player adapter. 47k chess players supported at scale; top-500 ATP/WTA trivial. Tennis style vectors could become covariates in the rating update — a "style-adjusted" handicap.

3. **Behavioural stylometry as side task** (McIlroy-Young 2022). Train model to identify player from rally patterns; resulting embedding = free per-player feature. MCP has enough data for top 200 players.

4. **Hierarchical decomposition with neural inner models** (Ingram 2019 + Maia). Keep serve/return Bernoulli outer probabilistic model, but let per-point probability be neural function of (rating embeddings, surface, score context, momentum). **Interpretability + expressive power.**

5. **GNNs for non-transitive matchups** (Clegg & Cartlidge 2025; Wang 2023). Elo assumes transitivity; tennis violates it (Nadal-Djokovic-Federer cycles). GNN over historical win-loss graph can learn corrections to Elo as a *residual* — neural handicap on top of base wglicko/WHR.

6. **Margin-of-victory extensions** (Kovalchik 2020; TrueSkill 2 2018). Both domains independently found *how much* you won by improves rating quality. Neural margin model (predicting dominance ratio from features) feeding Bayesian update K-factor is clean way to combine.

7. **Evaluate models with BT on pairwise self-play tournaments** (AlphaZero, OpenAI Five). For any tennis-prediction model shipped: pairwise tournament Elo on fixed dataset = useful simulation-free yardstick.

**Skippable**: deep-RL self-play paradigm (no tennis simulator). HLTV-style hand-tuned linear performance score is what wglicko already does implicitly, just better.

---

## 7. References

**Chess**
- McIlroy-Young et al. (2020). Maia. KDD 2020. arXiv:2006.01855. https://github.com/CSSLab/maia-chess [ABSTRACT/SECONDARY]
- Tang et al. (2024). Maia-2. NeurIPS 2024. arXiv:2409.20553. https://github.com/CSSLab/maia2 [READ FULL]
- McIlroy-Young et al. (2022). Individual Behavior in Chess. KDD 2022. arXiv:2008.10086 [ABSTRACT/SECONDARY]
- McIlroy-Young et al. (2022). Behavioral Stylometry. arXiv:2208.01366 [ABSTRACT/SECONDARY]
- Omi et al. (2025). Generative Modeling of Individual Behavior at Scale. ICLR 2025. arXiv:2502.14998 [READ FULL]
- Kapicioglu et al. (2020). Chess2vec. arXiv:2011.01014 [ABSTRACT/SECONDARY]
- Silver et al. (2018). AlphaZero. *Science* 362(6419). arXiv:1712.01815 [ABSTRACT/SECONDARY]

**Esports**
- Vinyals et al. (2019). AlphaStar. *Nature* 575:350-354 [ABSTRACT/SECONDARY]
- OpenAI (2019). OpenAI Five. arXiv:1912.06680 [ABSTRACT/SECONDARY]
- Minka, Cleven, Zaykov (2018). TrueSkill 2. MSR-TR-2018-8 [ABSTRACT/SECONDARY]
- HLTV (2017). Rating 2.0 [ABSTRACT/SECONDARY]
- Joshy (2024). OpenSkill. JOSS. arXiv:2401.05451 [ABSTRACT/SECONDARY]
- Wang (2023). GElo. arXiv:2304.08257 [READ FULL]

**Tennis**
- Ingram (2019). Point-based Bayesian hierarchical. *JQAS* 15(4):313-325 [ABSTRACT/SECONDARY]
- Clegg & Cartlidge (2025). Tennis GNN. arXiv:2510.20454 [READ FULL]
- Buhamra & Groll (2025). Statistical Enhanced Learning. arXiv:2502.01613 [READ FULL]
- Penn, Michael, Bhatt (2025). Dynamic Graph Forecasts. arXiv:2508.15956 [READ FULL]
- Klaassen & Magnus (2001). *JASA* 96(454):500-509 [ABSTRACT/SECONDARY]
- Klaassen & Magnus (2003). *EJOR* 148:257-267 [ABSTRACT/SECONDARY]
- Kovalchik (2016). GOAT. *JQAS* 12(3):127-138 [ABSTRACT/SECONDARY]
- Kovalchik (2020). MOV Elo. *IJF* 36(4):1329-1341 [ABSTRACT/SECONDARY]
- Glickman (1999). *Applied Statistics* 48:377-394 [ABSTRACT/SECONDARY]

**Team sports**
- 538 Soccer SPI (2009/2014) [ABSTRACT/SECONDARY]
- Heaton & Mitra (2021). MLB form embeddings. arXiv:2109.05280 [READ FULL]
- NBA2Vec. arXiv:2302.13386 [ABSTRACT/SECONDARY]
- Alcorn (2021). baller2vec. arXiv:2102.03291 [ABSTRACT/SECONDARY]

**Board games / AI**
- Silver et al. (2017). AlphaGo Zero. *Nature* [ABSTRACT/SECONDARY]
- Egri-Nagy & Törmänen (2020). Go strength assessment. arXiv:2009.01606 [READ FULL via abs]
