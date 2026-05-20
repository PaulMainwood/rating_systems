# Agent 2: TrueSkill Family and Extensions, with Bayesian Neural Hybrids

**Honesty note (from agent):** WebFetch permitted for some URLs (mainly arXiv abstracts) and denied for others (MSR TrueSkill 2 PDF, Coulom WHR PDF, model-based ML book, OptMatch PDF, KDD page). **The Maystre/Kickscore arXiv PDF was retrieved as substantial prose [READ FULL]**. Everything else is [ABSTRACT/SECONDARY] from abstracts, Wikipedia, semantic-scholar metadata, blog digests, search snippets, or paper-implementation READMEs. Equations not independently verified.

---

## 1. Executive Summary

TrueSkill family forms one of three dominant lineages of probabilistic rating: (i) Elo/BT, (ii) Glicko/Glicko-2, (iii) TrueSkill/TTT/TrueSkill 2. Distinguished by **Thurstone-style Gaussian latent-skill formulation, team additivity, explicit draw margin, Bayesian inference by EP on a factor graph**.

Frontier moved in three directions:
1. **Maystre's kickscore (KDD 2019)** — GP priors on latent score functions; retains EP-style linear-time inference; unifies TTT, WHR, decayed history.
2. **OpenSkill** — Weng-Lin Plackett-Luce approximations; patent-free; order of magnitude faster.
3. **Neural-Bayesian hybrids (QuickSkill, Action2Score, GElo, OptMatch, DATELINE)** — introduce embeddings, attention, RL into match prediction/matchmaking; almost none place coherent Bayesian belief over neural parameters.

**No fully published system combines TrueSkill factor graph with Bayesian neural network for per-game observation model**, although all ingredients exist: neural-EP message learners, variational message passing on factor graphs, deep Plackett-Luce.

---

## 2. TrueSkill and TTT Foundations

### TrueSkill (Herbrich, Minka, Graepel 2006/2007) [ABSTRACT/SECONDARY]
- **Citation**: NIPS 2006 / NeurIPS 19, MIT Press pp. 569-576. MSR-TR-2006-80.
- **Model**: skill `s_i ~ N(μ_i, σ_i²)`, performance `p_i ~ N(s_i, β²)`, team performance = sum, outcome by draw margin ε.
- **Factor graph + EP**: chain of Gaussian factors (skill, perf, team sum) + non-Gaussian inequality factors. EP iterates, approximating non-Gaussian factors by moment-matched Gaussians. Skill posteriors remain Gaussian.
- **Display rating**: Xbox Live `μ₀=25, σ₀=25/3`, leaderboards show `μ_i - 3σ_i`.
- **Code**: sublee's Python `trueskill`, j2kun's pure-Python, moserware's C# `Skills`, ts-trueskill, saulabs Ruby. **US patent valid until 9 April 2029.**

### TrueSkill Through Time (Dangauthier et al. 2007) [ABSTRACT/SECONDARY]
- **Citation**: NIPS 2007 / NeurIPS 20.
- **Idea**: replace filtering with smoothing over entire time series. Skill at year `t` linked to `t-1, t+1` by Gaussian transitions `s_t ~ N(s_{t-1}, τ²)`. EP message passing along chains and across matches.
- **Application**: ~3.5M chess games over 150 years; recovers Kasparov/Capablanca/Lasker trajectories; corrects era inflation.
- **Limitations**: single τ², no covariates, expensive for huge graphs without sparse approx.
- **Code**: CRAN R package `TrueSkillThroughTime`. User's numba TTT is in this lineage.

---

## 3. TrueSkill 2 — the Proper Deep-Dive

[ABSTRACT/SECONDARY — PDF not retrieved. Detail from Wikipedia, Murphy/Hislaw blog digest, Semantic Scholar, search-snippet quotations, SkylakeXx GitHub README.]

- **Citation**: Tom Minka, Ryan Cleven, Yordan Zaykov. MSR Tech Report MSR-TR-2018-8, 22 March 2018. (Cleven was at The Coalition / Halo studio.)
- **Extensions over TrueSkill**:
  1. **Individual contribution from in-game stats**: kills/deaths/assists/score as observed leaf factors attached to player's performance node. Each stat has own noise and scale. **Most important change for modern shooter matchmaking.**
  2. **Squad/party effect**: co-queued subset gets positive bias factor on team performance.
  3. **Quit modelling**: mid-match quit = surrender for rating; posterior shifted down conditional on observed pre-quit state. Discourages ragequit-for-rating-protection.
  4. **Cross-mode skill correlation**: skills in modes A, B coupled by Gaussian factor with `ρ_AB`. New mode → priors informed by other modes. Solves a cold-start variant.
  5. **Biased skill evolution for new players**: `s_{t+1} ~ N(s_t + γ_t, τ²)` with positive drift `γ_t` decaying with games-played.
  6. **Batch/online modes**: can run online (forward propagation, like original) or batch (TTT-style smoothing).
- **Inference**: EP on extended factor graph; same moment-matching family.
- **Empirical**: Halo 5 — **TrueSkill 2 = 68% accuracy vs original TrueSkill 52%.** Integrated into Halo 5 ranked May 2018, Gears of War 4.
- **Limitations**: many parameters to fit offline; in-game stats assumed conditionally independent given performance node (not strictly true); still Gaussian-EP so heavy tails / multimodal squashed.
- **Code**: No official Microsoft release. `SkylakeXx/TrueSkill2` is third-party. `mmooyyii/trueskill` (Erlang).

**For tennis project**: most reusable ideas are (a) per-match individual-stat factors (serve-points-won, return-points-won, break-points-saved attached to player's match-performance node) and (b) **cross-surface skill correlation** (direct analogue of cross-mode correlation). Squad/quit machinery does not transfer.

---

## 4. Kickscore and Maystre's Flexible Time-Dynamics

[READ FULL — arXiv PDF rendered substantially into prose; section structure, kernel list, inference outline, experimental results all seen.]

- **Citation**: Lucas Maystre, Victor Kristof, Matthias Grossglauser. "Pairwise Comparisons with Flexible Time-Dynamics." KDD 2019. arXiv:1903.07746.
- **Idea**: generalise TTT and WHR to arbitrary temporal kernels. **Gaussian-process prior on each player's latent score function `s_i(t)`**. Observation = Bernoulli with logistic link on `s_i(t) - s_j(t)`. Draw margin can be added.
- **Kernels (composable)**:
  - Constant (recovers static BT)
  - Wiener / Brownian (recovers TTT / WHR)
  - Matérn (1/2, 3/2, 5/2 — differentiability control)
  - Exponential (OU-like decay)
  - Periodic (season cycles)
  - Polynomial trend
- **Inference**: coordinate-ascent EP. Markovian kernel structure → **linear time per iteration** with closed-form Kalman-filter/RTS-smoother messages. **The key trick that makes GP-based rating tractable on millions of observations.**
- **Empirical**: NBA, ATP tennis, football leagues, chess. Superior out-of-sample log-loss vs TrueSkill, Elo, decayed-Elo, especially with structured skill (career arcs, season cycles) that Wiener walk can't capture.
- **Code**: https://github.com/lucasmaystre/kickscore (MIT, Python). PyPI + conda.
- **Why it matters**: closer to true Bayesian inference than TrueSkill; richer prior on time-evolution but less flexible observation model (logistic only, not Thurstonian draw margin). **For tennis: natural next step from TTT; sum-of-kernels composition is the cleanest way to layer surface effects + career arc + within-season fluctuation.**
- **Limitations**: Bernoulli only; no individual-contribution stats; no covariates other than through kernel; no native multi-player teams.

### Related Maystre work [ABSTRACT/SECONDARY]
- Maystre, Grossglauser (2015). Fast and Accurate Inference of Plackett-Luce Models. NeurIPS 2015. Markov chain stationary-distribution → linear-time spectral inference. Underpins OpenSkill's speed.
- Maystre (2018). *Efficient Learning from Comparisons.* PhD thesis, EPFL. (Thesis Distinction award.)
- Maystre, Grossglauser (2017). Just Sort It! ICML 2017 / arXiv:1502.05556. Active comparison strategies.

---

## 5. OpenSkill and Other Plackett-Luce Variants

### OpenSkill (Joshy 2024) [ABSTRACT/SECONDARY]
- **Citation**: Vivek Joshy. *JOSS* 9(94):5901, 2024. arXiv:2401.05451.
- **Idea**: Python lib implementing Weng-Lin (2011) approximations to TrueSkill + Plackett-Luce variant (Guiver-Snelson 2009) extended with variance params. Five models: BradleyTerryFull/Part, ThurstoneMostellerFull/Part, recommended PlackettLuce. Closed-form Gaussian updates without factor-graph EP.
- **Why exists**: sidesteps TrueSkill US patent (valid until 2029) using Plackett-Luce updates Weng-Lin proved competitive empirically.
- **Empirical**: ≥3× speed-up over Heungsub Lee's `trueskill` Python pkg with comparable accuracy. Used by o!TR osu! tournament project.
- **Code**: `openskill.me`, Python pip + npm port. MIT-style.
- **Limitations**: closed-form Gaussian only; no covariates, no time-varying transition, no TrueSkill-2-style individual-contribution.

### Related Plackett-Luce / Neural [ABSTRACT/SECONDARY]
- **DATELINE (Han et al. 2018, arXiv:1812.05877)** — Deep Plackett-Luce with uncertainty. Neural net maps each instance to PL score, weighted by per-annotator quality vectors. Crowdsourced ranking, but conceptually closest to "Plackett-Luce + deep features + uncertainty."
- **Schäfer & Hüllermeier (2017/2018)** — PLNet / dyad ranking. PL on joint dyad embeddings.
- **Ma et al. (2020, arXiv:2006.05067)** — PL with partitioned preference for fast neural LTR. Scales to millions of items. IR-oriented.

---

## 6. Bayesian / Neural Hybrids for Skill Rating

**There is much less here than the question presupposes. No public production system combines TrueSkill-style Bayesian factor graph with Bayesian neural network observation model.** Partial steps:

### QuickSkill (Zhang et al., CIKM 2022) [ABSTRACT/SECONDARY]
- **Citation**: arXiv:2208.07704. CIKM 2022 Applied Research Track.
- **Idea**: novice cold-start. Deep net (LSTM) takes sequential per-game features from first few games, predicts converged skill, injected as prior into matchmaking. "Neural surrogate for first 10 games of TrueSkill."
- **Architecture**: LSTM over per-game stat vectors; trained with ground-truth = converged TrueSkill value after many games.
- **Uncertainty**: implicit; point estimate only. **Main weakness: not a coherent Bayesian extension.**
- **Empirical**: lower team-skill disparity in cold-start on NetEase mobile-game data.
- **Code**: not publicly released.

### Action2Score (Jang, Woo, Kim 2022) [ABSTRACT/SECONDARY]
- **Citation**: CHI Play 2022. arXiv:2207.10297.
- **Idea**: embed per-event actions (LoL match logs) into GRU; loss calibrated so per-action scores sum to contribution-to-victory. Decouples individual contribution from team outcome.
- **Relation to rating**: feature-extraction step, not rating system. Could plug into TrueSkill-2-like factor graph as per-player observed-stat node, but paper doesn't.

### GElo (Wang 2023) [ABSTRACT/SECONDARY]
- arXiv:2304.08257. Skill-gap graph from match history, DeepWalk embeddings, cosine similarity adjusts Elo updates. Augments Elo but transferable to TrueSkill. No uncertainty; embeddings computed offline, new players cannot be embedded without retraining.

### OptMatch (Gong et al., KDD 2020) [ABSTRACT/SECONDARY — PDF fetch denied]
- **Citation**: KDD 2020. https://linxiagong.github.io/OptMatch/
- **Idea**: two-stage matchmaking. Offline: player embeddings (relational + raw stats) + multi-head self-attention for team-level high-order interactions → win-probability for candidate lineup. Online: planning module assigns players to maximise engagement utility. 3v3 dataset, ~850k matches (Fever Basketball).
- **Relation**: neural replacement for matchmaking-prediction step. Embedding+attention architecture **is exactly what you'd plug into Bayesian factor graph as team-likelihood node — and nobody has, in print.**
- **Code**: `github.com/fuxiAIlab/OptMatch`.

### EOMM (Chen et al., WWW 2017) [ABSTRACT/SECONDARY]
- arXiv:1702.06820. From EA Sports research lab.
- **Idea**: "fair = balanced 50/50 = engaging" assumption is wrong. Treat matchmaking as optimisation over disengagement-risk via churn model + graph matching to maximise expected engagement.
- **Empirical**: 1v1 EA dataset, 36.9M matches, 1.68M players. Significant churn reduction.
- **Connection to TrueSkill 2**: TrueSkill 2's improvement-biased early drift is engagement nudge inside rating; EOMM does analogous nudge inside matchmaker.

### Bayesian Deep Learning Lurking Nearby [ABSTRACT/SECONDARY]
- **Neural EP**: Heess et al. 2013 trained NNs to learn EP messages, replacing moment-matching with learned function. **Conceptually closest to "Bayesian neural TrueSkill" but not applied to skill rating.**
- **Gaussian BP on deep factor graphs**: arXiv:2311.14649.
- **Variational message passing**: Akbayrak, Bocharov, de Vries (Entropy 2021).
- **Bayesian hierarchical tennis models**: Kovalchik & Reid (JQAS 2019). Stan-based hierarchical models with serve/return latent skills + surface + tournament intercepts. Bayesian but not neural; closest sport-specific Bayesian to user's project.

**Frontier in short**: all ingredients exist (neural EP messages, deep PL, GP-based skill, learned player embeddings). **Coherent published system fusing them — "TrueSkill 2 with team-likelihood factor replaced by Bayesian NN trained variationally" — does not, as far as I can find, exist.**

---

## 7. Industry / Production Systems

[ABSTRACT/SECONDARY for all — production systems publish little.]

- **Microsoft / Xbox Live**: TrueSkill (2007-2018) and TrueSkill 2 (2018-present, Halo 5, Gears of War 4). MSR tech report unusually transparent.
- **Riot Games (LoL, Valorant)**: hidden MMR distinct from displayed rank, persists across resets. Public statements imply TrueSkill-2-style covariate use ("more than KDA and W/L") but proprietary, no academic publication.
- **Valve (CS:GO/CS2, Dota 2)**: Glicko-2-based for CS:GO ranked per public Valve comms. CS2 introduced "CS Rating" — details not public. **Bober-Irizar/Dua/McGuinness (arXiv:2410.02831, 2024)** "Skill Issues: Analysis of CS:GO Skill Rating Systems" benchmarks Elo, Glicko-2, TrueSkill on historical CS:GO; finds TrueSkill superior on tight matches with data-efficiency caveats.
- **EA (FIFA, Apex)**: EOMM developed at EA Digital Platform. USPTO patents 11478716 "Deep learning for data-driven skill estimation" and 12364929 describe Siamese network trained on player stats (patent, not peer-reviewed).
- **NetEase / Fuxi AI Lab**: OptMatch (KDD 2020), EnMatch (AAAI 2024). Most-published industrial matchmakers since EA's EOMM.
- **DeepMind/OpenAI AlphaStar/OpenAI Five**: RL agents, not rating systems. Internal Elo-like evaluation for bot strength. **Not relevant to tennis question.**

---

## 8. Gaps and Frontier

1. **Clean Bayesian-neural hybrid for sport rating**: Nobody has published "TrueSkill 2 with Bayesian NN as per-match observation factor." Neural extensions are either point-estimate feature extractors (QuickSkill, Action2Score, GElo, OptMatch) or Bayesian without neural (kickscore, hierarchical Stan). **The move**: VMP on factor graph whose match-likelihood node is small Bayesian NN taking pre-match covariates. Neural-EP literature (Heess et al.) provides recipe.
2. **Multi-output kickscore with covariate kernels**: extend kickscore-style inference to condition on per-match covariates by additive kernels `s_i(t,x) = s_i^base(t) + h_surface(t) + β^T x`. Mathematically straightforward, not published.
3. **Cross-discipline / cross-surface correlation**: TrueSkill 2's cross-mode correlation = cross-surface in tennis. Bayesian factor-graph implementation with EP on tennis dataset would be natural derivative.
4. **Calibration**: all systems output beliefs; rarely audited out-of-sample. CS:GO paper (Bober-Irizar) is one of few. **Calibration-aware VI is a real research opportunity.**
5. **Replace EP with neural amortisation**: neural EP message-learners 12 years old now; modern transformer-message networks should learn TrueSkill-style updates and generalise across factor structures. Adding covariates would become trivial.
6. **Tennis-specific**: nearest published work to user's project = Kovalchik & Reid point-based Bayesian (JQAS 2019, 68.8% on 2014 ATP). User's TTT numba is in this family. **Unfilled niche**: add (a) point-level covariates, (b) TrueSkill-2-style individual-contribution machinery.

---

## 9. References

**TrueSkill family**
- Herbrich, Minka, Graepel (2007). TrueSkill. NIPS 2006 / MSR-TR-2006-80. [ABSTRACT/SECONDARY]
- Dangauthier, Herbrich, Minka, Graepel (2007). TrueSkill Through Time. NIPS 2007. [ABSTRACT/SECONDARY]
- Minka, Cleven, Zaykov (2018). TrueSkill 2. MSR-TR-2018-8. https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/ [ABSTRACT/SECONDARY — PDF fetch denied]

**Maystre / Kickscore**
- **Maystre, Kristof, Grossglauser (2019). Pairwise Comparisons with Flexible Time-Dynamics. KDD 2019. arXiv:1903.07746. https://github.com/lucasmaystre/kickscore [READ FULL]**
- Maystre, Grossglauser (2015). Fast and Accurate Inference of Plackett-Luce Models. NeurIPS 2015. [ABSTRACT/SECONDARY]
- Maystre, Grossglauser (2017). Just Sort It! ICML 2017. arXiv:1502.05556. [ABSTRACT/SECONDARY]
- Maystre (2018). *Efficient Learning from Comparisons.* PhD thesis, EPFL. [ABSTRACT/SECONDARY]

**OpenSkill / Plackett-Luce**
- Joshy (2024). OpenSkill. JOSS 9(94):5901. arXiv:2401.05451. https://openskill.me/ [ABSTRACT/SECONDARY]
- Han et al. (2018). DATELINE. arXiv:1812.05877. [ABSTRACT/SECONDARY]
- Schäfer, Hüllermeier (2017/2018). Dyad ranking with PL. [ABSTRACT/SECONDARY]
- Ma et al. (2020). PL with partitioned preference. arXiv:2006.05067. [ABSTRACT/SECONDARY]

**Other rating systems**
- Glickman. Glicko / Glicko-2. http://www.glicko.net/glicko.html [ABSTRACT/SECONDARY]
- Coulom (2008). WHR. https://www.remi-coulom.fr/WHR/WHR.pdf [ABSTRACT/SECONDARY — PDF fetch denied]
- Ebtekar, Liu (2021). Elo-like for Massive Multiplayer. arXiv:2101.00400. [ABSTRACT/SECONDARY]
- Tang et al. (2025). Is Elo Reliable? arXiv:2502.10985. [ABSTRACT/SECONDARY]
- Bober-Irizar, Dua, McGuinness (2024). Skill Issues: CS:GO. arXiv:2410.02831. [ABSTRACT/SECONDARY]

**Neural / matchmaking hybrids**
- Zhang et al. (2022). QuickSkill. CIKM 2022. arXiv:2208.07704. [ABSTRACT/SECONDARY]
- Jang, Woo, Kim (2022). Action2Score. CHI Play 2022. arXiv:2207.10297. [ABSTRACT/SECONDARY]
- Wang (2023). GElo. arXiv:2304.08257. [ABSTRACT/SECONDARY]
- Gong et al. (2020). OptMatch. KDD 2020. [ABSTRACT/SECONDARY — PDF fetch denied]
- Chen et al. (2017). EOMM. WWW 2017. arXiv:1702.06820. [ABSTRACT/SECONDARY]

**Neural EP / Bayesian deep learning bridges**
- Heess, Tarlow, Winn (2013). Learning to Pass EP Messages. NeurIPS 2013. [ABSTRACT/SECONDARY]
- Akbayrak, Bocharov, de Vries (2021). VMP and Local Constraint Manipulation. Entropy 23(7):807. [ABSTRACT/SECONDARY]
- Wilkinson et al. (2021). Sparse Algorithms for Markovian GPs. arXiv:2103.10710. [ABSTRACT/SECONDARY]
- Learning in Deep Factor Graphs with Gaussian BP. arXiv:2311.14649. [ABSTRACT/SECONDARY]

**Tennis-specific Bayesian**
- Kovalchik, Reid (2019). Point-based Bayesian hierarchical tennis model. *JQAS* 15(4):313-325. [ABSTRACT/SECONDARY]
- Gollub (2021). Forecasting serve performance. *J. Sports Analytics*. [ABSTRACT/SECONDARY]
