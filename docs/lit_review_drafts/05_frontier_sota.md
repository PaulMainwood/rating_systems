# Agent 5: Frontier 2023-2026 — Deep Learning, Foundation Models, LLMs + Paired-Comparison Ratings

**Honesty note (from agent):** No full paper PDF body was read this session. **Every "ABSTRACT" entry** = arXiv landing page retrieved (title, authors, abstract, dates). Methodological descriptions beyond the abstract = search summaries / aggregator snippets, may contain transcription errors. **[SECONDARY]** = could not even retrieve landing page. WebFetch denied for the Maia-2 PDF on Toronto's domain; arXiv abs page substituted. Some "2026" arXiv numbers (e.g. 2602.10286, 2603.15212, 2604.05460) may be future-dated index artefacts; not verified.

---

## 1. Executive Summary: Where is the Frontier (2024-2026)?

Intersection of deep learning with paired-comparison rating has exploded since late 2023, but **almost entirely on the LLM-evaluation / RLHF side, not classical sports rating**. Five frontiers:

1. **Bradley-Terry is now de facto rating model of LLM evaluation.** LMSYS/Chatbot Arena formally switched from incremental online Elo to BT MLE with bootstrap CIs in late 2023; codified in Chiang et al. 2024 (arXiv:2403.04132). Most newer "LLM leaderboard" papers fit BT coefficients + style/feature regression to mitigate confounding.

2. **DPO and descendants reframe RLHF as a Bradley-Terry problem.** Rafailov et al. DPO (NeurIPS 2023) shows policy training itself fits a BT likelihood; KTO, IPO, NLHF, SPPO, GPM each relaxed/generalised the BT assumption. **Dominant theoretical area at NeurIPS/ICML/ICLR.**

3. **Critiques of BT's transitivity assumption are everywhere in 2024-2026.** GPM, NLHF, Neural BT, 2026 statistical survey (Fang et al.) all argue intransitive preferences need latent / preference-embedding representations.

4. **Foundation models for sports/games are emerging** (Foundation Model for Soccer, RisingBALLER, Maia-2) but mostly *outcome/action* predictors. Rarely instantiate an explicit BT/Glicko/TrueSkill head. Closest: **Maia-2's skill-aware attention** conditioning a transformer on coarse Elo bucket.

5. **Little frontier work directly modernises Glicko/TrueSkill/WHR with deep learning.** Most 2024-2026 extensions still classical (MOVDA margin-of-victory, am-ELO with annotator abilities, GElo 2023). **Real gap.**

**User's tennis stack (Elo/Glicko/Stephenson/TrueSkill/WHR/TTT) sits in a research area substantially bypassed by the LLM-driven re-discovery of paired comparison.**

---

## 2. LLM Evaluation as Rating: Chatbot Arena and Beyond

### 2.1 Foundational Platform Papers

**Zheng, Chiang et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685** [ABSTRACT]
Original "Chatbot Arena meets MT-Bench" paper. MT-Bench = 80 expert-curated multi-turn questions. **Claim**: GPT-4 as judge ≥80% agreement with humans, matching inter-human agreement. Documents major biases: position, verbosity, self-enhancement. Original ratings used incremental online Elo. Released 30K conversations + human prefs. (FastChat repo.)

**Chiang, Zheng, Sheng, Angelopoulos et al. (2024). Chatbot Arena. ICML 2024. arXiv:2403.04132** [ABSTRACT]
Methodology paper. By March 2024: >240K votes over >50 models. **Central move: switched from streaming Elo to static Bradley-Terry MLE on full vote graph**, mapped to Elo scale, with **bootstrap confidence intervals**. Bias-aware sampling and discriminating-prompt selection. Validates crowdsourced votes agree with experts. Limitations: no formal mitigation for vote-rigging or style confounds.

### 2.2 Style Control and Confounding

**LMSYS Style Control Blog (Li, Frick et al., Aug 2024)** [SECONDARY]
Extended BT regression with **style features as independent confounders**: normalised difference in response length, markdown headers, lists. After style control, GPT-4o-mini and Grok-2-mini drop while Claude 3.5 Sonnet/Opus and Llama-3.1-405B rise. **Standard "Style-Controlled Arena" leaderboard variant.**

**Length Bias in LLM-Based Preference (arXiv:2407.01085, 2024)** [SECONDARY]
GPT-4-as-judge systematically prefers longer responses.

**Dubois et al. (2024). Length-Controlled AlpacaEval. arXiv:2404.04475** [SECONDARY]
Length-controlled AlpacaEval: Spearman correlation with human Chatbot Arena leaderboard 0.94 → 0.98.

### 2.3 Critiques and Robustness

**Zhao, Rush, Goyal (2024). Challenges in Trustworthy Human Evaluation of Chatbots. arXiv:2412.04363** [ABSTRACT]
Three classes of bad annotations (apathetic, adversarial, malicious) can corrupt Arena rankings. **10% poor-quality votes can shift rankings by up to 5 places.** PoC attack: programmatically voted on Chatbot Arena Oct 13 2024, bypassed bot detection. **BT MLE is annotator-fragile.**

**Vyas et al. (2025). Dropping a Handful of Preferences Can Change Top LLM Rankings. arXiv:2508.11847** [SECONDARY]
Top-end BT/Elo rankings sensitive to removal of small numbers of pairwise votes. **MT-Bench more robust than Arena due to expert annotators / curated prompts.** Direct implication for sparse-data sports ratings.

**Min et al. (2025). Improving Your Model Ranking on Chatbot Arena by Vote Rigging. arXiv:2501.17858** [SECONDARY]
Active attack on BT-based Arena scores.

**Liu et al. (2025). am-ELO: Stable Framework for Arena-based LLM Evaluation. ICML 2025. arXiv:2505.03475** [ABSTRACT]
Two contributions:
- **m-ELO**: replaces iterative Elo with MLE → theoretical consistency guarantees.
- **am-ELO**: modifies BT/Elo so **annotator ability jointly estimated with model scores** — IRT for raters.
**Directly relevant**: weighting noisy annotators (or judges, or umpires).

**Wu et al. (2024). Arena-Hard and BenchBuilder. arXiv:2406.11939** [ABSTRACT]
Arena-Hard-Auto: 500 difficult prompts auto-curated via BenchBuilder, judged by GPT-4.1/Gemini-2.5. **98.6% correlation with Chatbot Arena human preferences, 3× model separation vs MT-Bench. Production cost ~$20.** Closest 2024 thing to "Elo on demand for new models."

**Zhuo et al. (2024). Auto-Arena. arXiv:2405.20267** [SECONDARY]
LLMs play peer-battle pairs, committee of LLM judges votes after deliberation. **92.14% correlation with human preferences with zero human effort**, uses BT-style aggregation.

**Boubdir et al. (2024). Judging the Judges: Position Bias in LLM-as-a-Judge. arXiv:2406.07791** [SECONDARY]
Position bias varies sharply across judges/tasks; modulated by quality gap.

**Li et al. (2024-2025). A Survey on LLM-as-a-Judge. arXiv:2411.15594** [SECONDARY]
Bias coverage and mitigation.

---

## 3. RLHF, DPO, IPO, KTO, NLHF, SPPO — Bradley-Terry as Loss Function

### 3.1 DPO and the BT Closed-Form Trick

**Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (2023). Direct Preference Optimization. NeurIPS 2023. arXiv:2305.18290** [ABSTRACT]
Pivotal paper. DPO **assumes BT preference model**, re-parameterises optimal RLHF reward through closed-form mapping to policy log-ratios. Policy training becomes binary cross-entropy on (chosen, rejected) pairs. Implicit reward `r(x,y) = β log π(y|x)/π_ref(y|x) + const`. **No PPO loop, no separate reward model.** Matches/exceeds PPO-RLHF on summarisation, sentiment. **Conceptual contribution to rating: BT log-likelihood now used directly to train function approximators (LMs) — a 70B-parameter "neural Bradley-Terry rater".**

### 3.2 First Departures: ΨPO and IPO

**Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos (2024). ΨPO / IPO. AISTATS 2024. arXiv:2310.12036** [ABSTRACT]
DeepMind's DPO response. ΨPO = generic preference-learning objective; DPO and RLHF are special cases. **IPO** drops BT logistic link; trains so log-probability gap between chosen and rejected hits *fixed* margin. **More robust to overfitting when preferences are deterministic.**

### 3.3 KTO: Kahneman-Tversky Prospect Theory

**Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela (2024). KTO. ICML 2024. arXiv:2402.01306** [ABSTRACT]
Frames alignment through Kahneman-Tversky utility rather than BT preferences. **Needs only binary thumbs-up/down, not paired comparisons.** Matches/beats DPO at 1B-30B. **Most radical break — explicit denial that paired comparison is necessary.**

### 3.4 Reward Modelling Beyond BT

**Sun, Shen, Ton (2025). Rethinking Bradley-Terry Models in Preference-Based Reward Modeling. ICLR 2025. arXiv:2411.04991** [ABSTRACT]
Convergence rates for BT reward models on deep embeddings. **Argues BT not required: only order consistency matters for downstream policy opt; any monotone transform suffices.** Proposes simpler upper-bound classifier. 12,000+ setups, 6 base LLMs, 2 datasets. Code: github.com/holarissun/RewardModelingBeyondBradleyTerry.

**Zhang, Zhang, Wu, Xu, Gu (2025). Beyond Bradley-Terry: GPM. ICML 2025. arXiv:2410.02197** [ABSTRACT]
General Preference Model: embeds each response in learned latent space whose inner-product structure encodes **intransitive (cyclic) preferences with 100% accuracy** — where BT necessarily does chance. **Outperforms BT on RewardBench by up to 5.6%.** Companion GPO algorithm. Code: github.com/general-preference/general-preference-model.

**Munos, Valko, Calandriello, Azar, Rowland, Guo et al. (2023/2024). Nash Learning from Human Feedback. arXiv:2312.00886** [ABSTRACT]
Recasts alignment as **symmetric two-player preference game** with preference oracle; solution = Nash equilibrium, not BT max. Nash-MD (mirror descent) and deep-learning variant. **Explicit motivation: BT assumes transitivity, real preferences intransitive.** Spawned SPPO, Iterative Nash PO, Magnetic PO, Proximal Point NLHF, Multiplayer Nash PO.

**Wu, Sun, Yuan, Ji, Yang, Gu (2024). Self-Play Preference Optimization. NeurIPS 2024. arXiv:2405.00675** [ABSTRACT]
SPPO = constant-sum two-player game with iterative policy updates approximating Nash. **Bypasses BT entirely.** 60K UltraFeedback prompts + PairRM oracle → Mistral-7B reaches **28.53% length-controlled win-rate vs GPT-4-Turbo on AlpacaEval 2.0.** Beats iterative DPO/IPO on MT-Bench, Arena-Hard, Open LLM. Code: github.com/uclaml/SPPO.

**RewardBench (Lambert, Pyatkin, Morrison et al., arXiv:2403.13787, 2024) and RewardBench 2 (arXiv:2506.01937, 2025)** [ABSTRACT for v1]
Standard benchmark for reward models. Evaluates both BT classifiers and DPO-implicit rewards. Code: github.com/allenai/reward-bench.

**Joint BT + regression heads (arXiv:2510.15242, 2507.07375, 2506.01937)** [SECONDARY]
**2025-2026 wave argues BT and regression heads should be trained jointly** in shared backbone, combining ordinal + cardinal signals.

### 3.5 Reward Over-Optimisation as Rating Problem

**Rafailov et al. (2024). Scaling Laws for Reward Model Overoptimization in Direct Alignment. arXiv:2406.02900** [SECONDARY]
DPO and friends still over-optimise their implicit BT reward.

**PURM (arXiv:2503.22480) and BNRM (arXiv:2602.10623)** [SECONDARY]
Generalise BT to learn **distribution over rewards**, enabling uncertainty-aware policy training. **Bayesian-deep-learning analogue of Glicko's variance estimates.**

---

## 4. Recent Neural Rating Systems and Extensions (2023-2026)

**Honest finding: much thinner than LLM literature.**

**Fujii (2024). Neural Bradley-Terry Rating. ICAART 2024. arXiv:2307.13709** [ABSTRACT]
Cleanest "deep BT". Embeds BT score as output layer of NN; cross-entropy = BT likelihood. Supports multiway comparisons, feature conditioning, end-to-end training. **Useful template** if user wants learnable strength head conditioned on player covariates.

**Wang (2023). GElo. arXiv:2304.08257** [ABSTRACT]
Skill-gap graph → node2vec embeddings → cosine similarity + activeness adjust Elo updates.

**Lin and Wu (2025). Online Learning of Counter Categories and Ratings in PvP Games. arXiv:2502.03998** [ABSTRACT]
Extends Elo for RPS-style intransitivity via **dynamically learned counter categories**. Updates online without offline NN training. Maintains scalar interpretability while absorbing intransitivity into discrete categories.

**Lin et al. (2024b). Neural Rating Table / Neural Counter Table** [SECONDARY]
Referenced in PvP-counter paper. Siamese network on team features + vector quantisation of residual win values into interpretable counter categories. **No clean arXiv anchor found.**

**Tang et al. (2024). Maia-2. NeurIPS 2024. arXiv:2409.20553** [ABSTRACT]
Most relevant deep-learning paper for rating-conditioned modelling. Single transformer with **skill-aware attention** taking player's Elo bucket as input. **Architecture consumes a rating to condition predicted move distribution, rather than producing one.**

**Joshi et al. (2024). Chess Rating Estimation from Moves and Clocks (CNN-LSTM). arXiv:2409.11506** [SECONDARY]
**Inverts direction**: CNN-LSTM consumes positions + clocks, **predicts player's Elo move-by-move**, MAE 182 rating points on 1.2M Lichess games. First move-by-move rating predictor with no hand-crafted features. **Near-direct analogue of what user could build for tennis.**

**Shorewala and Yang (2025). MOVDA: Margin of Victory Relative to Expectation. arXiv:2506.00348** [ABSTRACT]
Scaled-tanh model of expected margin given rating differential; rating updates weighted by deviation between observed and expected MOV. **On 13,619 NBA games (2013-2023): reduces Brier-score error by 1.54% vs TrueSkill, +0.58% accuracy, 13.5% faster convergence, Elo-comparable compute.** Classical not deep, but cutting-edge in classical-rating sub-field. **Interesting for tennis (set scores).**

**Tognon et al. (2024). Skill Issues: CS:GO Skill Rating Systems. arXiv:2410.02831** [SECONDARY]
Empirical comparison of Elo/Glicko/TrueSkill variants on CS:GO data.

**Maitra (2025). Empirical Parameterization of Elo. arXiv:2512.18013** [SECONDARY]
Data-driven Elo tuning vs TrueSkill and Glicko; NN extensions mentioned.

**Adapting Skill Ratings to Luck-Based Hidden-Information Games (arXiv:2512.18858, 2025)** [SECONDARY]
Modified Elo for incomplete-information games.

**Clegg & Cartlidge (2025). Intransitive Player Dominance in Tennis Forecasting (GNN). arXiv:2510.20454** [ABSTRACT]
**Directly relevant.** Tennis-specific temporal directed graph + GNN; explicit intransitive A>B>C>A representation. 65.7% acc / 0.215 Brier on intransitive matchups; **3.26% ROI with Kelly staking** over 1903 bets vs Pinnacle.

**A Statistical Framework for Ranking LLM-Based Chatbots (arXiv:2412.18407, 2024)** [SECONDARY]
Generalises BT with covariates for LLM-rating setting.

**LLM Evaluation as Tensor Completion (arXiv:2604.05460, possibly future-dated)** [SECONDARY]
Claims low-rank semiparametric structure underlies model-prompt evaluation matrices.

---

## 5. Foundation Models for Sports / Games

### 5.1 Soccer / Football

**Baron, Hocevar, Salehe (2024). A Foundation Model for Soccer. arXiv:2407.14558** [ABSTRACT]
Transformer trained on three seasons of pro soccer to predict next action given input sequence. Baselines: Markov, MLP. **Open source: github.com/danielhocevar/Foundation-Model-for-Soccer. No explicit rating head — pure action prediction.**

**Adjileye (2024). RisingBALLER: A player is a token, a match is a sentence. StatsBomb Conf 2024. arXiv:2410.00943** [ABSTRACT]
Treats each match as sequence of player tokens. **Masked Player Prediction (BERT-style) pre-training.** Downstream: Next Match Statistics Prediction. **Player embeddings encode skill-like structure** useful for scouting, team-cohesion, similarity retrieval. Beats strong baseline on performance forecasting.

**ScoutGPT — Modeling Matches as Language (arXiv:2603.15212)** [SECONDARY]
NanoGPT-based, events as tokens, next-token prediction for counterfactual valuation.

**Axial Transformer for In-Game Outcome (arXiv:2511.18730, 2025)** [SECONDARY]
Joint, time-step-recurrent prediction of expected action totals for players and teams.

**EventGPT (arXiv:2512.17266, 2025)** [SECONDARY]
GPT-style player-conditioned next-event prediction.

### 5.2 Chess as Test-Bed

- **Maia 1** (KDD 2020) — rating-bucketed neural baseline.
- **Maia-2** (NeurIPS 2024) — unified skill-aware variant.
- **Ruoss et al. (2024). Amortized Planning with Large-Scale Transformers. NeurIPS 2024. arXiv:2402.04494** [SECONDARY]. 270M-param transformer trained on ChessBench, no search, **Lichess blitz Elo 2895** vs humans. Demonstrates transformer skill measured *in* Elo (rating downstream, not internal).
- Chessformer / ChessFormer sequence models [SECONDARY] — multiple recent transformer encoders consuming move sequences to predict both outcome and Elo.

### 5.3 Other

**player2vec (Wang & Asadi, arXiv:2404.04234, April 2024)** [SECONDARY] — long-range transformer over player action sequences, self-supervised; extended to chess.

**Basketball forecasting (arXiv:2508.02725, 2025)** [SECONDARY] — transformer vs LSTM for NCAA Division 1; transformer with BCE achieves AUC 0.8473.

---

## 6. In-Context Learning, Foundation Tabular Models, and Rating

**Could not find a paper directly asking "Can an LLM estimate Elo-like skill from match histories purely in-context?"** Adjacent:

- **ICPL (arXiv:2410.17233, 2024)** [SECONDARY] — LLM-as-preference-oracle in PbRL.
- **TabPFN / TabPFN v2 and prior-fitted networks** [SECONDARY] — transformers pre-trained on synthetic priors, approximate Bayesian inference in-context. **No published Elo-specific application yet; obvious open opportunity (a "PFN-Elo" trained on synthetic match histories generated from known TrueSkill prior).**
- **Transformers as Game Players (arXiv:2410.09701, 2024)** [SECONDARY] — theoretical guarantees that pre-trained transformers can approximate Nash equilibrium for 2-player zero-sum games in-context.

**Real gap.** User's setup (tennis match histories with OnCourt IDs and rating indices) is unusually well-suited to train PFN-style rater on simulated WHR/TTT priors.

---

## 7. Diffusion Models for Skill / Outcomes

**Found no work using diffusion models for player skill or game outcomes.** Diffusion entered alignment only via Diffusion-DPO / D3PO (BT loss applied to denoising policy). **Diffusion-for-skill remains open area.**

---

## 8. Bayesian Deep Learning and Generative Ratings (2023-2026)

- **Papamarkou et al. (2024). Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI. ICML 2024** [SECONDARY] — manifesto. Motivates uncertainty quantification, active learning, continual learning — all classic Glicko/TrueSkill concerns.
- **PURM** (arXiv:2503.22480) and **BNRM** (arXiv:2602.10623) [SECONDARY] — model reward as distribution. **Direct analogue of Glicko's RD.**
- **Bayesian approach to time-varying latent strengths in pairwise comparisons** (PMC, 2021) [SECONDARY] — closest classical Bayesian dynamic strength model.

---

## 9. Best Recent Surveys (Must-Reads)

**Fang, Han, Luo, Xu (Jan 2026). Recent advances in the Bradley-Terry Model: theory, algorithms, and applications. arXiv:2601.14727** [ABSTRACT]
**Most current statistical survey.** Asymptotic theory in large-objects/large-comparisons regime, MLE/spectral algorithms, Bayesian/mixture methods, LLM benchmarking, generative-model evaluation, sports, human-in-the-loop reward modelling. **Single recent survey explicitly bridging classical BT with LLM-RLHF use case.**

**Zhong, Shen, Li et al. (2025). A Comprehensive Survey of Reward Models. arXiv:2504.12328** [ABSTRACT]
Taxonomy of preference collection, reward modelling, usage patterns, benchmarking.

**Kaufmann, Weng, Bengs, Hüllermeier (2023; TMLR 2025). A Survey of RLHF. arXiv:2312.14925** [ABSTRACT]
Standard RLHF survey. Preference-based RL theory, LLM + robotics, feedback integration.

**Xiao et al. (2024-2025). A Comprehensive Survey of DPO. arXiv:2410.15595** [ABSTRACT]
Definitive DPO survey including BT connections, KTO, IPO.

**Li et al. (2024-2025). A Survey on LLM-as-a-Judge. arXiv:2411.15594** [SECONDARY]
Bias and reliability of LLM-as-judge.

**Older foundational**:
- The many routes to the ubiquitous Bradley-Terry model. arXiv:2312.13619, 2023. [SECONDARY]
- TrueSkill family (Herbrich 2006; Dangauthier 2007; Minka 2018) — no major new theoretical survey post-2018.

---

## 10. Frontier and Gaps Relevant to User's Tennis Project

1. **No published "neural Glicko / TrueSkill" with deep skill heads.** Cleanest gap. User could publish "deep Glicko" using small transformer over match-history tokens to predict posterior over skill, supervised on classical Glicko targets but conditioned on covariates (surface, recent form, sets won).

2. **No paper applies prior-fitted networks (TabPFN style) to rating estimation.** "Rating-PFN" pre-trained on synthetic WHR/TTT match streams → zero-shot Bayesian rating updates with user's surface-conditioned priors. **Low-hanging frontier project.**

3. **Intransitivity discussion is hot.** GPM, NLHF, SPPO, Neural BT, Clegg & Cartlidge GNN all argue scalar skill insufficient. User's tennis rating systems are all scalar; **adding learned latent-vector head per player (as user's WWHR/WTTT extensions move toward) places project in the 2024-2026 mainstream.**

4. **Style/feature regression as BT covariates** (Chatbot Arena's style-control trick) is structurally identical to "BT plus surface advantage" in tennis — **user's existing weighted/covariate systems are doing what LMSYS only adopted in mid-2024.**

5. **Foundation models for sports** nascent and almost entirely about *action* prediction, not rating. "FoundationTennis" akin to RisingBALLER, with masked-player or masked-match pre-training, BT head fine-tuned on win/loss → **no obvious published competitor.**

6. **Robustness of BT/Elo ratings to noisy/adversarial annotators** now documented LLM-side problem (Zhao 2024; Vyas 2025; Liu am-ELO 2025). User's data has cleaner annotators (match results) but **am-ELO joint annotator-ability framework translates straightforwardly to weighting tournaments/data sources of varying reliability.**

7. **No diffusion-for-skill, no in-context Elo from match histories.** Two genuinely open problems.

---

## 11. Honesty Audit

- **WebFetch denied** for Maia-2 PDF on Toronto's domain; substituted arXiv abstract.
- **No full paper PDF body read this session.** Every "ABSTRACT" = arXiv landing page only. Methodological descriptions beyond abstract from search summaries / aggregator snippets, may contain transcription errors. "[SECONDARY]" = could not retrieve landing page either.
- **Cannot guarantee exact wording of empirical claims.** 98.6% Arena-Hard correlation, 28.53% SPPO win-rate, 65.7% Clegg-Cartlidge accuracy etc. from search-result summaries (LMSYS / authors' blog posts / aimodels.fyi), not direct PDF inspection.
- **2026 arXiv numbers** (2601.14727 verified genuine Jan 2026; 2602.10286, 2603.15212, 2604.05460 may be index artefacts, not verified).
- **One unverified area**: "Can LLMs estimate Elo-like skill from match histories in-context?" — no direct published paper found. Reported as open gap rather than inventing citation.

---

## 12. References (chronological)

(Selected — see body for full list)
- Herbrich/Minka/Graepel TrueSkill (2006). [SECONDARY]
- Coulom WHR (2008). [SECONDARY]
- Dangauthier TTT (2007). [SECONDARY]
- Minka/Cleven/Zaykov TrueSkill 2 (2018). [SECONDARY]
- McIlroy-Young Maia 1 (2020). [SECONDARY]
- Fujii Neural BT Rating. arXiv:2307.13709. [ABSTRACT]
- Wang GElo. arXiv:2304.08257. [ABSTRACT]
- Rafailov DPO. NeurIPS 2023. arXiv:2305.18290. [ABSTRACT]
- Zheng/Chiang MT-Bench/Arena. NeurIPS 2023. arXiv:2306.05685. [ABSTRACT]
- Azar IPO. AISTATS 2024. arXiv:2310.12036. [ABSTRACT]
- Munos NLHF. arXiv:2312.00886. [ABSTRACT]
- Kaufmann RLHF Survey. arXiv:2312.14925. [ABSTRACT]
- Ethayarajh KTO. ICML 2024. arXiv:2402.01306. [ABSTRACT]
- Chiang Chatbot Arena. ICML 2024. arXiv:2403.04132. [ABSTRACT]
- Lambert RewardBench. arXiv:2403.13787. [ABSTRACT]
- Wang Asadi player2vec. arXiv:2404.04234. [SECONDARY]
- Dubois Length-Controlled AlpacaEval. arXiv:2404.04475. [SECONDARY]
- Wu SPPO. NeurIPS 2024. arXiv:2405.00675. [ABSTRACT]
- Zhuo Auto-Arena. arXiv:2405.20267. [SECONDARY]
- Rafailov Scaling Laws Reward Overoptim. arXiv:2406.02900. [SECONDARY]
- Boubdir Judging Judges. arXiv:2406.07791. [SECONDARY]
- Wu/Li Arena-Hard. arXiv:2406.11939. [ABSTRACT]
- Baron Foundation Model for Soccer. arXiv:2407.14558. [ABSTRACT]
- Length Bias. arXiv:2407.01085. [SECONDARY]
- LMSYS Style Control Blog Aug 2024. [SECONDARY]
- Tang Maia-2. NeurIPS 2024. arXiv:2409.20553. [ABSTRACT]
- Joshi CNN-LSTM Chess Rating. arXiv:2409.11506. [SECONDARY]
- Adjileye RisingBALLER. arXiv:2410.00943. [ABSTRACT]
- Zhang Beyond BT / GPM. ICML 2025. arXiv:2410.02197. [ABSTRACT]
- ICPL. arXiv:2410.17233. [SECONDARY]
- Sun Rethinking BT. ICLR 2025. arXiv:2411.04991. [ABSTRACT]
- Li LLM-as-Judge Survey. arXiv:2411.15594. [SECONDARY]
- Zhao Trustworthy Eval Challenges. arXiv:2412.04363. [ABSTRACT]
- Xiao DPO Survey. arXiv:2410.15595. [ABSTRACT]
- Statistical Framework for Ranking LLM Chatbots. arXiv:2412.18407. [SECONDARY]
- Min Vote Rigging. arXiv:2501.17858. [SECONDARY]
- Lin/Wu PvP Counter Categories. arXiv:2502.03998. [ABSTRACT]
- Buhamra/Groll Statistical Enhanced Tennis. arXiv:2502.01613. [SECONDARY]
- Zhong Reward Models Survey. arXiv:2504.12328. [ABSTRACT]
- Liu am-ELO. ICML 2025. arXiv:2505.03475. [ABSTRACT]
- Shorewala/Yang MOVDA. arXiv:2506.00348. [ABSTRACT]
- Vyas Dropping Preferences. arXiv:2508.11847. [SECONDARY]
- Clegg/Cartlidge Tennis GNN. arXiv:2510.20454. [ABSTRACT]
- NCAA Basketball arXiv:2508.02725 [SECONDARY]
- Axial Transformer arXiv:2511.18730 [SECONDARY]
- Maitra Elo Parameterization. arXiv:2512.18013 [SECONDARY]
- Luck-Based Hidden-Info Games. arXiv:2512.18858 [SECONDARY]
- Fang BT Survey. arXiv:2601.14727 (Jan 2026). [ABSTRACT]
