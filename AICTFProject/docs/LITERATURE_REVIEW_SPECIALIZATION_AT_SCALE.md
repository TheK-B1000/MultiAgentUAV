# Literature review: inducing and preserving opponent-conditioned specialization

**Date:** 2026-09-12 · **Status:** review only — licenses no experiment
**Framing question (frozen before searching):**

> How do MARL systems preserve or induce task/opponent-conditioned specialization
> when ordinary independent training converges to competent but insufficiently
> differentiated policies?

Searched as three independent tracks (curriculum+population · opponent
conditioning+roles · diversity+retention) so that convergence between tracks
carries evidential weight rather than reflecting one search path.

---

## 0. The headline, before the taxonomy

**Two independent tracks surfaced the same paper as a top-2 candidate, and it is
a hypothesis our ruled-out list does not exclude.**

Kuba et al., *Settling the Variance of Multi-Agent Policy Gradients*
(NeurIPS 2021, [2108.08612](https://arxiv.org/abs/2108.08612)) proves that the
excess variance of the multi-agent policy gradient **grows linearly in the number
of agents** (and quadratically in local advantages), and derives the
minimum-variance optimal baseline (OB).

Why this matters to us specifically: Δ is a *small payoff difference*. The
gradient signal that must be resolved to learn it is small by construction. If
estimator variance grows with N while the signal does not, then the
signal-to-noise ratio on precisely the quantity we measure degrades with team
size — with **identical architecture, identical parameter count, contested win
rates, and large behavioral divergence**. That is our observed pattern, line for
line, with no diversity mechanism, no capacity limit and no compression involved.

We have eliminated capacity, saturation, compression, and (for the B3-3 track)
early generalist dominance. We have **not** eliminated optimization SNR. It is
currently the cheapest unexcluded explanation, and it is a critic-side change
whose control arm is the runs we have already done.

**Second headline:** our JSD result is a published finding, twice over — see §3.
We retired the sibling-separation intervention on the right grounds.

**Third headline, for positioning:** no paper found studies *opponent-specific
specialist crossover degrading with team size*. The nearest work (§1) studies
*intra-team* specialization collapse. Our phenomenon appears to be unreported.

---

## 1. Curriculum / transfer across scale

### Predicting Multi-Agent Specialization via Task Parallelizability
[2503.15703](https://arxiv.org/html/2503.15703) — **the closest published result to our phenomenon**

| | |
|---|---|
| Problem | When do teams differentiate vs. converge to generalists? |
| Mechanism | Amdahl's Law ported to MARL. Parallelizability `S(N,C)` from subtask time fractions and per-subtask concurrency capacity. `S(N,C) = N` ⇒ generalism optimal; `S(N,C) < N` ⇒ bottlenecks force differentiation |
| Specialization signal | None added — it **predicts** differentiation from task structure. Measured by a JSD-based Specialization Index |
| PPO-compatible | Yes (independent PPO in JaxMARL) |
| Architecture change | None |
| Compute | A single prediction is free; their study was ~1,569 GPU-h |
| Across team sizes | **Yes — that is the axis.** SMAC: specialization index falls with team size, Pearson **r = −0.86**, mean SI 0.06 (near-total generalist collapse). MPE with unit-capacity bottlenecks: SI holds at 0.61 regardless of N |
| Evidence | Logistic prediction accuracy 1.0 (SMAC/MPE), 0.909 (Overcooked), 0.742 at largest scale |
| Relevance | Highest — but read the caveat |

**Caveat that must not be skipped:** their index is **intra-team** (agent *i* vs.
agent *j* inside one team). Ours is **inter-policy** (π_A vs. π_B across separate
runs). They do not test opponent-conditioned differentiation. So this is a
structural analogy, not a direct result about our quantity.

**Why it is still the first thing to do:** it predicts, with no GPU spend, that
an open-water pursuit task becomes *more* parallelizable as N grows, so
regime-specific play becomes optional rather than necessary. Their Experiment 4
frames differentiation as "a locally optimal solution agents use when exploration
is costly" — which would reframe Δ → +0.008 as *the environment no longer
requiring regime-specific play*, not the learner failing to find it.

**The tension worth confronting:** we *certified* strategic demand at 4v4 and
6v6 — a guard-style scripted response beats a breach-style one against pole A and
vice versa. That is evidence against "no demand at scale." But the certification
demonstrates demand **between two scripted responses**, not that the demand
survives into the space a learner actually explores, nor that it is large
relative to what a generalist can achieve. Parallelizability speaks to the
second. These are compatible, and separating them is a real question.

### DyMA-CL ([1909.02790](https://arxiv.org/pdf/1909.02790)) and EPC ([2003.10423](https://arxiv.org/abs/2003.10423))
Small→large agent-count curricula. DyMA-CL's three transfer mechanisms (buffer
reuse, curriculum distillation, model reload); model reload + a size-agnostic
network wins. **Our architecture is already size-invariant, so their hard part is
solved for us** — only the schedule is untested. EPC's stated core insight is
directly ours: "agents successfully trained at small population are not
necessarily the best candidates at scaled populations." EPC needs a population
and evolutionary selection (MADDPG-based).

**Ranking note:** curricula confound schedule, seeding and selection in one
change. Lower cleanliness than a single-term intervention.

### Confident negative
**No cross-scale work transfers a *strategy*; every one transfers competence, and
every one's metric is return or win rate.** Our exact failure — competent but
undifferentiated — would be *invisible* in all of these papers.

---

## 2. Opponent conditioning

### LIAM — Agent Modelling under Partial Observability
[2006.09447](https://arxiv.org/abs/2006.09447) (NeurIPS 2021) — **best first mechanistic test**

| | |
|---|---|
| Problem | Condition on other agents when their observations/actions are unavailable at execution |
| Mechanism | Recurrent encoder over the **controlled agent's own** obs-action trajectory → embedding `z_t`; a two-head decoder reconstructs the modelled agent's observation (MSE) and action (CE). Decoder discarded after training; `z_t` concatenated to the policy input |
| Specialization signal | The reconstruction loss. The policy **cannot ignore** opponent state because the shared trunk is graded on predicting it |
| PPO-compatible | Yes — A2C in the paper, on-policy, explicitly algorithm-agnostic |
| Architecture change | Encoder + throwaway decoder. No change to policy head, observation space, or RL objective |
| Compute | Negligible — one recurrent encoder + MLP decoder on data we already collect |
| Across team sizes | 2–4 agents only. **Untested at 6v6 — this is the gap our work would fill** |
| Evidence | Beats NAM, VariBAD, CBAM, CARL; approaches the full-information upper bound. Ablating either head significantly lowers return. Action prediction >90% where observable |
| Identity required | **None**, at training or execution — applicable to our schema as-is |

**Why this ranks so high for us:** reconstruction accuracy is a *direct readout*
of "does the representation carry opponent state," measurable **before** looking
at Δ. If a 6v6 encoder reconstructs opponent positions well and Δ stays ~0, our
current hypothesis is falsified cleanly and the failure is downstream, in credit
assignment or optimization — which points at §0. It is a mechanism test that
yields information under *either* outcome, satisfying Rule 10 by construction.

### Entity/attention encoders — REFIL ([ICML 2021](http://proceedings.mlr.press/v139/iqbal21a/iqbal21a.pdf)), HGAP (ICML 2024)
The cheapest structural fix to the schema limitation we audited. Cross-attention
over per-opponent entity vectors is permutation-invariant **without** destroying
geometry — unlike our min-distance scalar, which discards everything but one
number, and our set-scatter grid, which survives only at grid resolution.
REFIL's randomized entity-wise factorization explicitly targets generalization
across entity counts, i.e. our 2v2→6v6 cliff. No identity required.

### OMG (NeurIPS 2024) and MIX ([2605.31318](https://arxiv.org/html/2605.31318))
OMG infers opponent **subgoals** (CVAE) rather than actions; 5–20% task-success
gain against unknown opponents. Semantically the most faithful model of our
generator — regimes A and B are scripted, so they have literal subgoals. But it
"struggles under partial observability," and our task is partially observed.

MIX fuses four predictive experts (opponent action, opponent observation, future
state, **future ego reward** via InfoNCE) with a learned gate; **PPO backbone in
all experiments**; the future-reward expert ranks top-two in 3 of 4 environments.
Too many moving parts for a first experiment. The clean extraction is the single
**reward-prediction expert**: our Δ is a value difference, so an embedding
trained to predict future ego return is better aligned with crossover than one
predicting opponent actions.

### Grover et al. (ICML 2018)
Generative + discriminative (triplet-loss on **agent identity**) policy
embeddings. We *do* have regime labels A/B at training time even though the
observation lacks identity, so it is admissible — but with only two scripted
regimes the embedding space is a 2-point set and degenerates to a one-bit
indicator. Clean, but nearly vacuous.

### DRON (ICML 2016), Social Influence (ICML 2019)
DRON is the origin paper, DQN-only with hand-crafted opponent features — cite,
don't run. Social Influence rewards *influencing teammates* and needs access to
others' policies; poor fit. Do not spend a run.

---

## 3. Diversity preservation — where our own negative result is corroborated

### Liu et al., *Unifying Behavioral and Response Diversity*
NeurIPS 2021 / BD&RD-PSRO ([2106.04958](https://arxiv.org/abs/2106.04958))
— **found independently by two of the three tracks**

Separates **Behavioral Diversity** (BD — an *f*-divergence between occupancy
measures; this is our JSD) from **Response Diversity** (RD — defined on the
**payoff vector**: the Euclidean distance from a new policy's vector of returns
against each opponent to the convex hull of the population's payoff vectors).

**Their Table 2 ablation is our result.** PSRO+BD alone reaches exploitability
41.13 ± 1.06 versus **13.26 ± 0.24** for BD+RD. Behavioral divergence without
payoff divergence does not move the game-theoretic quantity.

This is direct published corroboration of our own measurement: **JSD = 0.338 nats
(≈49% of the ln 2 maximum) with payoff crossover still ≈ 0.** We retired
sibling-separation before spending seeds on it; the literature says that was
correct, and says why. Mechanism: an augmented intrinsic reward on a standard
single-agent best-response solve — PPO-compatible, no architecture change, cost ≈
one extra payoff-matrix evaluation per iteration. Tested up to 11v11 Google
Research Football.

### Yao et al., *Policy Space Diversity* (NeurIPS 2023) — [2306.16884](https://arxiv.org/abs/2306.16884)
The general theorem: a population more diverse under existing behavioral metrics
is **not** necessarily a better Nash approximation. This is the theoretical
warrant for retiring JSD repulsion, not merely an empirical one.

### CoMeDi — *Diverse Conventions for Human-AI Collaboration*
NeurIPS 2023, [2310.15414](https://arxiv.org/pdf/2310.15414) — **optimizes our actual quantity**

Loss: `L(π_n) = −J(π_n,π_n) + α·J(π_n,π*) − β·J_M(π_n,π*)` — maximize own-pairing
return, **minimize cross-pairing return**, plus a **mixed-play** term (start an
episode from self/cross transitions, then switch to self-play) that stops the
agent from *sabotaging* to game the cross-play term. MAPPO-based; separate actors
per convention, separate critics for self/cross. ~3 GPU-h per config on
Overcooked. 25-participant user study: CoMeDi 3.68 vs XP 2.48, ADAP 2.24, SP 2.08.
Ablation with β=0 produced **16% "handshake" sabotage failures** — i.e. the
naive version of this idea fails in a specific, documented way.

Their criticism of statistical diversity is worth quoting into our own writing:
policies "can take different actions that lead to the same outcome."

`−J(π_A, B)` is *literally* the second term of our Δ_A.

> **Methodological trap, flagged here because it is decisive for our program:**
> if we train on the crossover objective, **Δ is no longer a valid confirmatory
> measure** — it becomes the training loss. Adopting CoMeDi requires
> pre-registering a *held-out* criterion (a third scripted regime, or held-out
> pole parameterizations) before any seed is spent. Otherwise we would be
> optimizing for PASS, which is the one thing we have agreed not to do.

### DiCo (ICML 2024) — [2405.15054](https://arxiv.org/abs/2405.15054) — the decisive null test
Constrains behavioral diversity to an **exact** value of a metric (SND) by
architectural decomposition (shared + scaled per-agent), **leaving the learning
objective unchanged**; works with any actor-critic method. Requires an
architecture change, but yields a **dose–response curve**: sweep diversity, plot
Δ. **A flat Δ across the sweep closes the diversity hypothesis outright.** This
is the cleanest available way to *kill* a hypothesis rather than support one.

### Confident negatives — do not run
- **DIAYN / MI skill discovery**: well-replicated negative. MI objectives prefer
  static skills with poor state coverage ("yoga poses" in DM Control). Diversity
  without task relevance is the documented failure mode.
- **TrajeDi** (ICML 2021): generalized JSD over *trajectory* distributions — our
  retired intervention lifted from actions to trajectories. Given 49% of max JSD
  at the action level with Δ ≈ 0.03, expect no payoff effect.
- **DvD** (NeurIPS 2020) / **DPP diversity** (ICML 2021): determinantal volume
  over *behavioral* embeddings. Same class as what failed; ES/population-based,
  awkward under PPO.
- **QD / MAP-Elites**: TD3/ES-based, needs hand-designed behavior descriptors.
- **PSRO meta-game machinery**: our opponents are **fixed scripts**. The Nash
  solver is inert; there is nothing to iterate. Only the *diversity-regularized
  best-response oracle* transfers.
- **AlphaStar-style league training**: does not apply for the same reason —
  exploiters have nothing to exploit that we did not already specify.
- **Parameter-sharing remedies (SePS, DiCo-as-sharing-fix)**: we already have
  zero sharing and bit-exact dispatch.

---

## 4. Role specialization — a confident negative, cleanly

**ROMA, RODE, CDS, EOI, SePS, HyperMARL all solve a different problem:**
differentiating **teammates within one team** under parameter sharing, keyed on
**agent ID**. Their specialization signals (MI between identity and role,
identity-aware intrinsic reward, ID-based clustering) require an identity input
our schema lacks, and none targets opponent-conditioned best response or anything
resembling a crossover criterion. Surveyed scalability work notes that taking
agent IDs as inputs makes these methods **break when population size changes** —
the opposite of what we need.

This closes the family. Two salvage items only:

1. **HyperMARL's gradient-interference framing** — a plausible alternative
   account of why 4v4/6v6 specialists collapse toward a generalist even
   uncompressed. Note this echoes our own earlier K=2 finding that π_R was a
   dominant generalist.
2. **Role Diversity** (ICML 2022, [2207.05683](https://arxiv.org/abs/2207.05683))
   — action-/trajectory-/contribution-based diversity metrics with an error-bound
   decomposition. Usable as a **diagnostic** on existing 2v2 vs 6v6 checkpoints.
   Not a training method for us.

---

## 5. Optimization / retention

> **RESOLVED 2026-09-12 — this entire section is now DEPRIORITISED.** The
> conditional below was answered by `B3_CKPT_TRAJECTORY_4V4`: across all five
> checkpoints (200k→1M, n=24 shared seeds) **no large transient specialization
> phase was detected on either axis**, and every CI contains zero. There is no
> hump to have collapsed from, so the pattern these methods fix is unsupported.
> Deprioritised on *absence of the pattern they address*, not refuted as
> techniques. See `B3_CKPT_TRAJECTORY_4V4_READING.json` — note the reading
> selects the conservative `noisy_non_monotone` branch, so this is an
> underpowered negative, load-bearing only as the absence of the retention
> signature.

### Policy Consolidation (ICML 2019) — [1902.00255](https://arxiv.org/abs/1902.00255)
A cascade of hidden networks at geometric timescales, KL-coupled to the live
policy. No task boundaries needed, PPO-native, and — the reason it ranked —
**evaluated in competitive multi-agent self-play, where baseline PPO showed
performance collapse.** Cost: *k* extra network copies (memory, not compute).

~~Conditional on the trajectory diagnostic now running.~~ **Answered: Δ is never
high, so retention is the wrong family and this drops off the near-term list.**

### Kickstarting (2018) — [1803.03835](https://arxiv.org/abs/1803.03835)
Auxiliary KL to a teacher with an annealed coefficient; architecture-agnostic,
self-regulating. Use the highest-Δ checkpoint as teacher. Scientifically weakest
(confounds teacher quality with retention) but trivial to control.

### EWC — skip
Repeatedly reported to reduce but not eliminate forgetting under PPO, and it
constrains *weights*, not the payoff property we care about.

---

## 6. Ranking by scientific cleanliness

Ordered by how unambiguously the result would be interpretable — **not** by ease
of implementation, per standing instruction.

| # | Candidate | GPU cost | Why it ranks here |
|---|---|---|---|
| 1 | **Task-parallelizability prediction** (§1) | **zero** | Falsifiable, pre-registerable, no intervention. May explain the failure without any fix. Costs nothing to be wrong |
| 2 | **Optimal baseline / MAPG variance** (§0) | low | Single critic-side change; theory *predicts the N-dependence we observe*; control arm is the runs we already have; excludes a hypothesis we have not excluded |
| 3 | **LIAM auxiliary head** (§2) | low | Reads out mechanism *before* payoff. Informative under either outcome. Directly falsifiable against our current opponent-channel hypothesis |
| 4 | **DiCo diversity dose–response** (§3) | medium | The only candidate that can *close* a hypothesis outright rather than support one. Architecture change is the cost |
| 5 | **Entity-attention opponent encoder** (§2) | medium | Isolates schema information loss from everything else |
| 6 | **BD&RD-style response-diversity oracle** (§3) | medium | One term, one knob, and weight→0 recovers our current setup as an exact control |
| 7 | **CoMeDi cross-play objective** (§3) | medium | Optimizes the actual quantity — but **invalidates Δ as a confirmatory measure** unless a held-out criterion is pre-registered first |
| 8 | **Policy Consolidation** (§5) | medium | Clean, but only meaningful if the trajectory diagnostic shows Δ decaying |
| 9 | EPC / DyMA-CL curricula (§1) | high | Confounds schedule, seeding and selection |
| — | League training, PSRO meta-game, role methods, DIAYN, TrajeDi, EWC | — | Ruled out on structural grounds above |

**Note on #1 and #2 together:** they are not alternatives. Parallelizability asks
whether the *task* still demands differentiation at 6v6; optimal-baseline asks
whether the *learner* can resolve the demand that exists. Running #1 first is
free and tells us which question #2 is answering.

---

## 7. What the in-flight experiments decide

Both are diagnostics, both pre-registered, neither licenses an intervention.

- **`B3_CKPT_TRAJECTORY_4V4`** — ✅ **LANDED 2026-09-12.** Δ never high at any
  checkpoint; no transient hump detected on either axis; all CIs contain zero.
  **§5 (retention) is deprioritised.** Weight shifts to §0/§1 — but note the
  reading took the conservative `noisy_non_monotone` branch, so this
  *establishes the absence of the retention signature*, it does not by itself
  establish that acquisition is the failing stage. Full numbers:
  `B3_CKPT_TRAJECTORY_4V4_RESULT.json`; interpretation:
  `B3_CKPT_TRAJECTORY_4V4_READING.json`.
- **`OPP_ABLATION_6V6`** (ZERO_OPP / RANDOM_OPP vs FULL, 32 seeds × 4 cells) — if
  ablating the two opponent carriers does not move win rate, the policy does not
  materially use opponent position, and §2 (conditioning) is the indicated
  family. If it *does* move win rate, the policy uses opponent information and
  fails to *differentiate on it anyway* — which points at §0 and §3.

Neither outcome selects an intervention by itself. That is the point: the next
GPU spend should be justified by these two results plus #1, which is free.

---

## 8. Citation hygiene

Verify before any of these enters a manuscript. The following were reported but
not independently confirmed to full text during this review:

- Bettini et al. 2506.09434 ("When Is Diversity Rewarded…") and 2412.16244 —
  team-size claims **unverified**.
- MIX / Generalized Intention Modeling (2605.31318) — recent; confirm venue.
- OMG (NeurIPS 2024 poster) — confirm the camera-ready.

Everything in §6's top eight has a confirmed venue and arXiv id.

---

## 9. What this review does not license

No experiment. No training run. The PI has stated that random-init vs validated
2v2 warm-start at 4v4 remains the prior candidate for the first controlled
intervention, and that it is **not to be locked until the trajectory and
opponent-channel ablation land and the literature is checked**. The literature is
now checked; two of the three inputs are still in flight.
