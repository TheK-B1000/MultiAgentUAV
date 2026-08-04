# SEA-GUARD formalization (revision)

Replacement for the submitted Section 3. Written to close Reviewer 2's **C3**
("Red agents do not appear to have a reward function defined by the game model.
If this is the case, should the system be modeled as a dec-POMDP instead with
only blue agents… Also, the model seems to define $r_i$ as a reward function, but
then $R$ is used later") and **C4** (the nonstationarity claim), and to give the
league phase an objective it is actually approximating.

The key change: **the model depends on what red is.** The submitted paper asserted
one model for all opponents, which is both inaccurate and an invitation to the
exact objection C3 raised. Two regimes are stated instead.

---

## 3.1 Two regimes, two models

Let $N$ be the blue team and $M$ the red team, $|N| = |M| = n$ (symmetric $n$v$n$).

### Regime A — learned red: two-player zero-sum POSG

When red is itself a policy (self-play, a league snapshot, or a trained
best-response exploiter), each *team* is a player. The game is a two-player
zero-sum partially observable stochastic game parameterized by a configuration
$c$ drawn at episode start:

$$
\mathcal{G}(c) = \langle \mathcal{S}, \{\mathcal{O}_b, \mathcal{O}_r\},
\{\mathcal{A}_b, \mathcal{A}_r\}, \mathcal{T}, R, c \rangle
$$

- $\mathcal{S}$ — full maritime state (ASV poses and velocities, flag states, tag
  states, mine states, current field, scores, clock).
- $\mathcal{O}_b, \mathcal{O}_r \subseteq \mathcal{S}$ — the side-local
  observations produced by the environment's symmetric observation function; red
  observes the mirror of what blue observes.
- $\mathcal{A}_b = \prod_{i \in N}\mathcal{A}_i$ — the *joint* team action; each
  ASV emits a (macro, target) pair, so $\mathcal{A}_i$ is the macro–target
  product set. Likewise $\mathcal{A}_r$.
- $\mathcal{T}(s' \mid s, a_b, a_r)$ — stochastic transition, stochastic through
  current, drift, sensor noise, and tag resolution.
- $R(s_T) \in \{-1, 0, +1\}$ — terminal team reward from blue's perspective
  (win / draw / loss), with $R_{\text{red}} = -R_{\text{blue}}$.

Writing $\Gamma_c(\pi_b, \pi_r)$ for the trajectory distribution induced by the
two team policies under configuration $c$, the value is

$$
V(\pi_b, \pi_r) = \mathbb{E}_{c \sim \mathcal{U}(\mathcal{C})}
\big[\mathbb{E}_{\tau \sim \Gamma_c(\pi_b, \pi_r)}[R(s_T)]\big],
$$

and the objective the league phase approximates is the equilibrium

$$
(\pi_b^{\text{eq}}, \pi_r^{\text{eq}})
= \arg\max_{\pi_b} \arg\min_{\pi_r} V(\pi_b, \pi_r).
$$

**This is the sentence the submitted paper was missing.** It converts "we trained
against a varied opponent pool" — a design choice a reviewer can call arbitrary —
into "we approximate a maximin equilibrium in expectation over the configuration
space," which is a stated objective the baselines can be compared against.

### Regime B — scripted red: induced POMDP

When red is a fixed script (OP1–OP4) it has no reward function and no policy
parameters, and it is therefore **not** a player. It is part of blue's transition
dynamics. Fixing $\pi_r = \pi^{\text{script}}$ marginalizes red out and induces a
single-player problem for the blue team:

$$
\mathcal{M}(c, \pi^{\text{script}}) =
\langle \mathcal{S}, \mathcal{O}_b, \mathcal{A}_b,
\mathcal{T}^{\pi^{\text{script}}}, R, c \rangle,
\qquad
\mathcal{T}^{\pi^{\text{script}}}(s' \mid s, a_b)
= \mathbb{E}_{a_r \sim \pi^{\text{script}}(\cdot \mid s)}
\big[\mathcal{T}(s' \mid s, a_b, a_r)\big].
$$

Whether $\mathcal{M}$ is a **POMDP** or a **Dec-POMDP** is decided by blue's
execution architecture, not by red:

- **This work is a POMDP.** One centralized policy receives the joint team
  observation and emits the joint team action. There is a single decision maker.
- It would be a **Dec-POMDP** only under decentralized execution, where each ASV
  conditions solely on its own observation. That is future work and is scoped as
  such in the discussion (it is also what Reviewer 3's point 1 is really asking
  about — see the deployment-feasibility paragraph).

So C3's premise is right, and the correct answer is more precise than the one it
proposed: against scripted red the model *is* a single-agent problem, but a
POMDP rather than a Dec-POMDP, because blue is centralized.

### Notation fix (C3, second half)

The submitted paper used $r_i$ and $R$ inconsistently. There is exactly one
reward:

- $R(s_T)$ — the terminal zero-sum team outcome, defined above. This is the
  quantity every reported metric is a function of.
- $\tilde{R}(s, a, s') = R(s_T) + \sum_k \lambda_k \phi_k(s, a, s')$ — the
  *shaped training* reward, used only for optimization. Every $\phi_k$ and every
  $\lambda_k$ is tabulated in the appendix.

No per-agent $r_i$ exists: the team is optimized against a shared team signal.
The symbol is removed from the paper.

---

## 3.2 The configuration space

A configuration is drawn once per episode and held fixed:

$$
c = (\text{opponent}, \text{current profile}, \text{team size}, \text{episode seed})
\in \mathcal{C}.
$$

The map and the ruleset are **fixed constants** of the protocol, not varied
factors, and are stated as such. The implementation of record is
[`rl/configuration_space.py`](../../rl/configuration_space.py), which is the
single source of truth for the splits below.

| Factor | Values | Split |
| --- | --- | --- |
| Scripted opponent | OP1, OP2, OP3 | seen |
| Scripted opponent | OP4 | held out |
| Behavioral species | RUSHER, CAMPER, BALANCED | seen |
| Current profile | `calm` (0.02 cps), `nominal` (0.12 cps, 0.03 drift) | seen |
| Current profile | `strong` (0.20 cps, 0.06 drift), `severe` (0.28 cps, 0.10 drift) | held out |
| Team size | 2v2, 3v3, 4v4 | separate axis — see §3.4 |

The seen/held-out split is **derived from what training actually samples** rather
than chosen to hit a target cardinality: `rl/curriculum.py` defines
`VALID_PHASES = ("OP1", "OP2", "OP3")` and the league samples SCRIPTED OP1–OP3,
SPECIES, and SNAPSHOT opponents, so OP4 and any current profile outside the
stress schedule are provably unseen. Following VGC-Bench's set-theoretic
statement of the same idea:

$$
\mathcal{C}_{\text{seen}} = \bigcap_{k=1}^{|\Pi|} \mathcal{C}_k,
\qquad
\mathcal{C}_{\text{heldout}} \cap \bigcup_{k=1}^{|\Pi|} \mathcal{C}_k = \emptyset .
$$

Only `current_strength_cps` and `drift_sigma_cells` are used as the current-profile
knobs, because those are the only stress keys the simulator applies at runtime
(`BatchedCTFCore._apply_profile_runtime`). Claiming a sensor-noise axis would
overstate what the environment does.

---

## 3.3 Nonstationarity (C4)

The submitted text said a probabilistic opponent draw "induces a nonstationary
adversarial training distribution." Reviewer 2 correctly objected: sampling
opponents from a *fixed* distribution is stationary, merely stochastic. The
revision distinguishes the two and only claims nonstationarity where it exists.

**Stationary but stochastic.** The curriculum and the scripted/species categories
draw from a fixed distribution over a fixed opponent set. Blue faces a stochastic
but *time-invariant* environment. No nonstationarity claim is made here.

**Genuinely nonstationary.** The snapshot category is nonstationary, because the
sampling distribution's *support* changes during training:

1. new snapshots of the improving main agent enter the pool over time, so the
   opponent distribution at step $t$ is not the distribution at step $t'$;
2. PFSP re-weights toward opponents the current policy is losing to, so the
   weights are a function of the learner's own evolving performance;
3. in the double-oracle variant the meta-Nash is recomputed as the pool grows;
4. exploiters are trained *against the current main agent* and injected, which is
   nonstationarity by construction — the opponent is a best response to the
   learner's present weaknesses.

Sentences 1–4 are the ones the paper is entitled to, and the loose claim on page 4
is replaced by them.

**A related fix.** The submitted line "Training purely against a fixed opponent
often yields strategy overfitting, where $\pi$ adapts to a narrow opponent
distribution" is self-contradictory, as Reviewer 2's final small comment noted: a
fixed opponent is not a distribution. Replacement: *"Training against a fixed
opponent yields a best response to that opponent alone. Such a policy is
$\epsilon$-optimal against it and may be arbitrarily exploitable against any
other, which is what §5.3 measures directly."*

---

## 3.4 What team size does and does not show

Observation and action spaces are team-size dependent — `_make_obs_action_spaces`
builds `grid`/`vec`/`agent_mask` with a leading $n$ dimension and a
`MultiDiscrete([n_macros, n_targets] * n)` action space — so a policy trained at
2v2 cannot be loaded at 3v3 at all. Two claims must therefore be kept apart, and
`configuration_space.assert_team_size_compatible` raises rather than allow the
confusion in code:

- **Scalability** *(what we report)* — independently trained 2v2/3v3/4v4 policies,
  each evaluated at its own team size. This shows the method still works as the
  team grows.
- **Zero-shot team-size generalization** *(what we do not claim)* — one policy
  evaluated at a team size it never trained on. This requires a variable-team
  (tokenized or attention-based) architecture and is stated as future work.

Describing the existing runs as team-size generalization would be the single
easiest thing for a reviewer to falsify, so the paper does not.

---

## 3.5 Evaluation objectives

Three quantities, defined here and measured in §5. $\text{ms}(\pi_b, \cdot)$ is
the match score $(W + 0.5D)/(W + L + D)$.

**Performance.** $\; \mathbb{E}_{c \sim \mathcal{C}_{\text{seen}}}[\text{ms}(\pi_b, c)]$
— competence on configurations every compared method trained on.

**Generalization.** $\; \mathbb{E}_{c \sim \mathcal{C}_{\text{heldout}}}[\text{ms}(\pi_b, c)]$,
and the *generalization gap* = performance − generalization.

**Empirical exploitability.** With $\mathrm{BR}(\pi_b)$ a red best response trained
against frozen $\pi_b$ under a fixed budget,

$$
\widehat{\text{exp}}(\pi_b) = \max_{t \in \text{checkpoints}}
\text{ms}\big(\mathrm{BR}_t(\pi_b), \pi_b\big),
$$

taken over held-out validation episodes and reported as an **approximate lower
bound**: failure to find an exploit at this budget is not proof that none exists.
The oracle is controlled — identical architecture, identical budget, matched
validation seeds, multiple initializations, peak *validation* score, and the
target frozen throughout (`rl/eval_exploitability.py`).

---

## 3.6 Statistical protocol

- **Common random numbers.** Every compared method faces the identical episode
  seed list for a given configuration; each configuration gets a disjoint seed
  block (`configuration_space.episode_seeds`), so no two configurations share an
  episode and no method gets an easier draw.
- **Episodes.** 200 per training seed per configuration; 3 training seeds
  (42/43/44) → 600 games per method per configuration. Five seeds would be
  preferable and is the stated limitation.
- **Uncertainty across training seeds, not episodes.** Point estimates are the
  mean of per-seed means; intervals come from a paired bootstrap over shared
  episode indices. A binomial interval over pooled episodes would treat runs as
  independent samples and understate variance — the seed spread is reported
  separately for exactly this reason.
- **Checkpoint selection is stated.** The final checkpoint is evaluated, not the
  best-of-training checkpoint, and per-seed training curves are released. This is
  Reviewer 1's "they save only the best model" objection.
