# SEA-GUARD related work (revision)

Replacement for the submitted Section 2. Two reviewer complaints drive the
restructure:

- **R2 C1** — "The related work section lists related papers, but does not clarify
  limitations of those works (or how this paper differs from those works)… For
  example, how is the proposed method different from AlphaStar. Other works using
  some form of curriculum training or self-play, like [1] or [2], also seem to be
  missing."
- **R1** — "The bibliography is scarce, and a third of it corresponds to
  self-cites."

The fix is structural, not cosmetic. Every subsection follows the same three-move
pattern: *what the line of work established → what it does not address → what
this paper does about it.* Nothing is listed without a stated limitation. The
citation count roughly triples and the self-citation share drops well below a
third.

---

## 2.1 Population-based league training

**Established.** AlphaStar [vinyals2019] introduced league training for StarCraft
II: a main agent, main exploiters, and league exploiters, matched by prioritized
fictitious self-play (PFSP), reaching Grandmaster. OpenAI Five [berner2019]
reached professional Dota 2 play with large-scale self-play. ROA-Star
[huang2023roastar] improves on AlphaStar in two specific ways — *goal-conditioned*
exploiters that find weaknesses in the main agent and the league far more
effectively than AlphaStar's unconditioned exploiters, plus explicit opponent
modeling so the main agent responds to the opponent's real-time strategy.

**Limitation.** All three operate in discrete-action video-game domains with no
continuous vehicle dynamics, no environmental disturbance field, and no physical
deployment target. Their league compositions are tuned to a specific game's
strategy space; none reports how the composition transfers to a domain where the
opponent set is small and partly scripted. Critically, none of the three reports
**exploitability of the final agent** — league training is motivated as a
robustness mechanism but validated by ladder rank, which is a relative measure
against a population that the method itself shaped.

**This paper.** We port PFSP league training to maritime multi-robot CTF with
current, drift, sensor noise, and macro-action ASV dynamics, implement ROA-Star's
exploiter mechanism as our strongest baseline, and evaluate all methods with a
best-response oracle so the robustness claim is measured rather than assumed
(§5.3).

## 2.2 Curriculum and automatic opponent generation

**Established.** Curriculum learning over progressively harder opponents is
standard practice. PORTAL [wu2024portal] generates curricula for multi-agent RL
automatically rather than by hand. Robust league training [huang2023roastar]
adapts the opponent distribution to the learner's current weaknesses. In the
open-ended line, PAIRED [dennis2020paired] and POET [wang2019poet] co-evolve
environments with agents.

**Limitation.** Automatic curriculum generation assumes a parameterized task
space rich enough to search. Maritime CTF as deployed has a small, discrete,
partly hand-authored opponent set (scripted doctrines plus behavioral species),
because those opponents encode operational doctrine a defense scenario must be
tested against — they are not free parameters to be optimized away.

**This paper.** Our curriculum is deliberately hand-specified over doctrine-derived
opponents, and we treat that as a constraint of the application rather than a
contribution. The paper's claim is not that our curriculum is novel; it is that we
*measure* what each stage contributes, via the `no_curriculum` ablation (§5.5).

## 2.3 Empirical game-theoretic multi-agent learning

**Established.** The double oracle algorithm [mcmahan2003do] and PSRO
[lanctot2017psro] give a unified game-theoretic frame for population methods:
maintain a population, compute a meta-strategy over the empirical payoff matrix,
and train a best response to it. Fictitious play is the uniform-meta-strategy
special case. α-Rank [omidshafiei2019alpharank] ranks agents from a cross-play
matrix without assuming a unique equilibrium. Approximate exploitability
[timbers2020exploitability] estimates distance to equilibrium by learning a best
response, which is tractable where exact computation is not.

**Limitation.** This literature is developed and benchmarked almost entirely on
abstract or discrete games — poker [brown2017libratus], board games
[silver2016alphago], and card/matrix domains. Its application to physically
embodied multi-robot teams with continuous dynamics and disturbance is sparse,
and the methods are rarely compared *against each other* on a single embodied
task under a fixed budget.

**This paper.** We implement the full ladder — self-play, fictitious play, double
oracle, PFSP, and ROA-Star — inside one maritime environment with identical
architecture, identical budget, and identical evaluation protocol, so the
comparison isolates the opponent-selection rule (§5.4). We adopt α-Rank for
cross-play summarization and approximate best-response exploitability as a primary
metric.

## 2.4 Benchmarks for multi-agent generalization

**Established.** VGC-Bench [angliss2026vgc] argues that the interesting axis in
competitive multi-agent domains is not the environment but the *configuration*
drawn at episode start, and standardizes a triad of tests — performance on seen
configurations, generalization to held-out ones, and exploitability under a
learned best response — over a ladder of BC/SP/FP/DO baselines. Its central
finding is that agents strong in the single-configuration setting degrade sharply
as configuration diversity grows. SMAC [samvelyan2019smac], Neural MMO
[suarez2021neuralmmo], and Melting Pot [leibo2021meltingpot] provide multi-agent
benchmarks with held-out evaluation scenarios.

**Limitation.** None of these targets physically deployable robot teams. Their
configuration axes (team compositions, unit mixes, social scenarios) have no
counterpart in environmental disturbance, vehicle dynamics, or sensing
degradation, which are the axes that decide whether a maritime policy survives
transfer to hardware.

**This paper.** We define the maritime analogue of a configuration space —
opponent doctrine, current profile, team size, episode seed — with a seen/held-out
split *derived from what training actually samples*, and adopt the same three-test
evaluation triad. We contribute the environment, the baseline suite, the
configuration splits, and the evaluation scripts as an open benchmark package.

## 2.5 Multi-robot and maritime multi-agent RL

**Established.** Aquaticus [novitzky2018aquaticus] established maritime CTF as a
human–robot competitive testbed with real ASVs. Pyquaticus [pyquaticus2023]
provides a lightweight simulator for it. Multi-robot pursuit–evasion and
perimeter-defense work [shishika2020defense] gives control-theoretic guarantees
under restrictive dynamics assumptions. Jacob et al. [jacob2024uav] apply
multi-agent RL to a UAV capture-the-flag variant.

**Limitation.** The control-theoretic results assume simplified dynamics and known
adversary models, which is precisely what an adaptive learned opponent violates.
The learning-based maritime work evaluates against fixed scripted opponents, so
it cannot speak to robustness against an adapting adversary. Jacob et al. differ
in dynamics (aerial, no current field), sensing, and rules; the submitted version
of this paper did not state how their method was adapted to our setting, which
Reviewer 1 rightly flagged as making the comparison untrustworthy.

**This paper.** We state the adaptation of [jacob2024uav] explicitly — which
components were ported unchanged, which were necessarily re-specified for ASV
dynamics and the maritime ruleset, and what was retuned — in an appendix, and we
release that adaptation as code. Robustness is evaluated against learned
adversaries rather than only scripted ones.

## 2.6 Positioning of this paper

Given the above, the contribution is deliberately **not** "a new MARL algorithm."
PPO is used unmodified. The claim is:

> A reproducible maritime multi-robot capture-the-flag benchmark and evaluation
> protocol for opponent-aware population training, with a controlled comparison of
> curriculum league training against self-play, fictitious play, double oracle,
> and ROA-Star across performance, generalization, and exploitability.

That framing is what the AE's novelty objection actually requires: the value is in
the environment, the protocol, the baseline ladder, and the empirical findings —
including the negative ones — not in a new optimizer.

---

## Notes for assembling the bibliography

- `references.bib` in this directory holds the entries cited above. Entries marked
  `TODO-VERIFY` need their exact venue and page numbers checked against the
  publisher record before submission — do not submit unverified citations.
- The self-citation share should be recomputed after the merge; R1's complaint was
  that self-cites were roughly a third of a short list, and the fix is a longer
  list of genuinely relevant work, not the removal of legitimate self-citations.
- R2's small comments still to apply in the main text: define OP4 on first use;
  the abstract should read "over those two settings," not three; define $\pi$ and
  the horizon $H$; drop the $1{:}|N|$ subscript on the joint action; define MCTF
  before use; fix Figure 1's legend (what the circles, diamonds, squares and
  shaded circles are, and which ASV is scoring); enlarge all figure fonts (R3);
  and display Equations 1–4 rather than inlining them, which is what pushes
  Equation 4 outside the margin.
