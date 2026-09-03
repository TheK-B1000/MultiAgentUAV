# ICRA 2027 — manuscript skeleton

**Deadline** 2026-09-15 23:59 PST · **Limit** 8 pages *including references* · no planned extension.

Bones only. Created as insurance so the "the causal-commitment experiment may improve the paper
but not endanger it" rule is structural rather than aspirational. Not prose. Do not polish.

**Supersedes** `docs/archive/` history and the removed 2026-08-31 skeleton (git history at
`ce5315f9` and earlier; deleted from the canonical path 2026-09-02 because its figure plan and
"V3 optional" framing predate V4, the CCP successor, and CCP-S2 entirely and would mislead a
future session into treating that state as current).

---

## Current scientific state (2026-09-02)

| Result | Status |
|---|---|
| Strategic demand (two poles genuinely require different play) | CONFIRMED |
| Specialist crossover (π_A, π_B complementary) | CONFIRMED |
| Latent differentiation (behavioral) | CONFIRMED |
| Trajectory identity (full-episode) | CONFIRMED |
| Strategic payoff preservation under compression | NOT CONFIRMED |
| Commitment-level causal treatment (CCP-S2) | IN PROGRESS |

## Governing sentence

> **The paper is submission-ready on the confirmed specialist and compression-limit results.
> The causal commitment experiment may strengthen the ending but is not a prerequisite for
> submission.**

---

## Working title

*Strategic Demand and the Limits of Latent Compression in Multi-Agent Maritime CTF*

**One-sentence claim.** In a 2v2 maritime CTF task we establish that two scripted-opponent
poles create genuine strategic demand, that specialist policies trained per pole are
complementary and individually strong, and that compressing them into one latent-conditioned
policy reliably reproduces both *behavioral* and *full-trajectory* identity without reproducing
*strategic* advantage — a compression limit that diversity metrics computed at the policy or
trajectory level do not detect. A commitment-level causal treatment (CCP-S2), evaluated
separately and not required for the claim above, tests whether supervising only the
interventions shown to causally improve terminal payoff at real macro-action commitment
boundaries can recover the missing crossover.

**The claim does not depend on CCP-S2.** If CCP-S2 (or a tightly-scoped successor) confirms
crossover, the ending strengthens. If not, the claim above is already fully supported by frozen
results.

---

## Section plan and page budget

| § | Content | Pages | Status |
|---|---|---|---|
| I | Introduction, contributions | 1.0 | stable core |
| II | Related work | 0.5 | to write |
| III | Task, poles, strategic demand | 0.75 | results frozen |
| IV | Learning complementary specialists (SAPPO) | 1.0 | results frozen |
| V | Compressing complementary strategies | 1.5 | results frozen |
| VI | Why behavioral identity is not strategic value (central finding) | 0.75 | results frozen |
| VII | Commitment-level causal strategy preservation (CCP-S2) | 1.0 | **in progress** |
| VIII | Discussion / limitations | 0.5 | to write |
| IX | Conclusion | 0.25 | two variants below |
| — | References | 0.75 | — |
| | **Total** | **8.0** | |

If CCP-S2 does not confirm, §VII shrinks (per-result reporting only, no claim of success) and
its freed space returns to §VIII. Section VII is never cut entirely — unlike the old skeleton's
treatment of V3, CCP-S2 is the paper's live causal-method contribution and gets reported
either way, just at different length.

---

## I. Introduction

- Multi-agent strategic specialization: a policy that adapts its *strategy* to the opponent,
  not merely one that is good on average.
- **Stable core:** SAPPO produces complementary specialists with confirmed crossover.
- Contributions:
  1. A strategic-demand protocol certifying two poles genuinely require different play.
  2. SAPPO specialists with confirmed complementary crossover.
  3. A compression study isolating *where* latent compression fails — extended past behavioral
     identity to full-trajectory identity and private critic capacity, all confirmed, crossover
     still not confirmed.
  4. **The central negative result**, stated positively: neither policy-level nor
     trajectory-level differentiation is sufficient evidence of distinct long-horizon strategy.
  5. A commitment-level causal treatment that changes the supervision target from behavioral
     resemblance to measured incumbent-relative payoff improvement at real decision boundaries.

## II. Related work

MAVEN (temporally committed latent) · OPRE (strategic response hierarchy) ·
ROMA (role identifiability) · DGPO (performance-aware diversity) ·
SHPPO (latent-generated heterogeneous actor parameters) · semi-MDP / options literature for
the commitment-boundary framing · causal credit assignment for the CCP-S2 estimator.

> Positioning note (carried from the prior skeleton, still true): our persistence is *stronger*
> than MAVEN's episode-level commitment and still insufficient on its own — under-committing is
> not the failure mode. SHPPO and ROMA remain the load-bearing comparisons.

## III. Multi-agent maritime CTF and strategic demand

2v2, `map_a`, `RULESET_V3_M1_OWN_FLAG_HOME`. Pole A = OP6 (TURTLE archetype) — favored response
GUARD; Pole B = OP7 (SWITCHER archetype) — favored response BREACH. Demand established, not
assumed (`docs/research-progress-tracker.md`: "OP7 BREACH-GUARD +0.531 PASS; OP6 GUARD-BREACH
+0.094 FAIL").

**Figure 1** (task and poles) — built, QA-passed.

## IV. Learning complementary specialists

SAPPO method; the confirmed crossover; π_A and π_B as the specialist pair used throughout the
rest of the paper.

## V. Compressing complementary strategies

Oracle-Gated (one-sided) → Paired Latent (OG-PSP) → Trajectory-Guided (H-OG-PSP V3) →
Private-Critic (H-OG-PSP V4). Each step adds capacity or supervision; none confirms crossover.

**Figure 2** (crossover forest plot, all methods incumbent-relative, 95% CI) — built,
provenance-backed, quantitative centerpiece of the paper.

## VI. Why behavioral identity is not strategic value — central finding

```
teacher imitation  ->  latent differentiation  ->  trajectory identity  ->  strategic payoff
   ESTABLISHED            CONFIRMED                    CONFIRMED              BROKEN
```

This chain does not need its own figure now that Figure 2 carries the quantitative burden — a
compact table or callout suffices. Carry forward the prior skeleton's three supporting facts
(capacity was present, differentiation was real not inert, the cross-pole asymmetry inverted
between mechanism-diagnostic prediction and EVAL) and its methodological note: a confounded
comparison that scores exactly 1.0000 by identifying the opponent rather than the strategy is a
warning about diversity diagnostics generally, not specific to one method.

## VII. Commitment-level causal strategy preservation

Semi-MDP discovery (per-agent asynchronous commitment boundaries, `commit_ticks_left <= 0`) and
CCP-S2: incumbent-relative causal advantage measured at prospectively-selected boundaries,
routed to whichever teacher improves on the incumbent (or neither), causally-weighted
warm-start fine-tuning, fresh sealed EVAL.

**Figure 3** (commitment boundary + causal branching) — built, real environment identifiers,
QA-passed. **If CCP-S2's result is not strong enough, shorten this section's reporting rather
than rebuilding the paper** — the method and figure stand regardless of outcome; only the
strength of the claim in the text changes.

## VIII. Discussion / limitations

*To be written once VII's result is known — placeholder.*

## IX. Conclusion — two variants, pick after VII resolves

**(a) causal-success ending.** Commitment-level causal supervision, not behavioral or
trajectory resemblance, is the objective that recovers complementary strategic crossover after
capacity, data, persistence, and memorization were each independently excluded as the limiting
factor.

**(b) compression-limit ending — default, safe regardless of VII.** Across an increasingly
capable sequence of behavioral- and trajectory-identity-preserving treatments, we fix latent
differentiation and full-trajectory identity without recovering strategic advantage. The limit
is not capacity, data, persistence, or memorization — each independently excluded. Policy- and
trajectory-level differentiation are not sufficient evidence of distinct long-horizon strategy;
preserving the decisions that causally determine terminal payoff, rather than the behavior that
merely correlates with it, is the direction this negative result points toward.

---

## Figures / tables

| # | Content | Role | Status |
|---|---|---|---|
| F1 | Task and strategic poles | Required | **built, QA-passed** |
| F2 | Crossover forest plot (all methods, incumbent-relative, 95% CI) | Required, quantitative centerpiece | **built, provenance-backed** |
| F3 | Commitment boundary + causal branching | Current causal-method figure; final inclusion/length depends on manuscript space and CCP-S2's result | **built, QA-passed** |
| T1 | Compression ladder: differentiation / trajectory identity / private actor / private critic / crossover, per method | Compresses the chronological method history into one progression | to build from frozen records |
| T2 | Final quantitative results: V(z0), V(z1), Δ, CI per pole per method | Numeric backing for Figure 2 | to build once CCP-S2 resolves |

All three figures share one style pipeline (`paper/figures/figure_style.py`): Times New Roman,
embedded/subsetted fonts (verified via `pdffonts`), IEEE column widths, CVD-safe redundant
color/marker/linestyle encoding, vector PDF canonical + 600dpi PNG preview.

---

## Provenance for every claimed number

Carried forward from the prior skeleton (not re-audited tonight, inherited from an
already-frozen, git-committed document):
`OG_PSP_EVAL_B_SIDE_INTEGRITY.json` (a2162f77) ·
`OG_PSP_MECHANISM_DIAGNOSTIC.json` (a59ee84a) ·
`OG_PSP_MODEL_FROZEN.json` (75e1e696) ·
`TEACHER_TRAJECTORY_SEPARABILITY_PROBE.json` (bdd23fc4) ·
`LATENT_PROGRAM_POSTMORTEM.json` (3178c804)

Independently verified tonight (commit exists AND its diff created the cited file, checked via
`git show --stat`, not merely trusted from a source report):
`artifacts/strategic_demand/sappo_crossover/summary.json` (17fd8f87) ·
`ORACLE_GATED_K2_EVAL_RESULT.json` (ff4d36ef) ·
`OG_PSP_EVAL_RESULT.json` (5183f296) ·
`HOG_PSP_V3_EVAL_RESULT.json` (9e1684a6) ·
`HOG_PSP_V4_EVAL_RESULT.json` (ab9ddd7e) ·
`CCP_SUCCESSOR_EVAL_RESULT.json` (698d5ea6, also the source of the trajectory-identity numbers
paired with mechanism commit 4a6e4239)

CCP-S2 in-progress records (not yet a final result): `CCP_S2_SPEC.json` (bf41980a) ·
`CCP_S2_STATE_MANIFEST.json` (c366f317) · `CCP_S2_COMPUTE_BUDGET_AMENDMENT.json` (f1aaab17).

Figure 2's exact numbers are re-derived from these files at build time by
`paper/data/extract_fig2_data.py` — never hand-typed — with commit provenance computed via
`git log`, not hardcoded, so this table and the figure cannot silently drift apart.

> Every number in the manuscript must trace to a frozen artifact. No number enters the paper
> that is not in one of these records.
