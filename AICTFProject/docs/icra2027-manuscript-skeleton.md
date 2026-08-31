# ICRA 2027 — manuscript skeleton

**Deadline** 2026-09-15 23:59 PST · **Limit** 8 pages *including references* · no planned extension.

Bones only. Created as insurance so the "V3 may improve the paper but not endanger it"
rule is structural rather than aspirational. Not prose. Do not polish.

---

## Working title

*Strategic Demand and the Limits of Latent Compression in Multi-Agent Maritime CTF*

**One-sentence claim.** In a 2v2 maritime CTF task we establish that two scripted-opponent
poles create genuine strategic demand, that specialist policies trained per pole are
complementary and individually strong, and that compressing them into one latent-conditioned
policy reliably reproduces *policy-level* differentiation without reproducing *strategic*
advantage — a compression limit that single-state diversity metrics do not detect.

**The claim does not depend on V3.** If V3 confirms, the ending strengthens. If not, the
claim above is already fully supported by frozen results.

---

## Section plan and page budget

| § | Content | Pages | Status |
|---|---|---|---|
| 1 | Introduction, contributions | 1.0 | stable core |
| 2 | Related work — MAVEN, OPRE, ROMA, DGPO, SHPPO | 0.5 | to write |
| 3 | Task, poles, strategic demand experiment | 1.0 | results frozen |
| 4 | SAPPO method + complementary specialists | 1.5 | results frozen |
| 5 | Compression study: V1 → OG-PSP | 1.5 | results frozen |
| 6 | The three-layer analysis (central finding) | 1.0 | results frozen |
| 7 | *optional* H-OG-PSP V3 | 0.5 | **ONLY IF CONFIRMED** |
| 8 | Deployment realism | 0.5 | placeholder |
| 9 | Conclusion | 0.25 | two variants below |
| — | References | 0.75 | — |
| | **Total** | **8.0** | |

If V3 is not confirmed, §7's 0.5 pages return to §6 and §8.

---

## 1. Introduction

- Multi-agent strategic specialisation: why one policy that adapts its *strategy* to the
  opponent is the goal, not one policy that is merely good on average.
- **Stable core:** SAPPO produces complementary specialists with confirmed crossover.
- Contributions:
  1. A strategic-demand protocol that certifies two poles genuinely require different play.
  2. SAPPO specialists with a confirmed complementary crossover.
  3. A compression study isolating *where* latent compression fails.
  4. **The central negative result**, stated positively: policy-level differentiation is not
     sufficient evidence of distinct long-horizon strategy.

## 2. Related work

MAVEN (temporally committed latent) · OPRE (strategic response hierarchy) ·
ROMA (role identifiability) · DGPO (performance-aware diversity) ·
SHPPO (latent-generated heterogeneous actor parameters).

> Positioning note: our persistence was *stronger* than MAVEN's episode-level commitment
> (one fixed latent per env for 1M steps) and still insufficient — so we are not simply
> under-committing. SHPPO and ROMA are the load-bearing comparisons.

## 3. Task and strategic demand

2v2, `map_a`, `RULESET_V3_M1_OWN_FLAG_HOME`. Pole A = OP6 + `SDS2_A_payoff_INIT_3`
overlay (conditional defender); Pole B = OP7. Demand established, not assumed.

## 4. SAPPO and complementary specialists

Method; the confirmed crossover; π_A and π_B as the specialist pair used throughout.

## 5. Compression study

- **V1 (oracle-gated, one-sided).** `CROSSOVER_NOT_CONFIRMED`. Postmortem: the objective was
  satisfiable by a latent-independent map; crossed comparison showed z1 matched π_A nearly as
  well as z0 (93.0% vs 94.7%); JSD 0.0051 nats.
- **OG-PSP (paired).** Both specialist targets on the same state under different latent IDs.
  Fixed layer two: matched-CALIB JSD 0.006589 → 0.042102 nats (6.39×), no collapse FIT→CALIB.
- **OG-PSP EVAL.** `CROSSOVER_NOT_CONFIRMED`. δ_A +0.0625 [−0.1250, +0.2500];
  δ_B +0.0000 [−0.1875, +0.1875]. Retention 1.1452.

## 6. The three-layer analysis — **central finding**

```
teacher imitation  →  latent differentiation  →  strategic payoff
   ESTABLISHED            FIXED by OG-PSP           BROKEN
```

Three facts that make this a finding rather than a null result:

1. **Capacity was present.** Retention 1.1452; z0 beats π_A on π_A's own pole (0.8750 vs
   0.7500). Not a lossy copy.
2. **Differentiation was real, not inert.** B-pole cells differ on 19/32 EVAL seeds, with
   5 seeds each way cancelling exactly. Differently-behaving *and equally good*.
3. **The asymmetry inverted.** The mechanism diagnostic predicted B separates (+10.0 pp
   crossed gap) and A does not (+1.32 pp); EVAL delivered the opposite ordering.

**Supporting probe.** With the opponent controlled, the specialists are separable from
held-out trajectories at 0.9531 (Pole A) / 0.9688 (Pole B). Single-state teacher actions
provide relatively sparse disagreement, while aggregate trajectories carry highly
discriminative specialist identity — motivating trajectory-level supervision.
*(Metrics are on different scales; do not present as a ratio.)*

**Methodological contribution worth its own paragraph.** The confounded comparison
(π_A@PoleA vs π_B@PoleB) scores exactly **1.0000** — a discriminator wins by identifying the
opponent, learning nothing about strategy. Diversity diagnostics in MARL must control the
opponent, or they certify a signal that does not exist.

## 7. *Optional* — H-OG-PSP (**INCLUDE ONLY IF CONFIRMED**)

Private per-latent actor capacity + frozen teacher-grounded trajectory discriminator.
If `HOG_PSP_V3_CROSSOVER_NOT_CONFIRMED`, **cut this section entirely** and reallocate pages.
Do not report it as a partial or promising result.

## 8. Deployment realism

*Placeholder — scope to be decided.*

## 9. Conclusion — two variants, pick one

**(a) latent-success ending.** Compression to complementary latent strategies is achievable,
but requires latent-private actor capacity and trajectory-grounded identity; state-level
conditioning and supervision are insufficient, as the V1/OG-PSP progression shows.

**(b) compression-limit ending — default.** Across two preregistered treatments we fix
latent differentiation without recovering strategic advantage. The limit is not capacity,
data, persistence, or memorisation — each independently excluded. Policy-level
differentiation is not sufficient evidence of distinct long-horizon strategy, and diversity
metrics computed at single states can certify separation that carries no strategic value.

---

## Figures / tables

| # | Content | Source | Status |
|---|---|---|---|
| F1 | Task + two poles | — | to draw |
| F2 | Strategic demand result | frozen | ready |
| F3 | SAPPO crossover matrix | frozen | ready |
| T1 | V1 vs OG-PSP vs (V3) cross-eval + δ with LCB95 | frozen | ready |
| F4 | Three-layer diagram w/ break at layer 3 | §6 | to draw |
| F5 | Matched-CALIB JSD 6.39× + crossed gaps | frozen | ready |
| T2 | What was excluded: demand, difference, ignored-z, bank, memorisation, persistence | frozen | ready |

## Provenance for every claimed number

`OG_PSP_EVAL_RESULT.json` (5183f296) · `OG_PSP_EVAL_B_SIDE_INTEGRITY.json` (a2162f77) ·
`OG_PSP_MECHANISM_DIAGNOSTIC.json` (a59ee84a) · `OG_PSP_MODEL_FROZEN.json` (75e1e696) ·
`ORACLE_GATED_K2_EVAL_RESULT.json` (ff4d36ef) ·
`TEACHER_TRAJECTORY_SEPARABILITY_PROBE.json` (bdd23fc4) ·
`LATENT_PROGRAM_POSTMORTEM.json` (3178c804)

> Every number in the manuscript must trace to a frozen artifact. No number enters the paper
> that is not in one of these records.
