# Parameter–specialization tradeoff (2v2 & 4v4)

Paste-ready for professor. Δ cells use sealed/flagged crossover means [LCB95, UCB95]
(paired percentile bootstrap, n_boot=20000, α=0.05, rng=7).
Imitation = holdout argmax z0↔π_A / z1↔π_B. Reduction vs Share-0 (2v2) or Separated (4v4).

Δ_A = V(z0,A) − V(z1,A); Δ_B = V(z1,B) − V(z0,B).

## 2v2

| Condition | Params (M) | Reduction (%) | Imitation agreement | ΔA | ΔB |
|---|---:|---:|---:|---|---|
| Share-0 | 6.947 | 0 | 1.000 / 1.000 | +0.289 [+0.164, +0.406] | +0.258 [+0.141, +0.375] |
| Share-Encoder | 3.613 | 48.0 | 0.967 / 0.989 | +0.266 [+0.148, +0.383] | +0.164 [+0.047, +0.281] |
| Share-Backbone | 3.509 | 49.5 | 0.963 / 0.985 | +0.148 [+0.023, +0.266] | +0.141 [+0.016, +0.266] |
| Share-Macro | 3.508 | 49.5 | 0.936 / 0.969 | +0.125 [0.000, +0.250] **FAIL** | +0.156 [+0.031, +0.281] |
| Fully Shared+z | 3.457 | 50.2 | 0.965 / 0.988 | — (to be redone, same method as 4v4) | — (to be redone, same method as 4v4) |

## 4v4 (CLOSEST_DEFENDS k=2)

| Condition | Params (M) | Reduction (%) | Imitation agreement | ΔA | ΔB |
|---|---:|---:|---:|---|---|
| Separated (≈ Share-0) | 6.929 | 0 | — (not distilled) | +0.156 [+0.039, +0.273] | +0.406 [+0.297, +0.508] |
| Share-Encoder | 3.637 | 47.5 | 0.914 / 0.876 | **INVALIDATED** † | **INVALIDATED** † |
| Share-Backbone | 3.533 | 49.0 | 0.901 / 0.862 | **INVALIDATED** † | **INVALIDATED** † |
| Share-Macro | 3.531 | 49.0 | 0.899 / 0.859 | **INVALIDATED** † | **INVALIDATED** † |
| Fully Shared+z | 3.469 | 49.9 | 0.895 / 0.856 | **INVALIDATED** † | **INVALIDATED** † |

**Notes**
- † **INVALIDATED FOR THE INTENDED 4v4 SUITE COMPARISON** (`SUITE_4V4_DISTILLED_ARMS_POLE_B_INVALIDATION.json`): the distillation dataset and the crossover evaluations used **plain OP7** rather than the **certified B3-3 Pole B** (`lock_defender=10, enable_2v1`) that the frozen protocol specifies. Because the dataset itself was collected on the wrong pole, the students were *trained* on the wrong Pole-B states, so re-evaluating them would not repair the rows: the dataset must be recollected on B3-3 and the students redistilled. The sealed values remain provenance only ("this checkpoint, evaluated on plain OP7, gave these numbers") and are **excluded from paper claims**. For the record, on plain OP7: Encoder ΔA +0.359 / ΔB −0.219; Backbone +0.438 / −0.281; Macro +0.281 / −0.156; Fully Shared+z +0.281 / −0.109. Scope audited in `SUITE_4V4_POLE_B_IDENTITY_AUDIT.json`: the Separated row and the KL teacher pair were on certified B3-3 and **remain valid**.
- Share-Macro 2v2 fails because LCB95(Δ_A)=0 (joint gate).
- **Measured cross-scale nonconformance (2026-09-26).** An audit of the loaded checkpoints and frozen dataset manifests (`CROSS_SCALE_METHODOLOGY_IDENTITY_AUDIT.json`) found seven axes differing across scales, and 6v6 is affected as well as 2v2. The SA-PPO strategy anchor was **on** for the 2v2 teachers and **off** at 4v4/6v6; entity repair is **on** only at 4v4; the 2v2 foundation ran 1.5 M steps vs 1 M elsewhere; and only the 4v4 distillation dataset was collected under CLOSEST_DEFENDS with entity tensors and roles stored. Until the three scales are rebuilt to one target (4v4 is the reference), no Δ comparison across scale blocks in this table is a like-for-like comparison.
- **2v2 methodology mismatch (2026-09-26).** The 2v2 rows above are the *natural* 2v2 setup: no heuristic CLOSEST_DEFENDS allocator, teachers without entity repair, and a legacy distillation dataset. The 4v4 rows use the allocator, entity-repair teachers and the suite dataset. Because the contribution is one methodology run at every team size, the 2v2 rows are to be redone through the same pipeline (`SAME_METHODOLOGY_2V2_PORT_PLAN.json`). Until then the 2v2 and 4v4 blocks are not comparable rows of one figure.
- The 2v2 Fully Shared+z natural-setup crossover I started was stopped at 82/512 episodes and is not a paper row.
- 4v4 Backbone/Macro are a PI-authorized depth extension outside the locked cross-scale suite (which names Share-Encoder as the only 4v4 partial-share arm). Both were distilled from the same `SUITE_DISTILLATION_4V4` dataset under CD(k=2). Share-Macro's eval was interrupted at 60/256 by a reboot on 2026-09-25 and relaunched from scratch; it finished 2026-09-26.
- 4v4 Separated Δ is confirmatory **n=128**. Share-Encoder / Share-Backbone / Share-Macro / Fully Shared+z Δ are exploratory **n=64** under `SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC`; all four wrote `INTEGRITY_REQUIRED` (Δ_B ≤ 0). All four were then **sealed together post-hoc** through `run_state.seal` (13/13 gating audit checks each, every statistic re-derived from the rows on disk; `*_CROSSOVER_EVAL_RESULT.json` + `*_AUDIT.json`). **Sealed integrity is not a pass:** each record carries `scientific_verdict = FLAG`, and **it is not historical Rule-9 compliance either** — the seed blocks (22510001, 22511001, 22513001, 22514001; 64 each) were registered only after being spent, and each seal records `seed_registration_origin = RETROACTIVE_RECONCILIATION`, `historically_pre_registered = false`, read from the registry. Confirmatory n=128 not authorized.
- ~~Reading: all four distilled arms keep strong Pole-A specialization; all four reverse on Pole B.~~ **Withdrawn 2026-09-26:** the Pole-B cells were measured on plain OP7, not the certified pole (†). No reading of the four student rows stands. Separated still passes the joint gate, on certified B3-3.
- Pole win rates (n=64 each): Share-Backbone z0@A=0.875, z1@A=0.4375, z0@B=0.8906, z1@B=0.6094; Share-Macro z0@A=0.8125, z1@A=0.5312, z0@B=0.9688, z1@B=0.8125.
- Row-level integrity audit (`SUITE_4V4_SHARING_FLAGGED_ARMS_ROW_AUDIT.json`) on all four arms: rows, seed blocks, checkpoint pins and scores are consistent, forcing z changes the episodes (identical z0/z1 outcomes on only 13–33% of seeds), and each student's z labels agree with the right teacher (0.86–0.91 holdout). The audit does not seal any result.
- ~~Why Δ_B reverses: the students reproduce the teacher pair's 4v4 asymmetry.~~ **Withdrawn 2026-09-26.** That reading compared the teacher pair's Δ_B −0.156 [−0.266, −0.047] — measured on **certified B3-3** (live-attested at launch, hash `ee1cab77…`) — with student Δ_B measured on **plain OP7**. The caveat previously noted here ("I did not verify they used the same Pole-B overlay") turned out to be exactly the problem: they did not. The student Δ_B reversal may be an artifact of the uncertified, easier pole (student z0@B won 0.94–0.97 vs π_A3's 0.74 on B3-3). The teacher pair's own 4v4 result is unaffected and still stands.
