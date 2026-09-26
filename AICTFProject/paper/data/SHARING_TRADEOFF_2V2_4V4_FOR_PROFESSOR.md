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
| Share-Encoder | 3.637 | 47.5 | 0.914 / 0.876 | +0.359 [+0.188, +0.516] | −0.219 [−0.344, −0.109] **FLAG** |
| Share-Backbone | 3.533 | 49.0 | 0.901 / 0.862 | +0.438 [+0.297, +0.578] | −0.281 [−0.422, −0.125] **FLAG** |
| Share-Macro | 3.531 | 49.0 | 0.899 / 0.859 | — (eval pending) | — (eval pending) |
| Fully Shared+z | 3.469 | 49.9 | 0.895 / 0.856 | +0.281 [+0.125, +0.438] | −0.109 [−0.219, 0.000] **FLAG** |

**Notes**
- Share-Macro 2v2 fails because LCB95(Δ_A)=0 (joint gate).
- **Measured cross-scale nonconformance (2026-09-26).** An audit of the loaded checkpoints and frozen dataset manifests (`CROSS_SCALE_METHODOLOGY_IDENTITY_AUDIT.json`) found seven axes differing across scales, and 6v6 is affected as well as 2v2. The SA-PPO strategy anchor was **on** for the 2v2 teachers and **off** at 4v4/6v6; entity repair is **on** only at 4v4; the 2v2 foundation ran 1.5 M steps vs 1 M elsewhere; and only the 4v4 distillation dataset was collected under CLOSEST_DEFENDS with entity tensors and roles stored. Until the three scales are rebuilt to one target (4v4 is the reference), no Δ comparison across scale blocks in this table is a like-for-like comparison.
- **2v2 methodology mismatch (2026-09-26).** The 2v2 rows above are the *natural* 2v2 setup: no heuristic CLOSEST_DEFENDS allocator, teachers without entity repair, and a legacy distillation dataset. The 4v4 rows use the allocator, entity-repair teachers and the suite dataset. Because the contribution is one methodology run at every team size, the 2v2 rows are to be redone through the same pipeline (`SAME_METHODOLOGY_2V2_PORT_PLAN.json`). Until then the 2v2 and 4v4 blocks are not comparable rows of one figure.
- The 2v2 Fully Shared+z natural-setup crossover I started was stopped at 82/512 episodes and is not a paper row.
- 4v4 Backbone/Macro are a PI-authorized depth extension outside the locked cross-scale suite (which names Share-Encoder as the only 4v4 partial-share arm). Both were distilled from the same `SUITE_DISTILLATION_4V4` dataset under CD(k=2). Share-Macro's crossover eval was interrupted at 60/256 by a reboot and relaunched from scratch on 2026-09-25 (about 3 h); its Δ cells are pending.
- 4v4 Separated Δ is confirmatory **n=128**. Share-Encoder / Share-Backbone / Fully Shared+z Δ are exploratory **n=64** under `SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC`; all three wrote `INTEGRITY_REQUIRED` (Δ_B ≤ 0) — no `FROZEN_RESULT` seal. Confirmatory n=128 not authorized.
- Reading: all three distilled arms keep strong Pole-A specialization; all three reverse on Pole B (z0 beats z1 on B, so Δ_B = V(z1,B)−V(z0,B) is negative). Separated still passes the joint gate.
- Share-Backbone pole win rates: z0@A=0.875, z1@A=0.4375, z0@B=0.8906, z1@B=0.6094 (n=64 each).
- Row-level integrity audit (`SUITE_4V4_SHARING_FLAGGED_ARMS_ROW_AUDIT.json`) on Share-Encoder, Share-Backbone and Fully Shared+z: rows, seed blocks, checkpoint pins and scores are consistent, forcing z changes the episodes (identical z0/z1 outcomes on only 13–33% of seeds), and each student's z labels agree with the right teacher (0.86–0.91 holdout). Share-Macro will be audited when its eval finishes. The audit does not seal any result.
- Why Δ_B reverses: the teachers these students imitate (4v4 entity-repair specialists π_A3 → z0, corrected π_B3 → z1) already fail Δ_B at 4v4: Δ_A +0.273, Δ_B −0.156 [−0.266, −0.047] (n=128, `4V4_ENTITY_REPAIR_CORRECTED_CROSSOVER_READING.json`). π_A3 wins on both poles (0.81 on A, 0.74 on B) and π_B3 is weak on both (0.54, 0.59), so z0 behaves like a generalist. The student reversal is consistent with faithful imitation of that teacher pair and is not evidence that sharing removes Pole-B specialization. The Separated row is the split ATTACK/DEFEND composite, a different system from these distillation teachers, so the drop in Δ_B from Separated to the students should not be read as the cost of sharing. Caveat: the teacher figures use different seeds and n, and I did not verify they used the same Pole-B overlay as the student evals.
