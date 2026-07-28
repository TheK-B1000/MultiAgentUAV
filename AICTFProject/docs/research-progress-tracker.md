# Research Progress Tracker

**Owner:** This file is the single source of truth for *current* run
status, *open* decisions, and *recommended next* experiments. It is the
working logbook of the research effort; it is updated when a run
launches, finishes, fails, or its interpretation changes.

It is **not** the source of truth for:

* The scientific method definition →
  [`summer-method-spec.md`](summer-method-spec.md).
* Fidelity rules, classification, or proposal templates →
  [`summer-fidelity-rules.md`](summer-fidelity-rules.md).
* Per-preset facts, aliases, or resolved deltas →
  [`latent-preset-registry.md`](latent-preset-registry.md).
* Launch / eval / statistical protocols →
  [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md).

> **Last updated:** 2026-07-27 (UTC-4)

---

## 1. Run-status legend

| Tag         | Meaning                                                                                                  |
|-------------|----------------------------------------------------------------------------------------------------------|
| `RUNNING`   | Active training job; check the terminals folder for the live tail.                                       |
| `COMPLETED` | Training finished at the configured budget; checkpoints + CSVs are on disk; eval may or may not be done. |
| `EVALUATED` | Training finished and the §4 / §5 protocol in `experiment-and-evaluation-protocol.md` has been run.       |
| `FAILED`    | Training did not reach the budget, or completed but a §7 trainer-side invariant was violated.            |
| `PLANNED`   | Preset / launch designed but not started; the *Proposed Preset Review* template must be on file.         |
| `DEFERRED`  | Designed but explicitly paused pending a decision from a parent comparison.                              |

A row is "ready to be cited in a paper claim" only when status is
`EVALUATED` and the §9 checklist in
[`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)
is satisfied.

---

## 2. Current run status (v4 / v5 ladder, 4v4, OP5/OP6/OP7)

All rows use the §1 invariants of
[`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)
unless otherwise noted (4v4, OP5/OP6/OP7 uniform, 1 M steps, `n_envs=32`,
`n_epochs=6`, `--seed 0`, `--device cuda`).

### 2.1 v5i4 (conditional-entropy paper-faithful interpretation) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i4_paper_faithful` (conditional-entropy paper-faithful interpretation)                             |
| Classification                          | `PAPER-FAITHFUL`                                                                                       |
| Status                                  | `COMPLETED` — 1,000,000 / 1,000,000 decision steps (2 h 05 m wall).                                   |
| Run tag on disk (artifact filename)     | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4`                                                   |
| Post-fix preset run tag                 | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4`                                                   |
| Discrepancy                             | Run was launched before the `_2m_` → `_1m_` tag fix; artifacts retain `_2m_`. See [`latent-preset-registry.md`](latent-preset-registry.md) §7.1. |
| Final checkpoint                        | `AICTFProject/checkpoints/4v4/final_v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4.zip`            |
| Periodic checkpoints                    | 50k stride → `ckpt_..._50000.zip` … `ckpt_..._1000000.zip`                                            |
| Final rollout WR (training)             | `64.4 %` (W=6442, L=3403, D=155, n=10 000)                                                            |
| Final ev (explained variance)           | `0.605`                                                                                                |
| Final `zH` (q_phi entropy on resample)  | `0.642` (vs `ln(K) = 1.386`); end-of-run `H/ln(K) ≈ 0.46`                                              |
| Final `z_occ`                           | `[0.092, 0.137, 0.706, 0.065]` — **z=2 dominant**                                                      |
| Final per-`z` WR (rollout, non-causal)  | `[0.677, 0.712, 0.685, 0.696]`                                                                         |
| Final `MI_z_outcome` (E3)               | `0.0041`                                                                                               |
| Final `actor_z_jsd` (sep_JSD)           | `8.33e-4` (max_JSD `0.068`)                                                                            |
| Final `actor_input_dim`                 | `164` (matches paper-faithful R14 in `summer-fidelity-rules.md`)                                       |
| Eval status                             | `PENDING` — `plot/eval_checkpoint.py` matrix has not yet been run.                                    |
| Decisive comparison rows still needed   | `no_latent_v4i3_baseline` (same-everything-except-z, **`COMPLETED`** for v4i3 budget — needs 1M run at v5i4 budget); `v5_strict_summer` at same seed; `v5i4 random-matched` eval-time row. |

**Observations from rollout telemetry only (not causal):**

* The router collapsed to a z=2-dominant occupancy by 1 M, even with
  the on-policy categorical PPO term ON. This is the same failure mode
  v5i2 / v5i3 were designed to repair (v5i3 added a forced-z anneal as
  a *coverage* fix on top of v5i2; v5i4 deliberately omits any
  curriculum to stay literal-paper-faithful).
* `zH` decayed from ~1.0 by step ~720 k to `0.64` at 1 M; the entropy
  anneal floor (`latent_lam_h_end = 0.0002`) was reached well before
  the collapse stabilized.
* Per-`z` WR spread is modest (`max − min ≈ 0.035`) but the per-`z`
  values are **not** causal evidence — see
  [`AGENTS.md`](../../AGENTS.md) §8.7.
* MI(`z`; outcome) is `O(10^-3)` — informative as a floor only.

**Next actions for v5i4:**

1. Run the eval matrix (`experiment-and-evaluation-protocol.md` §4)
   on the final checkpoint at `--episodes 300`, `--map-sets train eval`,
   `--opponents OP5_RUSHER OP6 OP7`.
2. Run the matched-schedule routing-quality control:
   `--latent-selection router` vs `--latent-selection random-matched`
   with the same `--seed`, same checkpoint, same `--episodes`.
3. Run `tools/q_probe.py` (forced-z return contrast) and
   `tools/q_probe_local_counterfactual.py` (local Q-contrast) on the
   final checkpoint.
4. Assemble the per-checkpoint Markdown report via
   `tools/summer_proof_report.py`.
5. Compare to `no_latent_v4i3_baseline` (a re-launched 1 M-step
   baseline at the same seed; the existing v4i3 baseline is at the
   v4i3 budget).
6. Use this row as the conditional-entropy reference for v5i6 comparisons
   before adding more v5i4 seeds.

### 2.2 v5_strict_summer (literal-Summer ablation) — `COMPLETED` (earlier)

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5_strict_summer`                                                                                     |
| Classification                          | `ABLATION` (literal `docs/algorithm.md` loss; no task-reward PG channel on `q_phi`)                   |
| Status                                  | `COMPLETED` (earlier; pre-dates the v5i4 launch).                                                     |
| Result interpretation                   | The literal equation alone (persistence + entropy on `q_phi`, no `L_strategy_PPO`) collapsed `q_phi` to a single z; rollout WR did not exceed the no-latent control by a paired-bootstrap-significant margin. This is the result that *motivated* v5i4 (adding the on-policy categorical PPO term). |
| Run tag on disk                         | `v5_strict_summer_OP5_OP6_OP7_2m_4v4` (legacy `_2m_` suffix; see §7.2 of the registry).                |
| Decisive comparison done                | v5_strict_summer vs `no_latent_v4i3_baseline`: not paired-bootstrap-significant. v5_strict_summer vs v5i4: pending (the v5i4 row above).                                                                                                                                                       |

### 2.3 v5i1_reward_credit_router (per-episode router PPO + dedicated AdamW) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i1_reward_credit_router`                                                                            |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION`                                                                          |
| Status                                  | `COMPLETED`.                                                                                          |
| Result interpretation                   | The dedicated AdamW + per-episode router PPO repaired the router collapse only partially; FiLM-less actor coverage remained the bottleneck. Motivated v5i2 (add FiLM). |

### 2.4 v5i2_stronger_z_conditioning (v5i1 + FiLM) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i2_stronger_z_conditioning`                                                                         |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION` (adds R11 FiLM)                                                          |
| Status                                  | `COMPLETED`; per-checkpoint files on disk through `_550000.zip`.                                       |
| Final-checkpoint observation            | `q_phi` collapsed to z=2 dominant within the first 200 k steps; z=1 reached `<5 %` occupancy by 540 k. The actor's per-`z` sensitivity grew steadily under FiLM, but only on the `z` values the router actually picked. |
| Result interpretation                   | Coverage problem (not credit-assignment); motivated v5i3 forced-z anneal.                              |

### 2.5 v5i3_balanced_warmup (v5i2 + forced-z anneal) — `COMPLETED` (partial)

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i3_balanced_warmup`                                                                                 |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION` (adds R27/R28 forced-z anneal on top of v5i2)                            |
| Status                                  | `COMPLETED` partial budget; on-disk checkpoints reach `_100000.zip` then stop (run was rotated to v5i4 once the v5i4 design was finalized). |
| Result interpretation                   | Not yet sufficient runtime to compare against v5i4; if the v5i4 forced-z-free row collapses, v5i3 is the established compound-extension fix to compare against. |

### 2.6 v4i3_summer_proof (arc-credit row) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v4i3_summer_proof`                                                                                    |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION` (carries v3i19 arc-credit; **not** literal paper-faithful — see [`AGENTS.md`](../../AGENTS.md) §3) |
| Status                                  | `COMPLETED` at `_1000000.zip` (and full periodic stride from `_50000.zip`).                            |
| Run tag on disk                         | `v4i3_summer_proof_OP5_OP6_OP7_2m_4v4` (budget-agnostic in current preset; on-disk has the historical `_2m_` suffix). |
| Role in the proof table                 | The *arc-credit row*. Not a substitute for v5i4.                                                       |

### 2.7 no_latent_v4i3_baseline (same-everything-except-z control) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `no_latent_v4i3_baseline`                                                                              |
| Classification                          | `DIAGNOSTIC` (matched no-latent control)                                                               |
| Status                                  | `COMPLETED` at the v4i3 budget; **not yet re-launched at v5i4's `--total-steps 1000000`** with `--seed 0` for the headline v5i4 comparison. |
| Required next                           | Re-launch at v5i4's exact budget and seed for the headline `v5i4` vs `no_latent` comparison.            |

---

## 3. Planned / proposed experiments

> Every row in this section must be backed by a completed *Proposed
> Preset Review* template (see
> [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §8). PLANNED
> rows that have not yet had the template filed are explicitly labeled
> as such.

### 3.0.0 OP6-OP12 strategic BT niches - `IMPLEMENTED, PENDING_EVAL`

**Status:** `IMPLEMENTED, PENDING_EVAL`. The scripted BT opponent pool now
uses a single active OP6-OP12 strategic-niche registry: OP6 immediate dual
rush, OP7 deep fortress, OP8 protected carrier escort, OP9 split-lane feint,
OP10 aggressive interceptor, OP11 adaptive exploiter, and OP12 late
converter. Old aliases remain compatibility handles but resolve to the new
identities.

**Boundary:** no reward labels, no oracle z targets, and no physical
family-clone hardening. The OP8/OP10/OP11 blue-carrier slowdown and red
overdrive constants are neutralized; separation is intended to come from role
gates, lock timings, lane usage, mines, escort/intercept/counter logic, and
adaptive memory for OP11/OP12. Forced-z behavior separation still remains the
router-readiness gate before making latent-strategy claims.

**OP6 calibration status (2026-07-26):** full scripted-style matrix collection
is paused. The first OP6/map_b block initially showed OP6 as universally hard,
so the current work is single-cell calibration rather than pool evaluation.
The OP6 long-name dispatch path is fixed and covered by tests. BLUE_RUSH,
BLUE_TURTLE, BLUE_SPLIT, and BLUE_ESCORT trajectory probes now validate their
intended coarse behaviors.

Current best development screen:
`artifacts/op6_failure_timeline_dev11_tag_counts_8seed`, OP6/map_b, 8 paired
development seeds, all four blue styles. TURTLE is the best response but not
yet an accepted counter: WR 1/8, mean margin -1.125. It is clearly better than
RUSH (-1.875), ESCORT (-2.125), and SPLIT (-2.625), and delays first red
capture to ~109.3 decision steps vs ~35-43 for the exposed styles.

The earlier "zero red deaths" interpretation was incomplete because tags do
not flip `red_alive`. The corrected diagnostic records tag transitions:
TURTLE averages 7.625 red tags and 2.875 red-carrier tags per episode. The
remaining OP6 problem is therefore not failed contact mechanics; turtle is
stopping carriers but OP6 still converts enough after tags/resets/respawns to
keep payoff slightly negative.

Post-tag counterattack development screen:
`artifacts/op6_failure_timeline_dev12_turtle_post_tag_counter_8seed` added a
BLUE_TURTLE-only 20-step counter window after a red-carrier tag or stopped
dual rush. Agent 0 remains defensive while agent 1 attacks the red flag unless
the blue flag is threatened again. No OP6, reward, speed, tag, PPO, LRO, or
router changes were made.

Result: TURTLE remains best and improves from mean margin -1.125 to -0.75 on
the same 8 paired development seeds. Post-tag counterattack launch and blue
flag-touch counts both average 2.125 per episode. However, red re-enters blue
territory about 1.0 step after carrier tags, while blue reaches the red flag
about 26.3 steps later, so blue still never captures before the next meaningful
red attack in this screen. The trajectory-style gate still passes: TURTLE keeps
the highest home-half occupancy and does not collapse into BLUE_RUSH.

Next OP6 step: do not tune contact. The current bottleneck is the post-tag
window being too short in practice because red pressure resumes immediately.
Choose one controlled follow-up: either make the turtle counter window trigger
earlier on first successful carrier stop/flag denial, or add an OP6-specific
post-failure regroup before the next dual rush. Do not change both in one run.
Do not resume the full matrix until OP6 passes a held-out payoff trade-off
check.

OP6 regroup follow-up:
`artifacts/op6_failure_timeline_dev13_op6_regroup_8seed` added an OP6-only
post-failure regroup after carrier tags or broad two-attacker stops. That
implementation overcorrected: TURTLE home-half occupancy fell from about 0.50
to about 0.30 and the trajectory gate failed. Treat dev13 as invalid for payoff
interpretation.

`artifacts/op6_failure_timeline_dev13c_op6_carrier_regroup_cooldown_8seed`
narrows the trigger to red-carrier tags only and prevents renewal while the
30-step regroup/cooldown is active. This preserves the style gate:
TURTLE remains the highest home-half style. The result is still not an OP6
pass: TURTLE mean margin is -0.875, WR 1/8, red first capture about 95.7 steps,
and blue captures before red reentry in 0.0 post-tag events. Regroup is active
for TURTLE (about 66.25 steps/episode) and creates blue flag touches during
regroup, but not enough captures. This is a valid negative/weak result for the
regroup hypothesis at 30 steps.

Interpretation: the missing factor is probably not the existence of a failed
rush tempo cost. The counterattacking blue agent still does not convert the
window efficiently. Next OP6 work should inspect why the TURTLE attacker fails
to score during regroup before increasing delay or changing OP6 again.

**OP6 held-out confirmation screen (dev12 frozen):**
`artifacts/op6_failure_timeline_dev12_heldout16_seed361001`, 16 disjoint
paired seeds, OP6/map_b only. The OP6 regroup experiment was reverted; regroup
metrics are zero in this artifact. BLUE_TURTLE keeps the intended defensive
identity: the held-out trajectory gate passes and TURTLE has the highest
home-half occupancy.

Held-out margins:

```text
BLUE_TURTLE  WR=0/16   mean_margin=-1.5000
BLUE_SPLIT   WR=1/16   mean_margin=-2.0625
BLUE_ESCORT  WR=0/16   mean_margin=-2.1875
BLUE_RUSH    WR=1/16   mean_margin=-2.4375
```

TURTLE delays first red capture to about 77.1 steps, versus about 36.4 for
SPLIT, 47.4 for RUSH, and 52.2 for ESCORT. It also produces many more red
carrier tags (2.625/episode) than the exposed styles.

Paired margin advantage on the 16 matched seeds:

```text
TURTLE - RUSH    mean +0.9375, bootstrap 95% CI [ +0.3750, +1.5000 ]
TURTLE - ESCORT  mean +0.6875, bootstrap 95% CI [ +0.0625, +1.3750 ]
TURTLE - SPLIT   mean +0.5625, bootstrap 95% CI [ -0.3125, +1.3125 ]
pooled vs others mean +0.7292, bootstrap 95% CI [ +0.3333, +1.1042 ]
```

Decision: OP6 is a **provisional defensive niche**, not a fully accepted
single-cell lock. It passes the intended "TURTLE avoids the larger loss"
criterion against RUSH and ESCORT and pooled alternatives, but the individual
TURTLE-vs-SPLIT CI is not above zero at 16 held-out seeds. Do not spend more
development tuning on OP6 now. Move to OP7 calibration; revisit OP6 only if
the final pool-level cross-style statistic needs stronger separation.

**OP7 development screen (2026-07-26):**
`artifacts/op7_failure_timeline_dev1_8seed`, OP7/map_b, 8 paired development
seeds. Locked intended contract before tuning: OP7_DEEP_FORTRESS should punish
RUSH and concentrated ESCORT, while SPLIT or patient pressure should be the
best response because deep concentration leaves lanes/sustained pressure
vulnerable.

Current result does not show that contract:

```text
BLUE_TURTLE  WR=0/8  mean_margin=-0.125
BLUE_RUSH    WR=0/8  mean_margin=-0.250
BLUE_SPLIT   WR=0/8  mean_margin=-0.375
BLUE_ESCORT  WR=0/8  mean_margin=-0.750
```

Margins are too compressed and TURTLE is slightly best instead of SPLIT.
Paired TURTLE advantages are small: +0.125 vs RUSH, +0.250 vs SPLIT, +0.625
vs ESCORT. The OP7 trajectory gate also failed one style check because SPLIT
crossed midfield one step before RUSH on the probe seed, although TURTLE,
SPLIT, and ESCORT otherwise retained their expected signatures.

Decision: do not run OP7 held-out confirmation yet. OP7 needs calibration.
First inspect whether OP7 is truly a deep fortress under the audited long-name
BT path, then make one controlled OP7 decision-structure change that makes
deep concentration punish direct/concentrated attacks while exposing lanes to
SPLIT/patient pressure. Do not change blue scripts, rewards, PPO, LRO, or the
full matrix budget.

`artifacts/op7_failure_timeline_dev2_lane_commit_8seed` tested the first
OP7-only lane-commitment hysteresis change. This attempt is rejected and was
reverted. It made OP7 broadly exploitable instead of selectively vulnerable:
RUSH WR 8/8 mean margin +3.0, ESCORT WR 8/8 mean margin +2.875, SPLIT WR 8/8
mean margin +2.625, TURTLE WR 5/8 mean margin +1.125. The trajectory gate also
failed after this change: TURTLE no longer had highest home-half occupancy and
SPLIT no longer had greatest lateral separation on the probe seed.

Interpretation: freezing one defender as a flag anchor plus one lane defender
removed too much fortress coverage. The next OP7 change, if attempted, should
not disable normal fortress defense globally. It should first instrument red
lane assignment/retargeting under SPLIT, then introduce a narrower lateral
retarget delay only after a single-lane overcommit is observed.

`artifacts/op7_baseline_lane_audit_dev1b_8seed` reran unchanged OP7 with
defender-assignment/open-lane telemetry. This audit points away from "no lane
opening exists." SPLIT creates substantial open-lane opportunity:

```text
BLUE_SPLIT uncovered-lane steps:          40.375 mean
BLUE_SPLIT max consecutive uncovered:     10.5 mean
BLUE_SPLIT uncovered progress past mid:   4.11 cells mean
BLUE_SPLIT flag touch during uncovered:   6.625 mean
BLUE_SPLIT min blue0/red-flag distance:   0.0
BLUE_SPLIT min blue1/red-flag distance:   0.0
```

Red target telemetry also shows frequent same-blue targeting under SPLIT
(`both_red_target_same_blue_steps` about 62.5), so OP7 already creates the
structural opening the contract wanted. SPLIT fails despite reaching the flag
area and touching during uncovered windows. Therefore the next OP7 change
should not weaken OP7 further or add retarget delay yet. The immediate next
diagnostic should inspect why BLUE_SPLIT fails to convert existing openings:
carrier pickup/capture contact timing, whether the free attacker is tagged
after touching, whether return routing crosses into the defended lane, and
whether both split agents collapse after flag contact.

`artifacts/op7_split_touch_funnel_dev1_8seed` reran unchanged OP7 with a
BLUE_SPLIT touch-to-capture funnel and an event-level
`split_pickup_events.csv`. This clears the first conversion stage: SPLIT is not
merely grazing the flag. It averaged 8.125 distinct flag touches and 7.0
successful pickups per episode, with about 1.17 touches per pickup. The failure
is after possession: capture-given-pickup was 0.0, carrier lifetime averaged
about 6.45 steps, max return progress averaged about 4.46 cells, and carrier
loss was almost always tag (`53/56` pickup events; 3 other losses). Red
retarget latency after pickup was effectively immediate (mean about 0.10
steps). Separation also shrank after pickup (`8.72` before vs `6.03` after),
with non-carrier support/convergence present in several pickups.

Interpretation: OP7 should not be weakened. The fortress already exposes the
opposite lane and SPLIT converts touches into pickups. The bottleneck is
BLUE_SPLIT post-pickup escape/support behavior under immediate red retargeting.
The next controlled change, if made, should be BLUE_SPLIT-only: preserve lane
separation after pickup, route the carrier through the least-defended legal
return lane, and make the non-carrier distract/intercept rather than collapse
into escort. Keep OP7, rewards, PPO, LRO, and the full matrix blocked.

`artifacts/op7_split_post_pickup_route_lock_dev2_8seed` tested that
BLUE_SPLIT-only first change: at pickup, the carrier scores upper/lower return
lanes by defender clearance and path length, locks the selected route briefly,
and the non-carrier remains on the opposite lane instead of converging into an
escort. OP7, rewards, tag rules, PPO, LRO, and the other blue styles were
unchanged.

Compared with the unchanged OP7 funnel, the intended extraction metrics moved:

```text
SPLIT mean margin:              -0.375 -> +1.000
SPLIT WR:                        0/8   -> 5/8
capture given pickup:            0.000 -> 0.160
carrier lifetime after pickup:   6.45  -> 9.64 steps
max return progress:             4.46  -> 6.64 cells
carrier tag losses:              53/56 -> 45/57 events
mean separation after pickup:    6.03  -> 8.89
```

RUSH, TURTLE, and ESCORT margins stayed unchanged in this matched run, so the
payoff improvement is localized to SPLIT extraction rather than a general OP7
weakening. However, the existing single-seed trajectory gate still reports
overall FAIL because SPLIT becomes the most aggressive style by the gate's
early aggregate after pickups start. The style-specific checks that matter for
this change pass: SPLIT remains the most laterally separated style and ESCORT
remains the closest carrier-teammate style.

Decision: route-lock extraction is a promising OP7/SPLIT development result,
not a held-out acceptance. Before confirming OP7, either update the trajectory
gate to separate pre-pickup aggression from post-pickup extraction, or inspect
matched trajectory traces to ensure SPLIT still has two-lane identity before
flag pickup. Do not weaken OP7.

`diagnose_v6i26_blue_style_trajectories.py` is now phase-aware for this gate:
RUSH/TURTLE/SPLIT identity is checked before first successful pickup where
that phase is meaningful, while post-pickup checks verify that SPLIT remains
less clustered than ESCORT instead of treating extraction aggression as a style
violation. On OP7/map_b seed 360726 the updated gate passes: SPLIT has the
highest pre-pickup y-separation and simultaneous lane penetration, TURTLE has
the highest home-half occupancy, ESCORT has the smallest carrier-teammate
distance, and SPLIT remains much less clustered than ESCORT while carrying.

**OP7 held-out confirmation screen (route-lock frozen):**
`artifacts/op7_split_route_lock_heldout16_seed461001`, 16 disjoint paired
seeds, unchanged OP7/map_b. The attempted 32-seed run exceeded runtime budget
without finalizing artifacts and was stopped; this 16-seed run completed and is
the current held-out evidence.

Held-out margins:

```text
BLUE_SPLIT   WR=8/16   mean_margin=+0.9375
BLUE_RUSH    WR=0/16   mean_margin=-0.2500
BLUE_TURTLE  WR=0/16   mean_margin=-0.4375
BLUE_ESCORT  WR=0/16   mean_margin=-0.7500
```

Paired SPLIT margin advantage on the 16 matched seeds:

```text
SPLIT - RUSH    mean +1.1875, bootstrap 95% CI [ +0.5000, +1.8750 ]
SPLIT - TURTLE  mean +1.3750, bootstrap 95% CI [ +0.7500, +2.0625 ]
SPLIT - ESCORT  mean +1.6875, bootstrap 95% CI [ +1.0625, +2.3125 ]
pooled vs others mean +1.4167, bootstrap 95% CI [ +1.0417, +1.7917 ]
```

Extraction metrics also hold out: SPLIT averages 6.625 pickups/episode,
capture-given-pickup about 0.204, carrier lifetime about 11.74 steps, max
return progress about 7.37 cells, and post-pickup separation about 9.26. This
supports OP7 as a SPLIT niche under the current scripted-probe protocol.

Decision: OP7 is **ACCEPTED** as a held-out SPLIT niche for the pool matrix.
Preferred scripted response is BLUE_SPLIT; held-out evidence PASS; phase-aware
style identity PASS; runtime behavior FROZEN. Do not tune OP7 further unless
the later pool-level statistic shows insufficient distributed crossover.

Current strategic surface:

```text
OP6: provisional TURTLE niche
OP7: ACCEPTED SPLIT niche
OP8-OP12: pending
```

Next calibration target: OP8_PROTECTED_CARRIER_ESCORT.

Locked OP8 hypothesis before baseline:

```text
OP8 strength: protected, concentrated carrier push
Punishes: fragmented or isolated defense
Structural weakness: red agents cluster around the carrier, leaving the red
flag and alternate lanes exposed
Candidate best response: BLUE_RUSH
```

BLUE_RUSH is only the predeclared hypothesis, not a required answer. The
acceptance condition remains: one style is significantly better against OP8,
at least one other style remains meaningfully worse, and the result survives
held-out paired seeds. Use the same sequence: run unchanged OP8 on 8 paired
development seeds, confirm trajectory gates, diagnose the failure stage, make
at most one localized change, confirm on 16 disjoint paired seeds, then freeze
or reject.

OP8 baseline result: the predeclared BLUE_RUSH hypothesis is rejected. The
unchanged OP8 surface instead strongly favors BLUE_SPLIT, likely because the
protected escort cluster leaves alternate lanes and red-base pressure exposed.
No OP8 or blue-controller tuning was required after the baseline screen.

Development screen:
`artifacts/op8_baseline_dev1_8seed`, 8 paired seeds, unchanged OP8/map_b.

```text
BLUE_SPLIT   WR=8/8   mean_margin=+2.500
BLUE_RUSH    WR=0/8   mean_margin=+0.000
BLUE_TURTLE  WR=0/8   mean_margin=-0.125
BLUE_ESCORT  WR=0/8   mean_margin=-0.375
```

Paired development CIs all favored SPLIT: SPLIT-RUSH mean +2.5 with 95% CI
[+1.75,+3.0], SPLIT-TURTLE +2.625 CI [+2.0,+3.25], SPLIT-ESCORT +2.875 CI
[+2.5,+3.25]. Trajectory gates pass after the phase-aware diagnostic update.

Held-out confirmation:
`artifacts/op8_baseline_heldout16_seed561001`, 16 disjoint paired seeds,
unchanged OP8/map_b.

```text
BLUE_SPLIT   WR=15/16  mean_margin=+2.4375
BLUE_RUSH    WR=0/16   mean_margin=+0.0000
BLUE_TURTLE  WR=0/16   mean_margin=-0.0625
BLUE_ESCORT  WR=2/16   mean_margin=-0.1250
```

Paired SPLIT margin advantage on the 16 matched seeds:

```text
SPLIT - RUSH    mean +2.4375, bootstrap 95% CI [ +1.9375, +2.8750 ]
SPLIT - TURTLE  mean +2.5000, bootstrap 95% CI [ +2.0000, +2.9375 ]
SPLIT - ESCORT  mean +2.5625, bootstrap 95% CI [ +2.0625, +3.0000 ]
pooled vs others mean +2.5000, bootstrap 95% CI [ +2.2083, +2.7708 ]
```

Decision: OP8 is **ACCEPTED** as a held-out SPLIT niche for the pool matrix.
Preferred scripted response is BLUE_SPLIT; held-out evidence PASS; style
identity PASS; runtime behavior FROZEN. This means OP7 and OP8 currently share
the same preferred scripted response, so the pool still needs OP9-OP12 to add
different preferred styles before claiming broad distributed crossover.

**OP9 development screen (2026-07-27):** `artifacts/op9_dev1_8seed`,
OP9_SPLIT_LANE_FEINT/map_b_split_lane, 8 paired development seeds
(base-seed 501001). Worked out of documented sequence order (OP8 is still
pending) per explicit direction.

Locked intended contract before tuning, from the BT profile
(`gpu_env/_core/_bt_profiles.py`, profile 9): `enable_defender=True`,
`enable_intercept=True`, `intercept_feasibility_ratio=0.88` (defender only
commits to an intercept when it is quite confident, unlike OP7's more
reflexive coverage), `enable_counter=True` but `counter_when_trailing=True`
only (no proactive counter-press), `lane_amplitude_frac=0.55` (OP9's own
attacker swings between lanes far more than OP7/OP8's profiles -- the "feint"
in its name). Hypothesis: OP9 should punish a style that reactively chases or
tracks the feint (TURTLE's intercept logic could get drawn out of position by
a fake), while a style that does not react to the feint at all -- either
committing directly regardless of red's movement (RUSH) or pressuring both
lanes simultaneously so no single feint matters (SPLIT, possibly re-exposing
the same lane-overload mechanism that worked for OP7) -- should do better.
This is a hypothesis to test, not an assumed result: OP7's own contract
("SPLIT should win") was initially wrong-looking in its first dev screen
(TURTLE was marginally ahead) before the real mechanism was found, so the
dev-screen numbers below are what actually drives the next step, not this
paragraph.

OP9 baseline result: unlike OP7, no calibration was needed. The unchanged OP9
surface already shows a clean, decisive, mechanistically obvious split: RUSH
and TURTLE produce an exact 0-0 stalemate on every single development seed
(OP9's defender appears to fully lock down any single, undivided threat given
its high `intercept_feasibility_ratio`), ESCORT is a wash, and SPLIT converts
consistently by 2-3 points every episode. No OP9 or blue-controller tuning was
attempted or required.

Development screen: `artifacts/op9_dev1_8seed`, 8 paired seeds, unchanged
OP9/map_b_split_lane, base-seed 501001.

```text
BLUE_SPLIT   WR=8/8   mean_margin=+2.125
BLUE_ESCORT  WR=1/8   mean_margin=+0.000
BLUE_RUSH    WR=0/8   mean_margin=+0.000
BLUE_TURTLE  WR=0/8   mean_margin=+0.000
```

Raw scores confirm this is a real mechanism, not a scoring artifact: RUSH and
TURTLE are literal 0-0 on all 8 seeds; ESCORT has one win (1-0), one loss
(1-2), rest 0-0; SPLIT wins every seed by (3,1)/(3,0)/(2,1)/(2,0)-style
margins.

Held-out confirmation: `artifacts/op9_split_heldout16_seed511001`, 16
disjoint paired seeds (base-seed 511001, no overlap with the dev screen),
unchanged OP9/map_b_split_lane.

```text
BLUE_SPLIT   WR=16/16  mean_margin=+2.6250
BLUE_RUSH    WR=0/16   mean_margin=-0.0625
BLUE_TURTLE  WR=0/16   mean_margin=-0.1875
BLUE_ESCORT  WR=2/16   mean_margin=-0.0625
```

Paired SPLIT margin advantage on the 16 matched held-out seeds:

```text
SPLIT - RUSH     mean +2.6875, bootstrap 95% CI [ +2.3125, +3.0000 ]
SPLIT - TURTLE   mean +2.8125, bootstrap 95% CI [ +2.3750, +3.2500 ]
SPLIT - ESCORT   mean +2.6875, bootstrap 95% CI [ +2.1875, +3.1875 ]
pooled vs others mean +2.7292, bootstrap 95% CI [ +2.3333, +3.1042 ]
```

Decision: OP9 is **ACCEPTED** as a held-out SPLIT niche for the pool matrix.
Preferred scripted response is BLUE_SPLIT; held-out evidence PASS (stronger
than OP7/OP8: 16/16 WR vs their 8/16 and 15/16); style identity PASS (SPLIT's
existing signatures were not touched -- no controller change was made or
needed); runtime behavior FROZEN. Do not tune OP9 further.

**OP9 BLUE_PROBES_V2 reconfirm (2026-07-28):** after RUSH/ESCORT controller
freeze, re-ran untouched held-out on fresh paired seeds
`artifacts/op9_split_heldout16_blue_probes_v2_seed521001` (base-seed 521001,
disjoint from 511001; protocol tagged `BLUE_PROBES_V2`).

```text
BLUE_SPLIT   WR=16/16  mean_margin=+2.8125
BLUE_ESCORT  WR=4/16   mean_margin=+0.1250
BLUE_RUSH    WR=0/16   mean_margin= 0.0000
BLUE_TURTLE  WR=0/16   mean_margin=-0.1875
```

Paired SPLIT advantages (bootstrap 95% CI, all clear):

```text
SPLIT - ESCORT  mean +2.6875, CI [ +2.2500, +3.0625 ]
SPLIT - RUSH    mean +2.8125, CI [ +2.6250, +3.0000 ]
SPLIT - TURTLE  mean +3.0000, CI [ +2.6875, +3.3125 ]
pooled vs best-other mean +2.5625, CI [ +2.1875, +2.8750 ]
```

Verdict: **RECONFIRM_PASS** (`reconfirm_verdict.json`). OP9 remains the
canonical SPLIT niche under BLUE_PROBES_V2. Single-column `delta_pool` is
still zero by definition; this does not claim pool-level crossover.

**Cross-opponent concern, now with three data points instead of two:** OP7,
OP8, and OP9 all currently share BLUE_SPLIT as their held-out-confirmed
preferred response. Only OP6 (provisional, TURTLE, not yet fully accepted)
points a different direction so far. Three SPLIT niches out of the four
opponents examined is no longer a coincidence to shrug off -- it raises a
real possibility that SPLIT is simply the strongest all-around scripted style
against this whole BT-opponent family (dual, spread pressure beating any
single-target-tracking defender), rather than each opponent demanding a
genuinely distinct response. If OP10-OP12 also converge on SPLIT, the correct
conclusion is not "four niches" but "one dominant style, zero pool-level
crossover" -- exactly what the `no_dominating_blue_style` /
`best_response_diversity` gates are already flagging FAIL on every
individual-opponent run above (those gates need multiple red presets scored
together to be meaningful, not a single-opponent run in isolation -- but the
individual FAILs are consistent with, not contradicted by, this concern).
Do not treat OP9 as evidence of pool-level crossover by itself; it is
evidence that OP9, individually, is a real (not noise) SPLIT-favorable
opponent. The pool-level crossover question stays open until OP10-OP12 are
examined and at least one of them prefers a style other than SPLIT.

**OP10 development screen (2026-07-27):** `artifacts/op10_dev1_8seed`,
OP10_AGGRESSIVE_INTERCEPTOR/map_b_split_lane, 8 paired development seeds
(base-seed 521001).

Locked intended contract before tuning, from the BT profile (profile 10):
`enable_counter=False` (no counter-attack at all -- purely defense/intercept,
unlike OP9/OP11), `intercept_feasibility_ratio=0.70` (commits to intercepting
MORE readily than OP9's 0.88 -- less picky, not waiting for high confidence),
`lock_intercept=28` (very long commitment once locked on a target, vs OP9's
shorter, unspecified lock), `intercept_block_base=0.88` with
`intercept_block_trailing_bonus=0.36` (very strong blocking once committed),
`threat_radius=11.0` (wide detection), `lane_amplitude_frac=0.24` (narrow --
little of OP9's feinting behavior). Hypothesis: OP10's identity is "commit
early and hard to whatever looks like the threat, then stay locked a long
time." That should make it a strong stalemate/lockdown opponent against a
SINGLE clearly-identifiable threat (RUSH, ESCORT's concentrated push) --
similar to how OP9's defender stalemated RUSH/TURTLE -- but the 28-step lock
duration is a bigger liability against two genuinely independent threats than
OP9's shorter commitment was: once OP10 locks onto one SPLIT lane, the other
lane should have an unusually long uncontested window. Restating the
standing caveat: this is the hypothesis to test, not an assumed result, and
OP7/OP8/OP9 all confirming SPLIT means the base rate for "SPLIT wins again"
is now high enough that a clean SPLIT win here would be the LEAST
informative outcome for the pool-level crossover question -- watch
specifically for whether RUSH or TURTLE actually punishes OP10, since that
is what would matter most right now.

Development screen result:

```text
BLUE_SPLIT   WR=4/8   mean_margin=+0.875
BLUE_ESCORT  WR=1/8   mean_margin=+0.125
BLUE_TURTLE  WR=0/8   mean_margin=-0.375
BLUE_RUSH    WR=0/8   mean_margin=-0.875
```

Notably weaker/more contested than OP7-9: SPLIT wins only half the time here,
not near-100%, and RUSH is clearly punished hardest (-0.875, matching the
"long lock shuts down a single obvious threat" half of the hypothesis). No
OP10 or blue-controller tuning was attempted -- the spread was already wide
enough (best vs worst = 1.75) to warrant going straight to confirmation
rather than more dev iteration, matching the OP9 precedent.

Held-out confirmation: `artifacts/op10_split_heldout16_seed531001`, 16
disjoint paired seeds (base-seed 531001), unchanged OP10/map_b_split_lane.

```text
BLUE_SPLIT   WR=12/16  mean_margin=+1.6250
BLUE_ESCORT  WR=0/16   mean_margin=-0.4375
BLUE_TURTLE  WR=0/16   mean_margin=-0.6875
BLUE_RUSH    WR=0/16   mean_margin=-0.8750
```

```text
SPLIT - RUSH     mean +2.5000, bootstrap 95% CI [ +1.7500, +3.1875 ]
SPLIT - TURTLE   mean +2.3125, bootstrap 95% CI [ +1.7500, +2.8750 ]
SPLIT - ESCORT   mean +2.0625, bootstrap 95% CI [ +1.4375, +2.6875 ]
pooled vs others mean +2.2917, bootstrap 95% CI [ +1.7078, +2.8542 ]
```

Decision: OP10 is **ACCEPTED** as a held-out SPLIT niche for the pool matrix.
Held-out evidence PASS (WR rose from dev's 50% to held-out's 75%, and every
paired CI clears zero comfortably); no controller change made or needed;
runtime behavior FROZEN.

**Cross-opponent concern, escalated: four for four.** OP7, OP8, OP9, and now
OP10 all confirm BLUE_SPLIT as their held-out preferred response. Only OP6
(provisional TURTLE, never fully accepted) points anywhere else, and even
that one never passed a full 16-seed held-out screen. At four consecutive
confirmed SPLIT niches, "SPLIT is just the strongest all-around scripted
style against this opponent family" is no longer a concern to flag for
later -- it is the more likely reading of the evidence than "each opponent
creates a genuinely distinct niche." OP10's own numbers are consistent with
this: the identity that was supposed to make it a *harder* matchup for SPLIT
(a long, sticky lock) only weakened SPLIT's margin, it did not create an
opening for a different style to win instead. Before running OP11 or OP12
under the assumption that this exercise is still discovering distinct
niches, it is worth deciding explicitly whether the goal has quietly shifted
to "confirm SPLIT is pool-dominant" (a real, useful, but different finding
than the Summer plan's crossover claim) versus continuing to search OP11/12
for the first non-SPLIT confirmed niche.

**Direction (2026-07-27, locked sequence):** Stop all new latent training and
router work. Immediate job is to **build a payoff surface where four blue
strategies are actually necessary**.

**Canonical four-niche assignment (robustness OPs optional later):**

```text
OP6  → TURTLE niche
OP9  → SPLIT niche   (**RECONFIRM_PASS** under BLUE_PROBES_V2, seed 521001)
OP11 → ESCORT niche  (parallel session: make ESCORT uniquely best)
OP12 → RUSH niche    (parallel session: make RUSH uniquely best)
```

OP7 / OP8 / OP10 do **not** each need a unique blue best response; keep them as
later robustness / generalization opponents only.

**Per-niche acceptance (dev seeds → freeze → untouched held-out paired seeds):**
intended blue must (1) be uniquely best, (2) beat SPLIT and other styles by a
meaningful margin, (3) have paired CI above zero vs competitors, (4) not win
only because the red was weakened overall, (5) still lose somewhere to another
style (trade-off preserved). Target crossover sketch:

| Blue   | OP6 | OP9 | OP11 | OP12 |
|--------|----:|----:|-----:|-----:|
| RUSH   | low | low |  low | **best** |
| TURTLE | **best** | low | low | low |
| SPLIT  | low | **best** | low | low |
| ESCORT | low | low | **best** | low |

**After all four niches pass independently:** full paired-seed matrix; require
`LCB95(delta_pool) > 0` with `all_blues_protected`. Sample training by niche
(25% each of TURTLE/SPLIT/ESCORT/RUSH), not by raw opponent count.

**Before LRO:** train four independent PPO specialists (one per niche), build
their payoff matrix, require external PPO oracle > best single PPO. If learned
policies do not cross over, the issue is still env / obs / reward — do not blame
latent architecture.

**Then K=4 LRO:** birth one z per niche (freeze others); each must show
controlled movement, action distinction, target-niche improvement, **worse
elsewhere**, and positive multi-latent oracle over best fixed latent. Only then
router training, then matched non-latent PPO comparison.

**Exact sequence:**

```text
1. OP11 → ESCORT niche
2. OP12 → RUSH niche
3. OP6  → TURTLE niche (revisit after blue/controller freezes)
4. OP9  → confirm SPLIT niche
5. Full scripted matrix + LCB(delta_pool)>0
6. Independent PPO specialist matrix
7. Positive learned-policy oracle
8. K=4 LRO births with balanced niche sampling
9. Forced-z four-branch crossover
10. Router training
11. Compare against matched non-latent PPO
```

**LRO scope reminder:** LRO preserves specialists once niches exist; it does
not create niches when one style already wins everywhere. Separation of
problems: red pool creates trade-offs → PPO learns responses → LRO preserves
branches → router chooses.

**Immediate focus:** make OP11 and OP12 pass their ESCORT and RUSH held-out
niche gates. Everything downstream depends on those payoff teeth being real.
No expensive K=4 PPO/LRO birth / router campaign until steps 1–7 clear.

**OP11 development screen (2026-07-27):** `artifacts/op11_dev1_8seed`,
OP11_ADAPTIVE_EXPLOITER/map_b_split_lane, 8 paired development seeds
(base-seed 541001).

Locked intended contract before tuning, from the BT profile (profile 11):
`adaptive_enabled=True` (the only profile with this flag -- OP11 is meant to
be the hardest, most reactive opponent), `enable_2v1=True` (can commit both
red agents to double-team a single identified threat -- OP6-OP10 cannot),
`enable_counter=True` with `counter_always=True` (proactive counter-press,
not just when trailing), short locks throughout (4-8 steps, vs OP10's 28) --
reactive/flexible, not sticky. Hypothesis: the 2v1 mechanism should be able
to double-team a SINGLE concentrated threat (RUSH, ESCORT) hard, which is bad
news for those two styles specifically -- but it may also be able to
reallocate quickly enough to cover BOTH SPLIT lanes, which would finally
break SPLIT's run. If SPLIT's margin collapses here while nothing else picks
it up, that supports redesigning OP11/12 deliberately per the direction
above, rather than continuing to search for a natural non-SPLIT niche.

OP11 is now treated as DEVELOPMENT / TUNING ONLY for a missing protected
style, preferably ESCORT. If SPLIT remains best, do not accept OP11 as another
SPLIT matchup; tune OP11 to punish split play and expose an escort-compatible
weakness before held-out confirmation.

**OP12 development target (2026-07-27):** OP12_LATE_CONVERTER is the current
RUSH-candidate. Locked hypothesis before tuning: OP12's late conversion should
punish slow, passive, or over-defensive blue styles; its intended exploitable
weakness is early tempo before the late conversion loop stabilizes. Candidate
best response is BLUE_RUSH. This is DEVELOPMENT / TUNING ONLY until a frozen
OP12 variant clears held-out paired seeds with BLUE_RUSH uniquely best and
beating SPLIT by a positive paired CI.

**OP12 development screens (2026-07-27):** baseline dev1
`artifacts/op12_dev1_8seed` rejected the RUSH hypothesis: SPLIT was best
(`+1.125`, WR 7/8), TURTLE second (`+0.625`), RUSH negative (`-0.500`),
ESCORT negative (`-0.375`). Single-column `delta_pool` is zero by definition
and is not a pool result.

The first OP12-only anti-SPLIT attempt added an observable-position split
detector and post-trigger carrier-denial response. Dev4
`artifacts/op12_dev4_structural_split_detector_8seed` is rejected as a payoff
result: SPLIT became even stronger (`+1.375`, WR 8/8) and RUSH stayed negative
(`-0.500`). Telemetry showed the detector was over-broad, firing on RUSH in
7/8 episodes (`mean_trigger=14.43`, active steps `5.88`) as well as SPLIT in
8/8 episodes (`mean_trigger=13.62`, active steps `23.0`). The runner now logs
detector trigger step, active steps, max lateral separation, max teammate
distance, conversion first step, and intercept attempts from in-episode state
instead of terminal reset state.

Dev5 `artifacts/op12_dev5_tight_split_detector_8seed` fixed detector
selectivity but not payoff. The tightened detector fired for SPLIT in 8/8
episodes (`mean_trigger=20.38`, active steps `6.88`) and for RUSH/TURTLE/ESCORT
in 0/8, so the structural classifier is doing the intended job. Payoff still
failed the RUSH niche: SPLIT remained best (`+1.375`, WR 7/8), TURTLE second
(`+0.625`), RUSH negative (`-0.500`), ESCORT negative (`-0.375`).

Dev6 `artifacts/op12_dev6_split_dual_denial_8seed` strengthened only the
post-trigger OP12 response by committing both red agents to carrier denial
after split evidence. This moved SPLIT directionally down (`+1.375` -> `+1.000`,
WR `7/8` -> `6/8`) while keeping detector selectivity clean (SPLIT 8/8,
RUSH/TURTLE/ESCORT 0/8). It still failed the RUSH-niche goal: RUSH stayed
negative (`-0.500`) and SPLIT remained uniquely best. Classification:
detector correctness PASS, anti-SPLIT punishment DIRECTIONAL_BUT_INSUFFICIENT,
early RUSH vulnerability FAIL. Do not run held-out OP12 confirmation yet.

Dev8 `artifacts/op12_dev8_opening_gate_4seed` added an OP12-only opening gate
that suppresses generic retrieval/intercept/counter/defender behavior before
step 20 unless the split detector has fired. This created the missing early
tempo window but overcorrected: RUSH improved to `+0.750`, but SPLIT and ESCORT
both reached `+1.500` and TURTLE reached `+1.250`. Classification:
early vulnerability EXISTS, but it is too general and not RUSH-protected.

Current unconfirmed code candidate adds OP12-only post-pickup denial for
escort-like carrier clusters while leaving the split detector frozen. Focused
tests pass. A one-seed probe showed RUSH `+3`, SPLIT draw, TURTLE `-1`, but
ESCORT also `+3`, so the escort-cluster denial is not yet sufficient evidence
for a RUSH niche. Next OP12 step should diagnose/strengthen the post-pickup
anti-ESCORT response or narrow the opening so it rewards direct RUSH timing
without giving ESCORT the same conversion path.
`v6i9_arc_credit_running_mean_feedforward_hardpool` (aliases
`v6i9_arc_credit_feedforward`,
`plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool`),
`SUMMER-COMPATIBLE EXTENSION` (arc-credit row), parent
`v6i9_mapaware_router_feedforward_hardpool`.

**Motivation (credit audit, 2026-07-02):** the feedforward router control
routes q_phi credit through the main-loop strategy-PPO term on
`router_advantages = router_return − V_critic`. The critic overestimates
V(s_0) by ~+2.71 and, because most single-decision chunks skip the
`numel>1` advantage normalization, that constant bias survives into the
loss (chronically negative advantages, ~41% sign flips). The treatment
**replaces** that channel: `latent_strategy_ppo_coef` 0.1→0.0 (magnet
removed) plus arc credit with a detached running-mean EMA baseline that
auto-centers advantages. Resolved diff vs control = exactly 4 keys
(pinned by `tests/test_v6i9_arc_credit_feedforward.py`).

**Next actions (do NOT scale until these pass):**
1. One-update treatment smoke from the repertoire anchor with a fresh
   optimizer: `experiments/run_arc_credit_treatment_smoke.py`. Gates:
   arc-credit source active, baseline=running_mean, valid router
   decisions > 0, raw arc advantage finite, frozen actor/z hashes
   unchanged, router gradients > 0.
2. Three-update A/B mechanism test (control vs treatment, identical
   seed/budget): look for fewer all-negative decision batches, positive
   fraction moving toward balance, context-conditioned logit variation,
   argmax no longer always z1, MI(z;context) above noise.
3. Aligned episode-persistent credit audit: require Q1 negative / Q4
   positive credit and near-zero critic/global bias shift.
4. Cheap corrected ablation (learned / exact-histogram shuffled /
   uniform / fixed z2); proceed only when learned > shuffled.

### 3.0.1 v6i9 feedforward credit-patch mechanism + continuation — `EVALUATED` (context-blind collapse)

**Status:** `EVALUATED`. This is the *minimal credit patch* on the control
preset `v6i9_mapaware_router_feedforward_hardpool` (feedforward router,
35-dim team-geometry context, K=4, frozen repertoire) — **not** the
§3.0 arc-credit A/B. The patch makes the feedforward strategy-PPO update
consume `router_advantages` (erroring if absent) and applies conditional
router entropy only at decision steps (`router_ent_coef`, kept separate
from `latent_lam_h`). No context change: **credit + entropy changed,
context did not.**

**Runs (2v2, opponent pool OP8/OP9/OP10, seed 1, frozen repertoire):**

1. Wiring smoke (1 update) — all gates green (router_advantages selected,
   decision mask active, entropy subordinate, frozen repertoire hashes
   unchanged).
2. Mechanism run — 3 updates from the repertoire anchor with a fresh
   router optimizer (`final_v6i9-router-credit-mechanism-seed1_2v2.zip`).
3. Continuation — 7 more updates (base_step 1,769,472 → 2,169,472) via a
   clean resume that **preserves** the learned router + optimizer
   (`--router-reinitialize-on-load false`; behavioral-equivalence check
   PASS, argmax_diff=0; freeze intact)
   (`final_v6i9-router-credit-mechanism-cont-seed1_2v2.zip`).

**Continuation trend (updates 28→34):** `strategy_entropy` fell
monotonically `1.356 → 1.256`; **`MI_z_obs` stayed pinned at ≈0.001**
throughout; sampled occupancy concentrated `z0/z1 → z2/z3`
(`[0.24,0.21,0.28,0.28] → [0.12,0.16,0.34,0.38]`). Signature of a
**context-independent preference shift (mode-collapse toward z3)**, not
context-dependent routing.

**Cheap held-out ablation (OP8/OP10 Ã— {map_b, map_b_split_lane_v2},
10 seeds/cell = 160 eps).** Cross-episode histogram-preserving shuffle
control added (`build_cross_episode_shuffled_mapping_from_learned_traces`).

* On the mechanism (3-update) checkpoint (base_seed 12000): eval argmax
  = z3 in 188/188 opportunities, mean-max-prob 0.300, entropy 1.376.
* On the continuation (10-update) checkpoint (fresh base_seed 14000):
  eval argmax = z3 in **177/177** opportunities across all cells;
  mean-max-prob **0.30 → 0.462**; entropy `1.226`. Returns (n=40):
  `uniform −3.231 > learned/shuffled −3.399 > fixed_z2 −3.763`. Both
  shuffle controls are byte-identical to learned (constant output â‡’
  shuffles are no-ops). Promotion gates all fail
  (`learned_beats_uniform=False`, `learned_beats_shuffled=False`,
  `proceed_to_250k=False`).

**Conclusion (decisive).** With provably healthy plumbing (correct
credit source, ~85% positive advantage fraction, entropy subordinate at
grad-ratio 11–24, repertoire frozen, resume clean) the feedforward
router trained on the **geometry-only 35-dim context collapses to a
context-independent constant z3**. More updates increase *confidence*
(max-prob 0.30→0.46) but **not context sensitivity** (`MI_z_obs≈0`
throughout; eval argmax constant). The under-training hypothesis is
refuted; this is the **Fail branch = context insufficiency**. Note the
two known ablation caveats (do not over-read the trace-based trust
gates): (a) the cross-episode shuffle grouping keyed on
`(opponent, episode_seed)` yields singleton cells, and (b) a pre-existing
trace-summary join keys episode rows on `cell_seed` vs traces on
`episode_seed` so only `episode_index=0` joins — both inflate
`same_z_sequence`; the return-based verdict is unaffected.

**Open decision (fork presented, user deferred):** (i) offline best-z
predictability probe on the real 35-dim decision-time context (cheap;
separates "context lacks signal" from "router can't extract it") before
spending compute; (ii) lift the deferral and add opponent/map identity
to the context, then retrain (the Fail-branch action); (iii) stop
feedforward routing on geometry-only context. Artifacts:
`artifacts/router_credit_mechanism_ablation_crossep/`,
`artifacts/router_credit_mechanism_cont_ablation/`,
`artifacts/v6i9_router_credit_mechanism_cont_seed1.log`.

### 3.0.2 Recurrent running-mean arc-credit A/B + collapse-visibility tooling — `EVALUATED` (mechanism repaired, router collapsed)

**Status:** `EVALUATED`. Recurrent-GRU A/B from the repertoire anchor
(2v2, OP8/OP9/OP10, seed 1, frozen repertoire, 5 updates each). Control =
`v6i9_mapaware_router_sparse_hardpool` (sparse-GAE router credit);
treatment = `v6i9_arc_credit_running_mean_hardpool` (running-mean arc
credit, main-loop strategy PPO disabled). Same GRU, same initial router +
frozen-actor hashes, fresh optimizer both arms, credit channel the only
functional difference (`compare_ab_router_credit.py` launch contract:
PASS).

**Mechanism result (treatment):** running-mean baseline works — final
`raw_adv_mean ≈ −0.09`, `positive_fraction ≈ 0.57`, per-z spread ≈ 0.067,
router gradients active. The chronic critic-bias advantage offset is
removed and the credit signal is two-sided.

**Behavioral result (fresh held-out base_seed 15000, 150 eps/condition):**
`fixed_z2 −3.09 > learned −3.58 > uniform −3.78`. Learned beats uniform
and within-episode shuffle but **loses to fixed_z2** and does not beat the
cross-episode shuffle. Root cause: the router **collapsed** — argmax = z3
in the large majority of opportunities, z0/z2 never selected. Repaired
grades did not (yet) produce better *choices*.

**Tooling fixes landed with this entry:**

1. **Cross-episode shuffle regrouping (caveat (a) in §3.0.1 fixed).**
   `build_cross_episode_shuffled_mapping_from_learned_traces` now groups
   cells by `(opponent, map)` instead of `(opponent, episode_seed)`. Under
   a unique-seed-per-episode protocol the old key produced singleton cells
   → `can_reassign=False` → a structural no-op that made the
   `learned_beats_cross_episode_shuffled` gate vacuous (delta 0.0). Map is
   threaded through the opportunity trace
   (`inference_policy.set_current_map`, set per cell in
   `eval_v6i9_router_diagnostic_ablation._run_condition`); absent map falls
   back to per-opponent grouping. Note: for a *collapsed* router the
   shuffle is still legitimately a no-op (identical signatures), which is
   now an honest collapse signal rather than a grouping artifact.
2. **Decision-point selected-z occupancy telemetry.**
   `router_selected_z_occupancy_z{0..K-1}` (+ `_max`, `_unique_count`,
   `_dominant`, `_decision_count`) computed every update in
   `_latent_rollout_stats` for both credit channels, independent of the
   entropy mode (unlike `router_rollout_soft_argmax_occupancy_*`, which
   only runs on the marginal-entropy path). Surfaced in the A/B runner and
   in `compare_ab_router_credit.py` (`router_not_collapsed` signal). Makes
   router collapse visible per-update during training, not only at the eval
   shuffle gate. Pinned by
   `tests/test_router_occupancy_and_cross_episode.py`.

### 3.0.3 v6i9 arc-credit *specialize* preset (entropy-balance) — `EVALUATED` (mechanism FAIL: global z3 collapse, MI≈0)

**5-update result (2026-07-03, recurrent, seed 1, from repertoire anchor;
`artifacts/ab_router_specialize/treatment/`).** Integrity PASS (frozen-actor
hash match, router moved `q_phi_grad` 0.041→0.073, fresh optimizer, arc
credit active, old strategy-PPO channel off, resolved entropy path correct).
Specialization/coverage FAIL:

| upd | H_marg | H_cond | MI_proxy | margin | argmax_frac |
|-----|--------|--------|----------|--------|-------------|
| 1 | 1.3845 | 1.3844 | 0.0001 | 0.065 | z1=1.00 |
| 2 | 1.3725 | 1.3722 | 0.0004 | 0.018 | z2=0.68/z3=0.32 |
| 3 | 1.3661 | 1.3649 | 0.0011 | 0.248 | z3=1.00 |
| 4 | 1.3521 | 1.3494 | 0.0027 | 0.343 | z3=1.00 |
| 5 | 1.3203 | 1.3182 | 0.0022 | 0.482 | z3=1.00 |

`MI_proxy` peaked at ~0.0027 nats (~0.2% of log 4) — `H_cond ≈ H_marg`
throughout, i.e. the router is **context-independent**. Deterministic
argmax collapsed to **z3=100%** by update 3; `q_bar` drifted z3 0.25→0.40,
z0 0.23→0.15 while `H_marg` fell 1.385→1.320 (coverage eroding). The
growing top1−top2 margin is a **global** logit bias, not contextual
confidence. This is the "false diversity → global collapse" pattern:
`latent_lam_h=0.01` marginal coverage too weak to hold the distribution
while reduced `router_ent_coef` let a global z3 preference form.

**Conclusion.** With the entropy path now correctly wired (bug in §3.0.3
fixed pre-launch) the two-axis entropy balance changed *which* latent and
increased *confidence* but not *context sensitivity* — reproducing the
§3.0.1 context-insufficiency finding on the recurrent router. No entropy
knob converts a context-independent router into a context-dependent one
when MI(z;context)≈0. Behavioral gate expected to be near-vacuous
(deterministic z3 everywhere â‡’ cross-episode shuffle likely
`can_reassign=False` / `cross_episode_gate_untestable=true`). Recommended
next: the offline best-z predictability probe on the real 35-dim
decision-time context (separate "context lacks signal" from "router can't
extract it") before adding opponent/map identity to the context.

**Learned-only preflight (`base_seed=18000`, 8 eps/cell, learned router
only, `experiments/preflight_learned_trace.py` →
`artifacts/ab_router_specialize/treatment/preflight_s18000.json`).** Ran a
cheap 48-episode learned-only trace instead of the full 900-episode
behavioral exam, to decide whether the cross-episode shuffle is even
testable. Result confirms the collapse prediction:

```text
argmax_z_histogram        : {3: 230}   (100% z3, all decisions)
distinct_z_values         : [3]
non_constant_episode_count: 0          (no episode ever switches z)
cross_episode_gate_untestable = true
```

Verdict: **STOP — do not run the full behavioral exam.** The cross-episode
shuffle is an identity permutation: every episode plays z3 at every router
opportunity, so fixed_z2 / uniform / shuffled conditions cannot prove
contextual routing (nothing to shuffle). Note a tooling lesson: the
preflight's first auto-verdict was a *false* PROCEED because
`build_cross_episode_shuffled_mapping_from_learned_traces` returned
`can_reassign=True` — but that came purely from episodes having different
*lengths* (`[3,3,3]` vs `[3,3,3,3,3]`), not different z *values*. The
preflight gate was corrected to require ≥2 distinct z **values**
(`non_constant_episode_count` / `distinct_z_values`), not length-distinct
signature tuples. This is the definitive answer for the specialize arm:
the behavioral gate is untestable; the next lever must create z-value
variation (offline best-z context probe, or context enrichment), not
another entropy-knob run.

### 3.0.3b v6i9 arc-credit *specialize* preset (entropy-balance) — original PENDING_LAUNCH notes

**Preset:** `v6i9_arc_credit_specialize_hardpool` (aliases
`v6i9_arc_credit_specialize`,
`plan_faithful_latent_v6i9_arc_credit_specialize_hardpool`), parent
`v6i9_arc_credit_running_mean_hardpool` (recurrent GRU router, running-mean
arc credit, BPTT PPO disabled). `SUMMER-COMPATIBLE EXTENSION`.

**Hypothesis:** can the router become *decisive within each context*
(lower H(z|context)) while still *using all four latents across the
context distribution* (preserve marginal coverage)- Two-axis entropy
balance on top of the repaired running-mean credit channel.

**Resolved-config diff vs the running-mean parent (exactly 6 keys, pinned
by `tests/test_v6i9_arc_credit_specialize.py`):**

```text
router_ent_coef          : 0.005 -> 0.001   (weaker conditional entropy)
latent_lam_h             : 0.0   -> 0.01     (marginal coverage weight)
latent_entropy_mode      : conditional -> marginal
latent_entropy_objective : none -> maximize
h_mode                   : conditional -> marginal  (legacy alias, kept consistent)
run_tag                  : ...specialize...
```

**Bug found and fixed before any launch (2026-07-03).** The preset
originally set only the legacy `h_mode="marginal"` field. The runtime
entropy path (`rl/custom_ppo/update/entropy_objectives.py::RolloutMarginalPrep`)
and the audit banner both key off `latent_entropy_mode`, which stayed
`"conditional"`, and the arc-credit parent had zeroed
`latent_entropy_objective` to `"none"`. Net effect of the buggy config:
the rollout-level marginal-coverage loss never engaged, and `latent_lam_h`
acted as a **conditional entropy-maximization** term (pushing q_phi toward
uniform *per context*) — the exact opposite of the intended "decisive
within each context." Fix sets `latent_entropy_mode="marginal"` and
`latent_entropy_objective="maximize"` so the rollout-level
`rollout_marginal_entropy_loss` path (AGENTS.md aggregation contract)
actually runs. Verified: resolved config now yields
`marginal-path would_apply = True`. Snapshot regenerated (additive: 6 new
arc-credit/specialize entries, 0 existing presets changed).

**Launch (5-update mechanism run, recurrent, from the repertoire anchor):**

```powershell
uv run python experiments/run_ab_router_credit.py --arm treatment `
  --preset v6i9_arc_credit_specialize_hardpool `
  --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip `
  --n-updates 5 --device cuda --seed 1 `
  --out-dir artifacts/ab_router_specialize --force
```

Mechanism success (per-update telemetry): `H_marg` high (≈log 4≈1.386),
`H_cond` falling below `H_marg`, `MI_proxy = H_marg − H_cond` rising,
`q_bar` still spread over all four z, and — via the new
`router_selected_z_occupancy_z*` telemetry — the decision-point argmax
histogram becoming context-dependent (z0/z2 used, not only z1/z3).

**Behavioral gate (fresh held-out seeds — do NOT reuse 15000):**

```powershell
uv run python experiments/eval_v6i9_router_diagnostic_ablation.py `
  --checkpoint artifacts/ab_router_specialize/treatment/final_treatment.zip `
  --anchor-checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip `
  --opponents OP8 OP9 OP10 --maps map_b map_b_split_lane_v2 `
  --episodes 25 --base-seed 18000 --device cuda `
  --out-dir artifacts/ab_router_specialize/treatment/shuffle_eval_s18000
```

Promotion requires: `learned > cross_episode_shuffled`, `learned > uniform`,
`learned` closer to `fixed_z2`, and `cross_episode_gate_untestable = false`
(now emitted explicitly by the evaluator; the corrected (opponent, map)
cell grouping means a genuine reassignment/delta or an explicit
untestable flag — never a silent identity tie).

### 3.1 v5i6 — canonical marginal-entropy interpretation (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i6_paper_faithful_marginal_entropy` (apply function
`apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`
in `rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests live in
`tests/test_v5i6_paper_faithful_marginal_entropy.py`. Snapshot
regenerated. Audit banner prints `[PPO] v5i6 paper-faithful audit:`
and reports
`entropy maximization: ON (mode=marginal, aggregation=rollout, objective=maximize)`.

**Scientific delta:** v5i6 inherits v5i4 directly and replaces mean
conditional entropy `E_s[H(q_phi(z|s))]` with **rollout-level**
marginal entropy `H(E_s[q_phi(z|s)])` aggregated over **every**
resample-decision point in the rollout (not per-PPO-minibatch — see
`summer-method-spec.md` §8.1 for the Jensen rationale). The marginal
loss is computed once per PPO inner epoch by
[`rl/latent_losses.py::rollout_marginal_entropy_loss`](../rl/latent_losses.py),
applied to the first minibatch of that epoch's `latent_loss`, and uses
the same `lambda_H` schedule as v5i5 (`0.003 -> 0.001` over
`0..300_000`) so the v5i6-vs-v5i5 comparison isolates only the
entropy-reduction interpretation. No actor, critic, sampling,
router-PPO, persistence, curriculum, label, FiLM, episode-credit,
preference, distillation, or auxiliary-head channel changes.

**Resolved diffs:** v5i4 -> v5i6 is exactly
`{latent_entropy_mode, latent_lam_h_end, run_tag}`. v5i5 -> v5i6 is
exactly `{latent_entropy_mode, run_tag}`.

**Aggregation contract (post-Jensen-bias-fix):** the marginal entropy
loss must be taken over the **full** rollout resample subset (~1024
states for the standard 32-env Ã— 2048-step rollout with cadence 64).
The deprecated per-minibatch helper `strategy_marginal_entropy_loss`
(active in earlier v5i6 prototypes) systematically over-applied the
loss by Jensen `E_B[KL(q_bar_B || U)] >= KL(q_bar_rollout || U)` and
the gap was closed by the gradient softening individual `q_phi(z|s)`
toward uniform — the conditional-entropy regression v5i6 was meant
to replace. The current rollout-level path is pinned by
`tests/test_latent_losses.py::RolloutMarginalEntropyLossTests` and
`tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6RolloutMarginalEntropyContractTests`.

**Required evidence to declare v5i6 successful:**

1. High aggregate usage: `router_rollout_soft_marginal_entropy_nats /
   ln(K)` (and the sampled-z analogue `latent_marginal_entropy_nats /
   ln(K)` / `effective_num_latents`) remain high late in training.
2. Confident routing: `router_rollout_soft_conditional_entropy_nats`
   (per-state `H(q_phi(z|s))` averaged over the same rollout resample
   subset) stays meaningfully below the marginal value, producing
   positive `router_rollout_soft_mi_proxy_nats` rather than uniform
   per-state indecision. This is the "broad and state-specific"
   pattern; the failure mode "broad but indecisive" appears as both
   marginal and conditional approaching `ln(K)`.
3. Soft-argmax occupancy stays balanced:
   `router_rollout_soft_argmax_occupancy_max < ~0.50` and
   `router_rollout_soft_argmax_occupancy_ratio` close to `1`.
4. Forced-z evaluations show distinct behaviors under matched seeds.
5. `router` beats `random-matched` at eval time without a meaningful
   win-rate loss versus v5i4/v5i5.

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i6_paper_faithful \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --n-epochs 6 \
  --e3-step-telemetry \
  --checkpoint-dir checkpoints/4v4 \
  --fresh-metrics-csv \
  --periodic-checkpoint-steps 50000
```

### 3.2 v5i5 — conditional entropy-floor ablation (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i5_paper_faithful_entropy_floor` (apply function
`apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor`
in `rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests in
`tests/test_v5i5_paper_faithful_entropy_floor.py` (20 tests, all green).
Snapshot regenerated. Audit banner is family-aware
(`rl/training/banner.py`) and prints `[PPO] v5i5 paper-faithful audit:`
when the run tag matches `v5i5_paper_faithful*`. **Ready to launch.**
The *Proposed Preset Review* template should be filed retroactively in
[`summer-fidelity-rules.md`](summer-fidelity-rules.md) §"Open /
unresolved".

**Motivating observation (§2.1):** v5i4 collapsed to z=2-dominant
occupancy at 1 M steps (~64% on `z2`, ~7% on `z3`). The entropy schedule
`lam_h: 0.003 → 0.0002` over `0..300_000` reached its floor
`0.0002` ~700 k steps before the collapse stabilized. A higher floor
preserves `H(q_phi) ≥ Îµ` without changing the architecture, the
loss objective, or the resampling cadence.

**Locked single-axis design:** the only resolved field changed against
v5i4 is `latent_lam_h_end` (raised from `0.0002` to `0.001`). The
`v5i4 → v5i5` resolved-config diff is **exactly two keys**
(`latent_lam_h_end`, `run_tag`); this is enforced by
`tests.test_v5i5_paper_faithful_entropy_floor.V5i5PresetInheritanceTests.test_v5i5_minimal_diff_vs_v5i4`.
`latent_lam_h_start` stays at `0.003`, the anneal window stays
`0..300_000`, and the floor `0.001` remains *inside* the
`[0.001, 0.01]` Summer-plan range, so R20 / R21 are still satisfied.

**Classification:** `PAPER-FAITHFUL`. The change is a hyperparameter
inside the documented Summer-plan entropy range; no fidelity rule
(R1..R42 in [`summer-fidelity-rules.md`](summer-fidelity-rules.md))
flips state. Actor stays embedding-concat, router stays main-loop PG,
no forbidden channel enabled. The launch-time audit banner prints the
family-prefixed `v5i5 paper-faithful audit` block.

**New diagnostics added with v5i5 (no new gradient channel, no new
loss term):** `effective_num_latents`, `latent_marginal_entropy_nats`,
`latent_occupancy_min`, `latent_occupancy_max`,
`latent_occupancy_ratio`, `mean_strategy_duration` -- all logged to
the per-update metrics CSV.
[`rl/custom_ppo/latent_diagnostics.py`](../rl/custom_ppo/latent_diagnostics.py)
`_latent_rollout_stats` is the only function that grew. The CSV
header was extended in `rl/custom_ppo/csv_writers.py`. A schema
test (`V5i5OccupancyDiagnosticSchemaTests`) pins the new columns
plus the previously-existing per-z and per-z-per-opponent columns
so the v5i5 telemetry contract cannot regress silently.

**Required evidence to declare v5i5 successful:**

1. End-of-run `H(q_phi) / ln(K) ≥ 0.6` (vs v5i4's `0.46`).
   Equivalently: `effective_num_latents ≥ exp(0.6Â·ln 4) ≈ 2.30`.
2. Final `latent_occupancy_max â‰¤ 0.50` and
   `latent_occupancy_ratio â‰¤ ~5` (vs v5i4's ~9).
3. Headline WR is not worse than v5i4 by paired-bootstrap
   `Δ-CI > 0` at 95% (i.e. the entropy floor is not a free reward
   sacrifice).
4. `router` vs `random-matched` Δ at eval time is not worse than
   v5i4's.

**Required baselines:** v5i4 at the same seed/budget,
`no_latent_v4i3_baseline` at v5i4's exact budget and seed.

**Launch command:**

```bash
python rl/train_ppo.py \
  --config v5i5_paper_faithful_entropy_floor \
  --total-steps 1000000 \
  --seed 0
```

(All other knobs come from the preset; do not override
`--latent-strategy-ppo-coef`, `--latent-lam-p`, etc., or the v5i5
contract is broken.)

### 3.3 v5i7 — Summer-faithful entropy-floor split-lane row (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i7_summer_faithful_entropy_floor_split_lane` (apply function
`apply_plan_faithful_latent_v5i7_entropy_floor_split_lane` in
`rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests live in
`tests/test_v5i7_entropy_floor_split_lane.py`. Snapshot regenerated.
Audit banner prints `[PPO] v5i7 paper-faithful audit:` and reports
`entropy maximization: ON (mode=conditional, aggregation=per-state, objective=maximize)`.

**Scientific delta:** v5i7 inherits v5i5 directly and changes only the
environment geometry to `map_b_split_lane`. It keeps v5i5's conditional
entropy floor (`0.003 -> 0.001`), concat-only actor, main-loop
categorical PPO term on `q_phi`, persistence, sparse 64-decision
resampling, opponent pool, no forced-z curriculum, no FiLM/adapter, no
marginal entropy, no auxiliary heads, and no extra `q_phi` gradient
channel.

**Resolved diff:** v5i5 -> v5i7 is exactly `{map_layout, run_tag}`.

**Required evidence to declare v5i7 successful:** use the v5i5 occupancy
criteria plus matched-seed forced-z and router-vs-random-matched evals on
the split-lane map. Compare only against split-lane matched controls when
making causal claims about the latent method.

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i7 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4
```

### 3.4 v5i8 - Summer-faithful split-lane v2 task-pressure row (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i8_split_lane_v2_task_pressure` (apply function
`apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure` in
`rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests live in
`tests/test_v5i8_split_lane_v2_task_pressure.py`. Snapshot regenerated.
Audit banner prints `[PPO] v5i8 paper-faithful audit:` and reports
`entropy maximization: ON (mode=conditional, aggregation=per-state, objective=maximize)`.

**Scientific delta:** v5i8 inherits v5i7 directly and changes only the
environment geometry to `map_b_split_lane_v2`. It keeps v5i7's v5i5
conditional entropy floor (`0.003 -> 0.001`), concat-only actor,
main-loop categorical PPO term on `q_phi`, persistence, sparse
64-decision resampling, opponent pool, no forced-z curriculum, no
FiLM/adapter, no marginal entropy, no auxiliary heads, and no extra
`q_phi` gradient channel.

**Resolved diff:** v5i7 -> v5i8 is exactly `{map_layout, run_tag}`.

**Map-side intent:** reduce wall-bump noise and make route choices more
legible. The v2 wall is narrower/shorter than v5i7's split-lane wall,
route guidance uses larger clearance around the obstacle, and OP5/OP6/OP7
stress different lane-pressure patterns through normal scripted opponent
movement. The episode CSV adds route-context counters for attack, return,
and intercept crossings so route behavior can be grouped by `latent_z`
without assigning meanings to latent IDs.

**Required evidence to declare v5i8 successful:** use the v5i7 occupancy
criteria plus lower obstacle-collision counts, nontrivial attack/return/
intercept route distributions by `latent_z`, matched-seed forced-z evals,
and router-vs-random-matched evals on the split-lane-v2 map. Compare only
against split-lane-v2 matched controls when making causal claims about the
latent method.

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i8 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4
```

**Post-training forced-z evaluation command:**

```bash
python tools/v5i8_forced_z_eval.py \
  --checkpoint checkpoints/4v4/<v5i8_final>.zip \
  --metrics-csv checkpoints/4v4/<v5i8_metrics>.csv \
  --map-layout map_b_split_lane_v2 \
  --opponents OP5 OP6 OP7 \
  --episodes-per-mode 100 \
  --device cuda
```

This is the required evidence harness for the v5i8 latent-strategy claim.
It keeps training unsupervised and tests learned `z` behavior after the
checkpoint is frozen.

### 3.5 v5i9 - CSIA guided specialization extension (IMPLEMENTED, PENDING_EVIDENCE)

**Status:** `IMPLEMENTED, PENDING_EVIDENCE`. Preset committed as
`v5i9_csia_guided_specialization` (apply function
`apply_plan_faithful_latent_v5i9_csia_guided_specialization` in
`rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Focused tests live in
`tests/test_csia.py` and
`tests/test_v5i9_csia_guided_specialization.py`.

**Classification:** `SUMMER-COMPATIBLE EXTENSION`, not
`PAPER-FAITHFUL`. v5i9 inherits v5i8 but enables detached CSIA reward
feedback. Once gates pass, PPO trains on `reward_total + reward_csia`.

**Scientific delta:** v5i9 asks whether causal strategic-impact feedback
from frozen forced-z evaluation improves opponent-adaptive latent
specialization. It does not add labels, role targets, opponent ID inputs,
auxiliary heads, actor FiLM/adapters, forced-z curriculum, or a new
router optimizer.

**Resolved diff:** v5i8 -> v5i9 is exactly
`{csia_enabled, csia_reward_coef, run_tag}`.

**Required evidence before launch:** run the v5i8 forced-z harness and
save both:

```text
*_qualitative_rollout_by_z.csv
*_strategy_evidence.csv
```

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i9 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4 \
  --csia-payoff-csv checkpoints/4v4/qualitative/<stem>_qualitative_rollout_by_z.csv \
  --csia-strategy-evidence-csv checkpoints/4v4/qualitative/<stem>_strategy_evidence.csv \
  --fresh-metrics-csv
```

**Success criteria:** `csia_bonus_active = 1`, gates A/B/C pass, and
post-training forced-z eval shows behavioral differences plus
opponent-dependent performance or macro-behavior differences. If v5i9
only improves win rate without forced-z behavior spread, the extension
improved performance shaping but did not prove latent strategy
specialization.

### 3.6 v5i4 multi-seed (PLANNED)

**Status:** `PLANNED`. After the v5i4 single-seed eval matrix
(§2.1) and the `no_latent_v4i3_baseline` matched-budget re-launch,
add **two more v5i4 seeds** (`--seed 1`, `--seed 2`) and two more
`no_latent_v4i3_baseline` seeds to reach the §5.4 headline minimum
of three seeds per row.

### 3.7 v5i4 random-matched eval (PLANNED, eval-time only)

**Status:** `PLANNED`, no training cost. Run
`plot/eval_checkpoint.py --latent-selection router` and
`--latent-selection random-matched` against every saved v5i4
checkpoint with identical `--seed` and identical `--episodes`. The
delta is the matched-schedule routing-quality control
([`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md) §4.2).

### 3.8 v4i4post_periodic_router_distill comparison (DEFERRED)

**Status:** `DEFERRED`. Counter-factual router distillation is the
honest next step *only* if v5i4 fails its gates (§2.1 eval matrix +
§3.2 multi-seed). If v5i4 passes the §4.2 routing-quality control with
a paired-bootstrap-significant delta, v4i4post is icing and is
deprioritized.

### 3.9 v6i10 episode-router exploration preset — `EVALUATED` (smoke PASS; 5-update mechanism HARD-STOP, MI≈0)

**Status:** `EVALUATED`. Committed at `696bfb1` (code + tests; tracker doc
follow-up `d03f7e0`). The one-update runtime smoke passed **every** gate
and the five-update mechanism run then triggered two hard-stop conditions
(MI pinned at the smoke floor; per-z advantage rankings flip randomly).
Verdict: **reject for promotion; do not run the behavioral grid.** The
35-dim geometry-only context yields no extractable routing signal — the
third independent confirmation after §3.0.1 (v6i9 continuation) and
§3.0.3 (v6i9-specialize).

**Smoke (1 update, `--load-weights-only` from anchor, run_tag
`v6i10-episode-router-explore-smoke-seed1`):** all 11 gates green — one
decision/episode (486 opportunities − 454 finalized arcs = 32 open =
one per env; `arc_mean_length=138.8`); z fixed until termination
(`strategy_switch_count=0`); behavior = `0.8Â·q_phi + 0.2Â·U` with stored
old-log-prob = behavior mixture (config `router_uniform_exploration_prob=0.2`,
`router_sampling.py:730`, unit-test pinned); all four z sampled
(`unique=4`); `latent_arc_running_mean_count=454`; marginal entropy active
(`rollout_marginal_active=1.0`, `main_loop_q_phi_grad_norm=3.2e-5>0`);
router grads nonzero (`q_phi_grad_norm=0.0253`); frozen actor+z grads/deltas
exactly 0; frozen tensor hash byte-identical to anchor (`f332687…`);
checkpoint round-trips bit-exact. Deterministic argmax already 0.909-
concentrated on one z despite near-uniform `q_bar` (the v6i9 precursor).

**Five-update mechanism (`experiments/run_ab_router_credit.py --arm
treatment`, from anchor, fresh optimizer, `source_commit=d03f7e0`,
`artifacts/v6i10_episode_router_explore/treatment/`):**

```text
              u1       u2       u3       u4       u5
mi_proxy    2.19e-5  2.13e-5  2.40e-5  2.30e-5  2.31e-5   FLAT at smoke floor
top1-top2   0.0212   0.0157   0.0124   0.0107   0.0118    SHRINKING (logits flatten)
argmax z2   0.934    0.865    0.705    0.638    0.715     (z0,z3 NEVER argmax)
best per-z  z3       z1       z3       z2       z3        ranking flips randomly
frozen hash f332687 == f332687 unchanged; router moved YES; unique z=4 each update
```

`H_marginal ≈ H_conditional ≈ ln4` every update â‡’ the router emits a
near-uniform distribution for every state; the per-episode z is therefore
effectively random (near-uniform router + 20% floor), so arc credit chases
noise and per-z ordering never stabilizes. The `argmax<0.90` guard passed
but is a **false positive**: argmax softened (0.93→0.72) because logits
flattened (indecision), not because context emerged. Plumbing verified
perfect (exact frozen hash, auto-centered arc advantage, ~85–99% positive
fraction). Artifacts: `summary.json`, `run_meta.json`, `final_treatment.zip`.

**Next lever (unchanged from §3.0.1 conclusion):** context enrichment
(opponent/map identity) or the offline best-z predictability probe on the
real 35-dim context — **not** another entropy/credit/exploration knob run.

### 3.10 v6i11 contextual Q-value return router — `EVALUATED` (three pre-run bugs fixed; 15-update diagnostic = FLAT on a VALID dataset)

**Result (2026-07-03, `artifacts/v6i11_q_router_run2_seed1/summary.json`):**
`routing_verdict = FLAT`, `promotion_status = NOT_A_CANDIDATE`,
`reliably_separating_opponents = 0/3`. The dataset is **valid, not
insufficient**: replay_size 7038, no duplicates, all z + all opponents
represented, `min_cell_arcs = 527` (â‰« 20/cell bar), `return_variance = 11.08`,
`mean_arc_length ≈ 139`, `terminal_finalized_fraction = 1.0`,
`frozen_actor_ok = true`. So `FLAT` here is a genuine negative under the
tightened semantics, not a swallowed pipeline failure.

*Why FLAT (the reliability gate did its job).* Empirical row-spreads
(OP8 0.262, OP9 0.240, OP10 0.256) exceed the 0.10 magnitude threshold, but
every best-vs-second-best gap's bootstrap CI **includes zero** (OP8 gap 0.151
CI[-0.15,0.45]; OP9 0.030 CI[-0.34,0.38]; OP10 0.036 CI[-0.36,0.43]). Episode-
return std (≈ 2.6–3.9 per cell) swamps the ≈0.15–0.26 per-z mean gaps even at
~530–630 arcs/cell. Predicted-Q spread stayed tiny (0.01–0.04) — the network did
**not** invent confident spreads — and best-z agreement was 2/3 (OP8âœ“, OP9âœ—,
OP10âœ“), i.e. suggestive but unreliable.

*What FLAT does and does not mean here.* It does **not** re-open repertoire
diversity (already established by counterfactual actor-logit differences,
forced-z behavioural separation, and the +2.37 forced-z EPISODE oracle gap). The
key tension: the oracle gap is a **paired, matched-seed within-episode**
quantity, whereas this Q-router regresses an **unpaired between-episode**
expectation `E[return | episode-start context, z]`. Between-episode variance
(map, spawn, opponent stochasticity) dominates the per-z effect, so an unpaired
replay-mean target cannot resolve the latents at this SNR/budget — and/or
episode-start geometry is only weakly predictive of which z wins *this* episode.
FLAT = "the current unpaired Q-formulation failed to resolve the latents under
this dataset/horizon/context/budget," not "the latents don't differ."

*Consequence for the held-out gate.* Per the recommended sequence, the held-out
prospective evaluator (`experiments/eval_v6i11_q_router_heldout.py`, built) is
run only when the diagnostic is at least `WEAK_SEPARATION`. FLAT does **not**
meet that bar, so the held-out gate is **not** run and `map_id` instrumentation
is **not** warranted yet. A productive redesign (not yet actioned) would target
the paired signal directly — e.g. a per-context/per-episode baseline-subtracted
(advantage-style) target rather than a raw between-episode return mean.

**Pre-run status (retained):** 15-update diagnostic ran from the clean anchor
(seed 1, cuda, ~13 min/update ≈ 3.3 h wall). Update 1 validated coverage
(29–51 arcs/cell, balanced `count_by_z`, `arc_length ≈ 138`, `term_frac = 1.0`,
`records_after_update = 0`). Preset
`v6i11_q_router_hardpool` (aliases `v6i11_q_router`,
`plan_faithful_latent_v6i11_q_router_hardpool`), experiment
`experiments/run_v6i11_q_router.py`, external model `rl/router/q_value_router.py`.
Classification: **SUMMER-COMPATIBLE EXTENSION** — off-policy value regression
over online experienced returns, plus a 3-way opponent one-hot as an *input*
feature (not opponent-identity supervision). Targets are experienced returns
from sampled actions; no hindsight forced-z labels, no best-z labels, actor +
adapters frozen. **Not yet run.**

**Scientific delta:** replace BPTT PPO logit routing (which repeatedly turned
tiny logit biases into one-latent argmax collapse in §3.0.1/§3.9) with a
separate return-prediction model learning `context + selected z → expected
EPISODE return` from a replay buffer. Separates "estimate which latent has
higher value" (Q-router) from "execute the selected latent" (frozen actor).

**Three pre-run bugs found and fixed (2026-07-03):**

1. **Target-horizon mismatch.** The first draft inherited the cadence-32
   recurrent lineage (`strategy_interval=32`, `latent_resample_every_n=32`,
   `latent_arc_credit_min_len=32`), so each arc was a ~32-step MID-EPISODE
   segment and the target became "which z produced the best *local* arc
   return-" — NOT the episode-persistent forced-z EPISODE return validated by
   Probe A / the +2.37 oracle gap. **Fix:** re-parent to
   `v6i10_episode_router_explore_hardpool` (episode-persistent contract →
   `strategy_interval=0`, `latent_resample_every_n=0`, `min_len=1`), so
   arc == episode, `global_state_0` == episode-start context, `arc_return` ==
   total episode return. `arc_length` telemetry now printed per update to
   confirm arc ≈ episode length.

2. **Arc-extraction after drain.** The script read
   `rollout_strategy_arc_records` *after* `trainer.update()`, but
   `post_update.py` drains that buffer via `reset_arc_credit_rollout_state()`
   at the end of every update → the Q-router would have trained on **zero
   arcs** every step (silent no-op). **Fix:** extract arcs between
   `collect_rollout()` and `update()`.

3. **Opponent identity never captured (opponent one-hot always zero).** The
   rollout `arc_open` (router_sampling) and the episode-end `arc_finalize`
   (collector) both omitted `opponent_ids`, so `arc_open_opponent_id` stayed at
   its `-1` sentinel and every arc record carried `opponent_id = -1`. That
   zeroed the Q-router's opponent one-hot — collapsing the context back to
   geometry-only and defeating v6i11's premise — and forced every per-opponent
   cell to `count = 0 / mean = NaN` (an automatic `INSUFFICIENT_DATA`). A
   *second* half of the bug: the Q-router assumed OP8/9/10 → ids 8/9/10, but the
   canonical `_opponent_id_int_from_info` (`csv_writers._OPPONENT_TAG_TO_ID`,
   scheme OP_N → N-1) yields **7/8/9**, so even a threaded id would have been
   unmapped. **Fix:** (a) collector stamps the episode-end `arc_finalize` with
   `_opponent_id_int_from_info` per env (opponent is episode-constant, so the
   finalize-time value is exact for arc == episode); (b) `_OPPONENT_ID_TO_IDX`
   in the experiment and `_DEFAULT_OPPONENT_ID_TO_IDX` in the Q-router corrected
   to `{7:0, 8:1, 9:2}`; (c) `q_value_router` display labels now route through
   `_opponent_tag_from_id` so rows read OP8/OP9/OP10, not OP7/OP8/OP9. Verified
   live: update-1 `count_OP*_z*` all populated (29–51/cell), `mean_return_OP*_z*`
   real. Pinned by `V6i11OpponentContextWiringTests` (canonical one-hot rows,
   zero one-hot for -1/unmapped, default-map scheme).

**Hardening pass (2026-07-03, before trusting `summary.json`):**

* **Stable record IDs + rejection dedup.** `arc_finalize` now stamps each
  record with `env_index` + a monotonic `arc_uid`
  (`arc_credit.py`). The replay buffer dedups by identity
  `(rollout_index, env_index, arc_uid)` and **rejects** (does not insert)
  duplicates — content-hash dedup could collide two legitimate episodes.
  `push_many` returns `{inserted, duplicates_rejected, size_before,
  size_after}`.
* **Hard guards abort the run** (`check_arc_guards` → `ArcIntegrityError`)
  every update: `records_before_update > 0`, `inserted > 0`,
  `size_after > size_before`, and `records_after_update == 0` (proves the
  drain happened after we copied). A broken pipeline writes
  `routing_verdict = INVALID` and exits — it **never** emits `FLAT`.
* **Deep-copied extraction** (`copy_arc_record`) so the post-update reset
  cannot mutate the captured records; copy+push happen **before** `update()`.
* **Verdict is now 5-state** (`decide_verdict`): `INVALID` (zero arcs / dup
  contamination / horizon mismatch via terminal-finalized fraction / frozen
  actor drift), `INSUFFICIENT_DATA` (missing z, missing opponent, zero
  variance, or <20 arcs in the smallest cell), `FLAT`, `WEAK_SEPARATION`,
  `SEPARATING`. `FLAT` explicitly does **not** re-open repertoire diversity
  (proven by counterfactual logits, forced-z separation, +2.37 oracle gap);
  it means the Q-formulation failed to resolve the latents. Adding `INVALID`
  and `INSUFFICIENT_DATA` prevents `FLAT` from swallowing tooling failures.
* **Reliability gate:** an opponent separates only if row-spread ≥ threshold
  **and** the bootstrap CI on the best-vs-second-best mean-return gap excludes
  zero (`best_second_gap_ci`). Raw spread alone is insufficient.
* **Replay validity report** (`validity_report`): count-by-z, count-by-opponent,
  per-cell count/mean/std/sem, return variance, mean arc length,
  **terminal-finalized fraction** (episode-horizon check — should be ≈1.0),
  and the duplicate-rejection guard. Per-update coverage gate warns on z
  starvation by update ≥3.
* **`map Ã— z` coverage remains NOT_INSTRUMENTED**: the arc record carries no
  `map_id` (threading it through the shared arc lifecycle touches every
  arc-credit preset). `count_by_opponent Ã— z` is reported instead. Adding
  `map_id` is the prerequisite for a map-aware held-out grid.
* **Promotion is gated, not asserted:** a positive data verdict yields
  `promotion_status = SEPARATING_CANDIDATE` (not "wire in") and
  `heldout_gate = REQUIRED_NOT_RUN`. The decisive gate is the held-out
  prospective test (argmax-Q vs fixed-z2 / uniform / cross-episode-shuffled-Q /
  oracle; decisive = Q-router > shuffled-Q), a separate post-training step.
* Pinning tests: `tests/test_v6i11_q_router.py` (15 cases) — horizon contract,
  extraction-before-drain, zero-arc/no-insert abort, record_id dedup,
  terminal-fraction/arc-length, coverage→INSUFFICIENT_DATA, reliable
  separation→SEPARATING, noisy overlap→not SEPARATING, plus opponent-context
  wiring (canonical `{7:0,8:1,9:2}` one-hot rows, zero one-hot for -1/unmapped).

**Held-out prospective evaluator (built, not yet run):**
`experiments/eval_v6i11_q_router_heldout.py` is the decisive behavioural gate.
Matched-seed design: per `(opponent, map, seed)` held-out episode it reads the
legal t=0 context, predicts `Q(context, z)`, and runs ALL FOUR forced-z rollouts
once on fresh matched-seed envs; every condition (Q-router argmax, cross-episode
histogram-preserving shuffled-Q, uniform episode-persistent, fixed-z2, oracle)
is derived from the SAME four paired returns. Cross-episode shuffle permutes
chosen-z assignments *within* each `(opponent, map)` cell and reports
`cross_episode_gate_untestable = true` if no cell can be reassigned (all choices
identical) rather than a spurious zero-delta tie. Fresh `base_seed = 30000`,
disjoint from Probe A (42), the v6i9 diagnostic (4242), and v6i11 training
(seed 1). Decisive gate: paired `Q-router > shuffled-Q` (bootstrap CI excludes
0); then `> uniform`; then approaches/beats fixed-z2. Frozen-actor hash checked
before/after. It loads `q_router_final.pt`; run only after the diagnostic is at
least `WEAK_SEPARATION`.

**Next step:** await the running 15-update diagnostic
(`artifacts/v6i11_q_router_run2_seed1/summary.json`). If validity holds and the
verdict is at least `WEAK_SEPARATION`, run the held-out evaluator above; add
`map_id` instrumentation only before a full map-aware grid, per the recommended
sequence. Snapshot regenerated (adds the 3 v6i11 aliases only; no other preset
changed).

### 3.12 v6i13 delayed-commit opening-window advantage router — `EVALUATED` (20-update = FLAT; baseline_r2 ~0.18–0.21 held; z-residual advantage genuinely ~0 on single map)

**5-update mechanism result (2026-07-03,
`artifacts/v6i13_opening_window_advantage_router_5u_seed1/summary.json`, seed 1,
2341 arcs):** net-positive on the information axis, not yet on the routing axis.
`baseline_r2` rose and held across accumulation
(`0.127 → 0.193 → 0.193 → 0.200 → 0.213`), and `advantage_target_std` fell
(`0.932 → 0.883`), both far better than v6i12's `0.03` / `~0.99` plateau. z
coverage broad (`{577,617,589,558}`), `dup=0`, `terminal_frac=1.0`,
`min_cell_arcs=178`, commit locked at 32, frozen actor unchanged. **BUT**
verdict is still `FLAT` (0/3 reliably separating): final advantage-gap CIs
include zero (OP8 `+0.077 [-0.062,+0.221]`; OP9 `+0.010 [-0.157,+0.182]`; OP10
`+0.064 [-0.134,+0.252]`), though they narrowed sharply vs the smoke (OP8 width
0.71→0.28 as n grew 44→200). Empirical spreads (OP8 0.145, OP10 0.154) exceed
v6i12's final (~0.09).

**Interpretation:** the opening window confirms the *information* hypothesis —
the episode-start context was genuinely missing return-predictive signal
(V(context) RÂ² 0.03 → 0.21). The remaining gap is that the *z-conditional
residual advantage* is small (~0.07) relative to per-cell noise at n≈200. This
is the pre-registered "promising" branch (smoke signal survived + strengthened,
separation unresolved), so the decision is: **thread `map_id`, then run the
20-update diagnostic**; hold the GRU/history encoder. `hidden=256` /
`train_steps=100` are already the defaults, so the "strengthen V/A training"
lever is spent — the open levers are (a) more data + narrower CIs at 20 updates,
and (b) an extra context axis (`map_id`) for the advantage to separate on.

**Next step — BLOCKER on the map_id plan:** the pre-registered "thread map_id"
step is **infeasible as-is**. `rl/training/env_factory.py` builds one
`GPUFieldConfig` with a single `map_layout`, and the v6i9 split-lane preset note
(`v6_router_adapters.py`) states explicitly: "The current training system passes
a single map_layout per run; there is no built-in map-pool sampling." v6i13 runs
on `map_b_split_lane` with only a 0.5 vertical-mirror flip. A `map_id` field
would therefore be **constant** across every arc → zero information for V/A.
Options considered: (a) run 20-update WITHOUT map_id (pure more-data /
CI-narrowing test on the ~0.07 gaps); (b) add genuine map-pool sampling
(`map_pool` field + per-episode env sampling) so map actually varies, then
thread map_id — larger infra change that shifts the training distribution;
(c) use the per-episode vertical-mirror polarity as a lightweight varying axis if
exposed in `info`; (d) escalate to a compact history/temporal encoder since the
baseline signal is already strong.

**Decision taken (2026-07-03):** option (a) — launched the 20-update V6I13
diagnostic without map_id (`artifacts/v6i13_opening_window_advantage_router_20u_seed1/`,
seed 1).

**20-update diagnostic result (2026-07-04,
`artifacts/v6i13_opening_window_advantage_router_20u_seed1/summary.json`, seed 1,
9352 arcs):** `FLAT`, 0/3 reliably separating — a **decisive negative** on the
"more data will narrow CIs enough" hypothesis. Dataset fully valid (`dup=0`,
`terminal_frac=1.0`, `min_cell_arcs=741`, z balanced `{2302,2308,2335,2307}`,
frozen actor unchanged, commit locked at 32). `baseline_r2` held in the
`~0.17–0.21` band (peaked 0.213 at u5, final u20 = 0.188); `adv_std` stayed
`~0.88–0.90` (well below v6i12's ~0.99). **But advantage gaps regressed toward
zero as n grew** — the early ~0.07 spreads were noise-inflated: final gaps OP8
`+0.012 [-0.060,+0.087]`, OP9 `+0.039 [-0.042,+0.126]`, OP10 `+0.009
[-0.090,+0.107]`; empirical spreads compressed to OP8 0.066, OP9 0.124, OP10
0.067. No held-out eval (requires ≥ `WEAK_SEPARATION`).

**Interpretation — what V6I13 proved and closed:**
1. **Information hypothesis CONFIRMED:** opening-window context (`[s0,s32,delta]`)
   carries return-predictive signal V(context) can absorb (`baseline_r2 ~0.18–0.21`
   vs v6i12's 0.03). The router was missing information, not just training pressure.
2. **Z-residual routing hypothesis REFUTED on single map:** the post-commit
   z-conditional advantage is genuinely ~0 at n≈740/cell — more data did not reveal
   separation, it *removed* the illusion of it. The forced-z oracle gap (+2.37) is
   not recoverable from unpaired post-commit returns on `map_b_split_lane` alone
   via this V/A formulation.
3. **map_id as planned is still blocked:** single fixed map per run; threading
   `map_id` without map-pool sampling remains dead instrumentation.

**Next fork (pre-registered):** do NOT run held-out delayed-router eval. The
open levers are now infrastructure-level, not "more updates":
* **(b) map-pool sampling** — add `map_pool` + per-episode env sampling so map
  actually varies, then thread map_id into arc records and re-run V6I13-class
  diagnostic; or
* **(d) compact history/temporal encoder** — richer opening summary than
  `[s0,s32,delta]` if the residual advantage needs trajectory dynamics, not just
  endpoint snapshots; or
* **(c) vertical-mirror polarity** as a lightweight binary geometry axis if
  exposed in `info` (secondary — mirror is the only within-run geometry variation
  today).

### 3.13 v6i14 contract-specialist repertoire birth — `EVALUATED_FAIL`

**Scientific delta vs v6i13/v6i12:** the delayed-router diagnostics showed the
measurement pipe works but the z-conditional residual advantage is effectively
flat at large n. v6i14 stops treating the router as the next bottleneck and
tests whether explicit temporary z contracts can birth real reusable
specialists before routing resumes.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). This is not a
paper-faithful row and not a Summer-compatible extension. It deliberately adds
handcrafted z-role reward terms during repertoire training.

**Parent:** `v6i9_mapaware_repertoire_hardpool`. Router is off, z assignment is
balanced by episode, and the v6i9 repertoire trainable scope remains active
(shared actor trunk frozen; z-specific modules trainable).

**Resolved-config diff vs parent:** exactly `{experiment_id,
latent_contract_specialist_coef, latent_contract_specialist_enabled, run_tag}`.
The runtime adds default-off `latent_contract_specialist_enabled`,
`latent_contract_specialist_coef`, and `latent_contract_specialist_clip` to
`PPOConfig`; only v6i14 enables the contract bonus.

**Contract map:** `z0` opening pressure, `z1` home defense / recovery, `z2`
friendly-carrier support, `z3` carrier conversion. The reward is computed from
existing normalized global-state features and stored as
`reward_contract_specialist`.

**Gate before router work resumes:** forced-z behavior fingerprints must exist.
If z0/z1/z2/z3 do not separate on their contract metrics, router training is
not meaningful. If fingerprints exist but forced-z returns do not differ by
opponent/context, the specialists are different but not useful. Only if both
pass should a selector be trained.

**1-update smoke (2026-07-04,
`artifacts/v6i14_contract_specialists_smoke_metric/`):** PASSED. Warm-started
from `final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip` with
`--additional-steps 1024`, `n_envs=4`, `n_steps=256`, `n_epochs=1`. Run config
resolved `latent_assignment_mode=balanced_episode`,
`train_router_when_forced=False`, `v6i9_training_stage=repertoire`,
`latent_contract_specialist_enabled=True`, and `latent_contract_specialist_coef=0.25`.
The metrics CSV exposes `reward_contract_specialist_mean=0.05616` on the
smoke update, proving the contract bonus flows through the real rollout/update
path. `shared_actor_max_abs_delta=0.0`; z-specific update telemetry is present
for the contract-repertoire path.

**20-update diagnostic (2026-07-04,
`artifacts/v6i14_contract_specialists_20upd_diag/`):** MECHANISM PASSED,
SPECIALIST GATE FAILED. The run completed updates 17-36 from the v6i9 hardpool
checkpoint with router off, `balanced_episode` z assignment, contract reward
enabled, and shared actor frozen (`shared_actor_max_abs_delta=0.0`). Contract
reward stayed active (`reward_contract_specialist_mean=0.04987`, min `0.03503`,
max `0.06106`), and episode z coverage was balanced (z0=36, z1=34, z2=35,
z3=36). The forced-z fingerprint sniff was partial (44/48 episodes) and showed
weak separation only: mean pair distance `0.0453`, max `0.0717`, pairs above
threshold `0`. Do not train the router from this checkpoint.

**50-update continuation (2026-07-04,
`artifacts/v6i14_contract_specialists_50upd_cont/`):** COMPLETED, SPECIALIST
GATE STILL FAILED. The run continued from
`artifacts/v6i14_contract_specialists_20upd_diag/final_v6i14_contract_specialists_20upd_diag_2v2.zip`
for 50 update rows (updates 37-86, timesteps 1,070,080 -> 1,120,256). Router
remained off, z coverage stayed balanced (z0=92, z1=91, z2=92, z3=93), contract
reward remained active (`reward_contract_specialist_mean=0.05162`, min
`0.03822`, max `0.06473`), and shared actor drift stayed zero. The complete
small forced-z behavior grid
(`artifacts/v6i14_contract_specialists_50upd_cont/forced_z_fingerprint_eps2_complete/`,
48/48 episodes) did not improve separation: mean pair distance `0.0431`, max
`0.0623`, min `0.0238`, pairs above threshold `0`, all z represented. The next
move is not router training; strengthen the contract-specialist reward/loss
before any selector work resumes.

**Verdict (2026-07-04):** `EVALUATED_FAIL`. Contract-specialist diagnostic
wiring passed: contract reward live, shared actor frozen, balanced z assignment
active, training stable. Specialist-birth gates failed: forced-z behavior pair
distances remained far below threshold and decreased after continuation
(20upd mean `0.0453` → 50upd mean `0.0431`); Stage-C complementarity failed
with saturated forced-z win rates and no best-z variation (`best_z=0` in every
cell, all forced-z win rates `1.0`). Tiny oracle gap is not promotion-worthy
due to saturated returns and near-identical behavior fingerprints. Router
promotion blocked. Next step: V6I15 contract-pressure / capacity /
harder-surface ablation.

### 3.14 v6i15 contract-pressure sweep -- `EVALUATED_FAIL` (phase-1: 5-update coefficient arms)

**Scientific delta vs v6i14:** v6i14 proved the contract path is wired but
that the mild coefficient did not birth behaviorally separated specialists.
v6i15 keeps the same scaffold and asks whether the current z-specific actor
pathway responds when the contract reward is made loud.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). This is still
handcrafted z-role reward shaping, so it is not paper-faithful and not a
Summer-compatible extension.

**Parent:** `v6i14_contract_specialists`. Router remains off,
`balanced_episode` z assignment remains active, and the v6i9 repertoire-stage
trainable scope remains active (shared actor trunk frozen; z-specific modules
trainable).

**Resolved-config diff vs v6i14:** exactly `{experiment_id,
latent_contract_specialist_coef, run_tag}`. The 3x, 6x, and 10x arms set
`latent_contract_specialist_coef` to `0.75`, `1.50`, and `2.50` respectively.
`v6i15` and `v6i15_contract_pressure` resolve to the 3x arm. The 1x baseline
is v6i14 (`coef=0.25`).

**Phase-1 protocol (completed 2026-07-04):** 5-update coefficient arms from
the same v6i9 anchor
(`checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip`,
`--load-weights-only`, `--additional-steps 5120`), each followed by a complete
48-episode forced-z fingerprint grid (`episodes=2` per cell, OP8/OP9/OP10 Ã—
`map_b` / `map_b_split_lane_v2`).

| Arm | `coef` | `reward_contract_specialist_mean` (final update) | forced-z `behavior_pair_distance_mean` | pairs above threshold |
|-----|--------|--------------------------------------------------|----------------------------------------|----------------------|
| v6i14 (1Ã— baseline) | 0.25 | ~0.05 | 0.0431 (50upd cont) | 0 |
| 3Ã— (`artifacts/v6i15_contract_pressure_3x_5u_seed1/`) | 0.75 | 0.149 | 0.0436 | 0 |
| 6Ã— (`artifacts/v6i15_contract_pressure_6x_5u_seed1/`) | 1.50 | 0.298 | 0.0409 | 0 |
| 10Ã— (`artifacts/v6i15_contract_pressure_10x_5u_seed1/`) | 2.50 | 0.497 | 0.0409 | 0 |

**Mechanism checks passed on all arms:** contract reward scales with coefficient
(~linear vs v6i14), shared actor frozen (`shared_actor_max_abs_delta=0.0`),
balanced z assignment, training stable, win rate saturated (~100%).

**Specialist-birth gate failed on all arms:** forced-z behavior pair distances
stayed in the v6i14 band (~0.04) with zero pairs above threshold; 6Ã— and 10Ã—
eval fingerprints were identical (`mean=0.0409`). Stage-C still shows
`best_z=0` in every cell with all forced-z win rates `1.0`. Contract reward
rose but behavior did not separate — the model is collecting contract crumbs
without changing forced-z behavior.

**Verdict:** coefficient pressure alone does not birth specialists. Do **not**
continue any arm to 20 updates. Do **not** resume router training. Next fork:
Arm C (z-specific capacity / adapter design) and/or Arm B (harder eval surface
with non-saturating margin metrics).

**Promotion gate:** router training remains blocked. If a future capacity arm
still fails at 10Ã— pressure, treat the z pathway as underpowered or the
contract features as misaligned — not a routing problem.

### 3.15 v6i16 capacity + sharp-contract ablation -- `EVALUATED_FAIL`

**Scientific delta vs v6i15:** v6i15 showed that louder contract reward moves
behavior distance somewhat but quickly hits a ceiling. v6i16 tests the next
diagnosis: the current contracts may be satisfiable by one generic policy, and
the z-specific actor pathway may not have enough leverage.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). This is
handcrafted z-role reward shaping plus actor z-pathway capacity tuning. It is
not paper-faithful and not a Summer-compatible extension.

**Parent:** `v6i15_contract_pressure_3x`. All arms keep 3x contract pressure
(`latent_contract_specialist_coef = 0.75`), router off, `balanced_episode` z
assignment, `v6i9_training_stage = "repertoire"`, OP8/OP9/OP10 hard pool, and
the frozen shared actor trunk.

**Arm matrix:**

| Arm | Preset | Delta vs v6i15 3x |
|-----|--------|-------------------|
| A | `v6i16_sharp_contracts` | `latent_contract_specialist_variant = "sharp"` |
| B | `v6i16_capacity` | `latent_z_gate_init = 0.08`, `latent_actor_z_adapter_enabled = True`, `latent_actor_z_adapter_scale = 0.10`, `latent_actor_z_adapter_init_std = 0.05` |
| C | `v6i16_capacity_sharp_contracts` (`v6i16`) | Arm A + Arm B |

**Sharp contract map:** `z0` pressure / interception / enemy-carrier
disruption; `z1` escort / carrier support / conversion support; `z2`
home-flag defense / returns / denial; `z3` spacing / lane control / split
pressure.

**Phase-1 protocol (completed 2026-07-04):** 5 updates per arm from the same
v6i9 anchor checkpoint, followed by the same 48-episode forced-z fingerprint
grid (`episodes=2` per cell, OP8/OP9/OP10 x `map_b` /
`map_b_split_lane_v2`). Artifacts:
`artifacts/v6i16_sharp_contracts_5u_seed1/`,
`artifacts/v6i16_capacity_5u_seed1/`, and
`artifacts/v6i16_capacity_sharp_contracts_5u_seed1/`.

| Arm | forced-z `behavior_pair_distance_mean` | max pair distance | pairs above threshold | `unique_best_z_count` | Stage-C |
|-----|----------------------------------------|-------------------|-----------------------|-----------------------|---------|
| A sharp contracts | 0.0436 | 0.0617 | 0 | 1 | FAIL |
| B capacity | 0.0450 | 0.0608 | 0 | 1 | FAIL |
| C capacity + sharp contracts (`v6i16`) | 0.0450 | 0.0608 | 0 | 1 | FAIL |

**Mechanism checks passed:** all arms trained from the intended anchor with
router off, balanced episode z assignment, contract reward active, and
forced-z eval artifacts complete. Stage-C win rate stayed saturated
(`100%` for every forced-z cell), so binary win rate remains unusable.

**Specialist-birth gate failed:** sharper contracts, larger z pathway
capacity, and the combined arm all remained in the same ~0.04-0.045 behavior
distance band with zero pairs above threshold. The OP/map best-z surface stayed
constant (`best_z=0` in every cell), and Stage-C failed on every arm despite a
matched-seed oracle gap around `+0.78` to `+0.80`. The script's
`ORACLE_GAP_PLUS_CONTEXT` ladder line is not accepted as promotion evidence
here because the stricter v6i16 gates require actual behavior fingerprints and
context-varying best-z cells.

**Verdict:** capacity + sharp contracts did not produce an interaction effect.
Do **not** train a router from v6i16. The next fork should change the training
surface, not keep stacking scalar knobs: harder or asymmetric opponents,
non-saturating score-margin/tempo objectives, map-pool variation, or contexts
where defense, escort, interception, and pressure genuinely trade off.

### 3.16 v6i17 surface-pressure diagnostic -- `EVALUATED_FAIL`

**Scientific delta vs v6i16:** v6i16 ruled out louder contracts, sharper
contracts, larger z-pathway capacity, and the combined capacity + sharp
contract arm on the current saturated OP8/OP9/OP10 surface. v6i17 tests the
next hypothesis: the arena is too easy or too symmetric, so the same
generalist behavior wins without role tradeoffs.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). This inherits
handcrafted z-role contract rewards and v6i16 z-pathway capacity changes, then
changes the opponent surface. It is not paper-faithful and not a
Summer-compatible extension.

**Parent:** `v6i16_capacity_sharp_contracts`. Router remains off,
`balanced_episode` z assignment remains active, `latent_contract_specialist`
stays enabled at 3x with `variant="sharp"`, z-specific pathways remain
trainable, and the shared actor trunk remains frozen through
`v6i9_training_stage = "repertoire"`.

**Resolved-config diff vs v6i16 combined:** exactly `{experiment_id,
opponent_pool, run_tag}`.

| Field | v6i16 combined | v6i17 |
|-------|----------------|-------|
| `experiment_id` | `v6i16` | `v6i17` |
| `opponent_pool` | `("OP8", "OP9", "OP10")` | `("OP8", "OP9", "OP10", "OP11", "OP12")` |
| `run_tag` | `v6i16_capacity_sharp_contracts_3x_OP8_OP9_OP10` | `v6i17_surface_pressure_diagnostic_OP8_OP9_OP10_OP11_OP12` |

**Launch caveat (2026-07-04):** the first two attempted runs under
`artifacts/v6i17_surface_pressure_5u_seed1/` and
`artifacts/v6i17_surface_pressure_5u_seed1_op8_op12/` are invalid as v6i17
surface evidence. Runtime validation silently filtered the OP11/OP12 preset
surface back to OP8/OP9/OP10. Fixed by extending the training opponent
allowlist to preserve OP11 and OP12, pinned by
`tests/test_v6i17_surface_pressure_diagnostic.py`.

**Corrected 5-update diagnostic (2026-07-04,
`artifacts/v6i17_surface_pressure_5u_seed1_op8_op12_validated/`):** COMPLETED.
Warm-started from
`checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip`
with `--load-weights-only`, `--additional-steps 5120`, `n_envs=4`,
`n_steps=256`, `n_epochs=1`, CUDA. The launch banner and audit confirm
OP8/OP9/OP10/OP11/OP12 as the active training pool. Mechanism checks passed:
contract reward live (`reward_contract_specialist_mean=0.1688` final update),
shared actor frozen (`shared_actor_max_abs_delta=0.0`), router gradients zero
as intended, balanced episode z assignment active, and OP11/OP12 appeared in
training telemetry.

**Forced-z fingerprint grid (2026-07-04,
`artifacts/v6i17_surface_pressure_5u_seed1_op8_op12_validated/forced_z_fingerprint_eps2_op8_op12/`):**
COMPLETE, SPECIALIST GATE FAILED. Grid: OP8/OP9/OP10/OP11/OP12 x
`map_b`/`map_b_split_lane_v2` x z0..z3 x 2 episodes = 80 episodes. All
forced-z cells still won (`WR=100%`). Stage-C failed:
`oracle_wr=100%`, `best_fixed_wr=100%`, `best_z=0` in every OP/map cell,
`unique_best_z_count=1`. Behavior fingerprints did not improve versus v6i16:
mean pair distance `0.0392`, max pair distance `0.0533`, pairs above threshold
`0`, all z represented. Matched-seed oracle gap increased to `+1.3625`, but
that is not promotion evidence without behavior fingerprints or context-varying
best-z cells.

**Verdict:** harder/asymmetric OP11/OP12 surface did not birth specialists
under the current contract/capacity scaffold. Do **not** train a router from
v6i17. The next surface fork needs stronger consequence changes than simply
adding OP11/OP12 to this map surface: non-saturating margin/tempo objectives,
handicap/asymmetry, shorter-horizon pressure, or map-pool/layout variation
where different roles cannot all win by the same generalist behavior.

### 3.17 v6i18 margin/tempo surface diagnostic -- `EVALUATED_FAIL` (5-update live + forced-z, 2026-07-04)

**Scientific delta vs v6i17:** v6i17 showed that harder OP11/OP12 opponents
alone do not break win-rate saturation or force role tradeoffs. v6i18 keeps the
specialist-birth machinery fixed and changes only the consequence surface: the
arena now grades score margin, capture tempo, near-cap conversion, enemy flag
touches, and enemy carrier progress instead of relying on binary win/loss.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). This inherits
handcrafted z-role contract rewards and v6i16 z-pathway capacity changes, then
adds noncanonical margin/tempo reward pressure. It is not paper-faithful and
not a Summer-compatible extension.

**Parent:** `v6i17_surface_pressure_diagnostic`. Router remains off,
`balanced_episode` z assignment remains active, OP8/OP9/OP10/OP11/OP12 remain
active, `latent_contract_specialist` stays enabled at 3x with
`variant="sharp"`, z-specific pathways remain trainable, and the shared actor
trunk remains frozen through `v6i9_training_stage = "repertoire"`.

**Resolved-config diff vs v6i17:** exactly `{env_stalemate_max_steps,
env_surface_blue_capture_tempo_bonus, env_surface_blue_near_cap_bonus,
env_surface_red_carrier_progress_penalty, env_surface_red_flag_touch_penalty,
env_surface_score_margin_coef, experiment_id, max_decision_steps, run_tag}`.

| Field | v6i17 | v6i18 |
|-------|-------|-------|
| `experiment_id` | `v6i17` | `v6i18` |
| `max_decision_steps` | `320` | `240` |
| `env_stalemate_max_steps` | `120` | `80` |
| `env_surface_score_margin_coef` | `0.0` | `0.15` |
| `env_surface_blue_capture_tempo_bonus` | `0.0` | `0.25` |
| `env_surface_red_flag_touch_penalty` | `0.0` | `0.20` |
| `env_surface_red_carrier_progress_penalty` | `0.0` | `0.025` |
| `env_surface_blue_near_cap_bonus` | `0.0` | `0.015` |
| `run_tag` | `v6i17_surface_pressure_diagnostic_OP8_OP9_OP10_OP11_OP12` | `v6i18_margin_tempo_surface_OP8_OP9_OP10_OP11_OP12` |

**Launch command (completed):**

```powershell
uv run python rl/train_ppo.py --preset v6i18 --load checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --load-weights-only --additional-steps 5120 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i18_margin_tempo_surface_5u_seed1 --checkpoint-dir artifacts\v6i18_margin_tempo_surface_5u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
```

**Checkpoint:** `artifacts/v6i18_margin_tempo_surface_5u_seed1/final_v6i18_margin_tempo_surface_5u_seed1_2v2.zip`

**Training mechanism gates (final update 21):** PASS — `reward_contract_specialist_mean=0.157`, `shared_actor_max_abs_delta=0.0`, `rollout_win_margin_mean=2.38`, `strategy_wr_spread=0.5` (first non-zero z WR spread in this fork chain), OP8–OP12 present in telemetry, router/q_phi effectively off, balanced z assignment active. Surface coefs resolved in run config (`env_surface_*` nonzero; shorter `max_decision_steps=240`, `env_stalemate_max_steps=80`). Surface components are not logged as separate CSV columns; they flow through the GPU env reward path.

**Forced-z fingerprint eval (completed):**

```powershell
uv run python experiments/run_forced_z_eval.py --checkpoint artifacts\v6i18_margin_tempo_surface_5u_seed1\final_v6i18_margin_tempo_surface_5u_seed1_2v2.zip --out-dir artifacts\v6i18_margin_tempo_surface_5u_seed1\forced_z_fingerprint_eps2 --opponents OP8 OP9 OP10 OP11 OP12 --episodes 2 --oracle-metric win_margin --device cuda --progress-every 8
```

**Eval caveat:** canonical forced-z protocol uses `max_decision_steps=400` and does not replay v6i18 surface reward coefs; margin/tempo gates are therefore measured on behavior telemetry and episode outcomes under the standard eval env, not the training surface.

| Gate | Target | v6i18 result | Pass- |
|------|--------|--------------|-------|
| `behavior_pair_distance_mean` | `>0.06` | **0.0391** | FAIL |
| pairs above threshold | ≥1–2 | **0** | FAIL |
| `unique_best_z_count` | `>1` | **1** (`z0` every cell) | FAIL |
| forced-z WR | informative only | **100%** all 80 eps | saturated |
| score margin by z | differs | z0=2.40, z1=2.30, z2=2.65, z3=2.30 (spread 0.35) | weak |
| time-to-first-score by z | differs | z0=35.4, z1=46.4, z2=47.7, z3=39.5 (spread 12.3 steps) | weak |
| intercept/escort by z | role ownership | intercept 0.088–0.216, escort 0.260–0.317 | weak |

Stage-C gates: oracle WR advantage 0%, best-z varies = FAIL. Global best fixed-z by margin is z2 (2.65) but per-cell oracle picks z0 everywhere under `win_margin` metric.

**Verdict:** `EVALUATED_FAIL` on promotion gates. Margin/tempo surface changed training telemetry (`strategy_wr_spread`, rollout win margin) but did **not** produce forced-z specialist separation above the v6i14/v6i15 ~0.04 ceiling. The answer to â€œdo margin/tempo consequences create z-specialist separation where harder opponents alone did not-â€ is **no** at 5 updates.

**Router training:** remains **blocked** (ignore ladder verdict `ORACLE_GAP_PLUS_CONTEXT` from eval script; user gate is margin/tempo/role separation).

**Recommended next fork (per user decision tree):** explicit arena handicaps/asymmetry or real `map_pool` layout variation — **not** another contract or surface coefficient sweep.

**Artifacts:**
- Training: `artifacts/v6i18_margin_tempo_surface_5u_seed1/`
- Forced-z: `artifacts/v6i18_margin_tempo_surface_5u_seed1/forced_z_fingerprint_eps2/`

### 3.18 v6i19 map-pool surface diagnostic -- `EVALUATED_FAIL` (2026-07-04)

**Scientific delta vs v6i18:** v6i18 failed on forced-z fingerprints under a fixed
layout even with margin/tempo surface pressure. v6i19 keeps the v6i18 scaffold
fixed and adds only per-episode `map_pool` sampling so opponent x map context can
create layout-driven role tradeoffs.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). Router remains
off; contracts, z capacity, shared-actor freeze, and surface coefs unchanged.

**Resolved-config diff vs v6i18:** exactly `{experiment_id, map_pool, run_tag}`.

**Training (valid complete):** relaunched 5-update run from v6i9 anchor.
Checkpoint `artifacts/v6i19_map_pool_surface_5u_seed1/final_v6i19_map_pool_surface_5u_seed1_2v2.zip`.
Both layouts appeared in episode telemetry (`map_b_split_lane` 19 eps,
`map_b_split_lane_v2` 13 eps). Map-pool plumbing gate passed.

**Forced-z eval (authoritative, surface-matched):**
`artifacts/v6i19_map_pool_surface_5u_seed1/forced_z_fingerprint_eps2/`
(`--inherit-training-config`, `max_decision_steps=240`, OP8–OP12 Ã—
`map_b` + `map_b_split_lane_v2`, 2 eps/cell).

| Gate | Result |
|------|--------|
| `unique_best_z_count` | **1** (z0 in all 10 opponentÃ—map cells) |
| `behavior_pair_distance_mean` | **0.0413** (≈ V6I18 0.0412) |
| `pairs_above_threshold` | **0** |
| WR | **100%** saturated (all cells) |
| Global best-fixed z | z2 (margin 2.65); per-cell oracle still z0 |
| Stage C gate 2 (best-z varies) | **FAIL** |

**Verdict:** map-pool layout variation did **not** break the clone wall. Same
failure shape as V6I14–V6I18. Ladder `ORACLE_GAP_PLUS_CONTEXT` is **not** a
router unblock signal.

**Next fork (if continuing specialist-birth line):** explicit arena
asymmetry/handicap — not more map/reward/surface polishing. Router training
remains blocked.

### 3.19 v6i20 asymmetry-handicap surface diagnostic -- `EVALUATED_FAIL` (2026-07-04)

**Scientific delta vs v6i19:** v6i19 proved that map-pool infrastructure and
surface-matched eval work, but layout variation did not break the clone wall.
v6i20 keeps the v6i19 scaffold fixed and strengthens only asymmetric
consequence pressure: red flag touches and red carrier progress are more
expensive, while blue fast-capture and near-cap conversion pressure are
stronger.

**Fidelity classification:** `DIAGNOSTIC` (non-Summer scaffold). This inherits
handcrafted z-role contract rewards, v6i16 z-pathway capacity changes,
v6i18 margin/tempo surface rewards, and v6i19 map-pool sampling. It is not
paper-faithful and not a Summer-compatible extension.

**Parent:** `v6i19_map_pool_surface_diagnostic`. Router remains off,
`balanced_episode` z assignment remains active, OP8/OP9/OP10/OP11/OP12 and the
two-layout `map_pool` remain active, `latent_contract_specialist` stays enabled
at 3x with `variant="sharp"`, z-specific pathways remain trainable, and the
shared actor trunk remains frozen through `v6i9_training_stage = "repertoire"`.

**Resolved-config diff vs v6i19:** exactly
`{env_surface_blue_capture_tempo_bonus, env_surface_blue_near_cap_bonus,
env_surface_red_carrier_progress_penalty, env_surface_red_flag_touch_penalty,
experiment_id, run_tag}`.

| Field | v6i19 | v6i20 |
|-------|-------|-------|
| `experiment_id` | `v6i19` | `v6i20` |
| `env_surface_blue_capture_tempo_bonus` | `0.25` | `0.45` |
| `env_surface_red_flag_touch_penalty` | `0.20` | `0.50` |
| `env_surface_red_carrier_progress_penalty` | `0.025` | `0.075` |
| `env_surface_blue_near_cap_bonus` | `0.015` | `0.035` |
| `run_tag` | `v6i19_map_pool_surface_diagnostic_OP8_OP9_OP10_OP11_OP12` | `v6i20_asymmetry_handicap_surface_OP8_OP9_OP10_OP11_OP12` |

**Training (valid complete):** 5-update run from v6i9 anchor succeeded.
Checkpoint `artifacts/v6i20_asymmetry_handicap_surface_5u_seed1/final_v6i20_asymmetry_handicap_surface_5u_seed1_2v2.zip`.
Mechanism gates passed: stronger surface coefs resolved, map pool active,
`shared_actor_max_abs_delta=0.0`, router gradients zero, contract reward live.
`strategy_wr_spread=0.5` appeared once on update 20 — **not** promotion
evidence (5 updates, noisy training-time metric).

**Forced-z eval (authoritative, completed):**
`artifacts/v6i20_asymmetry_handicap_surface_5u_seed1/forced_z_fingerprint_eps2/`
(`--inherit-training-config`, `max_decision_steps=240`, stronger v6i20 surface
coefs inherited).

Grid: OP8..OP12 x `map_b` / `map_b_split_lane_v2` x z0..z3 x 2 episodes =
80 episodes.

| Gate | Target | v6i20 result | Pass- |
|------|--------|--------------|-------|
| `unique_best_z_count` | `>1` | **1** (`z0` every opponent x map cell) | FAIL |
| `behavior_pair_distance_mean` | `>0.06` | **0.0413** | FAIL |
| `behavior_pair_distance_max` | above prior ceiling | **0.0570** | FAIL |
| pairs above threshold | >=1 | **0** | FAIL |
| forced-z WR | informative only | **100%** all 80 eps | saturated |
| Stage-C | best-z varies and oracle beats fixed WR | **FAIL** | FAIL |

Tradeoff table by z:

| z | WR | margin | time-to-first-score | intercept-near-carrier | escort | defense pressure |
|---|----|--------|---------------------|------------------------|--------|------------------|
| z0 | 1.000 | 2.350 | 35.4 | 0.171 | 0.314 | 0.736 |
| z1 | 1.000 | 2.150 | 46.4 | 0.207 | 0.265 | 0.733 |
| z2 | 1.000 | 2.650 | 41.3 | 0.090 | 0.264 | 0.720 |
| z3 | 1.000 | 2.350 | 39.6 | 0.186 | 0.304 | 0.751 |

Stage-C details: oracle WR `100%`, best-fixed WR `100%`, WR advantage `0%`,
oracle margin `2.85`, best-fixed margin `2.35`, best fixed z by Stage-C is z0,
global best fixed z by margin summary is z2. The ladder verdict
`ORACLE_GAP_PLUS_CONTEXT` is not accepted as a router-unblock signal because
the strict behavior and context-variation gates failed.

**Verdict:** explicit asymmetric consequence pressure did not break the clone
wall. V6I20 has the same failure shape as V6I19: the arena/eval plumbing works,
but the forced-z repertoire still lacks behaviorally distinct, context-varying
specialists. Router training remains blocked.

**Next fork:** do not run more coefficient polish on this scaffold. If
continuing the specialist-birth line, move to a stronger intervention:
explicit environment handicap mechanics, limited shared-layer unfreeze under
asymmetry, or separate specialist pretraining / role-conditioned scenario
curricula.

Failure band to beat: V6I18/V6I19 (`distance ≈ 0.04`, `unique_best_z=1`, z0
everywhere, WR saturated).

### 3.20 v6i21 adaptive OP8-OP12 hardpool calibration -- `IMPLEMENTED` (2026-07-04)

**Scientific delta vs v6i20:** reward/surface polish failed to break the clone
wall. v6i21 upgrades **OP8-OP12 in place** (same IDs, no OP13-OP17) to adaptive
hardpool v2: intra-episode memory tracks blue lane preference, escort density,
overcommit, near-cap patterns, and fast conversions; red roles/routes shift to
punish repetition (intercept lane bias, escort split pressure, counter on
overcommit, emergency near-cap collapse).

**Fidelity classification:** `DIAGNOSTIC`. Engine change, not paper-faithful.
Router remains blocked.

**Resolved-config diff vs v6i20:** exactly `{experiment_id, run_tag}`.

**Comparability note:** pre-v6i21 OP8-OP12 forced-z / WR results are **not**
directly comparable to post-v6i21 OP8-OP12.

**Implementation:** `gpu_env/_core/_bt_adaptive.py`, profile/dynamics updates in
`_bt_profiles.py` and `opponent_params.py`. Preset aliases: `v6i21`,
`v6i21_adaptive_op8_op12_hardpool_calibration`.

**Calibration command (first gate — WR band, not specialist birth):**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 25 --device cuda --out-dir artifacts\v6i21_adaptive_hardpool_calibration
```

Target: mean blue WR 35-65%, hard cells below 50%, no cell 95%+, red scores
sometimes. Do **not** launch router or specialist training until calibration
passes.

**Calibration eval (v6i9 generalist, 2026-07-04):** `EVALUATED_FAIL` — still
too easy. 10 cells Ã— 25 episodes; mean blue WR **99.2%**; **0/10** cells in
35–65% band; **10/10** cells ≥95%. Red scores occasionally (0.28–0.92 mean) but
never threatens wins. Blue margin ~2.6–3.0. OP8–OP12 show no meaningful
difficulty spread (all saturated). Artifact:
`artifacts/v6i21_adaptive_hardpool_calibration/calibration_report.json`.

| Cell | map | WR | blue | red |
|------|-----|-----|------|-----|
| OP8 | map_b | 100% | 3.00 | 0.40 |
| OP8 | split_lane_v2 | 96% | 2.76 | 0.44 |
| OP9 | map_b | 100% | 2.96 | 0.44 |
| OP9 | split_lane_v2 | 100% | 2.96 | 0.32 |
| OP10 | map_b | 100% | 2.88 | 0.80 |
| OP10 | split_lane_v2 | 100% | 3.00 | 0.92 |
| OP11 | map_b | 100% | 2.88 | 0.32 |
| OP11 | split_lane_v2 | 96% | 2.96 | 0.32 |
| OP12 | map_b | 100% | 3.00 | 0.28 |
| OP12 | split_lane_v2 | 100% | 2.88 | 0.64 |

**Verdict:** OP8–OP12 v2 adaptive hardpool is **not yet hard enough** for the
already-trained v6i9 champion. Do **not** touch router or z-specialist birth
until calibration passes.

**v6i21B pressure tuning (2026-07-04):** implemented as an in-place calibration
patch over the same OP8-OP12 IDs, not a new PPO preset. The patch lowers adaptive
trigger thresholds, makes near-cap collapse fire earlier, strengthens intercept
block points, lets OP12 counter-push on blue overcommit before a blue flag grab,
removes 2v2 sub-base red speed ranges for OP8-OP12, and applies a hardpool-only
blue carrier speed multiplier of 0.95 while blue carries the red flag. Touched
files: `gpu_env/_core/_bt_adaptive.py`, `gpu_env/_core/_bt_profiles.py`,
`gpu_env/_core/_step.py`, `opponent_params.py`. Calibration artifact target:
`artifacts/v6i21B_adaptive_hardpool_pressure_tuning`.

**v6i21B calibration command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 25 --device cuda --out-dir artifacts\v6i21B_adaptive_hardpool_pressure_tuning
```

**Calibration eval (v6i9 generalist, v6i21B engine, 2026-07-05):** `EVALUATED_FAIL`
— marginal improvement only. 10 cells Ã— 25 episodes; mean blue WR **98.0%**
(vs 99.2% pre-v6i21B); **0/10** in 35–65% band; **9/10** ≥95% (vs 10/10).
OP12 shows strongest red pressure (1.40–1.60 mean red vs 0.28–0.64 pre-v6i21B);
OP9 split_lane dropped to 92%. Still Bad Result A. Artifact:
`artifacts/v6i21B_adaptive_hardpool_pressure_tuning/calibration_report.json`.

| Cell | map | WR | blue | red |
|------|-----|-----|------|-----|
| OP8 | map_b | 100% | 2.88 | 0.52 |
| OP8 | split_lane_v2 | 100% | 2.92 | 0.48 |
| OP9 | map_b | 96% | 2.84 | 0.68 |
| OP9 | split_lane_v2 | 92% | 2.68 | 0.76 |
| OP10 | map_b | 100% | 3.00 | 0.72 |
| OP10 | split_lane_v2 | 100% | 3.00 | 0.96 |
| OP11 | map_b | 100% | 3.00 | 0.48 |
| OP11 | split_lane_v2 | 96% | 2.88 | 0.80 |
| OP12 | map_b | 100% | 3.00 | 1.60 |
| OP12 | split_lane_v2 | 96% | 2.96 | 1.40 |

**Calibration eval (v6i20 5u checkpoint, 2026-07-05):** `EVALUATED_FAIL` — mean
blue WR **99.6%**; **0/10** in-band; **10/10** saturated. Artifact:
`artifacts/v6i21_adaptive_hardpool_calibration_v6i20/calibration_report.json`.

**Calibration eval (v6i9 repertoire, 2026-07-05):** `EVALUATED_FAIL` — mean
blue WR **99.2%**; **0/10** in-band; **10/10** saturated. Artifact:
`artifacts/v6i21_adaptive_hardpool_calibration_v6i9_repertoire/calibration_report.json`.

**Multi-anchor verdict:** OP8–OP12 v2 adaptive hardpool is **not hard enough**
against any of the three blue anchors (v6i9 generalist, v6i9 repertoire, v6i20
surface). Saturation is checkpoint-agnostic, not anchor-specific.

**Status:** v6i21B calibrated and failed (98.0% mean, 9/10 saturated). Router and
specialist birth remain blocked.

### 3.21 v6i21C adaptive hardpool denial calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21B:** adaptive memory was active but too soft — blue
still autopilots to ~3 captures. v6i21C strengthens **denial** on the same
OP8-OP12 IDs: predictive intercept, earlier/larger near-cap collapse with longer
role locks, dual flag retrieval, stronger cap-lane blocking, aggressive OP12
counter on overcommit/carrier-loss, physical pressure (red speed 1.10-1.15,
interceptor near-flag boost 1.22, blue carrier 0.87Ã—, red respawn 0.80Ã—).

**Fidelity classification:** `DIAGNOSTIC`. Engine-only; router blocked.

**Resolved-config diff vs v6i21:** exactly `{experiment_id, run_tag}`.

**Tier-1 calibration gates:** mean blue WR below 90%; saturated fewer than 5/10;
at least 1 cell in 35-65%; red_score above 1.0 in a hard cell; blue_score not
pinned near 3.0.

**Calibration command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 25 --device cuda --out-dir artifacts\v6i21c_adaptive_hardpool_denial_calibration
```

**Calibration eval (v6i9 generalist, 2026-07-05):** `EVALUATED_FAIL` (tier-1 and
final). Mean blue WR **96.8%** (down from v6i21B 98.0%, v6i21 99.2%); mean blue
score **2.81** (denial metric moving — min cell **2.24** on OP9/split_lane).
**9/10** cells saturated; **0/10** in-band. Best cell: **OP12/split_lane** WR
**84%**, red **1.96**. Tier-1: passed hard-red gate (2 cells), failed mean WR
below 90%, saturated fewer than 5, in-band, blue-not-pinned (max still 3.0). Artifact:
`artifacts/v6i21c_adaptive_hardpool_denial_calibration/calibration_report.json`.

**Status:** calibrated; partial denial progress but arena still saturated. Router
and specialist birth remain blocked.

### 3.22 v6i21D adaptive hardpool brutal denial calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21C:** v6i21C is connected but still appears saturated
in the visible cells. v6i21D is an upper-bound pressure test over the same
OP8-OP12 IDs: harsher blue carrier speed penalty, real hardpool-only red speed
overdrive, stronger interceptor near-flag boost, faster red respawn, larger
near-cap collapse zone, longer collapse/retrieval locks, harder cap-lane
blocking, and stricter calibration gates.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only calibration. No router,
no PPO training, no specialist birth, no new OP IDs, no blue checkpoint change.

**Resolved-config diff vs v6i21C:** exactly `{experiment_id, run_tag}`.

**Break-saturation gates for D:** mean blue WR below 85%; no more than 3/10
cells at 95%+; at least 2 cells below 75%; at least 1 cell with red_score above
1.0; blue_score not pinned near 3.0 and at least one cell below 2.5.

**10-episode smoke command after v6i21C finishes:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --out-dir artifacts\v6i21D_adaptive_hardpool_brutal_denial_smoke10
```

If blue remains 95-100%, push harder. If blue drops to 0-20%, back off. If blue
lands roughly 35-80%, run the full 25-episode calibration:

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 25 --device cuda --out-dir artifacts\v6i21D_adaptive_hardpool_brutal_denial_calibration
```

**Status:** implemented + focused tests passed. v6i21C calibration is still
running at implementation time, so the D smoke has not been launched yet.
Router and specialist birth remain blocked.

**10-episode smoke eval (2026-07-05):** `PARTIAL_SUCCESS` — first real pressure
signal. Mean blue WR **80.0%** (vs 99.2% v6i21 / 98.0% v6i21B / 96.8% v6i21C);
**4/10** cells in 35-65% band; **5/10** saturated; mean blue score **2.54**.
In-band cells: OP9 (both maps), OP12 (both maps). Still saturated: OP8 map_b,
OP10 (both), OP11 (both). Borderline: OP8 split_lane 90%. Artifact:
`artifacts/v6i21D_adaptive_hardpool_brutal_denial_smoke10/calibration_report.json`.

**Verdict:** denial lever found; grid uneven. Router and specialist birth remain
blocked. Next: targeted OP8/OP10/OP11 hardening (v6i21E), not router.

### 3.23 v6i21E targeted denial balance calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21D:** v6i21D smoke proved the arena can pressure blue
but left OP8/OP10/OP11 saturated while OP9/OP12 landed in-band. v6i21E hardens
only the weak opponents: OP8 carrier-hunter + wider cap-lane collapse, OP10
earlier escort-break + carrier cutoff intercept, OP11 faster anti-repeat collapse.
OP9/OP12 engine constants and dynamics unchanged.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only calibration. No router,
no PPO training, no specialist birth, no new OP IDs.

**Resolved-config diff vs v6i21D:** exactly `{experiment_id, run_tag}`.

**10-episode smoke command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --out-dir artifacts\v6i21E_targeted_denial_balance_smoke10
```

Smoke target: mean WR 60-75%, 6+/10 in-band, at most 2/10 saturated. If smoke
passes balance gates, run full 25-episode calibration before unblocking router.

**Status:** **10-episode smoke `MIXED`/`FAIL`** (2026-07-05). Full 10 cells Ã— 10
episodes; mean blue WR **80.0%**; **4/10** in 35–65% band; **5/10** saturated;
mean blue score **2.60**. In-band: OP9 (both), OP12 (both). OP10 map_b improved
(100%→90%). OP8 100/100 (worse than D on split_lane). OP11 100/100. Tier-1 and
final pass both false. Artifact:
`artifacts/v6i21E_targeted_denial_balance_smoke10/calibration_report.json`.
Superseded for OP8 by v6i21F smoke.

### 3.24 v6i21F OP8 carrier denial calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21E:** v6i21E smoke showed OP9/OP12 in-band and OP10 map_b
improving (100%→90%), but OP8 remained 100/100 with rising red scores and blue
still pinned at 3.0 — red activity without conversion denial. v6i21F makes OP8 a
pure carrier-hunter / cap-lane denial monster: counter-capture and 2v1 scoring
disabled, dual intercept on carrier path, wider near-cap collapse, longer
interceptor locks, lower coordinated-attack probability. OP9–OP12 unchanged.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only OP8 patch. No router, no
PPO training, no specialist birth.

**Resolved-config diff vs v6i21E:** exactly `{experiment_id, run_tag}`.

**10-episode smoke command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --out-dir artifacts\v6i21F_op8_carrier_denial_smoke10
```

**Status:** **10-episode smoke `FAIL`** (2026-07-05). Mean WR **80.0%**; **4/10**
in-band; **5/10** saturated; mean blue **2.59**. OP9/OP12 unchanged from E
(in-band). OP8 still **100/100** but red scores **collapsed** (0.2/0.1 vs
0.9/1.2 on E) — denial posture reduced red scoring without breaking blue caps.
OP11 still 100/100. Tier-1 and final pass false. Artifact:
`artifacts/v6i21F_op8_carrier_denial_smoke10/calibration_report.json`. OP8
hypothesis not confirmed; next lever is OP11 (and OP10 split_lane), not more
global OP8 pressure.

### 3.25 v6i21G easy-cell conversion denial calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21F:** v6i21F made OP8 more carrier-focused but did not
deny conversion. v6i21G targets the remaining easy cells directly: OP8/OP11
restore cap-lane body-blocking during emergency collapse, OP10 cuts off the cap
path instead of blending heavily toward carrier chase, and OP8/OP10/OP11 2v2
speed/coordination pressure increases. OP9/OP12 are unchanged.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only calibration. No router,
no PPO training, no specialist birth.

**Resolved-config diff vs v6i21F:** exactly `{experiment_id, run_tag}`.

**10-episode smoke command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --out-dir artifacts\v6i21G_easy_cell_conversion_denial_smoke10
```

**Status:** implemented + focused tests passed. Full 10-cell smoke and a
patched-cell 3-episode smoke both exceeded tool timeouts without writing a
report and were stopped by the agent; no G calibration result should be inferred
from those partial attempts. Router and specialist birth remain blocked.

**Interrupted background smoke partials:** after relaunching G with captured
stdout, OP8, OP10, and OP11 map_b remained saturated:
OP8 map_b `100%/3.00/0.10`, OP8 split `100%/3.00/0.10`,
OP10 map_b `100%/3.00/0.20`, OP10 split `100%/3.00/0.50`,
OP11 map_b `100%/3.00/1.50`. OP9 stayed in-band (`50%`, `60%`). The run was
interrupted before final JSON. Conclusion: bespoke OP8/OP10/OP11 geometry still
fails; use calibrated surrogate shapes.

### 3.26 v6i21H saturation surrogate calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21G:** G confirmed the failed pattern. H replaces the
remaining saturated custom shapes with already-calibrated pressure shapes:
OP8 becomes OP9-like fortress pressure, OP10/OP11 become OP12-like counter
pressure, and the failed OP8 dual-denial, OP10 escort-break, and OP11
repeat-intercept adaptive route overrides are disabled.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only calibration. No router,
no PPO training, no specialist birth.

**Resolved-config diff vs v6i21G:** exactly `{experiment_id, run_tag}`.

**Targeted smoke command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --opponents OP8 OP10 OP11 --out-dir artifacts\v6i21H_saturation_surrogate_patched_cells_smoke10 --progress-every 1
```

**Status:** implemented + focused tests passed. Evaluation pending.

### 3.27 v6i21I OP8 extreme physical calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21H:** H restored red pressure for OP8 but blue still
won 100/100. v6i21I makes OP8 an explicit physical upper-bound test: OP8-only
blue carrier speed multiplier `0.35`, OP8 red speed multiplier `1.60`, OP8
near-flag interceptor boost `1.85`, and OP8 2v2 speed range `1.35-1.45`.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only OP8 calibration. No
router, no PPO training, no specialist birth.

**Resolved-config diff vs v6i21H:** exactly `{experiment_id, run_tag}`.

**OP8-only smoke command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --opponents OP8 --out-dir artifacts\v6i21I_op8_extreme_physical_smoke10 --progress-every 1
```

**Status:** **OP8-only smoke `PARTIAL_SUCCESS`** (2026-07-05). Extreme physical
pressure **broke OP8 saturation** for the first time: map_b **80%** WR (blue
2.90, red 1.90), split_lane **70%** WR (blue 2.80, red 2.00). Mean WR **75%**;
**0/2** saturated; red scores finally threaten conversions. Not yet in 35–65%
band; tier-1 and final pass false. Refutes "OP8 is structurally unblockable" —
issue was insufficient physical pressure, not scoring/tagging geometry. Artifact:
`artifacts/v6i21I_op8_extreme_physical_smoke10/calibration_report.json`. Next:
dial OP8 physical knobs toward in-band without overshooting OP9/OP12 balance.

### 3.28 v6i21J hardpool balance calibration -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21I:** OP8I proved OP8 is no longer structurally
saturated. v6i21J keeps OP8 hard and adds targeted physical pressure to OP10 and
OP11: OP8 blue carrier `0.30`, OP8 red speed `1.70`, OP8 interceptor boost
`2.00`; OP10/OP11 blue carrier `0.45`, red speed `1.45`, interceptor boost
`1.65`. OP9/OP12 unchanged.

**Fidelity classification:** `DIAGNOSTIC`. Engine-only hardpool calibration. No
router, no PPO training, no specialist birth.

**Resolved-config diff vs v6i21I:** exactly `{experiment_id, run_tag}`.

**Next calibration command:**

```powershell
uv run python experiments/run_v6i21_adaptive_hardpool_calibration.py --checkpoint checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --episodes 10 --device cuda --out-dir artifacts\v6i21J_hardpool_balance_smoke10 --progress-every 1
```

**Target before V6I22:** mean blue WR 50-75%, saturated cells 0-2/10, at least
5/10 cells in 35-75%, red scores meaningfully, and blue score not pinned at 3.0.
Router and repertoire-birth training remain blocked until this hardpool
calibration is acceptable.

**10-episode smoke eval (2026-07-05):** `GOOD_SMOKE` — pool usable. Profile proof
in report confirms OP8 `0.30/1.70/2.00`, OP10/OP11 `0.45/1.45/1.65`. Mean blue
WR **64.0%**; **7/10** in-band; **2/10** saturated (OP8 map_b, OP10 split_lane);
mean blue score **2.43**. OP8 split_lane **50%**; OP9 **50-60%**; OP11 **40-50%**
(red 2.2-2.5); OP12 **50%** (red 2.1-2.3). Weak spots: OP8 map_b still **100%**,
OP10 split_lane **100%**. Formal tier-1 fails only on `blue_score_not_pinned`
(max 3.0). Artifact:
`artifacts/v6i21J_hardpool_balance_smoke10/calibration_report.json`.

**25-episode calibration eval (2026-07-06):** `TIER1_PASS` — arena usable at n=25.
Mean blue WR **66.4%**; **6/10** in-band; **2/10** saturated (OP10 both maps);
mean blue score **2.43** (max **2.80**, min **1.48** — no longer pinned at 3.0).
OP8 map_b dropped to **76%** (smoke 100%); OP12 split_lane **36%** in-band.
Weak spot: OP10 still **96-100%**. `calibration_pass_tier1=True`;
`calibration_pass=False` (mean WR above 65% final band). Artifact:
`artifacts/v6i21J_hardpool_balance_calibration/calibration_report.json`.

**Status:** hardpool calibration tier-1 passed. Repertoire birth (v6i22) may
proceed as diagnostic; router remains blocked.

### 3.29 v6i22 adaptive hardpool repertoire birth -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i21J:** v6i21J is a calibration preset. v6i22 starts the
next label-free repertoire-birth fork over the same adaptive hardpool surface:
router off, `balanced_episode` z assignment, one z held for the episode, shared
actor trunk frozen by `v6i9_training_stage = "repertoire"`, and z-specific
modules plus critic trainable.

**Fidelity classification:** `SUMMER-COMPATIBLE EXTENSION`, not
`PAPER-FAITHFUL`. It inherits v6 staged/frozen/adapted hardpool machinery, but
adds no handcrafted z-role contracts, no opponent-ID supervision, no oracle-z
targets, no router distillation, and no auxiliary label head. The old contract
specialist scaffold is explicitly off:
`latent_contract_specialist_enabled = False`,
`latent_contract_specialist_coef = 0.0`.

**Resolved-config diff vs v6i21J:** exactly `{experiment_id,
latent_contract_specialist_coef, latent_contract_specialist_enabled,
latent_contract_specialist_variant, run_tag}`.

**User-requested gate override:** v6i21J calibration evaluation was still pending
when v6i22 was implemented. Treat v6i22 as a direct diagnostic jump, not as proof
that the hardpool calibration target already passed.

**First 5-update launch command:**

```powershell
uv run python rl/train_ppo.py --preset v6i22 --load checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --load-weights-only --additional-steps 5120 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i22_repertoire_birth_5u_seed1 --checkpoint-dir artifacts\v6i22_repertoire_birth_5u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
```

**Smoke gates:** banner shows `balanced_episode`; router remains off for forced
episodes; contract reward columns remain zero/inactive; shared-trunk hash is
unchanged; z-specific params move; all four z values are sampled; OP8-OP12 and
both map layouts appear in episode logs.

**Promotion gates:** forced-z evaluation over OP8-OP12 x both maps must show
real options before any router work: behavior pair distance above the old
0.04-0.05 ceiling, at least 1-2 pairs above threshold, margin/tempo/behavior
fingerprints separating by z, and `unique_best_z_count > 1`.

### 3.30 v6i22B context behavior diversity -- `IMPLEMENTED` (2026-07-05)

**Scientific delta vs v6i22:** V6I22 produced useful z consequences but not
strong forced-z behavior fingerprints. The 5-update run passed Stage-C
(`oracle_WR = 95%`, `best_fixed_WR = 80%`, `unique_best_z_count = 4`) while
behavior distance stayed below threshold (`mean = 0.0327`, `max = 0.0512`,
`pairs_above_threshold = 0`). The 20-update continuation kept Stage-C alive
but behavior distance did not improve (`mean = 0.0289`, `max = 0.0439`,
`pairs_above_threshold = 0`). V6I22B therefore tests label-free anti-collapse
pressure instead of more updates.

**Fidelity classification:** `SUMMER-COMPATIBLE EXTENSION`, not
`PAPER-FAITHFUL`. Router remains off; one unlabeled z is held per episode;
contract-specialist rewards stay disabled; no handcrafted z roles, supervised
strategy labels, oracle best-z targets, opponent-ID actor shortcut, router
distillation, or router training are added. The new signal is a small
success-gated behavior-contrast reward from trajectory fingerprints.

**Resolved-config diff vs v6i22 primary arm:** exactly `{experiment_id,
latent_behavior_contrast_coef, latent_behavior_contrast_margin, run_tag}`.
The primary arm is `v6i22b` / `v6i22b_behavior_diversity_coef003` with
`latent_behavior_contrast_coef = 0.03` and
`latent_behavior_contrast_margin = 0.06`. Sweep arms are `v6i22b_coef001` and
`v6i22b_coef005`.

**Runtime contract:** balanced-episode z assignments now feed the behavior
contrast ledger; the contrast bucket is opponent x map at terminal; failed
episodes do not update the centroid or receive the bonus. This avoids semantic
z labels while directly targeting the failed behavior-distance gate.

**5-update coefficient sweep commands:**

```powershell
uv run python rl/train_ppo.py --preset v6i22b_coef001 --load checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --load-weights-only --additional-steps 5120 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i22b_div001_5u_seed1 --checkpoint-dir artifacts\v6i22b_div001_5u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
uv run python rl/train_ppo.py --preset v6i22b --load checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --load-weights-only --additional-steps 5120 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i22b_div003_5u_seed1 --checkpoint-dir artifacts\v6i22b_div003_5u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
uv run python rl/train_ppo.py --preset v6i22b_coef005 --load checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip --load-weights-only --additional-steps 5120 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i22b_div005_5u_seed1 --checkpoint-dir artifacts\v6i22b_div005_5u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
```

**Promotion gates:** Stage-C must stay passing, `unique_best_z_count` must stay
above 1, WR/margin advantage must not collapse, `behavior_pair_distance_mean`
must beat the V6I22 20-update level and move back above 0.04, and at least one
arm should approach the 0.06 behavior-distance target or produce an
above-threshold pair. Router training remains blocked.

**5-update coefficient sweep training completed (2026-07-05):** all three arms
launched from
`checkpoints\2v2\final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip`
with `--load-weights-only`, `--additional-steps 5120`, `n_envs=4`,
`n_steps=256`, `n_epochs=1`, and CUDA. Final checkpoints:

| Arm | Artifact | Final contrast telemetry |
|-----|----------|--------------------------|
| `v6i22b_coef001` | `artifacts\v6i22b_div001_5u_seed1\final_v6i22b_div001_5u_seed1_2v2.zip` | `active_frac=1.0`, `distance_mean=0.22215`, `bonus_mean=0.00060`, `reward_behavior_contrast_mean=2.93e-06` |
| `v6i22b` / `coef003` | `artifacts\v6i22b_div003_5u_seed1\final_v6i22b_div003_5u_seed1_2v2.zip` | `active_frac=1.0`, `distance_mean=0.22215`, `bonus_mean=0.00180`, `reward_behavior_contrast_mean=8.79e-06` |
| `v6i22b_coef005` | `artifacts\v6i22b_div005_5u_seed1\final_v6i22b_div005_5u_seed1_2v2.zip` | `active_frac=1.0`, `distance_mean=0.22215`, `bonus_mean=0.00300`, `reward_behavior_contrast_mean=1.46e-05` |

Mechanism read: contrast ledger is active and coefficient scaling is correct.
Training traces were otherwise near-identical across arms, so do not infer
promotion from training telemetry alone. Next required step is matched forced-z
fingerprinting for all three final checkpoints over OP8-OP12 x both maps.

**Forced-z fingerprints completed (2026-07-06):** all three arms used the same
matched protocol as V6I22 20u: OP8-OP12, `map_b` and
`map_b_split_lane_v2`, four forced z values, `episodes=2` per cell,
`base_seed=42`, deterministic actions, inherited training reward surface, and
`max_decision_steps=240`.

| Arm | Stage-C | WR adv | Margin adv | Unique best z | Behavior mean | Behavior max | Pairs above threshold |
|-----|---------|--------|------------|---------------|---------------|--------------|-----------------------|
| `v6i22b_coef001` | PASS | +15.0% | +0.80 | 4 | 0.0327 | 0.0512 | 0 |
| `v6i22b` / `coef003` | PASS | +15.0% | +0.80 | 4 | 0.0327 | 0.0512 | 0 |
| `v6i22b_coef005` | PASS | +15.0% | +0.80 | 4 | 0.0327 | 0.0512 | 0 |

Best-z surface for all arms:
`OP8|map_b=z0`, `OP8|map_b_split_lane_v2=z2`, `OP9|map_b=z1`,
`OP9|map_b_split_lane_v2=z3`, `OP10|map_b=z0`,
`OP10|map_b_split_lane_v2=z0`, `OP11|map_b=z0`,
`OP11|map_b_split_lane_v2=z0`, `OP12|map_b=z0`,
`OP12|map_b_split_lane_v2=z2`.

Verdict: `PROMISING_CONSEQUENCE_LEAD / BEHAVIOR_GATE_FAIL`. V6I22B preserves
the V6I22 Stage-C consequence surface but does not improve visible behavior
fingerprints in the 5-update sweep. The anti-collapse reward is live, but the
coefficient range is too weak or too delayed to change the forced-z behavior
gate. Router training remains blocked.

### 3.31 v6i22C contextual outcome diversity -- `EVALUATED_FAIL` (2026-07-06)

**Scientific delta vs v6i22:** V6I22B used trajectory behavior fingerprints as
the anti-collapse reward input and did not move the forced-z behavior gate. V6I22C
keeps the label-free repertoire-birth scaffold fixed and changes the pressure to
generic context-conditioned outcomes: successful terminal episodes receive a
bounded bonus when their score margin differs from other z outcome centroids in
the same opponent x map bucket.

**Fidelity classification:** `SUMMER-COMPATIBLE EXTENSION`, not
`PAPER-FAITHFUL`. Router remains off; one unlabeled z is held per episode;
`balanced_episode` exposure stays active; contract rewards stay disabled;
behavior-contrast reward stays disabled; actor receives no opponent ID; no
supervised labels, role rewards, handcrafted z mapping, oracle best-z target, or
router distillation is added.

**Resolved-config diff vs v6i22 primary arm:** exactly `{experiment_id,
latent_outcome_diversity_coef, run_tag}`. The primary arm is `v6i22c` /
`v6i22c_outcome_diversity_coef003` with
`latent_outcome_diversity_coef = 0.03`, default margin `1.0`, EMA `0.9`, and
success-only updates.

**5-update diagnostic completed (2026-07-06):** checkpoint
`artifacts\v6i22c_outcome_div003_5u_seed1\final_v6i22c_outcome_div003_5u_seed1_2v2.zip`.
Training: 5 updates, cumulative WR 61.5%, frozen trunk confirmed.

**Forced-z fingerprint (eps2):** Stage-C PASS (oracle margin +0.80, unique
best-z 4/10). Birth gate FAIL: `behavior_pair_distance_mean = 0.033`,
`pairs_above_threshold = 0`. Outcome diversity nudged scoreboard variance but
not playstyle fingerprints.

**Verdict:** `EVALUATED_FAIL` for specialist birth. Stage-C alive. Router
blocked.

### 3.32 v6i22D strong behavior diversity -- `EVALUATED_FAIL` (2026-07-06)

**Scientific delta vs v6i22B/C:** V6I22 25u (`behavior mean = 0.020`), V6I22B
sweep coef `<= 0.05` (`behavior mean ~ 0.033`), and V6I22C outcome diversity
(`behavior mean = 0.033`) all failed the forced-z behavior birth gate. V6I22D
returns to the behavior-contrast channel with stronger coefficients: primary
`0.10` (novel) and paired control `0.05`.

**Fidelity classification:** `SUMMER-COMPATIBLE EXTENSION`, not
`PAPER-FAITHFUL`. Same label-free scaffold as V6I22B: router off,
`balanced_episode`, contract disabled, outcome-diversity disabled, trajectory
fingerprint contrast keyed by opponent x map, success-only updates.

**Resolved-config diff vs v6i22:** exactly `{experiment_id,
latent_behavior_contrast_coef, latent_behavior_contrast_margin, run_tag}`.
Primary arm `v6i22d` / `v6i22d_behavior_diversity_coef010` uses
`latent_behavior_contrast_coef = 0.10`. Sweep arm `v6i22d_coef005` uses
`0.05` (same coefficient as `v6i22b_coef005`, paired control).

**5-update coefficient sweep completed (2026-07-06):** both arms from v6i9
anchor, 5 updates each, cumulative WR 61.5%. Contrast reward scaled with coef
(`1.5e-05` at 0.05, `2.9e-05` at 0.10). Training-time
`forced_z_behavior_pair_distance_mean` stayed ~0.0155 for both arms.

**Forced-z fingerprints (eps2):** both arms produced **identical** surfaces:

| Arm | Stage-C | Oracle gap | Unique best-z | Behavior mean | Pairs above threshold |
|-----|---------|------------|---------------|---------------|-----------------------|
| `v6i22d_coef005` | PASS | +0.80 | 4/10 | 0.0331 | 0 |
| `v6i22d` / coef010 | PASS | +0.80 | 4/10 | 0.0331 | 0 |

Artifacts:
`artifacts/v6i22d_div005_5u_seed1/forced_z_fingerprint_eps2/`,
`artifacts/v6i22d_div010_5u_seed1/forced_z_fingerprint_eps2/`.

**Verdict:** `EVALUATED_FAIL` for specialist birth. Stronger behavior-contrast
pressure (0.05–0.10) did not move the forced-z behavior gate beyond the V6I22B/C
ceiling. Stage-C consequence remains alive. Router blocked.

### 3.33 v6i22E fixed-alpha adapters -- `EVALUATED_FAIL` (action-JSD) (2026-07-21)

**Scientific delta:** `h_z = h + Î± A_z(h)` with `Î±=0.1`, Kaiming init, no
learned gate. Parent `v6i22`. Classification: `SUMMER-COMPATIBLE EXTENSION`.

**5u result:** adapter weight L2 ~9.2 (was ~0.09); offline `Î±â€–Aâ€–/â€–hâ€–` ~5.8%.
Forced-z behavior mean ~0.144 (informal >0.06) but formal pairs≥0.35 still 0.
**CF action-JSD still FAIL** (mean ~0.0002). Magnitude trap confirmed broken;
shared frozen action_head still prevents stable `Ï€(a|s,z)` separation.

### 3.34 v6i23 population birth -- `IMPLEMENTED` (2026-07-23)

**Scientific delta vs v6i22e:** independent Stage-2-trainable per-z action heads
plus active-z-only residual forward. Same hardpool, fixed-Î±, router-off,
`balanced_episode` scaffold. No soft diversity rewards, no opponent-ID.

**Fidelity:** `SUMMER-COMPATIBLE EXTENSION`, not `PAPER-FAITHFUL`.

**Resolved-config diff vs v6i22e:** exactly `{experiment_id,
latent_population_birth_active_z_only,
latent_population_birth_per_z_action_heads, run_tag}`.

**Launch (5u smoke):**

> Note: `checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip`
> is missing on this machine. Warm-start from the completed V6I22E 5u final
> (same trunk lineage; per-z heads sync from shared `action_head` on load).

```text
uv run python rl/train_ppo.py --preset v6i23 --load artifacts\v6i22e_fixed_alpha_adapters_5u_seed1\final_v6i22e_fixed_alpha_adapters_5u_seed1_2v2.zip --load-weights-only --additional-steps 5120 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i23_population_birth_5u_seed1 --checkpoint-dir artifacts\v6i23_population_birth_5u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
```

**Gate:** CF action-JSD pair mean > 0.05 on ≥2 oracle cells (or head0 disagree
> 0.2 with non-tie). Router blocked until gate clears. Pinned by
`tests/test_v6i23_population_birth.py`. Helper:
`experiments/run_v6i23_population_birth.py`.

**5u smoke completed (2026-07-23):** warm-start from V6I22E 5u (v6i9 anchor
missing on disk). Load path: newly initialized `latent_action_heads` synced
from shared `action_head`; BE PASS (trunk-only bypass). Stage-2 froze 26
shared-trunk params; per-z heads trainable.

Diagnostic after 5u: head pairwise L2 vs head0 ≈ 0.035–0.040 (moved, but
small); adapter L2 ≈ 9.2; random-local forced-z logit max-abs ≈ 0.11–0.19.

CF action-JSD probe (10 cells, OP8–OP12 Ã— both split maps):
`mean_of_cell_jsd_means ≈ 0.00019`, `cells_with_any_pair_above_0_05 = 0`,
`gate_any_pair_jsd_gt_0_05 = false`. Some head0 disagree values reach ~0.15–0.28
with near-zero JSD (near-tie flips — same pattern as V6I22E; does **not** clear
the non-tie gate).

**Verdict so far:** architecture path is live; 5u is insufficient for the CF
action-JSD birth gate. Next: 25u continuation from this 5u final, then re-probe.
If still flat → fallback to full separate policies → distill (plan Path B).

```text
uv run python rl/train_ppo.py --preset v6i23 --load artifacts\v6i23_population_birth_5u_seed1\final_v6i23_population_birth_5u_seed1_2v2.zip --load-weights-only --additional-steps 25600 --n-envs 4 --n-steps 256 --n-epochs 1 --device cuda --run-tag v6i23_population_birth_25u_seed1 --checkpoint-dir artifacts\v6i23_population_birth_25u_seed1 --fresh-metrics-csv --episode-log-every 0 --periodic-checkpoint-steps 0 --no-progress-bar
```

**25u continuation:** `ABORTED` — replaced by pre-registered diagnostic
`v6i23_popbirth_prereg` (milestones 10u / 15u / 25u total, matched-seed
probes, stop/escalate rule). Open-ended artifact renamed
`artifacts/v6i23_population_birth_25u_seed1_ABORTED_open_ended/`.

**Pre-registered diagnostic (2026-07-23):** closed test of shared-trunk
population birth. Router locked. No new diversity losses. Probe CF
action-JSD (entropy-aware non-tie disagree), geometry (head L2 / adapter
ratio), and forced-z eps2 at 10u, 15u, 25u. Escalate to four independent
policies if JSD stays ~1e-4–1e-3 while head L2 rises.

```text
uv run python experiments/run_v6i23_population_birth_prereg_diagnostic.py --run
```

**10u milestone (2026-07-23):** completed. Geometry: head pairwise L2 mean
~0.063 (up from ~0.035 at 5u); adapter Î±-ratio ~0.065. CF action-JSD mean
still ~0.00020 (0/10 cells with pair >0.05). Stage-C PASS (oracle gap +0.40,
unique best-z 2); behavior pair mean 0.084 but formal pairs_above=0.

Auto-script initially emitted `PROMOTE` on a single-cell non-tie disagree
spike (max 0.44) despite flat JSD — **overridden**. Correct registered
verdict: `STOP_EARLY_ESCALATE` (head L2 rose while Ï€(a|s,z) stayed near-clone).
Do not train router. Do not add soft regularizers. Next: four independently
trained policies → verify payoff/trajectory separation → distill into
z-conditioned Summer architecture.

Artifacts: `artifacts/v6i23_popbirth_prereg/` (`ckpt_10u/`, `probes/10u/`,
`decision_log.json`).

### 3.35 v6i24 full-policy population diagnostic -- `ACTIVE` (2026-07-23)

**What V6I24 proves / does not prove (locked 2026-07-23):**

* **Can prove:** `G_available > 0` for independent teachers under distinct
  pressures (environment supports niches; external pressure can uncover them).
  Feasibility gate / teacher generator / strategic upper bound.
* **Cannot prove:** a single latent PPO discovered and used those strategies.
  Fixed-`z=0`, router off — four full policies, not Summer end-to-end.

**Claim distinction:**

* **Claim A (spontaneous latent emergence):** end-to-end `Ï€(a|s,z)` discovers
  niches. Teacher distillation does **not** prove this. Direct latent PPO
  remains a final-arm control.
* **Claim B (population-guided latent strategy learning):** discover
  repertoire externally → distill into `Ï€(a|s,z)` → route → beat non-latent
  PPO. V6I24 + distillation can prove Claim B. Preferred honest headline
  given V6I22–V6I25 negatives.

**Final proof arms (after teacher PASS + distill + router):**

| Arm | Role |
|-----|------|
| K=1 PPO | Non-latent baseline |
| Parameter-matched non-latent PPO | Capacity control |
| K=4 latent, fixed/random `z` | Capacity-without-routing control |
| K=4 Summer (learned niches + router) | Full method |

Same steps, maps, opponents, seeds, reward, PPO hparams, eval seeds.
Five requirements: controllability, competence, `G_available`, `G_realized`,
`G_latent` (multi-seed CIs). See protocol §7.3 / §8.

**Slim eval of existing 5u teachers (2026-07-23):** oversized 32-ep full-grid
eval killed; scored `probe_05u` zips on OP11/OP12 Ã— both maps @ 8 matched
eps, JSD skipped (`eval_gates_slim/`).

```text
WR matrix (n=8/cell):
              OP11/m1  OP11/v2  OP12/m1  OP12/v2
balanced        0.75     0.50     0.75     0.75
failure_cells   0.75     1.00     0.875    0.875
high_variance   0.50     0.75     0.625    0.50
complementary   0.875    0.00     0.75     0.00

Cross-fit oracle = best_fixed = 0.95; delta=0; CI=[0,0]
Primary: FAIL | Decision: TREND_EXTEND_TO_100K
Max row distance: 0.67 (driven by complementary collapse on v2)
Classifier acc: 0.86 (supporting)
```

Interpretation: not promote. Apparent cell winners exist, but
`G_available=0` after cross-fitting; complementary is **incompetent** on
v2 (0%), not a niche. Closest ladder outcome: **stop scaling this
confounded 5u setup → no-contract micro-probe** (fresh shared-core init).

**Locked next sequence (2026-07-23):**

```text
finish current 5u matrix (shared-contract confound; directional only)
→ classify: promote / stop / inconclusive
→ print experiment contract
→ no-contract --micro-probe (2u, OP11/OP12Ã—maps, 8 eps)
→ promote to 5u+larger eval OR stop Path C
```

Micro-probe outcomes: **Promote** = multiple cell winners + non-parallel rows + positive oracle point estimate; **Stop** = one policy everywhere or parallel rows; **Inconclusive** = unstable ranks → more eval seeds before more PPO.

**Process rule (2026-07-23):** maximize information per GPU hour.
See `experiment-and-evaluation-protocol.md` §8 (multi-fidelity ladder).
Next Path C spend after the current matrix: either stop, or
`--micro-probe` / `--disable-contract-specialist` — never another full
5u+32-eps launch without a five-line contract and a micro rejection filter.

**Status:** `CLOSED_AS_PRIMARY` (2026-07-23) — soft 5u Path C abandoned as
the method path. Retained as a landscape / feasibility probe only.
See `artifacts/v6i24_population_seed1/pathc_close_verdict.json`.

**Soft 5u close (slim OP11/OP12Ã—2 maps, 8 eps/cell):**

| Gate | Result |
|------|--------|
| Different best + margin ≥0.10 | PASS (2 unique; 4/4 cells) |
| Payoff row distance | PASS (max 0.67) |
| Cross-fitted oracle > best fixed | **FAIL** (0.95 − 0.95 = 0) |
| Donor→teacher mean KL | 0.011 (tiny; still near generalist basin) |
| Train success (directional) | Ï€0≈0.59, Ï€1≈0.23, Ï€2≈0.42, Ï€3≈0.67 |

Runner suggested `TREND_EXTEND_TO_100K`. **Rejected:** shared z=0 contract
confound + tiny KL + zero harvestable `G_available` means more soft PPO
hours will not answer Claim B. Primary path is **V6I26 LRO** (§3.37).

**Run:** 5u probe
(`artifacts/v6i24_population_seed1`, shared-core from V6I23 donor, seed 1).
Init competence gate skipped for this launch (`--skip-init-gate`): prior
multi-member eval hung mid-episode; lean gate + step-cap fixed in code for
later probes. Identity still guaranteed by shared-core template copy.

**Interpretation protocol (locked):** decisive quantity is
`V_context-oracle − V_best-fixed` with paired CI. Per-update train WRs and
intra-member latent JSD are not the evidence (all members use `fixed_z=0`).

* PASS niches despite shared contract → confirm with
  `--disable-contract-specialist` (new flag) from identical shared-core init.
* FAIL → do not declare Path C dead; restart no-contract (shared z=0 contract
  may have glued policies). Separators then = cell pressures + independent
  PPO only.
* One member best everywhere → quality gap, not niches; inspect narrow/hard
  pressures for competence loss.
* Varying winners but cross-fit fails → sampling noise; more eval eps only if
  payoff rows trend.

**Known confound on current 5u:** `latent_contract_specialist_enabled=True`
inherited from v6i21j (all members get the same z=0 contract). Pressures
were real (episode CSVs + `training_cell_distribution` in run_config); banner
previously mis-advertised uniform sampling — fixed to report cell
distribution when present.

**Why resumed:** V6I25 showed controllability without comparative advantage
(cross-fitted geometry oracle tied best-fixed `z=2` at 75%, delta 0). The
broken layer is **actor repertoire**, not router optimization. Independent
teachers under distinct context pressures are now evidence-justified, not a
sideways guess.

**Scientific delta (plain English):** Path C — K=4 independent policies
under fixed OP8–OP12Ã—map cell pressures. Parent `v6i21j`; optional
`--checkpoint-mode shared-core` from V6I23 donor.

**Primary success gate (locked):**

```text
Different contexts have different best policies (margin ≥ 0.10 on ≥2 cells)
AND
cross-fitted context oracle > best fixed policy
with paired bootstrap CI excluding zero
```

Action-JSD / trajectory classifier = supporting evidence only. Hindsight
per-cell `max_Ï€ R` is diagnostic, not the promotion gate.

**Progression after PASS:** distill teachers into `Ï€(a|s,z)` → re-test
distilled context oracle > best fixed `z` → only then train geometry router.

**Superseded as primary method path (2026-07-23):** soft Path C remains a
feasibility / teacher generator. The Claim B breakthrough implementation is
**V6I26 LRO-Summer** (§3.37): internal response-oracle branches, not four
external policies with handcrafted pressure mixtures.

### 3.37 v6i26 Latent Response-Oracle Summer (LRO) -- `ACTIVE` (2026-07-23)

**Status:** `ACTIVE` — finite proof ladder (not indefinite Summer polish).
Contract: `artifacts/v6i26_lro_round1_seed1/proof_ladder_contract.json`.
**Seeds 2–3 mid-flight: do not alter recipe.** Seed-1 Phase-2 causal claim failed.

**Map_a enablement (2026-07-25):** LRO may train/eval on `map_a_open` (default
arena). Preset forces `obstacle_obs_channel=True` so V6I23+ 8-channel
checkpoints keep a compatible CNN stem (obstacle plane is zeros on open maps).
Default landscape/birth surface: `LRO_DEFAULT_MAPS` =
`(map_a_open, map_b_split_lane, map_b_split_lane_v2)`. Prior map_b-only scans
remain valid; re-scan if map_a cells should enter target selection.

**Map_a measurement status (2026-07-25, seed-1, infrastructure unlock only):**

* Compat regression `artifacts/v6i26_map_a_obs_compat_regression_seed1.json`:
  **PASS** (after fixing diagnostic wrapper access:
  `CustomPPOInferencePolicy.model.state_dict()`, not the wrapper). 8ch both
  maps; map_a obstacle plane exactly zero; map_b nonzero; no CNN shape-skips;
  CNN weights identical across map_a/map_b loaders. First attempt FAILED only
  due to that script bug — not an obs-schema failure. Runtime landscape loads
  also report Behavioral-equivalence PASS with no shape-skipped CNN keys.
* Archive landscape `artifacts/v6i26_landscape_scan_mapa_only_seed1/`:
  competent policies (v6i24 balanced/failure_cells, v6i23) are **winrate=1.0
  on all 7 OP cells**; `G_available_point≈0.008`, `niche_signal=false`,
  `cells_with_margin≥0.1 = 0`. Cross-fitted oracle CI crosses 0
  (`gate_cross_fitted_oracle=false`). Auto-selected birth branch was the
  broken `v6i24_complementary` row (wr=0) — **not** a strategy niche.
  Interpretation: map_a is **not** a birth curriculum by itself (near-saturated
  for competent policies); keep for generalization / selector geometry only
  if forced-z shows crossover.
* Next (running): forced-z z0..z3 on map_a for V6I23; then three-map archive
  scan `artifacts/v6i26_landscape_scan_mapa_seed1/`. **No training** until
  crossover or a calibrated uncovered weakness is measured.

**Forced-z V6I23 on map_a only** (`artifacts/v6i26_forced_z_mapa_only_v6i23_seed1/`,
4 eps/cell, seed-1):

* Winrates mostly 1.0; OP6/OP9 at 0.75 for some z.
* **z0 and z3 are payoff-identical** on every OP cell (same WR and margin).
  z1≈z2 and slightly better on OP6/OP9. No z0↔z3 crossover.
* Stage-C Gate1 FAIL (oracle WR = best-fixed WR = 96.4%); margin gap ≈0.036.
* Unique best-z across cells: `{0,1}` only; z3 never uniquely best.
* Behavior pair mean ≈0.093 (some action distinction; not payoff complementarity).
* Decision: **do not launch birth on map_a**; **do not claim selector unlock
  from z0/z3** on this surface. Soft z0 vs z1 preference on OP6/OP9 is
  screening-only (4 eps). Three-map archive scan still running for
  cross-geometry candidate surface.

**Three-map archive scan complete** (`artifacts/v6i26_landscape_scan_mapa_seed1/`,
seed-1, 4 eps/cell):

* Formal decision string: `MANUFACTURE_VIA_LRO_STAGE1` (archives not harvestable;
  cross-fitted Δ negative, `gate_cross_fitted_oracle=false`).
* **map_a_open:** strong policies saturated — `balanced` and `v6i23` mean WR=1.0
  on all 7 OP cells; all best-vs-second margins `<0.1`. Useful for
  generalization / selector geometry, **not** an LRO birth curriculum.
* **map_b_split_lane:** real headroom — 5/7 cells have margin ≥0.1;
  `failure_cells` collapses (mean WR≈0.11); `balanced` vs `high_variance`
  unique-best split. This is where archive niches live.
* **map_b_split_lane_v2:** milder; margins mostly tiny; balanced often best.
* Map preference reversals exist (who is argmax changes map_a↔map_b) but
  map_a side is near-tied among strong policies — do not treat as causal
  complementarity for birth. **No training launched from this unlock.**

**Locked headline claim (replaces spontaneous emergence):**

> Summer uses response-oracle training to create complementary latent team
> strategies inside one decentralized PPO policy, then learns a persistent
> context router that selects among them and outperforms fixed-strategy and
> matched non-latent PPO agents.

Strategies remain unlabeled. LRO identifies weaknesses; PPO discovers actions
from task reward only.

**Finite proof ladder (doors lock behind us):**

```text
1. Stage 1 creates one latent response     ΔG > 0
2. Confirm seeds + ≥32 eps/cell            CI95(ΔG) > 0
3. Add specialists only if G rises again   (2–3 enough; not forced K=4)
4. Internal repertoire retention           G_available,internal > 0
5. Sparse context router                   G_realized > 0
6. Routed LRO vs matched non-latent PPO    G_latent > 0
```

`G = V_cross-fitted oracle − V_best fixed z`. Everything else is diagnostic.

**Phase 1 Stage-1 contract (current 25u — locked mid-flight):**

```text
Initial policy: V6I23
Selected branch: z3
Router: OFF | forced z3 | contract OFF | task reward only
Target: smoothed OP11/OP12 regret mixture
Inactive branches: frozen (active-branch-only)
Architecture: deep z trunks + per-z action heads
  (shared z-conditioned critic — no separate value heads this run)
```

**Accept z3 only if all hold:** `ΔG>0`, targeted OP11/OP12 improves, competence
floor, inactive branches do not drift, nonredundant payoff row, forced-z behavior
nonredundancy.
Not enough: KL alone, action JSD alone, one noisy 4-episode cell.

**Failure → one predefined response (no coefficient carousel):**

| Result | One allowed response |
|--------|----------------------|
| `ΔG>0` | Phase 2 larger confirm |
| `ΔGâ‰¤0`, tiny KL | **One** retry: more branch freedom/budget |
| Targetâ†‘, collapse elsewhere | Add fixed competence-anchor mixture fraction |
| Large KL, no targetâ†‘ | Stop OPÃ—map; move to possession/phase contexts |
| Two fair rounds flat | Redesign strategic regimes, not PPO machinery |

> **Superseded for post–Phase-2 seed-1 FAIL:** do **not** treat tiny behavior
> distance as automatic per-`z` value-head trigger. Use the senior-RL order
> and per-`z` precondition under â€œPost multi-seed protocolâ€ below.

**Phase 2 confirm:** matched seeds, ≥32 eps/cell, held-out cells not used for
mixture, ≥3 training seeds, `CI95(ΔG)>0`, plus payoff/competence/behavior
gates. 4 eps/cell is `PROMISING_DIRECTION` only — never `ACCEPT`.

**Phase 3–6:** add specialists only if `G` rises again (2–3 enough); retention
`G_available,internal>0`; sparse router (no opponent ID) → `G_realized>0`;
headline routed LRO > matched non-latent → `G_latent>0`. Minimal ablations:
no-LRO / shallow heads / fixed-or-random router.

**Permanently closed as primary fixes:** entropy-as-diversity, persistence-as-
specialization, MI/JSD headline, soft OPÃ—map birth, contract-specialist glue,
router before `G_available>0`, 4-eps cell winners, archive fishing, 5u-as-final,
simultaneous reward+router+arch+pool changes, V6I26a–z carousel.

**Paper vs extension (locked 2026-07-23):** Summer borrows history-aware state
processing, population BR, payoff matrices, and repeated adaptation. The paper
does **not** explicitly predict or choreograph future states. Summerâ€™s
repertoire, persistent router, event-based re-selection, and any
latent-conditioned `VÌ‚(c_t,z)` (or multi-feature future predictors) are
**beyond-paper**. Phases 1–6 use only implicit foresight (learned returns +
temporal context). Explicit future-value choreography is optional **after**
the ladder gates — not a Stage-1/router prerequisite.

**Opponent tag discipline (locked 2026-07-23):** short ``OP6``..``OP12`` are
aliases only (`OPPONENT_ALIASES`). Audited LRO pool is
`LRO_AUDITED_OPPONENT_POOL` (seven tags, seven distinct role-gate fingerprints).
Do not put every registry key into a payoff matrix. Stage-1 redo must use this
pool end-to-end (`G_before` / train / `G_after` same generation).

**Stage-0 niches redo (COMPLETE 2026-07-24):**
`artifacts/v6i26_landscape_scan_niches_seed1/landscape_scan.json`

```text
unique_best=3 (balanced, failure_cells, high_variance)
G_available_point=0.0089
G_available_effective=-0.075
max_row_distance=1.13  (competence gap, not repertoire)
decision=MANUFACTURE_VIA_LRO_STAGE1
```

Reading: niche opponents work as distinct tests; archive is quality tiers
(strong generalist `balanced`, weak/broken others), not opponent-specific
specialists. Stage-1 must manufacture complementary value inside one latent
policy. Directional breakthrough = `ΔG = G_after − G_before > 0`; strategy
acceptance also requires the forced-z behavior nonredundancy gate above.

**Stage-1 Round-1 (2026-07-24) — 4-eps PROMISING_DIRECTION, superseded by Phase-2:**

```text
artifacts/v6i26_lro_niches_round1_seed1/ROUND1_LOCK.json
checkpoint: final_v6i26_lro_z3_r1_25u_seed1.zip
SHA256: 83B798574E7C084FF7A0DA3F1EA38EAB7A83C37168608A7ADDA35D38D5292AEC
G_before=0.107  G_after=0.286  ΔG=+0.179   (4 eps/cell — PROMISING_DIRECTION only)
```

Do **not** overwrite this zip. Ignore stale
`artifacts/v6i26_lro_round1_niches_seed1`. Router/distill still forbidden.
The 4-eps screen is **not** an `ACCEPT`; it is `PROMISING_DIRECTION` only
and is treated as **noise / overestimation** after Phase-2.

**Phase-2 seed-1 STRICT CONFIRM — `FAIL` (LOCKED — decisive reading 2026-07-25):**
`artifacts/v6i26_lro_niches_round1_seed1/phase2_confirm/phase2_seed1_confirm.json`

```text
Numeric promotion:   FAIL
Strategy separation: FAIL
Overall:             PHASE2_STRATEGY_HOLD_OR_FAIL

G_before=0.0223  G_after=0.0000  ΔG=-0.0223
CI95(ΔG)=[-0.183, +0.170]   CI95_low>0 = False
branch z=3 nearest z=2 dist=0.0824 (thresh 0.35)  — nearly duplicate
all-pair behavior mean ≈ 0.12 does NOT rescue the candidate
```

Independent failure modes: (1) no reliable repertoire gain under bootstrap;
(2) no distinct candidate behavior. This is **not** “promising but
underpowered.” Preserve `z_3` as a **negative** result — stop treating it as
a specialist. Raw oracle-margin cell variation ≠ promotion statistic `G`.

**Clean interpretation (locked):** niche pool exposed a **pre-existing**
context-dependent repertoire at init (Stage-C / unique winners / oracle gap
still stand). This 25u forced-z3 LRO round did **not** improve that repertoire
or birth a behaviorally distinct `z_3`. Causal claim “LRO manufactured or
strengthened a distinct strategy in seed 1” is **false**.

**Paper boundary (locked):** the niche surface revealed latent payoff
variation, but tested LRO procedures have **not** yet manufactured a
statistically valuable and behaviorally distinct strategy. No promotion, no
router, no strategy-birth claim.

**Seeds 2–3 — `ABORTED_BY_PROTOCOL` (2026-07-24):** stopped mid seed-2
`G_before` (~14/56 cells). Partial artifacts preserved under
`artifacts/v6i26_lro_niches_round1_seed2/` (`ABORTED_PROTOCOL.json`). Seed 3
never started. Rationale: seed-1 already failed proper 32-eps causal confirm;
seeds 2–3 would repeat the same flawed recipe (fixed `z_3`, 4-eps landscape
targets, saturated cells, no learning-signal diagnostics) and could at best
yield `PROMISING_DIRECTION`. Replication postponed until a mechanism is worth
replicating.

**Next compute (locked -- do this, not finish flawed seeds):**

```text
1. Fix/verify branch-KL + learning diagnostics     DONE
2. WR-based saturated exclusion dry-run            DONE (gates FAILED; correct refuse)
3. Read-only headroom audit (WR + margins)         DONE -> Case A
4. If useful sensitive headroom: select by margin/score (WR safety gate)
5. Else build harder matched strategic contexts (~60-80% best WR)
6. Lock new surface -> fresh forced-z baseline -> unsaturated target -> 5u only
```

**Margin-sensitive selector (2026-07-24):**
Primary metric = recoverable headroom ``best_margin - candidate_margin``;
WR is competence/safety only; TTC descriptive only.
Threshold calibrated from matched-seed median cell SE
(``max(0.15, 2 * median_se)``).

Dry-run on before_32:
`artifacts/v6i26_margin_selector_dryrun_seed1/`

```text
target=OP9_SPLIT_LANE_FEINT|map_b_split_lane
branch=z0  best_margin_z=3
sensitive_headroom=1.4375  threshold~0.309
best_wr~0.969  mixture 75/25
selection_gates.all_pass=true
```

5u diagnostic pilot training COMPLETE; learning_signal=NO_USABLE_LEARNING_PRESSURE
with **broken_link=PARAMS_MOVE_KL_FLAT** (chain audit, 2026-07-24).

```text
reward/adv     : rollout_adv_std ≈ 1.14  (alive; not flat)
critic         : critic_grad_norm ≈ 1.67 ≈ joint grad_norm  (critic-dominated)
freeze mask    : shared Δ=0; z1–z3 adapter Δ=0; only z0 moves  (OK)
z0 step        : adapter/embed max|Δ| ≈ 4e-4–7e-4  (tiny)
policy         : approx_kl ≈ 1.85e-5, clip=0
```

**Logit control-authority probe (COMPLETE 2026-07-25):**
`artifacts/v6i26_margin_pilot_5u_seed1/logit_control_authority_probe.json`
Script: `experiments/run_v6i26_logit_control_authority_probe.py`
Fixed OP9 obs batch (n=128); trunk baseline = identity (absent from init ckpt).

```text
init→trained KL ≈ 5.5e-4   argmax_disagree ≈ 0.47
module 1× replay (isolated on trained graph):
  trunk   dθ=0.230  KL=2.0e-4  authority≈0.93   (MOVED; not dead identity)
  head    dθ=0.114  KL=8.8e-5  authority≈1.22   (healthy)
  adapter dθ=0.054  KL≈1e-7    authority≈0.08   (α=0.1 throttles this path)
  embed   dθ=0.005  KL≈0
  combined 1× KL=5.6e-4
scaled combined: 0.5×→1.4e-4, 1×→5.6e-4, 2×→2.3e-3, 5×→1.9e-2, 10×→0.14
α forward sweep vs birth: 0.0→5.3e-4 … 1.0→1.1e-3  (mild; not primary)
reading=VALID_DIRECTION_OPTIMIZER_STEP_TOO_SMALL
inactive trunks z1–z3 still exactly identity (freeze OK)
```

Interpretation update: the update **direction** is logit-valid and scales
smoothly; the 5u step is simply too small for usable policy KL. Deep trunk is
**not** stuck at identity on z0. Residual α is a secondary throttle on the
adapter only — most movement already went to trunk+head. Do **not** auto-
continue to 10u. Do **not** add per-z critics yet. Do **not** treat this as a
KL-logger bug (1× replay matches birth→trained KL).

**Preferred next single retry (after post-eval docs):** actor-step / LR (or
separate actor clipping) ablation — not higher-α first, not trunk redesign.
Post-eval `forced_z_after` still finishing for OP9 margin / anchors / behavior /
drift documentation only.

**Actor-step ablation contract (LOCKED 2026-07-25 — revised gates):**
Preset `v6i26_actor_step` = separate z-actor/critic clip + 2× z-actor LR.
Fresh 5u from V6I23 init on locked OP9/`z0` surface
(`artifacts/v6i26_margin_actor_step_5u_seed1/`).

```text
Learning pressure (primary = fixed-batch probe KL, not training≈probe equality):
  fixed_batch_init→final_kl >= 1e-3   (authority-probe protocol)
  training approx_kl > weak floor 1.85e-5; finite; not explosive (<1)
  clip_fraction < 0.5 (safety; NOT required >0)
  entropy field = metrics CSV ``entropy`` (summed action heads)
    stable vs weak mean 2.637 ± 0.3
  actor/critic grads > 0; z0 Δθ clearly above weak; inactive Δ≈0

Strategic:
  OP9 margin improves; OP11/OP12 hold; z0 behavior distance increases

Decision:
  learning fail → stop optimizer ablation
  learning pass, OP9 fail → active but strategically wrong
  learning+OP9, behavior flat → response not strategy birth
  all pass → continue THIS ckpt to 10u only
```

**Actor-step 5u 2× result (LOCKED NEGATIVE 2026-07-25):**

```text
recipe         = separate clip + 2× z-actor LR
dir            = artifacts/v6i26_margin_actor_step_5u_seed1/
fixed_batch_kl = 7.6e-4   (< 1e-3 gate) FAIL
approx_kl_mean = 3.4e-5   (above weak floor, still tiny)
clip_fraction  = 0.0      (safety OK; not required >0)
entropy        = 2.635    (stable on CSV field `entropy`)
z0 Δθ max      ≈ 8e-4     (not clearly above weak ~7.7e-4) FAIL
inactive drift = 0        OK
learning_pass  = false
10u            = NO
```

Preserve checkpoint + reports as a **negative optimizer-control result**.
Do **not** relaunch the identical 2× recipe. Post-eval may finish for
documentation only and **cannot** overturn the failed learning gate.
Separate clip + 2× LR helped slightly vs weak (KL 5.6e-4→7.6e-4) but did not
clear the predeclared movement gate. No threshold retune, no α/arch/opponent/
router/critic-head changes from this run.

**Next clean causal test (LOCKED recipe — only mult changes):** fresh 5u from
the same V6I23 init, `v6i26_actor_step` + `--z-actor-lr-mult 3`, new dir
`artifacts/v6i26_margin_actor_step_3x_5u_seed1/`. Gates unchanged
(fixed-batch KL floor `1e-3`, ceiling `1e-2`). LR schedule preserves
`lr_mult` (`updater.py`).

Ledger hygiene (locked): intentional kills of the duplicate 2× relaunch and
weak-run post-eval are **not** experiment failures. Only the fresh 3× result
matters for the causal comparison.

**3× judge order / fork (LOCKED):**

```text
1. Movement: fixed-batch KL ∈ [1e-3, 1e-2]
2. Stability: entropy healthy; clip not saturated; z1–z3 unchanged
3. Strategic: OP9 margin ↑; OP11/OP12 anchors hold
4. Specialization: z0 behavior distance from peers ↑

KL < 1e-3              → 3× insufficient; no 10u
KL > 1e-2              → step too large; stop LR escalation; no 10u
KL OK, OP9 fail        → moves but strategically wrong; stop LR escalation
KL+OP9, behavior flat  → refinement, not strategy birth
KL+OP9+behavior pass   → continue THIS exact ckpt to 10u only
```

**Actor-step 5u 3× result (LOCKED NEGATIVE 2026-07-25 — ceiling):**

```text
recipe         = separate clip + 3× z-actor LR (z_actor_lr=1.5e-3)
dir            = artifacts/v6i26_margin_actor_step_3x_5u_seed1/
fixed_batch_kl = 1.113e-2  (>= floor 1e-3, FAILS ceiling 1e-2
approx_kl_mean = 3.35e-4
clip_fraction  ≈ 7.8e-4   (not saturated)
entropy        = 2.673    (stable)
z0 Δθ max      ≈ 2.4e-3   (above weak)
inactive drift = 0        OK
learning_pass  = false    (ceiling)
10u            = NO
LR escalation  = STOP     (step too large; do not climb to 5×)
```

Movement is no longer the bottleneck (2× under-floor → 3× over-ceiling).
3× post-eval cannot change the seed-1 Phase-2 strategy verdict and cannot
authorize continuation (learning-safety gate already failed). Do not retune
the KL window after seeing this result. Do **not** climb to another LR rung.

**Three clean findings (LOCKED 2026-07-25):**

```text
Original z3 Phase 2:   no G improvement + behavior redundant
2× z0 actor step:      movement too small   (KL 7.6e-4)
3× z0 actor step:      movement too large   (KL 1.113e-2)
```

Nonlinear KL vs LR (2×→7.6e-4, 3×→1.113e-2) forbids another multiplier rung.

**Next optimizer control (LOCKED — not an LR rung):** target-KL early-stop /
checkpoint ladder on the same valid OP9/`z0` surface:

```text
same 3× actor LR (1.5e-3); critic 5e-4; separate clip
checkpoint every 1u; measure fixed-batch init→ckpt KL
stop at first checkpoint inside [1e-3, 1e-2]
evaluate ONLY that predeclared checkpoint
dir: artifacts/v6i26_margin_actor_step_3x_kl_ladder_seed1/
script: experiments/run_v6i26_actor_step_kl_ladder.py
```

Unchanged: target, architecture, reward, α, opponents, router.

**Target-KL ladder result (LOCKED 2026-07-25):**
`artifacts/v6i26_margin_actor_step_3x_kl_ladder_seed1/`

```text
u1 KL=2.26e-4  (below floor)
u2 KL=6.93e-4  (below floor)
u3 KL=1.51e-3  ← SELECTED (first in [1e-3, 1e-2]); early stop
z0 nearest peer = z2 dist≈0.0018  (≪ 0.35)  behavior FLAT
fork = KL pass + behavior flat → refinement, not strategy birth
```

Ignore forced_z_eval Stage-C banner “proceed to router training” — **not
authorized**. No router, no 10u on specialization grounds, no further LR rung.
Paper boundary unchanged.

**Usable-selector eval (LOCKED 2026-07-25) — `USABLE_REPERTOIRE_HOLD_OR_FAIL`:**
`artifacts/v6i26_margin_actor_step_3x_kl_ladder_seed1/usable_selector_eval_u3.json`
(z0 KL-ladder u3 vs locked z3 25u; 32 eps/context; leakage-free legal `c0` selector)

```text
V_z0=1.1875  V_z3=1.2875  best_fixed=1.2875
V_hindsight_oracle=1.3875  delta_oracle=+0.100  LCB>0  (upper bound only)
V_legal_selector=1.1875    delta_usable=-0.100  LCB=-0.238  FAIL
held-out picks: z0=80 / z3=0   (selector never chooses z3)
selection labels: z3_better=4 / z0_better_or_tie=76
```

Hindsight complementarity exists; **deployable** repertoire selection does not.
Do **not** promote, do **not** train router.

**Cross-checkpoint policy distinction (COMPLETE 2026-07-25):**
`diagnose_v6i26_cross_checkpoint_divergence.py`
z0(u3) vs z3(25u) on shared obs batch (n=1024):

```text
logit_L2 mean≈2.91  (distinct band; same-ckpt clones were ~0.3–0.4)
argmax disagree = 1.0 on all heads
JSD ≈ 0.008–0.016
```

Logit distinction **passes**; usable legal-selector **fails**. Combined
**Level-1 verdict = FAIL** (`LEVEL1_CLASSIFICATION.json`). No authorized
follow-on training from this pair — next train needs a new locked LRO birth
recipe, not router / not another LR rung.

**z1 OP8/OP12 v2 target-KL ladder screen (LOCKED 2026-07-25) -- `PROMISING_DIRECTION_NOT_ACCEPT`:**
`artifacts/v6i26_z1_op8_op12_v2_kl_ladder_seed1/`

Locked target recipe:
`artifacts/v6i26_z1_op8_op12_v2_locked_recipe_seed1/locked_response_target.json`
from the existing 32-episode forced-z matrix. Target branch is `z1`; target
contexts are OP8 protected-carrier escort and OP12 late-converter on
`map_b_split_lane_v2`; anchors are OP7 split-lane and OP10 split-lane-v2.

```text
u1 KL=3.01e-4  (below floor)
u2 KL=6.30e-4  (below floor)
u3 KL=7.36e-4  (below floor)
u4 KL=1.17e-3  <- SELECTED (first in [1e-3, 1e-2]); early stop
screen sample = 4 eps/cell only
best_fixed_z = z1
oracle WR = 1.000  best-fixed WR = 1.000  WR advantage = 0.000
oracle margin = 1.625  best-fixed margin = 1.375  margin advantage = +0.250
behavior_pair_distance_mean = 0.1299  max = 0.2182  threshold = 0.35
unique best-z values = [0, 1]
```

Strict classification artifact:
`artifacts/v6i26_z1_op8_op12_v2_kl_ladder_seed1/LEVEL1_CLASSIFICATION.json`.
The run is a movement-controlled screen only. It does **not** prove strategy
birth: no win-rate improvement over best fixed, no behavior-distance pass, no
32-episode confirmation, no CI pass, and no multi-seed repetition. Ignore any
legacy Stage-C router-training banner from this 4-episode path. Router remains
blocked.

Seed-1 action-level divergence diagnostic:
`artifacts/v6i26_z1_op8_op12_v2_kl_ladder_seed1/policy_divergence_z1_u4.txt`.
Same-checkpoint observation batch, n=1024:

```text
z0 vs z1: logit_L2=0.529  argmax=[0.499,0.500,0.480,0.728]  JSD~1.75e-4..3.18e-4
z1 vs z2: logit_L2=0.547  argmax=[0.540,0.999,0.777,1.000]  JSD~2.28e-4..3.43e-4
z1 vs z3: logit_L2=0.629  argmax=[0.459,1.000,0.379,1.000]  JSD~1.44e-4..4.52e-4
```

Interpretation: z1-u4 is above the local duplicate-policy band (~0.3-0.4 L2)
but far below the prior strong distinct z3 cross-checkpoint band (~2.9 L2),
with tiny JSD. Record as weak-to-moderate policy divergence, not a copy, and
not strategy proof. This partially softens the 7-D behavior failure but does
not override the failed WR complementarity, small sample count, missing CI, or
missing replication.

Replication status: no completed z1 seed2/seed3 KL-ladder replicas exist. Only
the seed1 V6I23 init checkpoint is present under artifacts/checkpoints; the
older seed2 z3 recipe artifact is marked `ABORTED_BY_PROTOCOL`. A clean
seed2/seed3 replication therefore needs either matching V6I23 seed2/seed3 init
checkpoints or an explicit decision that "seed2/seed3" means same seed1 init
with different rollout/training RNG.

**One-seed closeout (LOCKED 2026-07-25):**
`artifacts/v6i26_z1_op8_op12_v2_kl_ladder_seed1/CLOSEOUT.json`

```text
status = STOPPED_ONE_SEED_WEAK_DIRECTIONAL_SCREEN
optimization control = PASS
target margin direction = PROMISING
policy divergence = WEAK_TO_MODERATE
strong strategy distinction = FAIL
coarse behavior gate = FAIL
win-rate complementarity = NOT_SHOWN
Level 1 = NO
```

Current one-seed decision: preserve as a negative / weak directional screen.
Do not launch router training, strict 32-episode confirmation, per-z value-head
retry, or seed replication from this result alone. The seed2 V6I23 init artifact
created during replication-prep is not part of this one-seed closeout and should
not be used to reinterpret seed1.

**Next target nomination (LOCKED 2026-07-25) -- `NOMINATED_NOT_LAUNCHED`:**
`artifacts/v6i26_z1_op11_split_nomination_seed1/TARGET_NOMINATION.json`

Current 32-episode payoff matrix was rescored with `z3` treated as the incumbent
distinct policy. The only non-closed context where a generalist-cluster branch
beats `z3` by calibrated margin headroom is:

```text
target = OP11_ADAPTIVE_EXPLOITER|map_b_split_lane
branch = z1
z3 margin / WR = 1.03125 / 0.875
z1 margin / WR = 1.34375 / 0.96875
margin headroom vs z3 = +0.3125
required headroom = 0.3045
```

Proposed locked recipe if this is launched later: target OP11 split-lane at
75%, anchors OP11 split-lane-v2 / OP10 split-lane-v2 / OP7 split-lane at 25%
total, same 3x actor-step KL ladder, first checkpoint entering `[1e-3, 1e-2]`.
Immediate screen must include action-level divergence and two-branch
complementarity against incumbent `z3`. No training launched from this
nomination.

**Permanent keep — screening vs ACCEPT (locked):**

```text
4 eps/cell
→ screening only
→ PROMISING_DIRECTION

≥32 eps/cell + CI95 lower bound > 0
+ payoff nonredundancy
+ competence
+ behavioral separation
+ replication across ≥3 seeds
→ ACCEPT
```

Executable guardrail: Stage-1 screening cannot emit `ACCEPT` from the default
4 eps/cell path. Live seed-2/3 artifacts stay untouched mid-flight.

**Seed-1 failure diagnosis (locked 2026-07-24):**
`z_3` was trained on contexts it already mostly solved (OP8/OP10/OP11 near
ceiling; `z_3` already best fixed globally under 32-eps init). Almost no
economic reason to invent a new behavior → stayed in generalist basin
(dist to `z_2` = 0.082). Eval surface also near-saturated (~96% best-fixed
vs ~99% oracle WR) → little win-rate headroom.

Tiny behavior distance alone does **not** diagnose shared-critic failure.
It can come from: saturated target, wrong branch, near-zero advantages,
gradient/freezing bug, **or** shared-critic interference. A per-`z` critic
addresses only the last.

**Post multi-seed protocol (locked 2026-07-24 — senior-RL order):**

```text
seeds 2–3 finish unchanged
→ 1. Analyze multi-seed failure pattern
→ 2. Fix KL and inspect learning-signal diagnostics
→ 3. Recompute targets from current 32-eps forced-z payoff matrix
→ 4. Exclude saturated contexts
→ 5. Choose uncovered context + branch with headroom
→ 6. One frozen-branch response round with competence anchor
→ 7. Evaluate 5u-interval checkpoints (see pinning rules below)
→ 8. Strict causal + behavioral ACCEPT only
→ 9. Per-z value heads ONLY if precondition below holds
→ 10. Router ONLY once two strategies pass
```

**Per-`z` value-head precondition (tightened — not automatic on tiny distance):**

> tiny behavior distance **and** meaningful advantages **and** healthy
> gradients **and** properly selected unsaturated target
> → one controlled per-`z` value-head retry.

Do **not** jump to architecture because distance is tiny; that masks
target-selection failures.

**5u checkpoint pinning rules (implementation must obey):**

1. Checkpoint eval uses **identical matched seeds** and must **not** alter
   optimizer or RNG state for continuing training.
2. Final report records the **best predeclared checkpoint selection rule**,
   not a retrospective pick on the confirmation set. Clean rule:

```text
screen checkpoints at 4 eps/cell
nominate one checkpoint
confirm only that nominated checkpoint at ≥32 eps/cell
```

5u saves reveal whether a specialist never separates, separates then
collapses, improves early then overtrains, or becomes different without
becoming useful.

**Vs V6I24:** adaptive task-return response targets; iterative BR; four
branches in one model; no strategy-specific reward; distill optional only
after niche PASS.

**Implementation:** preset `v6i26` / `v6i26_lro`,
`experiments/v6i26_lro_core.py`, runners under `experiments/run_v6i26_*`,
`tests/test_v6i26_latent_response_oracle.py`.

### 3.36 v6i25 counterfactual-router diagnostic -- `FAIL_SIGNAL` (2026-07-23)

**Status:** closed as smoke `FAIL_SIGNAL` (optional larger-n confirm later).
V6I24 is now primary.

**Scientific question:** Is the existing Stage-C / oracle gap **predictable
from episode-start geometry** (permitted Summer context), and can
`q_phi(z|c)` recover that predictable gap-

```text
geometry c → z*(c) → return     (cross-fitted context oracle)
geometry c → q_phi(z|c) → R     (learned router)
```

**Not** the question answered by Path C (four independent teachers).

**Fidelity:** `DIAGNOSTIC` (Summer-compatible intent; not PAPER-FAITHFUL —
counterfactual all-z labels are unavailable to on-policy PPO).

**Corrected protocol (locked):**

1. Load V6I23 donor; freeze actor / adapters / per-z heads / critic;
   **reinitialize `q_phi` fresh**.
2. Matched-seed forced-`z` table for OP8–OP12 Ã— both maps; capture **real**
   episode-start `global_state` (fail loudly if missing / non-finite /
   all-zero / unique contexts â‰¤ 1). **No opponent ID** in router input;
   conflicting opponents under the same geometry are averaged into
   `QÌ‚(c,z)`.
3. **Stage A (signal gate):** on train seeds
   `z*(c)=argmax_z E[R|c,z]`; evaluate `R_heldout(c,z*(c))` vs best-fixed
   chosen on train. Require paired bootstrap CI for
   `(context-oracle − best-fixed)` excluding zero. If not → `FAIL_SIGNAL`
   (stop; do not train router).
4. **Stage B:** soft targets `p*(z|c)=softmax(QÌ‚_train(c,z)/Ï„)`;
   `L=−Î£ p* log q_Ï†`. Centered-advantage loss retained as ablation helper
   only. Ignore rows with negligible Q spread.
5. Held-out: router vs best-fixed vs uniform vs **cross-fitted** context
   oracle (never per-episode hindsight `max_z R`).
6. Fresh online rollouts on unused seeds.

**Gap recovery:**
`(R_router − R_best_fixed) / (R_context-oracle − R_best_fixed)`.

**Verdicts:**

| Verdict | Meaning |
|---------|---------|
| `PASS` | Stage A OK **and** router > best_fixed (CI) **and** recovery ≥ 50% |
| `PARTIAL` | Stage A OK, router > best_fixed, recovery < 50% |
| `FAIL_SIGNAL` | Context oracle cannot beat best_fixed → resume V6I24 / birth |
| `FAIL_ROUTER` | Stage A OK but router fails → fix `q_phi` / geometry encoding |

**Donor:** V6I23 Stage-C PASS zip (not V6I24).
**Implementation:** `rl/router/counterfactual_router.py`,
`experiments/run_v6i25_counterfactual_router_diagnostic.py`,
`tests/test_v6i25_counterfactual_router.py`.

**Launch (smoke):**

```text
uv run python experiments/run_v6i25_counterfactual_router_diagnostic.py \
  --checkpoint artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip \
  --output-dir artifacts/v6i25_cf_router_smoke_seed1 \
  --episodes-per-cell 8 --device cuda
```

**Smoke result (2026-07-23, 8 eps/cell):** `FAIL_SIGNAL` — decisive for
this checkpoint.

```text
Cross-fitted context oracle: 75%
Best fixed latent z2:         75%
Available routing gain:        0%
CI:                            [0%, 0%]
```

**Interpretation (locked):**

* Controllability: yes (`z0` vs `z2` moves outcomes).
* Competence: no (`z0` consistently weak).
* Comparative advantage: no (held-out geometry selection does not beat
  always-`z2`).
* Router utilization: not testable — no stable gain to harvest.

Latent structure ≈ one damaged branch + several near-equivalent strong
general modes — **quality differences, not strategic niches**. Apparent
cell winners in the raw WR table were mostly ties / 8-game noise; they
did not survive cross-fitting. Do **not** train the router longer,
enlarge it, add opponent IDs, increase `K`, or claim an oracle gap proves
strategies. Collapse framing: on-policy positive feedback from router
selection + shared team reward (not â€œPPO argmax creditâ€).

**Next:** V6I24 repertoire birth (§3.35). Optional 32–64 eps/cell confirm
of `FAIL_SIGNAL` is fine but not required before resuming Path C.

### 3.12-prerun v6i13 delayed-commit opening-window advantage router (implementation + smoke)

**Scientific delta vs v6i12 (plain English):** v6i12 refuted the hypothesis
"episode-start context can explain enough return variance for V/A routing"
(`baseline_r2` plateaued at ~0.03). v6i13 tests the better hypothesis: **the
first 32 decision steps reveal the missing routing information.** The router now
waits, observes the opening, then commits.

**Core contract (delayed commit):**

```text
steps 0–31:  execute a UNIFORMLY sampled warmup latent (no z gets a default edge)
step 32:     commit one router-selected latent; build opening-window context
post-commit: hold committed z to terminal; arc_return = POST-COMMIT return
context:     opening_context = [state_0, state_commit, state_commit - state_0]
             concatenated with the opponent one-hot (3*GLOBAL_STATE_DIM + 3)
```

**Fidelity classification:** SUMMER-COMPATIBLE EXTENSION. Preset
`apply_plan_faithful_latent_v6i13_opening_window_advantage_router` re-parents
from `v6i12_advantage_router_hardpool` and adds four keys
(`latent_episode_strategy_warmup_decision_steps=32`,
`router_warmup_uniform_z=True`, `router_arc_post_commit_only=True`,
`router_opening_context_mode="initial_commit_delta"`) plus
`experiment_id`/`run_tag`. The internal PPO router stays disabled exactly as in
v6i12; the external V/A diagnostic learns only from online post-commit returns.
No labels, opponent-ID supervision head, forced-z oracle target, hindsight
best-z target, auxiliary task, or actor training. Aliases include `v6i13`,
`v6i13_opening_window_advantage_router`,
`plan_faithful_latent_v6i13_opening_window_advantage_router`. Experiment
`experiments/run_v6i13_opening_window_advantage_router.py`; reuses the v6i12
external model (`rl/router/advantage_router.py`) with a 3Ã—-wide opening context.
Pinned by `tests/test_v6i13_opening_window_advantage_router.py` (6 cases).

**1-update smoke (preserved at
`artifacts/v6i13_opening_window_advantage_router_smoke_seed1/`):** the first real
evidence that richer temporal context helps. `baseline_r2 = 0.1266` (vs v6i12's
0.03 plateau), `advantage_target_std = 0.9317` (vs v6i12's 0.9995), commit locked
at step 32 (`commit_step_min=max=32`), all pipeline gates clean
(`records_after=0`, `dup=0`, `terminal_frac=1.0`, all four z, `frozen_actor_ok`).
Verdict `FLAT` as expected at 1 update (CIs wide with n≈30–44/cell).

**Implementation order (pre-registered):** (1) preserve smoke [done]; (2) run
5-update mechanism test [RUNNING, `artifacts/v6i13_opening_window_advantage_router_5u_seed1/`];
(3) add `map_id` to arc records if 5-update is promising; (4) 20-update run;
(5) held-out delayed-router eval only after ≥ `WEAK_SEPARATION`. Do NOT build a
GRU/history encoder yet — `[s0, s32, delta]` is the simplest surface; escalate
only if it fails.

**5-update pass gates:** `baseline_r2 > 0.05` consistently; `adv_std` below
v6i12, ideally `< 0.95`; ≥2 opponents/cells with reliable advantage gaps;
A-router not choosing the same z everywhere; all z sampled post-commit; frozen
actor hash unchanged. Real behavioral gate (later):
`delayed A-router > cross-episode-shuffled delayed A-router`.

**Next step:** await the running 5-update mechanism test; judge on `baseline_r2`
consistency (>0.05) and whether advantage gaps begin to separate. `map_id`
threading is the immediate follow-up if promising.

### 3.11 v6i12 paired-advantage router — `EVALUATED` (20-update diagnostic = FLAT on a VALID dataset; baseline_r2 plateaued at ~0.03)

**Result (2026-07-04, `artifacts/v6i12_advantage_router/summary.json`, seed 1,
20 updates, 9384 arcs):** `FLAT`, 0/3 opponents reliably separating. The
dataset is fully valid — `dup_rejected=0` every update, `terminal_frac=1.0`,
all four z represented (`{2327,2322,2367,2368}`), `min_cell_arcs=742`,
`frozen_actor_ok=True`. So this is a trustworthy negative, not a tooling
failure.

The decisive finding is at the **baseline stage, not the advantage stage.**
`baseline_r2` rose from `+0.0008` (u1) but plateaued at `~0.03` (final
`+0.031`; oscillated 0.024–0.037 over u14–u20). `advantage_target_std` fell
only from `0.9995` → `0.984` — a ~1.6 % variance reduction, exactly what
`sqrt(1 - 0.031)` predicts. Final advantage-gap CIs all include zero (OP8 gap
`+0.031` CI `[-0.052,+0.110]`; OP9 `+0.023` CI `[-0.077,+0.119]`; OP10 `+0.019`
CI `[-0.082,+0.124]`), and empirical advantage spreads compressed to ~0.09 as
per-cell counts grew past 700 (noise-inflated early spreads regressed toward the
true small effect, same pattern as v6i11).

**Interpretation — sharper diagnosis than v6i11.** v6i12's core bet was that
episode return variance is dominated by a *context-level* component that
`V(context)` could absorb, leaving a clean latent residual. The `baseline_r2 ≈
0.03` refutes that: the 34d episode-start geometry + opponent one-hot is nearly
uninformative about the eventual episode return. The variance that swamped v6i11
is therefore **within-context aleatoric variance** — how the episode actually
unfolds — which *no* baseline conditioned only on episode-start context can
remove. Double-centering cannot help when the baseline itself has almost nothing
to explain. The A-router leaned toward a mild global z1/z3 preference rather than
contextual routing, and its argmax disagreed with the empirical best-z in 2/3
opponents at the final update.

**Fork taken (per the pre-registered decision rule):** `FLAT` →
do NOT run the held-out prospective evaluator; the promotion gate requires at
least `WEAK_SEPARATION`. The next fix is richer context, not more updates or a
return to PPO-router credit: add `map_id` instrumentation to the arc record and
give V/A a **history/temporal encoder** (episode-start context alone is too weak
and too aleatoric). A scalar `A(context, z)` reparameterization and longer V/A
training are secondary levers. Only after context is enriched does re-running the
diagnostic make sense.

**Clarity fix (post-run):** `experiments/run_v6i12_advantage_router.py::_PRESET`
now uses the `v6i12_advantage_router_hardpool` alias so future launch banners /
run_tags advertise v6i12; this in-flight run's `summary.json` still records the
v6i11 alias (harmless — the resolved config is identical except
experiment_id/run_tag, pinned by `test_minimal_diff_vs_v6i11`).

---

**(pre-run notes below, retained for provenance)**

### 3.11-prerun v6i12 paired-advantage router — implementation + smoke

**Scientific delta vs v6i11 (plain English):** v6i11 regressed the *raw*
normalized episode return `Q(context, z)`; its 15-update diagnostic was a clean
`FLAT` because the ~2.6–3.9 std of episode-level return variance swamped the
0.15–0.26 per-z mean differences, so best-vs-second bootstrap CIs included zero.
v6i12 keeps the identical data-collection contract and adds a double-centering
external regressor:

```text
1. Global:  norm_ret = (episode_return - batch_mean) / (batch_std + eps)
2. Context: a_target = norm_ret - stopgrad(V(context))
Route: argmax_z A(context, z)
```

`V(context)` (a `ContextualVBaseline` MLP) absorbs the context-level return
component; `A(context, z)` (an `AdvantageRouter` MLP) isolates the latent
residual. This matches the original oracle evidence, which was a within-context
*paired* contrast, not a raw between-episode mean.

**Fidelity classification:** SUMMER-COMPATIBLE EXTENSION. The v6i12 preset
(`apply_plan_faithful_latent_v6i12_advantage_router_hardpool`) re-parents from
`v6i11_q_router_hardpool` with a resolved-config diff of **exactly two keys**
(`experiment_id`, `run_tag`); the trainer-side arc-collection contract is
byte-identical (frozen actor, episode-persistent one-z-per-episode, 50 % uniform
exploration, `latent_arc_credit_coef = router_ent_coef = latent_lam_h =
latent_lam_p = 0`). All learning is in the EXTERNAL diagnostic model from online
sampled returns — no forced-z oracle labels, no best-z supervision, no
opponent-ID prediction head. Pinned by `tests/test_v6i12_advantage_router.py`
(`test_minimal_diff_vs_v6i11`, 11 cases total).

**Aliases:** `v6i12`, `v6i12_advantage_router`,
`v6i12_advantage_router_hardpool`,
`plan_faithful_latent_v6i12_advantage_router_hardpool`. Experiment
`experiments/run_v6i12_advantage_router.py`; external model
`rl/router/advantage_router.py` (`ContextualVBaseline`, `AdvantageRouter`,
`train_advantage_router`, `advantage_gap_ci`, `advantage_matrix_from_replay`).

**1-update smoke (2026-07-03, `artifacts/v6i12_advantage_router_smoke_seed1`):**
PASSED every pipeline/wiring gate — `records_before=460`,
`records_after=0`, replay `0→460`, `dup_rejected=0`,
`terminal_finalized_fraction=1.0`, all four z sampled, V/A losses finite
(`0.995`/`0.367`), `v_grad_norm=0.047`, `a_grad_norm=0.110`, frozen-actor hash
unchanged. **Caveat:** the headline variance-reduction metric was null at
update 1 — `baseline_r2 = 0.0008`, `advantage_target_std = 0.9995` (vs the
unit-std normalized return). That is expected with only 460 samples over 20
gradient steps; the mechanism is proven correct by the unit test (drives
`baseline_r2 > 0.5`, `adv_target_std < 0.9` on context-predictive data). Whether
the *real* episode-start context can predict episode return is precisely what
the 20-update run tests.

**Leading indicator to watch:** `baseline_r2` across updates. If it climbs
above ~0 (V absorbs return variance) and `advantage_target_std` falls below 1.0,
the double-centering is working and advantage gap CIs may survive. If `baseline_r2`
stays near zero, that is the "episode-start context alone is too weak/noisy"
outcome — the next fix is adding `map_id` and possibly a history encoder, NOT
returning to PPO-router collapse.

**Verdict / promotion contract:** identical 5-state semantics as v6i11
(`INVALID`/`INSUFFICIENT_DATA`/`FLAT`/`WEAK_SEPARATION`/`SEPARATING`), scored on
the advantage gap CI (spread threshold lowered to 0.05 because advantages are
V-centered). `SEPARATING`/`WEAK_SEPARATION` → `SEPARATING_CANDIDATE` only;
promotion still requires the held-out prospective gate
(`A-router > cross-episode-shuffled-A-router`, then `> uniform`, then
approaches/beats fixed-z2). Held-out evaluator to be built only after a
`WEAK_SEPARATION`-or-better verdict; `map_id` instrumentation deferred until
after the diagnostic.

**Next step:** await the running 20-update diagnostic
(`artifacts/v6i12_advantage_router/summary.json`, seed 1). Judge on the
`baseline_r2` trajectory and whether ≥2 opponents' advantage gap CIs exclude
zero.

### 3.12 v6i13 opening-window advantage router — `IMPLEMENTED, PENDING_SMOKE`

**Scientific delta vs v6i12:** v6i12 asks the router to explain returns
from the episode-start context. v6i13 delays commitment until decision
step 32, after the opening has exposed movement, pressure, first-contact,
and flag-state deltas. The replay context is
`[state_0, state_commit, state_commit - state_0]`; the target is
post-commit return.

**Fidelity classification:** `SUMMER-COMPATIBLE EXTENSION`. The preset
(`apply_plan_faithful_latent_v6i13_opening_window_advantage_router`)
inherits v6i12 and changes exactly
`{experiment_id, latent_episode_strategy_warmup_decision_steps,
router_arc_post_commit_only, router_opening_context_mode,
router_warmup_uniform_z, run_tag}`. Actor and z-specific repertoire
parameters remain frozen; the internal router PPO remains disabled; the
external V/A model still learns only from online sampled returns.

**Mechanism contract:** steps 0..31 use a uniformly sampled warmup latent;
step 32 commits one router-selected latent; the committed z is held to
terminal; no warmup arc is inserted into replay; finalized records carry
`commit_step`, `opening_context`, selected `z`, post-commit `arc_return`,
opponent id, terminal reason, and `arc_uid`.

**Immediate smoke gates:** commit step reached for most episodes,
`commit_step` equals 32 for normal terminal records, one post-commit arc
per completed episode, `terminal_finalized_fraction` near 1.0, all four z
values sampled, no duplicate arc insertions, frozen actor hash unchanged,
`baseline_r2 > v6i12 baseline_r2`, and `advantage_target_std < v6i12`.

**Files:** preset and runtime are pinned by
`tests/test_v6i13_opening_window_advantage_router.py`; diagnostic entry
point is `experiments/run_v6i13_opening_window_advantage_router.py`.

### 3.9-orig v6i10 episode-router exploration preset (original PENDING_SMOKE notes)

**Status:** `IMPLEMENTED, PENDING_SMOKE`. Preset committed as
`v6i10_episode_router_explore_hardpool` (aliases `v6i10`,
`v6i10_episode_router_explore`,
`latent_v6i10_episode_router_explore_hardpool`,
`plan_faithful_latent_v6i10_episode_router_explore_hardpool`),
`SUMMER-COMPATIBLE EXTENSION`, parent
`v6i9_mapaware_router_feedforward_hardpool`.

**Scientific delta:** simplify router learning to one legal initial
context, one `z`, one full episode, one return. The v6i9 repertoire
checkpoint remains the experimental anchor:
`final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip`.
Actor and z-specific repertoire parameters stay frozen through
`v6i9_training_stage = "router"` and `router_freeze_actor = True`.

**Resolved diff vs feedforward parent:** exactly
`{experiment_id, h_mode, latent_arc_credit_baseline,
latent_arc_credit_enabled, latent_arc_credit_min_len,
latent_entropy_anneal_end, latent_entropy_anneal_start,
latent_entropy_mode, latent_entropy_objective, latent_lam_h,
latent_lam_h_end, latent_lam_p, latent_resample_every_n,
latent_strategy_ppo_coef, learning_rate, router_ent_coef,
router_uniform_exploration_prob, run_tag, strategy_interval}`.

**Mechanism contract:** `latent_resample_every_n = 0`,
`strategy_interval = 0`, `latent_strategy_ppo_coef = 0.0`,
`latent_arc_credit_enabled = True`,
`latent_arc_credit_baseline = "running_mean"`,
`latent_arc_credit_min_len = 1`, `learning_rate = 1e-4`,
`router_uniform_exploration_prob = 0.20`, `router_ent_coef = 0.002`,
`latent_lam_h = latent_lam_h_end = 0.015`,
`latent_entropy_mode = "marginal"`, and
`latent_lam_p = 0.0`.

**Immediate smoke gates:** all four z sampled in the behavior policy,
router gradients positive, frozen actor/z hashes unchanged, episode
credit finite, running-mean baseline active, and behavior log-probs
computed under `0.8 * q_phi + 0.2 * Uniform` rather than raw q_phi.

**Five-update mechanism gates:** no deterministic z above 80 percent for
two consecutive updates, at least two argmax z values, high marginal
entropy, falling conditional entropy, MI proxy above noise, and no
exploding logit margin. Hard stop: one z reaches 100 percent argmax for
two consecutive updates.

---

## 4. Open decisions

| ID  | Question                                                                                                          | Owner action                                                                                                                                |
|-----|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| D1  | Should the canonical entropy interpretation be conditional or marginal-                                           | Closed: v5i6 makes batch-marginal entropy canonical; v5i4/v5i5 remain conditional comparison rows. |
| D2  | The `_2m_` artifact-history suffix on the v5i4 in-flight run — rename or leave-                                   | Leave. Renaming changes embedded `run_tag` metadata; per [`latent-preset-registry.md`](latent-preset-registry.md) §7.1 we route by filename. |
| D3  | Re-launch `no_latent_v4i3_baseline` at v5i4's exact budget / seed, or rely on the v4i3-budget baseline already on disk- | Re-launch. The §1 invariants in `experiment-and-evaluation-protocol.md` require matched budget and seed for a headline comparison.           |
| D4  | If v5i6's routing-quality control (`router` vs `random-matched`) does **not** show a significant Δ, does the paper claim survive- | Open. Acceptable answers: (a) report the canonical row as `PAPER-FAITHFUL but inconclusive`; (b) escalate to an explicitly named extension and rerun. |
| D5  | Should `latent_strategy_ppo_coef = 0.10` (`c_Z`) be swept-                                                        | Open. Recorded as O3 in `summer-fidelity-rules.md` §7. A sweep is `SUMMER-COMPATIBLE EXTENSION` if any value â‰  `0.10` is used in a headline row. |
| D6  | Update [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §3's `GLOBAL_STATE_DIM = 19` paragraph to match the current `GLOBAL_STATE_DIM = 34` / `CONTEXT_STATE_DIM = 170`. | Open (O1 in `summer-fidelity-rules.md` §7). Code is authoritative; doc paragraph needs an update note.                                       |

---

## 5. Closed decisions (for audit history)

| ID  | Decision                                                                                                                              | Date         | Where recorded                                                                                                                                                       |
|-----|----------------------------------------------------------------------------------------------------------------------------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1  | v5i4 became the first operational paper-faithful conditional-entropy row; v4i3 is the arc-credit row, not the headline paper-faithful row. Superseded for canonical launch priority by C7. | 2026-06-15   | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.7. |
| C2  | The main-loop gate must trigger off `latent_router_optimizer is not None`, not off `latent_strategy_ppo_coef == 0`.                    | (pre-v5i4)   | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.4; `tests/test_marginal_baseline.py::MainLoopGatingTests`.                                       |
| C3  | The actor must read `z` only via `nn.Embedding(K, d_z)` concat in any paper-faithful row (FiLM / adapter / one-hot OFF).               | (with v5i4)  | [`summer-method-spec.md`](summer-method-spec.md) §5; `tests/test_v5i4_paper_faithful.py::V5i4ConcatOnlyActorTests`.                                                  |
| C4  | The forced-z resolver must be a pure function of `cfg` and the passed `global_step` (so resumes pick up the schedule correctly).       | (with v5i3)  | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.6; `tests/test_forced_z_anneal.py`.                                                              |
| C5  | v5i4's `run_tag` flips `_2m_` → `_1m_` so the tag advertises the actual `total_timesteps`. v5_strict_summer / v5i1 / v5i2 / v5i3 keep `_2m_` to preserve artifact paths. | 2026-06-15   | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §7; [`latent-preset-registry.md`](latent-preset-registry.md) §7; `tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests`. |
| C6  | In pool mode, the first env reset must use the first pool entry, not `cfg.fixed_opponent_tag` when the latter is out-of-pool.          | 2026-06-15   | `rl/train_ppo.py::_resolve_initial_opponent_and_phase`; `tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests`.                          |
| C7  | v5i6 is the canonical paper-faithful Summer interpretation: entropy protects the batch-marginal strategy repertoire while v5i4/v5i5 remain conditional-entropy comparison rows. | 2026-06-16   | [`summer-method-spec.md`](summer-method-spec.md) §8/§12; [`latent-preset-registry.md`](latent-preset-registry.md) §2/§6.9; `tests/test_v5i6_paper_faithful_marginal_entropy.py`. |
| C8  | v5i6's marginal entropy MUST be aggregated over the **full rollout** resample subset, not per-PPO-minibatch. The per-minibatch path is provably an upper bound on the intended objective by Jensen and the gap is closed by softening individual `q_phi(z\|s)` toward uniform — the conditional regression v5i6 was designed to replace. Implementation: [`rl/latent_losses.py::rollout_marginal_entropy_loss`](../rl/latent_losses.py) called once per PPO inner epoch from [`rl/custom_ppo/ppo_updater.py`](../rl/custom_ppo/ppo_updater.py); deprecated `strategy_marginal_entropy_loss` kept only for parity tests. | 2026-06-16 | [`summer-method-spec.md`](summer-method-spec.md) §8.1; `tests/test_latent_losses.py::RolloutMarginalEntropyLossTests` (Jensen demo); `tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6RolloutMarginalEntropyContractTests`. |

---

## 6. Recommended next experiments (priority-ordered)

### 0. Four-niche payoff surface (LOCKED 2026-07-27) — before any latent / router GPU

**Status:** latent training and router work **STOPPED**. Immediate work is
scripted niche construction only.

**Canonical niches:** OP6→TURTLE, OP9→SPLIT, OP11→ESCORT, OP12→RUSH.
OP7/OP8/OP10 = optional later robustness, not unique-BR requirements.

**Current blocker evidence:** OP6–OP10 joint acceptance
(`artifacts/op6_op10_br_diversity_acceptance_16seed`) FAIL — SPLIT uniquely
best on all five; `LCB(delta_pool) ≤ 0`.

**Now:** parallel OP11 (ESCORT) and OP12 (RUSH) held-out niche gates, then
revisit OP6 TURTLE, confirm OP9 SPLIT, then full four-column matrix +
`LCB95(delta_pool)>0`, then independent PPO specialist oracle, then LRO.

**Exact question for the four-column pool:**

> Do OP6/OP9/OP11/OP12 force blue to need different strategies, or can one
> strategy beat almost all of them?

Judge by blue best-response diversity + `LCB(delta_pool)>0`, not red BT
fingerprints. Sample future training 25% per niche.

* Module: `experiments/payoff_matrix_analysis.py`
* Collector: `experiments/run_scripted_style_payoff_matrix.py`
* Blue styles: `gpu_env/_core/_scripted_blue_styles.py`
* Gates: `all_blues_protected` + `delta_pool_lcb_positive` (+ support gates);
  `--min-br-diversity 4` for the four-style claim.
* Pinned by `tests/test_payoff_matrix_analysis.py`.
* Do not retrain latents until steps 1–7 of the Direction lock clear.

**First collector run checkpoint (2026-07-26):**
`artifacts/scripted_style_payoff_matrix_20260726_fixed/FIRST_BLOCK_CHECKPOINT.json`

The full 896-row matrix was stopped after the first red/map block reached 16
fully matched seeds:

```text
red/map = OP6_IMMEDIATE_DUAL_RUSH|map_b_split_lane
BLUE_RUSH    WR=0/16  mean_margin=-2.3125
BLUE_TURTLE  WR=0/16  mean_margin=-2.3750
BLUE_SPLIT   WR=0/16  mean_margin=-2.7500
BLUE_ESCORT  WR=0/16  mean_margin=-2.7500
```

Verdict: collector health PASS, crossover evidence none, OP6/map_b is a
`UNIVERSALLY_HOSTILE_CELL_WARNING`. This is a calibration problem, not a pool
admissibility result. Do not spend the remaining matrix budget until OP6/map_b
is weakened or the scripted-blue controllers are verified against easier cells.

---

### V6I2 staged gate protocol (frozen — confirmatory run pending)

| Step | Status |
|------|--------|
| v6i2 gate infrastructure + schedule clocks | **DONE** — 108+ gate/curriculum tests green |
| Short v6i2 smoke (wiring only) | **DONE** — `tests/test_v6i2_staged_integration.py` |
| Threshold calibration from v6i1 Î»_cf=0.01 / 1.0 runs | **DONE** — frozen thresholds recorded |
| Freeze [`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md) | **DONE** — stale-aware bounded online gate fingerprint `224f1aea9ab36319` |
| Full fresh enforce confirmatory run (1.0M → up to 1.3M) | **PLANNED** with frozen fingerprint |

**2026-06-18 preset fix:** default `v6i2` previously inherited `latent_cf_coef_max = 0.01`
from v6i1 (weak CF). Preset now sets `1.0`. The in-flight
`v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` run at ~393k trained under
weak CF unless restarted with the fixed preset or `--latent-cf-coef-max 1.0`; treat it
as wiring/smoke evidence, not confirmatory strong-CF.

**2026-06-19 pairwise objective/gate refinement:** v6i2 now requires competence
before actor-CF separation, tracks per-pair hinge/weight telemetry, applies
persistent weak-pair weighting, and adds a worst-pair hinge term. Matched-seed
behavioral realization reports route, task-behavior, performance, and aggregate
components independently; normalized aggregation uses frozen scales and raw
component floors so route distance cannot carry a pass. Gate fingerprint remains
`224f1aea9ab36319`.

### V6I4 router ablation protocol (evaluation-only — pending promoted v6i2 checkpoint)

v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol
over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently
planned/pending. No parameters are trained or updated.

It is not a replacement training row. The checkpoint's actor, critic,
`q_phi`, latent repertoire, reward configuration, opponent pool, and
evaluation environment stay fixed; only the latent-selection rule changes
under matched seeds. The evaluator must reject pre-promotion checkpoints:
the checkpoint evidence must verify v6i2 lineage, Phase A promotion
`PASS`, gate fingerprint, promotion step, checkpoint hash, and valid
confirmatory gate lineage.

The locked comparison rows are `learned_qphi_switching`,
`uniform_episode_fixed`, `uniform_random_at_router_opportunities`,
`preselected_global_fixed_z`, `fixed_z0` through `fixed_z3`,
`qphi_initial_only_no_switch`, `shuffled_qphi_outputs`, and the
non-deployable posthoc oracle rows.
Success is return and win-rate
advantage over uniform, fixed, initial-only, and shuffled controls,
reported aggregate and per opponent. MI, entropy, occupancy, argmax
stability, and event-associated switching remain diagnostics.

### V6I6 repertoire expansion (implemented - pending anchor evidence)

v6i6 is an evidence-gated Expansion Stage E1 over v6i5, not an automatic
next launch. It activates only after forced-z and state-conditioned branch
evaluations produce a hashed anchor-validation manifest with
`verdict = "VALIDATED"`, selected `anchors`, one `expansion_target`, and
any `dormant` latents.

The implementation intentionally does not hardcode `z0`, `z1`, `z2`, or
`z3`. Training validation rejects `--preset v6i6` unless
`--v6i6-anchor-validation-manifest <path>` is supplied and the manifest
latents are disjoint and in range. E1 uses fixed-z episodes for outcome
attribution, a frozen reference critic for opportunity weights, no-op
adapter initialization, and the declared trainable scope
`target_embedding_gate_adapter_only`.

Required evidence before launch: finish forced-z evaluation, finish
state-conditioned branch evaluation, generate the anchor-validation
manifest with report hashes, run the 2k invariant smoke with the
manifest, then run the 25k birth diagnostic before any longer launch.

### V6I3 local communication (frozen contract — confirmatory run pending)

| Slice | Status |
|-------|--------|
| Spec owner doc | **DONE** — [`v6i3-local-communication-spec.md`](v6i3-local-communication-spec.md) |
| Slice 1 transport + unit tests | **DONE** — `rl/custom_ppo/communication/` |
| Slice 2 policy / rollout / PPO | **DONE** |
| Slice 3–6 phases / telemetry / corruption / gates | **DONE** |
| v6i3 preset + registry row | **DONE** — fingerprint `9ef168d941f046fb` |
| Full fresh v6i3 confirmatory run | **PLANNED** — pre-freeze v6i3 artifacts are exploratory only |

V6I3 must not modify active v6i1 runs or frozen v6i2 lineages. `communication_enabled=False` preserves v6i2 behavior.

Calibration uses v6i1/v6i2 evidence only. Confirmatory launch must consume the
frozen table unchanged (`--fresh-metrics-csv`, `confirmatory_gate_lineage_valid=True`,
gate fingerprint `9ef168d941f046fb`).

1. **Launch full v6i2 enforce confirmatory run** — not a shortened budget.
2. **Run v6i4 router ablation** on the accepted promoted v6i2 checkpoint.
3. **Launch full fresh v6i3 confirmatory run** after v6i2 lineage is accepted;
   do not reuse pre-freeze v6i3 metrics as official.
4. **Launch v5i7 seed 0 (§3.3) if the immediate target is the best
   Summer-faithful latent model on split-lane geometry.** Compare only to
   split-lane matched controls.
5. **Launch v5i6 seed 0 (§3.1) for the canonical open-map paper-faithful
   row.** Preserve the §1 invariants in
   `experiment-and-evaluation-protocol.md`.
5. **Run the v5i6 eval matrix and random-matched control.** Use
   `plot/eval_checkpoint.py --latent-selection router` and
   `--latent-selection random-matched` on the same checkpoint, seed, and
   episode budget.
6. **Run forced-z behavioral probes for v5i6/v5i7.** Use matched seeds across
   `z=0..K-1` before making causal behavior claims.
7. **Run v5i6 vs v5i4/v5i5 comparisons.** v5i6 vs v5i4 tests the full
   marginal-entropy switch; v5i6 vs v5i5 isolates entropy reduction at
   the same lambda_H floor.
8. **`no_latent_v4i3_baseline` re-launch at v5i6's exact budget /
   seed.** Closes D3 for the new headline `v5i6 vs no-latent`
   comparison.
9. **v5i6/v5i7 multi-seed.** Add seeds 1 and 2 only after seed 0 passes the
   router-quality and no-loss checks.

### Scripted-style opponent-pool calibration (2026-07-26)

The 896-row scripted-blue x scripted-red payoff-matrix run remains stopped
after the first OP6/map_b block. The initial clean matched block showed
`OP6_IMMEDIATE_DUAL_RUSH|map_b_split_lane` was universally hostile:
16 matched episode seeds x four blue styles, all four styles at 0/16 wins,
with BLUE_RUSH only least negative by margin. This is calibration evidence,
not crossover evidence.

Follow-up calibration found a dispatch bug: canonical audited long names such
as `OP6_IMMEDIATE_DUAL_RUSH` were not routed through the BT profile path in
`gpu_env/_core/_scripted_red.py`; the matrix was exercising the legacy
scripted fallback for those names. Dispatch now canonicalizes opponent keys
and routes OP6-OP12 through BT targets. Focused tests passed:
`python -m unittest AICTFProject.tests.test_bt_strategic_niches AICTFProject.tests.test_scripted_style_payoff_matrix`.

The blue probe trajectory gate now passes on
`OP6_IMMEDIATE_DUAL_RUSH x map_b_split_lane`: RUSH crosses midfield first,
TURTLE has highest home-half occupancy, SPLIT has greatest y-separation, and
ESCORT has smallest carrier-teammate distance.

Current OP6 development rerun:
`artifacts/scripted_style_op6_mapb_calibration_dev4_probe_fixed`.
Result: OP6/map_b still fails the intended trade-off. All four styles remain
0/16 wins; mean margins were BLUE_RUSH -2.1875, BLUE_ESCORT -2.3750,
BLUE_SPLIT -2.6250, BLUE_TURTLE -2.8125. The next step is still OP6
calibration only. Do not resume the full pool, do not train PPO/LRO, and do
not treat this as evidence of crossover.

**OP6 failure timeline diagnostic:**
`artifacts/op6_failure_timeline_dev1` ran the same 16 paired episode seeds
across all four scripted-blue styles. Classification:
`CASE_1_TURTLE_CANNOT_STOP_INITIAL_RUSH`.

Key turtle evidence: mean first red midfield crossing 5.75 steps, both red
agents in blue territory 8.375, first red flag touch 14.25, first red capture
40.375. Mean blue counterattack start was 15.6875 and first blue flag touch
22.4375, but red deaths, blue deaths, red carrier deaths, and blue carrier
deaths were all exactly 0.0 across the block. The failure is not failed-rush
recovery yet; turtle is not mechanically intercepting the initial dual rush.

Next OP6 work should tune the first-rush interception geometry/combat
opportunity before adding recovery-window logic. The intended contract remains:
OP6 punishes BLUE_RUSH and BLUE_ESCORT, is countered by BLUE_TURTLE, and is
mixed against BLUE_SPLIT.

**Pre-touch interception geometry follow-up:**
`artifacts/op6_failure_timeline_dev2_intercept_geometry` added minimum
red-to-blue distance, one-/two-defender tag-range steps, turtle target counts,
and path-crossing counts before first red flag touch. Turtle's closest
red-to-blue distance was meaningful (`mean_pre_touch_min_any_red_to_blue` about
1.30 cells), but two-defender tag pressure was 0.0 steps and path crossing was
near-zero. This separated "not close enough to matter" from "combat trigger
not sustained."

`artifacts/op6_failure_timeline_dev4_turtle_collapse_8seed` tested a
development turtle-only layered-defense probe that collapses both defenders on
the urgent inbound rusher. The trajectory gate still passed and
two-defender pressure increased to about 3.375 pre-touch steps, with closest
distance about 0.73 cells, but red deaths and carrier deaths stayed 0.0 and
TURTLE remained 0/8. The current failure is therefore sustained-contact
duration under the two-defender tag-channel rule, not absence of proximity.

An OP6 direct-carrier-return experiment was also tested in
`artifacts/op6_failure_timeline_dev5_op6_direct_return_8seed`; it worsened
TURTLE (`mean_margin=-3.0`) and was reverted. Do not use that as the next OP6
tuning direction.

**Tag-mechanics isolation:**
`tests/test_aquaticus_tag_mechanics.py` now micro-tests the actual Aquaticus
tag channel. Two blue defenders held in range of one target red kill exactly
after 3 consecutive decision steps; moving one defender out resets the red tag
accumulator to 0.0. Focused validation:
`python -m unittest AICTFProject.tests.test_aquaticus_tag_mechanics AICTFProject.tests.test_bt_strategic_niches AICTFProject.tests.test_scripted_style_payoff_matrix`
passed 15 tests.

`artifacts/op6_failure_timeline_dev7_exact_tag_pressure_8seed` aligned the
diagnostic pressure definition with the game rule (`blue_can_tag`,
`red_targetable`, and tag radius). Current turtle gets proximity but not enough
consecutive qualifying pressure: mean max consecutive dual-defender contact is
1.625 steps, while the kill threshold is 3 consecutive steps; mean max red tag
accumulator is about 0.43/0.56 seconds for red0/red1, below the 1.0 second
threshold. Red deaths remain 0.0. Next controller work should add target-lock
hysteresis and a true pinch that lowers relative velocity, rather than further
generic proximity tuning.

`artifacts/op6_failure_timeline_dev11_tag_counts_8seed` corrected the tag
measurement. The target-lock/pinch fix produced real defensive stops:
TURTLE averaged 7.625 red tags and 2.875 red-carrier tags per episode, despite
`red_alive` never flipping. TURTLE remained the best probe on the 8 development
seeds (WR 1/8, mean margin -1.125), while RUSH/ESCORT/SPLIT stayed 0/8 with
worse margins. Red first capture was delayed to about 109 steps for TURTLE
versus about 35-44 steps for the exposed styles.

`artifacts/op6_failure_timeline_dev12_turtle_post_tag_counter_8seed` tested a
single blue-side post-tag counter window. TURTLE improved to mean margin -0.75
on the same seeds and still passed the style trajectory gate. The new
post-tag metrics show the window is real but too late to finish: TURTLE
averages 2.125 post-tag counter launches and 2.125 post-tag blue flag touches
per episode, but red re-enters blue territory after about 1.0 step while blue
needs about 26.3 steps to touch the red flag. Blue captures before renewed red
pressure in 0.0 post-tag events.

Treat this as an emerging OP6 payoff niche, not OP6 acceptance. The remaining
problem is no longer contact or first counter launch; it is the size of the
failed-rush exploitation window.

`artifacts/op6_failure_timeline_dev13c_op6_carrier_regroup_cooldown_8seed`
tested the OP6-specific failed-rush regroup with carrier-stop-only triggering,
no active-window renewal, and a 30-step cooldown. The style trajectory gate
passed, but payoff did not improve beyond dev12: TURTLE mean margin was -0.875
with WR 1/8. This does not support making the regroup window larger yet. The
next diagnostic should inspect TURTLE's counterattacker path/cancellation and
red-flag approach during regroup.

The regroup code was reverted after dev13c. Frozen OP6 status is now based on
the dev12 behavior plus the held-out confirmation screen above.

**OP12 opening audit (2026-07-27):**
`artifacts/op12_opening_audit_rush_vs_escort_8seed` compares current
BLUE_RUSH and BLUE_ESCORT against frozen OP12 on the same 8 paired seeds. This
is a development diagnostic only, not a payoff confirmation.

Result: the OP12 opening vulnerability exists, but it is too generic. RUSH and
ESCORT are behaviorally distinguishable during the first 20 steps, yet both
reach the red flag and score on essentially the same tempo. RUSH crosses
midfield slightly earlier and faster, while ESCORT stays more clustered:

```text
BLUE_RUSH:
  first midfield any/both      7.625 / 9.750
  first flag touch / pickup   14.625 / 15.125
  first blue score            42.125
  opening teammate dist        4.694
  opening lane sep             4.026
  opening forward velocity     0.667
  opening clustered frac       0.763

BLUE_ESCORT:
  first midfield any/both      8.500 / 15.250
  first flag touch / pickup   14.625 / 15.250
  first blue score            44.750
  opening teammate dist        3.679
  opening lane sep             2.199
  opening forward velocity     0.560
  opening clustered frac       0.863
```

Interpretation: this is not a blue-probe collapse. ESCORT is more compact and
slower to get both agents across midfield, so OP12 can in principle classify it
before pickup. The current failure is that the opening gate gives both styles
nearly equal flag access before OP12 distinguishes close support from raw
tempo. Next OP12 change should move anti-ESCORT recognition into the pre-pickup
opening phase and keep the SPLIT detector frozen. Do not run held-out OP12
confirmation yet; protected RUSH niche remains FAIL.

OP12 opening-escort detector follow-up:
`artifacts/op12_dev9h_opening_escort_detector_probe_2seed` is a telemetry smoke
after adding an opening-only pre-pickup lead/support detector and localized
response. Focused unit tests pass, but live detector telemetry does **not** yet
fire in the environment:

```text
BLUE_RUSH   core escort triggers 0/2, external detector triggers 1/2
BLUE_ESCORT core escort triggers 0/2, external detector triggers 2/2
```

Decision: do not run the 4-seed payoff pilot yet. The offline/diagnostic
geometry can see the ESCORT structure, but the core adaptive role path is not
recording the new detector in live episodes. Next OP12 work should inspect why
the BT adaptive detector state is not updating from the live role-assignment
path before tuning thresholds or interpreting payoff.

OP12 detector wiring follow-up:
`artifacts/op12_dev9p_opening_escort_detector_probe_8seed_max40` verifies the
core detector now records live in-episode events after fixing telemetry
accumulation and moving opening-escort persistence ownership to the BT role
path. Focused validation passes:
`python -m unittest AICTFProject.tests.test_bt_adaptive AICTFProject.tests.test_scripted_style_payoff_matrix`
ran 16 tests OK.

Current detector correctness:

```text
BLUE_ESCORT core escort trigger: 7/8, mean first trigger step 7.86
BLUE_RUSH   core escort trigger: 4/8, mean first trigger step 6.25
SPLIT detector on RUSH/ESCORT: 0/8 and 0/8
```

Decision: wiring/update-order bug is fixed, but OP12 pre-pickup ESCORT
recognition is still too broad. Do not run payoff yet. The next OP12 step is
not response tuning; it is adding a stronger live discriminator that reduces
RUSH false positives while preserving early ESCORT triggers. The audit suggests
forward velocity is useful in aggregate, but the first live velocity attempt was
not stable at the BT decision point.

OP12 history-score detector follow-up:
`artifacts/op12_dev10b_history_score_detector_probe_8seed_max40` replaces the
single hard predicate with a short-history score over compactness, lane
narrowness, stable leader/follower ordering, shared heading, and excessive
forward-speed penalty. This is still detector development only.

Current score separation:

```text
BLUE_ESCORT mean score: 3.376
BLUE_RUSH   mean score: 2.758
```

The score separates ESCORT from RUSH better than the previous binary geometry
gate, and RUSH no longer triggers the core escort detector in the 8-seed
opening-only probe. However, ESCORT also does not yet satisfy the 3-consecutive
activation gate before pickup, so the detector is **not accepted**. Focused
validation passes:
`python -m unittest AICTFProject.tests.test_bt_adaptive AICTFProject.tests.test_scripted_style_payoff_matrix`
ran 16 tests OK.

Decision: keep payoff blocked. Next OP12 work should tune the score activation
gate, not the red response, using the existing component telemetry. Do not move
to payoff until ESCORT triggers at least 7/8 and RUSH triggers at most 1/8 on
development seeds.

OP12 activation-gate sweep:
`artifacts/op12_dev11_escort_gate_sweep_8seed` evaluates predeclared detector
gates over live opening score traces: threshold with 3 consecutive steps,
threshold with 2 consecutive steps, 2-of-last-3, and rolling 5-step evidence.
Inputs are detector-only traces before pickup; no payoff run was launched.

Result: no usable operating point.

```text
Decision: NO_USABLE_OPERATING_POINT

Max score ranges by style:
BLUE_RUSH   1.443-2.900
BLUE_ESCORT 2.259-3.189
BLUE_SPLIT  1.149-2.931
BLUE_TURTLE 2.498-3.653
```

Interpretation: the current compactness/lane/leader/heading/speed score is not
sufficient. It separates ESCORT from RUSH on mean score, but the tails overlap
and TURTLE produces high scores because slow, compact motion can look
escort-like without being an offensive convoy. Do not tune the threshold/window
further on this score. The next detector feature should explicitly include
offensive convoy progress, such as leader nearing the red flag while the
follower remains behind within a controlled offset, or a projected same-corridor
flag approach. Response tuning and payoff remain blocked.

OP12 semantic convoy detector attempt:
`artifacts/op12_dev12f_convoy_speed_probe_4seed_max40` tested a conjunctive
state-machine detector with offensive progress, same-corridor movement, stable
leader/follower geometry, and a moderate centroid-speed clause. Focused tests
still pass (`python -m unittest AICTFProject.tests.test_bt_adaptive
AICTFProject.tests.test_scripted_style_payoff_matrix`, 16 tests OK), but the
development probe did not produce a usable activation pattern:

```text
BLUE_RUSH   escort triggers 0/4
BLUE_ESCORT escort triggers 0/4
```

Interpretation: the moderate-speed clause removes RUSH false positives but also
kills ESCORT recall at the live BT decision point. This confirms the issue is
not threshold tuning alone. The next OP12 detector should use a more direct
carrier/flag-route semantic feature, such as lead agent projected to the red
flag corridor with the follower trailing behind that route, or accept a
two-stage detector where pre-pickup only raises suspicion and post-pickup
confirms escort. Payoff and response tuning remain blocked.

OP12 convoy gap-stability probe:
`artifacts/op12_dev12g_convoy_gap_stability_probe_4seed_max40` replaced the
moderate-speed activation clause with semantic gap stability: offensive pair,
same corridor, stable leader/follower ordering, and controlled longitudinal-gap
jitter. Focused validation still passes:
`python -m unittest AICTFProject.tests.test_bt_adaptive AICTFProject.tests.test_scripted_style_payoff_matrix`
ran 16 tests OK.

```text
BLUE_RUSH   escort triggers 2/4
BLUE_ESCORT escort triggers 3/4
```

Interpretation: gap stability restores some ESCORT recall, but the pre-pickup
convoy signal still false-triggers on RUSH. Treat this as evidence that
pre-pickup convoy semantics alone are not clean enough on the live OP12 route.
The next controlled implementation should be the two-stage detector: pre-pickup
convoy raises suspicion only; post-pickup carrier plus trailing protector
confirms ESCORT. Do not enable payoff or tune OP12's punitive response until
the detector-only gate passes.

OP12 two-stage escort detector wiring:
Implemented detector-only two-stage telemetry in `_bt_adaptive.py`.
Pre-pickup convoy evidence now remains suspicion-only and no longer drives the
opening anti-escort hard response. Post-pickup carrier/protector confirmation
logs:

```text
escort_confirmation_step
escort_confirmation_active_steps
escort_confirmation_carrier_id
escort_confirmation_protector_id
escort_confirmation_distance
escort_confirmation_same_corridor_steps
escort_confirmation_to_episode_end_steps
pickup_to_confirmation_steps
```

Telemetry schema was added to both
`experiments/diagnose_op12_opening_trajectory.py` and
`experiments/run_scripted_style_payoff_matrix.py`. Focused validation passes:
`python -m unittest AICTFProject.tests.test_bt_adaptive AICTFProject.tests.test_scripted_style_payoff_matrix`
ran 18 tests OK.

Detector smoke:
`artifacts/op12_dev13b_two_stage_escort_detector_smoke_1seed_max40`

```text
BLUE_RUSH   suspicion step 8, confirmation step 20
BLUE_ESCORT suspicion step 6, confirmation step 32
```

Interpretation: the two-stage plumbing is live, but the current confirmation
rule is **not accepted**. On this seed it still false-confirms RUSH and confirms
ESCORT too late to be a reliable strategic counter. OP12 hard anti-escort
response remains disabled; payoff and response tuning remain blocked. Next
detector iteration should add a stronger post-pickup protector semantic, such
as defender-relative screening or non-carrier abandoning independent scoring,
rather than re-enabling a hard response.

OP12 post-pickup RUSH/ESCORT separability check:
`artifacts/op12_dev14_rush_escort_post_pickup_overlap_smoke_1seed_max40` added
detector-free post-pickup trajectory metrics to
`diagnose_op12_opening_trajectory.py`. On the smoke seed, RUSH and ESCORT are
actually separable after pickup:

```text
BLUE_RUSH:
  same-corridor frac:   0.227
  shadowing frac:       0.227
  independent frac:     0.364

BLUE_ESCORT:
  same-corridor frac:   1.000
  shadowing frac:       0.857
  independent frac:     0.000
```

The previous confirmation rule was using the wrong semantic shape. It
overweighted a trailing/leader condition that ESCORT does not consistently
satisfy after pickup, while RUSH can briefly satisfy it.

OP12 two-stage confirmation update:
`artifacts/op12_dev14c_post_pickup_confirm_probe_4seed_max40` replaces the
post-pickup confirmation rule with sustained close same-corridor carrier
support: six consecutive carrier/protector ticks with controlled distance and
same route corridor. Focused validation passes:
`python -m unittest AICTFProject.tests.test_bt_adaptive AICTFProject.tests.test_scripted_style_payoff_matrix`
ran 18 tests OK.

Detector-only development result:

```text
BLUE_RUSH   confirmation: 0/4
BLUE_ESCORT confirmation: 3/4
mean ESCORT confirmation step: 19.67
mean pickup-to-confirmation:   5.00
mean response-time remaining:  20.33 steps
```

Interpretation: post-pickup style separability is now demonstrated on a small
development slice, and the hard RUSH false-confirmation failure is fixed for
these seeds. This is still not the detector acceptance gate. Next run should
evaluate detector-only on all four blue styles, with the current gate:
ESCORT confirmation at least 7/8, RUSH 0/8, TURTLE 0/8, SPLIT at most 1/8, and
enough confirmation-to-capture/episode time for a response to matter. OP12 hard
anti-escort response remains disabled until that gate passes.

OP12 full four-style detector-only development gate:
`artifacts/op12_dev15_full_detector_gate_8seed_max40` ran OP12 against all four
blue scripted styles on the same eight paired development seeds. This remains
detector-only; OP12 hard anti-escort response is still disabled and no payoff
acceptance is inferred.

Predeclared gate:

```text
BLUE_ESCORT confirmation >= 7/8
BLUE_RUSH confirmation   = 0/8
BLUE_TURTLE confirmation = 0/8
BLUE_SPLIT confirmation  <= 1/8
useful response time remains after confirmation
```

Observed:

```text
BLUE_ESCORT confirmation: 6/8, mean step 25.17
BLUE_RUSH confirmation:   1/8, mean step 23.00
BLUE_TURTLE confirmation: 0/8
BLUE_SPLIT confirmation:  0/8
ESCORT pickup -> confirmation: 9.50 steps
ESCORT remaining response time in max40 smoke: 14.83 steps
```

Verdict: **FAIL / CLOSE**. The detector now rejects TURTLE and SPLIT and is
mostly selective for ESCORT, but it misses the 7/8 ESCORT recall gate and still
has one RUSH false confirmation. Do not freeze the detector and do not enable
the OP12 anti-escort response. Next diagnostic should inspect the one RUSH false
confirmation and the two ESCORT misses at row/trajectory level before changing
any thresholds.

OP12 dev15 failure inspection and targeted detector revision:
The three dev15 failures were isolated to:

```text
BLUE_RUSH seed 551005:
  false-confirmed after six loose same-corridor ticks
  support distance mostly 3.08-3.66 cells
  post-pickup independent frac 0.368

BLUE_ESCORT seed 551003:
  missed because protector was far from carrier for most of the return
  support distance 10-12 cells early after pickup
  classification: scripted ESCORT failed to form an escort promptly

BLUE_ESCORT seed 551006:
  missed despite clear close support
  old rule reset because agents became too close / narrowly missed six ticks
  classification: detector rule defect
```

One targeted detector change was made: post-pickup confirmation now requires
five sustained ticks of closer same-corridor carrier support, with support
distance narrowed from `[1.0, 4.0]` to `[0.75, 3.0]`. This rejects the loose
RUSH false-positive formation and admits the close-support ESCORT miss without
adding a broader rule family.

OP12 full four-style detector-only development gate after targeted revision:
`artifacts/op12_dev16_full_detector_gate_tight_support_8seed_max40`

```text
BLUE_ESCORT confirmation: 8/8
BLUE_RUSH confirmation:   0/8
BLUE_TURTLE confirmation: 0/8
BLUE_SPLIT confirmation:  0/8
ESCORT pickup -> confirmation: 11.25 steps
ESCORT remaining response time in max40 smoke: 13.38 steps
```

Verdict: **DEVELOPMENT DETECTOR CANDIDATE PASS**. Do not enable OP12's hard
anti-escort response yet. Freeze the detector rules/thresholds for the next
step and run fresh detector-only held-out seeds. Only if held-out also passes
should OP12 anti-escort response be enabled for a payoff development pilot.

OP12 frozen detector held-out gate:
`artifacts/op12_dev17_heldout_detector_gate_16seed_max40` evaluated the frozen
dev16 detector rules on 16 fresh paired seeds across all four blue styles.
No threshold edits were made after viewing this result; OP12 hard anti-escort
response remains disabled.

Predeclared held-out gate:

```text
BLUE_ESCORT confirmation >= 14/16
BLUE_RUSH confirmation   = 0/16
BLUE_TURTLE confirmation = 0/16
BLUE_SPLIT confirmation  <= 1/16
useful response time remains after confirmation
```

Observed:

```text
BLUE_ESCORT confirmation: 13/16
BLUE_RUSH confirmation:    3/16
BLUE_TURTLE confirmation:  0/16
BLUE_SPLIT confirmation:   0/16
ESCORT pickup -> confirmation: 9.08 steps
ESCORT remaining response time in max40 smoke: 14.38 steps
```

Verdict: **HELD-OUT FAIL**. The detector generalizes well against TURTLE and
SPLIT, but not sufficiently against RUSH, and ESCORT recall is just below the
held-out gate. Do not freeze or enable the response. Next step is to classify
the three RUSH false confirmations and three ESCORT misses on held-out rows.
If they reveal genuine RUSH/ESCORT trajectory overlap, fix the scripted blue
controllers or stop OP12 detector tuning; if they reveal one shared detector
mechanism, make at most one more targeted revision and restart development
validation before another held-out attempt.

OP12 dev17 held-out failure trace classification:
Per-step traces of the six failed rows show no single clean detector-threshold
defect.

RUSH false confirmations:

```text
BLUE_RUSH seed 552006: sustained carrier-shadowing run, ticks reached 5
BLUE_RUSH seed 552009: sustained carrier-shadowing run, ticks reached 5
BLUE_RUSH seed 552015: sustained carrier-shadowing run, ticks reached 5
```

In all three false positives, the RUSH non-carrier genuinely stayed in the same
return corridor with the carrier for five or more steps. This is behavioral
overlap, not just detector noise.

ESCORT misses:

```text
BLUE_ESCORT seed 552003: protector remained far from carrier for most of return
BLUE_ESCORT seed 552005: close support was fragmented by heading/distance resets
BLUE_ESCORT seed 552012: close support fragmented; never sustained confirmation
```

Interpretation: OP12's current detector is seeing the trajectories honestly.
The remaining failure is that BLUE_RUSH sometimes behaves like ESCORT after
pickup, while BLUE_ESCORT sometimes fails to maintain a stable escort. Do not
continue threshold tuning on OP12. Next controlled change should sharpen the
scripted blue probes:

```text
BLUE_RUSH non-carrier: no carrier shadowing; take independent/off-lane pressure
BLUE_ESCORT non-carrier: maintain close same-corridor support after pickup
```

Changing these probes invalidates direct comparison with older scripted-matrix
results that used the previous BLUE_RUSH/BLUE_ESCORT definitions, so affected
OP6-OP12 calibration rows must be rerun under the new frozen blue-controller
version before pool claims.

BLUE_RUSH / BLUE_ESCORT probe sharpening:
Updated `_scripted_blue_styles.py` so RUSH and ESCORT stay behaviorally
persistent after pickup:

```text
BLUE_RUSH non-carrier:
  moves to the opposite lane and keeps pressuring the red flag
  no carrier-shadowing target

BLUE_ESCORT non-carrier:
  targets a close same-corridor carrier offset
  stays with the carrier instead of chasing the carrier's evasion target
```

The OP12 post-pickup confirmation condition was also simplified after the probe
change: close same-corridor support no longer requires matching instantaneous
heading, because a tight convoy can turn around obstacles while still
protecting the carrier. Focused validation passes:
`python -m unittest AICTFProject.tests.test_bt_adaptive AICTFProject.tests.test_scripted_style_payoff_matrix`
ran 19 tests OK.

Sanity detector-only screen with new blue-controller definitions:
`artifacts/op12_dev19_new_blue_no_heading_confirm_sanity_4seed_max40`

```text
BLUE_ESCORT confirmation: 4/4
BLUE_RUSH confirmation:   0/4
BLUE_TURTLE confirmation: 0/4
BLUE_SPLIT confirmation:  0/4
ESCORT pickup -> confirmation: 4.50 steps
ESCORT remaining response time in max40 smoke: 20.25 steps
```

Interpretation: the measuring instrument is now much cleaner. This is only a
small development sanity screen, not a freeze. Next step is a full 8-seed
development detector gate under the new blue-controller version, then fresh
held-out seeds if that passes.

OP12 full detector-only development gate with sharpened blue controllers:
`artifacts/op12_dev20_new_blue_detector_gate_8seed_max40`

```text
BLUE_ESCORT confirmation: 8/8
BLUE_RUSH confirmation:   0/8
BLUE_TURTLE confirmation: 0/8
BLUE_SPLIT confirmation:  0/8
ESCORT pickup -> confirmation: 4.38 steps
ESCORT remaining response time in max40 smoke: 20.63 steps
```

RUSH post-pickup behavior is now clearly independent:

```text
BLUE_RUSH post-pickup independent frac: 0.772
BLUE_RUSH post-pickup same-corridor frac: 0.125
BLUE_ESCORT post-pickup shadowing frac: 0.951
BLUE_ESCORT post-pickup independent frac: 0.000
```

Verdict: **DEVELOPMENT PASS under new blue-controller version**. Do not enable
OP12 hard anti-escort response yet. Freeze the updated RUSH/ESCORT controller
definitions and detector rules for a fresh detector-only held-out gate. Older
scripted payoff matrices remain historical and must not be mixed with this
new blue-controller version.

OP12 BLUE_PROBES_V2 detector-only held-out gate:
`artifacts/op12_dev21_blue_v2_heldout_detector_gate_16seed_max40` evaluated the
frozen BLUE_PROBES_V2 RUSH/ESCORT controller definitions and frozen OP12
detector rules on 16 fresh paired seeds across all four blue styles. Artifact
rows and summary include `blue_probe_protocol = BLUE_PROBES_V2`.

Predeclared gate:

```text
BLUE_ESCORT confirmation >= 14/16
BLUE_RUSH confirmation   = 0/16
BLUE_TURTLE confirmation = 0/16
BLUE_SPLIT confirmation  <= 1/16
useful response time remains after confirmation
```

Observed:

```text
BLUE_ESCORT confirmation: 16/16
BLUE_RUSH confirmation:    0/16
BLUE_TURTLE confirmation:  0/16
BLUE_SPLIT confirmation:   0/16
ESCORT pickup -> confirmation: 5.13 steps
ESCORT remaining response time in max40 smoke: 19.19 steps
```

Verdict: **HELD-OUT DETECTOR PASS**. OP12 can legally recognize the BLUE_PROBES_V2
ESCORT formation without confusing it with RUSH, TURTLE, or SPLIT. This still
does not prove an OP12 RUSH payoff niche. Next step is to enable the smallest
OP12-only anti-escort response and run a payoff development pilot asking whether
BLUE_RUSH becomes uniquely best. Do not mix pre-BLUE_PROBES_V2 scripted payoff
matrices with this protocol.

OP12 confirmed-ESCORT response ablation:
`artifacts/op12_dev22_response_off_8seed` and
`artifacts/op12_dev22_response_on_8seed` used identical paired seeds
(`base_seed=556001`), OP12/map_b only, all four BLUE_PROBES_V2 styles, and
240 decision steps. The response was opt-in through
`--op12-confirmed-escort-response`; default OP12 remains detector-only.

Mean margins:

```text
            response OFF   response ON   delta
BLUE_ESCORT       -2.125        0.000    +2.125
BLUE_RUSH         -0.750       -0.625    +0.125
BLUE_SPLIT         1.250        1.250     0.000
BLUE_TURTLE        0.875        0.750    -0.125
```

Paired seed deltas show BLUE_ESCORT improved on 7/8 seeds when the response was
enabled. SPLIT stayed best at +1.250 mean margin and 8/8 wins. RUSH remained
negative and did not become uniquely best. The full 240-step horizon also showed
late RUSH/TURTLE escort confirmations that were absent in the max-40 detector
gate, so the detector remains clean for early response timing but not for
unbounded full-episode hard response.

Verdict: **RESPONSE ABLATION FAIL, OP12 RUSH NICHE UNPROVEN**. The detector
stays accepted; the current hard response is rejected because it helps ESCORT
instead of punishing it. Next OP12 change should alter the response mechanics
only, not the detector: target the protector/carrier separation in a way that
reduces carrier survival or pickup conversion instead of accidentally creating
a safer escort return.

OP12 response-failure trace:
`artifacts/op12_dev23_response_trace_escort_ep0_ep7` traced paired ESCORT
episodes 0 and 7 from dev22, where response ON improved BLUE_ESCORT by +3
margin points on each seed.

First causal divergence after confirmation:

```text
response OFF: red roles usually INTERCEPTOR + COUNTER
response ON:  red roles switch to INTERCEPTOR + INTERCEPTOR
```

In the traced seeds, OFF's default OP12 behavior kept one red agent in counter
pressure while the other handled carrier denial. ON replaced that useful
counter role with a second interceptor/protector-targeting maneuver. The
result was more containment near the carrier but less scoreboard pressure and
less disruption of ESCORT's episode-level plan.

Diagnosis: the failed response is not too weak; it displaces a good default
role. Next OP12 response should preserve the primary carrier interceptor and
the default counter/return-lane pressure. Do not chase the protector directly
unless a later trace shows the protector is the actual blocker. A safer next
design is:

```text
confirmed ESCORT
-> keep existing carrier-intercept role
-> keep or bias the other red agent's counter/return-lane pressure
-> optionally increase carrier priority without changing both roles
```

OP12 route-only response ablation:
`artifacts/op12_dev24_route_only_response_trace_escort_ep0_ep7` first verified
the modifier-only response on the two ESCORT seeds where dev22 helped ESCORT by
+3 margin. The new route-only response preserved the early default
`INTERCEPTOR + COUNTER` pattern and removed the large accidental ESCORT boost:

```text
episode 0: OFF -3, ON -3, delta 0
episode 7: OFF -2, ON -2, delta 0
```

Then `artifacts/op12_dev25_route_only_response_on_8seed` reran the full
8-seed response-ON development ablation against the same response-OFF baseline
from `artifacts/op12_dev22_response_off_8seed`.

Mean margins:

```text
            response OFF   route-only ON   delta
BLUE_ESCORT       -2.125          -1.750   +0.375
BLUE_RUSH         -0.750          -0.750    0.000
BLUE_SPLIT         1.250           1.250    0.000
BLUE_TURTLE        0.875           1.000   +0.125
```

Paired deltas:

```text
RUSH:   unchanged on 8/8 seeds
SPLIT:  unchanged on 8/8 seeds
TURTLE: +1 on one late-confirmed seed, otherwise unchanged
ESCORT: mean delta +0.375; still improved overall
```

Verdict: **ROUTE-ONLY RESPONSE SAFER BUT STILL REJECTED**. It fixed the
dev22 role-replacement bug, but it did not selectively hurt ESCORT. The current
response should not be promoted to payoff confirmation. Next OP12 work should
not touch the detector or replace roles; it should inspect the ESCORT seeds
where route-only ON still improved (`episode_index` 2, 3, 6) and identify
whether the carrier-lane bias delays red scoring, moves the interceptor to a
worse intercept point, or creates extra flag-retrieval churn.

OP12 dev26 trace diagnosis: `artifacts/op12_dev26_route_only_response_trace_escort_ep2_ep3_ep6`
paired OFF/ON traces (`experiments/trace_op12_response_ablation.py`,
base_seed=556001) for episodes 2, 3, 6. Confirmation step was IDENTICAL
between OFF/ON in all three (24, 23, 23) -- the detector's timing was never
the issue. The divergence was episode LENGTH: OFF closed out in 97/149/179
steps; ON ran to the 240-step cap (ep2, ep3) or nearly (ep6, 194). Per-step
inspection of episode 2 showed red0/red1 roles and distances IDENTICAL to
OFF through step ~40 (while the escort-denial-route modifier briefly applied
to whichever agent held ROLE_INTERCEPTOR), then diverging sharply once that
agent's role switched to ROLE_COUNTER: OFF's counter agent stayed within
4-8 units of the carrier and red scored a second point by step 80; ON's
counter agent drifted to 12-16 units away and red's score stayed stuck at 1
for the rest of the episode. Diagnosis: the unconditional carrier-lane route
was not making ESCORT survive better -- it left the modified agent in a
worse position for its OWN later counter-scoring duty, a small early
perturbation cascading into red's second scoring pass never landing. This
matches "red scoring is delayed" from the diagnostic decision tree, not "ON
approaches the carrier late/from behind."

OP12 ETA-gated interception fix (one revision, per the diagnostic decision
tree -- detector, roles, and INTERCEPTOR+COUNTER structure untouched): the
carrier-lane route (`escort_bx/escort_by`) is now only taken when its
Euclidean distance from the agent's current position is <= the default
intercept point's distance, so the specialized response can never replace an
already-superior default path.

`artifacts/op12_dev27_eta_gated_trace_escort_ep2_ep3_ep6` re-ran the same
three traced episodes: ep2 and ep6 fully resolved (`delta_margin=0`, ON
episode length now matches OFF almost exactly: 99 vs 97, 180 vs 179); ep3
still shows a residual effect (`delta_margin=1`, ON still runs to the 240
step cap). Not chased further per the "at most one route revision" guidance.

`artifacts/op12_dev28_eta_gated_response_on_8seed` reran the full 8-seed
response-ON development ablation with the fix:

```text
            response OFF   ETA-gated ON   delta
BLUE_ESCORT       -2.125         -1.875   +0.250 (was +0.375 pre-fix)
BLUE_RUSH         -0.750         -0.500   +0.250
BLUE_SPLIT         1.250          1.250    0.000 (untouched, as required)
BLUE_TURTLE        0.875          0.750   -0.125
```

Verdict: real, partial improvement (ESCORT's gap to the OFF baseline shrank
~33%; RUSH and TURTLE both moved in the desired direction as side effects;
SPLIT confirmed untouched). Still not a RUSH niche and not promoted to
held-out confirmation. Per the "larger reality check": ESCORT was already
the weakest response against OP12 before this fix, so further anti-ESCORT
tuning is not the lever most likely to produce a RUSH niche. The main
blockers remain SPLIT (+1.250, unchanged since dev6) and TURTLE (+0.750).
Prior SPLIT-suppression history (dev1/4/5/6/8, all pre-2026-07-27): dev5's
detector is cleanly selective (SPLIT 8/8, others 0/8 trigger rate) but dev6's
post-pickup dual-carrier-denial response only moved SPLIT from +1.375 to
+1.000 (DIRECTIONAL_BUT_INSUFFICIENT); dev8's broader pre-pickup opening
gate created an early RUSH window but overcorrected, pushing SPLIT/ESCORT to
+1.500 and TURTLE to +1.250. Next OP12 SPLIT work should not repeat either
of those exact approaches unchanged.

OP12 dev26 root-cause inspection (episodes 2, 3, 6):
`artifacts/op12_dev26_route_only_response_trace_escort_ep2_ep3_ep6` per-step
traces (already collected) were read directly rather than re-run. Consistent
pattern across all three episodes, not just one:

```text
episode  OFF: red_score reaches  ON: red_score reaches  OFF steps  ON steps
2        3 (by step 97)          1 (never converts a 2nd/3rd)  97   240
3        3 (by step 149)         2 (2nd score same step as OFF; no 3rd)  149  240
6        3 (by step 179)         2 (2nd score DELAYED 142->153; no 3rd)  179  194/240
```

`blue_score` stays 0 in every OFF trace and in 2 of 3 ON traces (ep6/ON is the
lone exception, blue_score=1). The margin "improvement" is not from ESCORT
being denied more captures -- it is from OP12's OWN counter-role scoring
converting less often and later. Role-occupancy counts make the mechanism
visible: in ep2, OFF has red0 mostly COUNTER (52/97 steps) and red1 mostly
INTERCEPTOR (36/97); ON has this almost fully reversed (red0 INTERCEPTOR
190/240, red1 COUNTER 195/240), yet the COUNTER role -- despite occupying it
for far MORE total steps in ON -- converts fewer scores, more slowly. The
route-only response (`escort_denial_route`, gated on `blue_carry_any` +
confirmed_escort, biasing the INTERCEPTOR's target to
`ec + (home-ec)*0.35`) never touches role ASSIGNMENT directly (that bug was
already fixed going from dev22 to dev24), but moving the INTERCEPTOR's
position appears to perturb the generic utility-based role-assignment on
SUBSEQUENT steps enough to flip which agent ends up COUNTER vs INTERCEPTOR,
and the resulting role churn/reassignment is what delays red's own scoring --
an indirect, emergent side effect of the position change, not a designed
denial mechanism.

Decision: **REJECT this response mechanism** rather than continue tuning it.
It does not create a genuine ESCORT-specific weakness; it accidentally
degrades OP12's own counter-attack conversion whenever the ESCORT detector
fires, which is not a legitimate niche (a real denial mechanism should show
up as ESCORT's specific captures being stopped, not as red's own unrelated
scoring machinery breaking). `op12_confirmed_escort_response_enabled`
defaults to False already (dev25/26 were explicit opt-in ablations for
diagnosis only), so no code change is needed to "turn this off" -- the
decision is simply to not promote it, and not spend further iterations
tuning ESCORT denial for OP12. Per the standing OP12 contract, the higher-
leverage remaining work is suppressing SPLIT (+1.250) and TURTLE (+0.875)
while preserving RUSH's early window -- and RUSH currently sits at -0.750,
i.e. blue's WORST-performing style against unchanged OP12, the opposite of
the intended "RUSH exploits an early vulnerability" niche. That inversion
(RUSH should be blue's BEST response, not its worst) is the real next
problem, larger in scope than the ESCORT-response question just closed here.

## OP12 RUSH-niche redesign (locked contract, 2026-07-28)

Redesign OP12 around its identity as a late converter: vulnerable to
immediate concentrated pressure (RUSH), dangerous against slow play
(punishes TURTLE via late conversion), able to punish sustained split
pressure (SPLIT), ESCORT already weak (no extra response). The accepted
ESCORT detector stays as telemetry only; its hard response (`dev24-26`,
rejected above) stays permanently off. Three isolated stages, tested
independently before combining. Hard budget: at most two complete
three-stage redesign rounds; if RUSH is not uniquely best after both, close
OP12 as a failed RUSH-niche candidate and reassign the niche elsewhere.

**Round 1, Stage 1 -- early RUSH window (2026-07-28):**
`artifacts/op12_stage1_rush_window_dev1_8seed`, 8 paired seeds (base_seed
556001, same seeds as the dev22/dev25 baseline for direct comparability),
unchanged blue controllers, all four BLUE_PROBES_V2 styles.

Root cause identified before changing anything: `_bt_assign_roles`'s
existing `op12_opening` window (first 20 sim steps, unless split pressure
already active) already forces BOTH OP12 agents into ROLE_ATTACKER and
already suppresses FLAG_RETR/INTERCEPTOR via `late_or_not_op12` -- but
Priority 2 (ESCORT own carrier) was NOT gated by `late_or_not_op12`. Net
effect: during the opening, both OP12 agents race for blue's flag, and the
instant either grabs it, the other instantly becomes its escort -- an
efficient, protected attacker+escort conversion with no analogous protection
for blue's RUSH, which plausibly out-races RUSH before RUSH can do anything.
This means the opening weakness the contract wants was not primarily about
OP12 defending too well (defense was already mostly off in the opening) --
it was about OP12's OWN offense converting too efficiently and uncontested.

One isolated change: gate Priority 2's `have_carrier` on `late_or_not_op12`
too (`gpu_env/_core/_bt_red.py`), using the same time-based,
opponent-agnostic signal already used for FLAG_RETR/INTERCEPTOR -- not the
blue style ID. During the opening, an OP12 carrier is now unescorted.

```text
                 baseline (dev22/25)   stage-1 ON   delta
BLUE_RUSH             -0.750  (0/8)     -0.500 (3/8)   +0.250
BLUE_SPLIT            +1.250  (8/8)     +1.250 (8/8)    0.000
BLUE_TURTLE           +0.875  (?/8)     +0.625 (4/8)   -0.250
BLUE_ESCORT           -2.125  (0/8)     -2.000 (0/8)   +0.125
```

Isolation criterion met: RUSH improved meaningfully (WR 0/8 -> 3/8) while
SPLIT stayed bit-identical (+1.250, 8/8 both before and after) -- confirms
the change is scoped to the opening-carrier-escort mechanism and does not
touch SPLIT's own path to conversion. TURTLE softened somewhat (side effect,
not part of this stage's target, acceptable per the isolated-test contract
which only requires RUSH-vs-SPLIT independence). ESCORT essentially
unchanged (still OP12's worst matchup for blue, as intended -- "ESCORT
already weak, no extra response needed").

RUSH is still far from uniquely best (SPLIT still leads by 1.75). Stage 1
alone was never expected to fully solve this; Stages 2 (punish SPLIT) and 3
(punish TURTLE via late conversion) still need to land before recombining
and re-testing the full four-style matrix.

**Round 1, Stage 2 -- bounded sustained-split response (2026-07-28):**
`artifacts/op12_stage2_split_response_dev1_8seed`, same 8 paired seeds,
Stage 1 code left in place (this measures Stage 2's own marginal effect on
top of Stage 1, not Stage 2 in a vacuum).

The pre-existing `split_denial`/`split_denial_route` mechanism (added before
this session) already existed but had two problems relative to the locked
contract: (1) it was gated on the live, single-step `adapt_split_pressure`
flag, which resets to 0 on any one non-qualifying step -- the same flicker
bug already diagnosed and fixed for OP11 earlier this session; and (2) it
force-reassigned BOTH red agents to `ROLE_INTERCEPTOR` with no guard against
overwriting whichever agent currently held `ROLE_COUNTER` -- exactly the
"counter-role churn" failure mode the contract warned about (dev25/dev26
showed perturbing OP12's own counter-attacking agent breaks its scoring
cadence and reads as a fake "improvement" that's really self-sabotage).

Fix (one isolated change, `gpu_env/_core/_bt_adaptive.py`): added a new
bounded-duration state, `bt_adapt_split_response_expiry_step` -- every time
the already-debounced `split_pressure_active` signal (requires several
consecutive qualifying steps, the "persistence" requirement) re-fires, the
expiry is pushed to `sim_step_count + 40`; the response decays 40 steps
after the pattern stops instead of either latching forever (OP11's choice)
or flickering (the bug). The role-override's per-agent assignment now
explicitly skips any agent already `ROLE_COUNTER`
(`assign = split_denial & eligible[:, j] & (out[:, j] != ROLE_COUNTER)`),
and the `blue_carry_any` gate was dropped so the response can commit
pre-pickup too (the base routing table already falls an
INTERCEPTOR-without-a-carrier back to the same target logic as DEFENDER, so
this reads as "prioritize defending the exposed lane" rather than
whatever role -- often ATTACKER -- the agent held before). A new pre-pickup
route branch targets whichever blue agent is farther from the field's
lateral center (the actual split threat) instead of the generic "nearest
intruder" fallback, which can pick the wrong (nearer) attacker during a
two-lane approach.

```text
                 Stage-1-only        Stage-1+2       delta
BLUE_RUSH             -0.500 (3/8)    -0.500 (3/8)    0.000
BLUE_SPLIT            +1.250 (8/8)    +0.875 (6/8)   -0.375
BLUE_TURTLE           +0.625 (4/8)    +0.625 (4/8)    0.000
BLUE_ESCORT           -2.000 (0/8)    -2.000 (0/8)    0.000
```

Isolation criterion cleanly met: SPLIT dropped meaningfully (WR 8/8 -> 6/8,
margin +1.250 -> +0.875) while RUSH, TURTLE, and ESCORT are bit-identical to
the Stage-1-only numbers -- confirms the fix is scoped to sustained-split
pressure and does not perturb anything else, and in particular does not
regress Stage 1's RUSH improvement.

**Round 1, Stage 3 -- late conversion vs TURTLE (2026-07-28): two
hypotheses tried, both falsified by direct evidence, mechanism still
unresolved.**

*Attempt A (rejected):* hypothesized "little meaningful offensive pressure
by a predeclared time" as blue never having touched red's flag
(`bt_adapt_blue_first_touch_step < 0`) and never showing sustained split
pressure, checked at step >=100, with the response being a route-only lane
split of red's two ATTACKER agents (so OP12's own two attackers stop
sharing one corridor -- `lane_y_pref` in `_bt_route_target` is built from
`red_script_lane_sign`, a single per-EPISODE value, not per-agent, so both
attackers travel the same lane and get pinch-tagged together by
`_blue_turtle_targets`'s single-target defense). Result: **never fired**.
`episode_results.csv` showed `time_to_first_score` of 48-95 steps in 6/8
TURTLE episodes and `blue_score >= 1` in 6/8 -- TURTLE's counter-attack
unlocks and converts well before step 100 in most games; TURTLE-vs-OP12 is
a competitive back-and-forth, not a stalling standoff. The premise was
wrong.

*Attempt B (rejected):* re-pointed the trigger at OP12's own scoreboard
instead of inferring blue behavior -- `red_score == 0` at step >=100 (fully
legal, no opponent-behavior inference needed at all). Re-ran the identical
screen: **still zero effect**, numbers bit-identical to Stage-1+2. Direct
role trace (`trace_op12_turtle_roles.py`, episode_seed=556001, the episode
where `red_score` stayed 0 for the full 240 steps) showed why: after the
step-20 opening, red's two agents are **never simultaneously
`ROLE_ATTACKER` again** -- by step 40 one is already `ROLE_COUNTER`, and by
step 95-105 (squarely inside the trigger window, `red_score` still 0) both
are `ROLE_COUNTER`. `ROLE_COUNTER`'s own base route already uses the
*opposite*-signed lane offset from `ROLE_ATTACKER` (`alt_lane_y = lane_mid -
red_script_lane_sign * lane_amp` vs `lane_mid + ... `), so once roles
diverge, they are already on separate lanes without any adaptive help --
the lane-split fix had nothing left to do. Despite that natural lane
separation, OP12's first score didn't land until ~t110, and the game
continued grinding (blue's own first score around t~200) -- the real
bottleneck is somewhere other than shared-lane pinching, and is not yet
identified.

Given two mechanistic hypotheses formed from reading the BT code have both
been falsified by direct trace evidence, further guessing at the TURTLE
mechanism without a dedicated diagnostic pass (in the style of the OP7/8/9
work, not a code-reading guess) is not a good use of the remaining budget.
**Stage 3 is parked, unresolved, pending either a real diagnostic pass or
user direction.** (Superseded below -- the inert attempt-B code was later
removed rather than left in place once Round 2's diagnostic pass replaced
it with a real, tested mechanism.)

**Round 1 status after Stages 1-3:**

```text
             BLUE_RUSH   BLUE_TURTLE   BLUE_SPLIT   BLUE_ESCORT
Round 1        -0.500       +0.625       +0.875       -2.000
baseline       -0.750       +0.875       +1.250       -2.125
```

RUSH improved (+0.250) and SPLIT was punished (-0.375), both isolated and
validated. TURTLE is unresolved (only the incidental -0.250 from Stage 1's
side effect, not a real punish mechanism). **RUSH is still not uniquely
best** -- SPLIT still leads RUSH by 1.375, the dominating-blue-style gate
still fails. Round 1 is not yet a pass. One round remains under the hard
budget before OP12 must be closed as a failed RUSH-niche candidate per the
locked contract.

**Round 2 -- dedicated TURTLE diagnostic, then a third (also failed)
mechanism (2026-07-28):**

Per explicit user direction, spent Round 2 on a real trace/event diagnostic
instead of a third code-reading guess. Two new scripts:
`experiments/diagnose_op12_vs_turtle.py` (event stream of every red TAG /
PICKUP / SCORE across the 8 dev seeds) and a carrier-tag follow-up trace
(scratchpad-only, checked whether each "tagged while carrying" failure was
a dual-carry priority conflict and what role the other red agent held).

Findings across the 8 episodes: 27 pickups, 16 red scores, 10 "red tagged
while carrying" (wasted pickup) events. Breaking those 10 down by cause:

```text
2  Stage-1's already-accepted early-window cost (other agent role=ATTACKER,
   t<=20 -- the known, accepted trade the RUSH-window fix made)
2  dual-carry structural conflict (blue ALSO carrying red's flag at that
   instant -> Priority-1 FLAG_RETR correctly preempts Priority-2 ESCORT;
   only 2 agents, cannot do both at once -- not fixable without making
   something else worse)
2  FLAG_RETR lock lingering briefly past its trigger condition
3  ESCORT correctly assigned to protect the carrier, but still failed to
   prevent the tag  <- largest single bucket
1  unattributed / episode ended mid-carry
```

The 3-case bucket pointed at a real, well-grounded mechanism: the shared
base ESCORT route (`_bt_route_target`'s "interpose" branch, used by every
opponent with `enable_escort`) targets the midpoint between the carrier and
only the SINGLE *nearest* blue agent
(`near_threat_idx = argmin` over both blue agents). TURTLE's own defense is
an explicit two-agent pincer (`_blue_turtle_targets`: both patrol targets
converge on the same red agent from offset angles) -- an escort that only
accounts for the nearer blue agent is structurally blind to the second one
closing from the other side.

**Attempt C (tried, then reverted):** an OP12-only adaptive-layer route
override (never touching the shared base logic other opponents rely on):
while escorting a carrier, target the midpoint between the carrier and the
*centroid* of both blue agents, pulled to a tighter 30% blend (vs the base
branch's fixed 50%). Isolated 8-seed screen
(`artifacts/op12_stage3c_escort_geometry_dev1_8seed`, same base_seed
556001, Stage 1+2 code left in place):

```text
                 Stage-1+2      Stage-1+2+3c(ESCORT-geom)   delta
BLUE_RUSH             -0.500 (3/8)   -0.500 (3/8)            0.000
BLUE_SPLIT            +0.875 (6/8)   +0.875 (6/8)            0.000
BLUE_TURTLE           +0.625 (4/8)   +0.750 (5/8)           +0.125  (worse)
BLUE_ESCORT           -2.000 (0/8)   -2.375 (0/8)           -0.125  (worse)
```

RUSH and SPLIT stayed bit-identical (clean isolation from that angle), but
TURTLE moved the WRONG direction (better for blue, not worse) and ESCORT
also regressed. Averaging the escort's aim-point toward both blue agents'
centroid apparently dilutes its positioning against the closer, more
immediate threat without adding real coverage against the farther one --
net negative, not neutral. **Reverted** (`gpu_env/_core/_bt_adaptive.py`);
confirmed the revert exactly reproduces the Stage-1+2 numbers
(`artifacts/op12_stage3_reverted_confirm_8seed`, bit-identical to the
Stage-1+2 row above). The dead attempt-B late-conversion code (from Round
1, already confirmed inert) was removed at the same time rather than kept
as unused dead code.

**Round 2 conclusion: three independent Stage-3 mechanisms attempted
(blue-passivity gate, shared-attacker-lane split, ESCORT-centroid
geometry), diagnosed via both code-reading and dedicated event-trace
evidence. Two never fired; the third fired and made TURTLE and ESCORT
worse. No working TURTLE-punish mechanism was found.** Both rounds of the
hard budget are now spent. Final state (Stage 1 + Stage 2 only, Stage 3
absent):

```text
             BLUE_RUSH   BLUE_TURTLE   BLUE_SPLIT   BLUE_ESCORT
Final          -0.500       +0.625       +0.875       -2.000
baseline       -0.750       +0.875       +1.250       -2.125
```

RUSH improved (+0.250, still negative) and SPLIT was punished (+1.250 ->
+0.875), both real and isolated. TURTLE and ESCORT are unchanged from
Round 1's incidental Stage-1 side effects. SPLIT still dominates
(distinct best responses: 1/1, `DOMINATING blue style: BLUE_SPLIT`); RUSH
is not uniquely best; `POOL ADMISSIBLE: False`.

**Decision: per the locked contract's hard stopping rule, close OP12 as a
failed RUSH-niche candidate.** Stage 1 (early RUSH window) and Stage 2
(bounded sustained-split response) are real, validated improvements and
should be kept -- they make OP12 a better-calibrated opponent regardless of
whether it ends up carrying the RUSH niche. The RUSH niche itself needs a
different opponent; that opponent choice and any further OP6/OP7-OP10
re-validation is open, pending user direction.

---

## 7. Cross-references

| Need                                              | Where to look                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Mandatory agent behavior                          | [`AGENTS.md`](../../AGENTS.md)                                                      |
| Scientific definition of the paper method         | [`summer-method-spec.md`](summer-method-spec.md)                                    |
| Fidelity rules / classification / proposal form   | [`summer-fidelity-rules.md`](summer-fidelity-rules.md)                              |
| Per-preset facts, aliases, deltas, run tags       | [`latent-preset-registry.md`](latent-preset-registry.md)                            |
| Launch / eval / statistical protocols             | [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)    |
| v6i2 frozen gate thresholds (pre-confirmatory)  | [`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md)                    |
| Codeâ†”manuscript trace                             | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)                    |
| Algorithm sketch                                  | [`../../docs/algorithm.md`](../../docs/algorithm.md)                                |
