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

> **Last updated:** 2026-09-26 — **METHODOLOGY IDENTITY LOCKED across 2v2/4v4/6v6.**
> PI: the contribution is **one methodology** — the same thing tested on different
> agent counts with **no other code/recipe changes**. Only scale knobs (`N`, `k`)
> may differ. Recorded in
> [`CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json`](../artifacts/strategic_demand/sppo/CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json)
> `#METHODOLOGY_IDENTITY_2026_09_26` (supersedes the old 2v2 “natural expert”
> allocator exception). Suite claims that mix legacy 2v2 Share-0 / Rung-1 rows with
> 4v4/6v6 split methodology are invalid until 2v2 is rebuilt under this identity.
> A paper sentence that needs a 2v2-only caveat is a methodology failure.
>
> **Conformance measured 2026-09-26 — the identity does NOT hold yet, and 6v6 fails it too.**
> `CROSS_SCALE_METHODOLOGY_IDENTITY_AUDIT.json`
> (`experiments/audit_cross_scale_methodology_identity.py`; settings read from the **loaded
> checkpoints**, since the 6v6 specialists carry no `run_config.json` and
> `sappo_anchor_lambda` is 0.10 by default everywhere — the real SA-PPO gate is a non-empty
> `sappo_anchor_dataset`). Seven axes disagree:
>
> | axis | 2v2 | 4v4 | 6v6 |
> |---|---|---|---|
> | SA-PPO anchor | **ON** | off | off |
> | entity repair | off | **ON** | off |
> | foundation steps | **1.5 M** | 1 M | 1 M |
> | dataset allocator | absent | **CD k=2** | absent |
> | dataset acting policy recorded | no | **yes** | no |
> | dataset entity tensors | absent | **yes** | absent |
> | dataset roles | absent | **yes** | absent |
>
> Matching across all three: 96 episodes/pole, 192 shards, decision-rows-only.
> Target (4v4 is the reference): anchor **OFF**, entity repair **ON**, 1 M foundation,
> dataset collected under CLOSEST_DEFENDS with entity+roles stored.
> **Open PI decision:** adopting it retrains the 2v2 suite teachers without the anchor,
> retiring the SAPPO-continuation 1.5 M lineage as suite teachers (the SAPPO V1 result itself
> is untouched). 6v6 needs entity repair (already in the school-PC pipeline) **and** a fresh
> dataset under CD k=1 (already directed by the suite spec).
> Plan + blockers + PI's certification decision: `SAME_METHODOLOGY_2V2_PORT_PLAN.json`.
>
> **PI ruling 2026-09-26 — 4v4 is the canonical reference; the old 2v2 SAPPO lineage is retired
> from the cross-scale suite (kept, not deleted).** Common recipe frozen in
> [`CROSS_SCALE_CANONICAL_RECIPE_V1.json`](../artifacts/strategic_demand/sppo/CROSS_SCALE_CANONICAL_RECIPE_V1.json):
> anchor OFF · 1 M base · entity repair ON · 200 k split with the N′ schedule reused · one shared
> collector under CLOSEST_DEFENDS storing entities+roles · 96 ep/pole · 192 shards ·
> decision-eligible only · same distillation path, evaluator and gates. Only \(N\) and \(k\) differ.
> **Key distinction:** the FINAL repaired artifacts enter the suite, never the historical ones that
> merely share a scale. SAPPO V1 stays a valid finding in its own right; the historical 6v6
> specialists remain legitimate warm-start bases (anchor OFF, 1 M), so **no 6v6 base restart**.
> **Runners generalized 2026-09-26 — layer 2 now passes: all 10 stages resolve to one
> cross-scale implementation (was 7/10 forked).** One code path, `--team-size 2|4|6`, only \(N\)
> and \(k\) changing. Removed: the `n == 2` legacy dataset route and legacy loader, the 2v2
> Share-Encoder reuse shortcut, the 2v2-only Pole-B overlay fork in the trainer, the `N != 2`
> branch in the separated crossover, and the 4v4-only collector/evaluator module constants.
> `collect_suite_distillation_states_4v4.py` → `collect_suite_distillation_states.py` (git mv);
> new `eval_suite_sharing_crossover.py`. Arm/scale authorization is now the `ARM_SCALES` table,
> not a team-size branch. **The certification gate was NOT weakened:** a 2v2 dry-run is now an
> accepted team size and then refuses with `FAIL-CLOSED: governing certification record not
> found: STRATEGIC_DEMAND_2v2_CERTIFICATION.json` (exit 1). 4v4 verified unchanged — all 36
> resolved profile fields identical with the non-live overlay added, and the pre/post dry-run
> differs only in `installed_overlay_keys`. Details and the remaining per-scale specs/seeds:
> `SAME_METHODOLOGY_2V2_PORT_PLAN.json#GENERALIZATION_DONE_2026_09_26`.
>
> ⚠ **Pre-existing test failure, unrelated to the above:**
> `tests/test_preset_resolution.py::test_resolved_configs_match_snapshot`. `tests/preset_snapshots.json`
> was last regenerated 2026-09-22 (`9ce9c840`) but five `PPOConfig` fields were added
> 2026-09-23 (`af8f76c2`): `fully_shared_z_conditioned_enabled`, `fully_shared_z_pole_match`,
> `role_conditioning_allow_pre_entity_base`, `role_k_defend`, `role_k_defend_choices`.
> 0 value diffs — purely missing keys. Not regenerated here: AGENTS.md requires a changelog
> entry in `Paper_experiment_alignment.md` §7 first.
>
> Attestation preflight: `experiments/attest_cross_scale_identity.py` →
> `CROSS_SCALE_IDENTITY_ATTESTATION.json`. It reads settings off the loaded checkpoints, treats a
> missing config as **UNKNOWN, never “same”**, and distinguishes PENDING (not built) from a
> mismatch. Current state: 4v4 matches on every invariant row; 2v2 and 6v6 are NOT YET ATTESTABLE
> (nothing built). Verified against a negative control (retired 2v2 + historical 6v6 artifacts →
> 10 mismatches caught, including every absent dataset field).
>
> ---
>
> Prior — 2026-09-25 — **4v4 Share-Backbone exploratory crossover DONE (FLAGGED); Share-Macro eval IN FLIGHT.**
> Share-Backbone (n=64, 3,532,656 params, holdout agree 0.901/0.862): Δ_A=+0.438 [+0.297,+0.578],
> Δ_B=−0.281 [−0.422,−0.125] — FLAG on Δ_B (z0 beats z1 on Pole B; z0 wins 0.875/0.891, z1 0.438/0.609).
> Record: `SUITE_SHARE_BACKBONE_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json`.
> Row-level integrity audit run for the three flagged 4v4 arms (Encoder, Backbone, Fully Shared+\(z\)):
> all mechanical checks pass (`SUITE_4V4_SHARING_FLAGGED_ARMS_ROW_AUDIT.json`,
> `experiments/audit_suite_sharing_4v4_flagged_arms.py`). Their KL teachers (π_A3→z0, corrected π_B3→z1)
> already fail Δ_B at 4v4 (Δ_B −0.156 [−0.266,−0.047], n=128), so the Pole-B reversal is consistent with
> imitating that pair and is not evidence that sharing removes specialization. No `FROZEN_RESULT` seal.
>
> **2v2 port under the locked identity** (owner: suite spec `#METHODOLOGY_IDENTITY_2026_09_26`; code state and
> blockers: `SAME_METHODOLOGY_2V2_PORT_PLAN.json`). The natural-setup 2v2 Fully Shared+\(z\) crossover launched
> 2026-09-25 21:02 was **stopped at 82/512** (partial retained; block 22900001–128 stays RESERVED, unspent; not a
> paper row). Existing 2v2 results stay as provenance. Blockers before any 2v2 training:
> `train_specialist_scale.py` supports only N∈{4,6}; no 2v2 pole-certification record for its gate; an
> `n == 2` branch in `_verify_live_pole`; 4v4-only collector and crossover runner to be generalized (after
> the running 4v4 Share-Macro eval finishes, so disk code equals run code). No training authorized yet.
> Share-Macro crossover eval: the chain launched it at 17:47 and a reboot at 18:29 killed it at 60/256
> (no partial results; killed logs kept as `crossover_eval.INTERRUPTED_reboot_1829.log*`).
> Relaunched from scratch 20:43 (pid 11464, seeds 22514001–064; 92/256 at 00:19). It ran 113–262 s/ep while
> a second eval shared the box and 26–53 s/ep alone; do not run another single-env eval beside it.
> Log: `suite_sharing/4v4/share_macro/crossover_eval.log.err`; watch: `depth_crossover_watch.log`.
> Its params (3,531,371) and imitation (0.899/0.859) are in the professor table; Δ cells pending.
> Table: [`paper/data/SHARING_TRADEOFF_2V2_4V4_FOR_PROFESSOR.md`](../paper/data/SHARING_TRADEOFF_2V2_4V4_FOR_PROFESSOR.md),
> [`paper/data/sharing_params_tradeoff_2v2_4v4.json`](../paper/data/sharing_params_tradeoff_2v2_4v4.json).
>
> ---
>
> Prior — 2026-09-25 — **4v4 Share-Backbone / Share-Macro depth extension IN FLIGHT.**
> PI-authorized diagnostic (same `SUITE_DISTILLATION_4V4` under CD \(k=2\)).
> Construction amendments frozen; suite distill runner extended. Both students froze
> at 11:10 / 11:23; sha pins written 11:24.
>
> ---
>
> Prior — 2026-09-25 — **4v4 suite exploratory crossovers DONE (both FLAGGED).**
> Fully Shared+\(z\): Δ_A=+0.281 [+0.125,+0.438], Δ_B=−0.109 [−0.219,0.000] — FLAG on Δ_B.
> Share-Encoder: Δ_A=+0.359 [+0.188,+0.516], Δ_B=−0.219 [−0.344,−0.109] — FLAG on Δ_B.
> n=64 exploratory; records:
> `SUITE_FULLY_SHARED_Z_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json`,
> `SUITE_SHARE_ENCODER_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json`.
> Professor table filled:
> [`paper/data/SHARING_TRADEOFF_2V2_4V4_FOR_PROFESSOR.md`](../paper/data/SHARING_TRADEOFF_2V2_4V4_FOR_PROFESSOR.md),
> [`paper/data/sharing_params_tradeoff_2v2_4v4.json`](../paper/data/sharing_params_tradeoff_2v2_4v4.json).
> Next suite blockers: 6v6 Separated seal; confirmatory n=128 not authorized.
>
> ---
>
> Prior — 2026-09-24 — **4v4 suite crossover evals IN FLIGHT**
> (exploratory n=64) for Fully Shared+\(z\) then Share-Encoder to fill Δ cells.
> Spec: [`SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json`](../artifacts/strategic_demand/sppo/SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json).
>
> ---
>
> Prior — 2026-09-24 — **4v4 Share-Encoder distilled student FROZEN.**
> Holdout agree \(0.914/0.876\), unique actor params \(3{,}636{,}592\), sha256
> `43e1a003…`. Record:
> `artifacts/strategic_demand/sppo/suite_sharing/4v4/share_encoder/STUDENT_FROZEN.json`.
> Same dataset as Fully Shared+\(z\) (`SUITE_DISTILLATION_4V4` under CD \(k=2\)).
> 4v4 sharing-family distill arms both frozen. Next suite blockers: 6v6 Separated
> seal, then Generalists / matched evals.
>
> ---
>
> Prior — 2026-09-24 — **4v4 Share-Encoder distillation IN FLIGHT**
> on the same `SUITE_DISTILLATION_4V4_DATASET` (CD \(k=2\)). Construction:
> [`SUITE_SHARE_ENCODER_4V4_CONSTRUCTION_AMENDMENT.json`](../artifacts/strategic_demand/sppo/SUITE_SHARE_ENCODER_4V4_CONSTRUCTION_AMENDMENT.json).
> Preflight PASS; unique params \(\approx 3.64\)M (shared CNN). Watch terminal `5534`.
>
> ---
>
> Prior — 2026-09-24 — **4v4 Fully Shared+\(z\) distilled student FROZEN.**
> Holdout agree \(0.895/0.856\), unique actor params \(3{,}468{,}503\), sha256
> `dfa82051…`. Record:
> `artifacts/strategic_demand/sppo/suite_sharing/4v4/fully_shared_z/STUDENT_FROZEN.json`.
> Dataset: `SUITE_DISTILLATION_4V4_DATASET` under CD \(k=2\). Next suite step:
> Share-Encoder at 4v4 on the same frozen set (after construction amendment).
>
> ---
>
> Prior — 2026-09-24 — **SUITE_DISTILLATION_4V4_DATASET FROZEN** under
> `CLOSEST_DEFENDS(k=2)`. Manifest:
> [`SUITE_DISTILLATION_4V4_DATASET.json`](../artifacts/strategic_demand/sppo/SUITE_DISTILLATION_4V4_DATASET.json)
> (192 shards; A 19904 / B 19949 decision rows; seeds `22520001..096` /
> `22520101..196`). Next: distill Fully Shared+\(z\) then Share-Encoder on that
> same set via `experiments/run_suite_sharing_distillation.py --team-size 4`.
>
> ---
>
> Prior — 2026-09-24 — **SUITE_DISTILLATION_4V4 collection IN FLIGHT**
> under `CLOSEST_DEFENDS(k=2)`. Spec frozen:
> [`SUITE_DISTILLATION_4V4_SPEC.json`](../artifacts/strategic_demand/sppo/SUITE_DISTILLATION_4V4_SPEC.json).
> Collector: `experiments/collect_suite_distillation_states_4v4.py` (Pole A =
> sealed DEFEND_ATTACK_SPLIT composite; Pole B = entity-repair `pi_B`; entities +
> roles stored). Seeds `22520001..096` / `22520101..196`. Smoke plumbing PASS.
> When the manifest freezes, distill Fully Shared+\(z\) then Share-Encoder on that
> same set.
>
> ---
>
> Prior — 2026-09-24 — **2v2 Fully Shared+\(z\) distilled student FROZEN.**
> Holdout agree \(0.965/0.988\), unique actor params \(3{,}456{,}599\), sha256
> `4aebecd8…`. Record:
> `artifacts/strategic_demand/sppo/suite_sharing/2v2/fully_shared_z/STUDENT_FROZEN.json`.
> Next suite step still: collect `SUITE_DISTILLATION_4V4` under CD \(k=2\).
>
> ---
>
> Prior — 2026-09-24 — **Suite fairness rule confirmed in ops.** Distillation
> arms fail-closed until matched state sets exist. Inventory:
> 2v2 Separated/Share-Encoder/Generalist available, Fully Shared+\(z\) training
> (unique actor params \(3{,}456{,}599\)); 4v4 Separated sealed, both sharing arms
> blocked on CD \(k=2\) dataset; 6v6 Separated critical path, sharing arms blocked.
> Next collectable dataset: `SUITE_DISTILLATION_4V4` under `CLOSEST_DEFENDS(k=2)`
> from the sealed mature split; 6v6 collection waits on Separated seal then \(k=1\).
>
> ---
>
> Prior — 2026-09-24 — **Suite distillation implementation started (do not wait
> on the 6v6 seal for the code).** `experiments/run_suite_sharing_distillation.py`
> + `rl/suite_fully_shared_distill.py`. **2v2 Share-Encoder reused** (sealed Rung-1).
> **2v2 Fully Shared+\(z\)** preflight 7/7; 20-epoch distillation launched on CPU
> (GPU was occupied). **4v4/6v6 both arms fail-closed** until
> `SUITE_DISTILLATION_{4,6}V{4,6}_DATASET.json` exists under `CLOSEST_DEFENDS`.
>
> ---
>
> **Last updated:** 2026-09-24 (cross-scale baseline suite FROZEN) —
> **Suite:** [`CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json`](../artifacts/strategic_demand/sppo/CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json)
> — same four arms at 2v2/4v4/6v6 (Generalist / Fully Shared+\(z\) / Share-Encoder /
> Separated), distillation identity for the sharing family, allocator fairness
> (`4v4: CLOSEST_DEFENDS k=2` / `6v6: k=1`; 2v2 natural). **Do not train suite arms
> now.** Critical path remains sealing repaired 6v6 Separated; then prioritize
> standardized Fully Shared+\(z\) at all three scales, then Share-Encoder under the
> same locks. Depth ladder (Backbone/Macro) stays **2v2-only**. Prior one-off
> fully-shared PPO SPECs are plumbing children under this suite, not complete
> baselines by themselves.
>
> **Final paper scale story:** `2v2 → 4v4 → repaired 6v6`, then suite curves.
>
> | Piece | Final paper status |
> |-------|--------------------|
> | 2v2 specialists + sharing ladder (depth) | **Include** |
> | 4v4 split crossover PASS | **Include** |
> | Old 6v6 Share-0 / Share-Encoder (no CLOSEST_DEFENDS) | **Exclude** from suite/narrative |
> | Repaired/mature 6v6 Separated | **Include once sealed** — critical path |
> | Cross-scale suite (4 arms × 3 scales) | **Frozen design; build after 6v6 seal** |
>
> Compute path unchanged: same algorithmic recipe, scale-specific config
> (`4v4: N=4,k=2` / `6v6: N=6,k=1`). Mature A/B → frozen ATTACK `π_A` → `π_D` +200k +
> identical N′ teacher → `CLOSEST_DEFENDS` → same `Δ_A`/`Δ_B` LCB95 gate.
> School PC: [`SCHOOL_PC_6V6_LOCKED_PIPELINE.json`](../artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_LOCKED_PIPELINE.json),
> [`experiments/SCHOOL_PC_6V6_README.md`](../experiments/SCHOOL_PC_6V6_README.md).
> Fail-closed preflight before grind. Home c2 STOPPED.
>
> ---
>
> Prior — 2026-09-23 (paper scale narrative LOCKED) —
> **Final paper scale story:** `2v2 → 4v4 → repaired 6v6`.
>
> | Piece | Final paper status |
> |-------|--------------------|
> | 2v2 specialists + sharing ladder | **Include** |
> | 4v4 split crossover PASS | **Include** (reverses 2026-09-11 “exclude 4v4” draft note) |
> | Old 6v6 Share-0 / Share-Encoder compression results | **Exclude** from narrative (archive/provenance only) |
> | Repaired/mature 6v6 split (school-PC pipeline) | **Include once sealed** — only 6v6 result in the main story |
> | Fully shared strategy-conditioned baseline | **Run after 6v6, then include** |
>
> Compute path unchanged: same algorithmic recipe, scale-specific config
> (`4v4: N=4,k=2` / `6v6: N=6,k=1`). Mature A/B → frozen ATTACK `π_A` → `π_D` +200k +
> identical N′ teacher → `CLOSEST_DEFENDS` → same `Δ_A`/`Δ_B` LCB95 gate.
> School PC: [`SCHOOL_PC_6V6_LOCKED_PIPELINE.json`](../artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_LOCKED_PIPELINE.json),
> [`experiments/SCHOOL_PC_6V6_README.md`](../experiments/SCHOOL_PC_6V6_README.md).
> Fail-closed preflight before grind. Home c2 STOPPED.
>
> ---
>
> Prior — 2026-09-23 — **6v6 closure criterion is the 4v4 recipe, not SDS observability.**
> Finish line: `CLOSEST_DEFENDS(k)` + frozen ATTACK + learned `π_D`, then the same sealed
> crossover gate (`Δ_A>0`, `Δ_B>0`, both LCB95>0). `V3_STRATEGIC_DEMAND_NOT_VALIDATED`
> stays a separate negative observability experiment
> ([`HUMAN_DECISION_REQUIRED_OBSERVABILITY.md`](../artifacts/strategic_demand/HUMAN_DECISION_REQUIRED_OBSERVABILITY.md),
> [`CONFIRMATION_SDS_G1_4_RESULT.json`](../artifacts/strategic_demand/CONFIRMATION_SDS_G1_4_RESULT.json)).
> It is not rewritten as a PASS, and it is not a gate on this closure path.
> Scale story: 2v2 crossover PASS; 4v4 heuristic allocation + learned strategy crossover PASS;
> 6v6 the same gate, still open.
>
> ---
>
> Prior — 2026-09-23 — **Fully Shared Strategy-Conditioned plumbing READY
> (train still deferred).** Implementation landed under the frozen SPECs:
> `rl/custom_ppo/fully_shared_z.py`, `experiments/train_fully_shared_strategy_conditioned.py`
> (`--authorize-launch` required for non-smoke), `experiments/eval_fully_shared_z_crossover_scaled.py`
> (forced-z + CLOSEST_DEFENDS + `run_state.seal`), `tests/test_fully_shared_z_conditioned.py`
> (13 passed). **Do not launch** while `CLOSEST_SPLIT` 6v6 viability owns the GPU.
> Seed reservation + 200k train + n=64 eval wait on C11 clear / other PC.
>
> ---
>
> Prior — 2026-09-23 — **Fully Shared Strategy-Conditioned baseline SPECs frozen
> (queue only; no train on this PC).** Sharing-axis port under identical `CLOSEST_DEFENDS`:
> [`FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.json`](../artifacts/strategic_demand/sppo/FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.json)
> (primary vs sealed separated 4v4 PASS) and contingent
> [`FULLY_SHARED_STRATEGY_CONDITIONED_6V6_V1_SPEC.json`](../artifacts/strategic_demand/sppo/FULLY_SHARED_STRATEGY_CONDITIONED_6V6_V1_SPEC.json)
> (k pins deferred to sealed separated 6v6 composition). Single network `π_φ(a|o,z)`, no
> router, forced-z crossover, same Δ/LCB95/seal machinery. **Do not launch while
> `CLOSEST_SPLIT` 6v6 viability owns the GPU.** Implementation remaining is listed in the
> 4v4 SPEC; this is freeze-only.
>
> ---
>
> Prior — 2026-09-23 — **4v4 = DONE** under the frozen closure rule
> (*heuristic role allocation + learned strategy* → confirmatory crossover PASS).
> Evidence already on disk: `DEFEND_ATTACK_SPLIT_POLICY_A_V1` trained `pi_D` under
> `CLOSEST_DEFENDS` (allocator on during training; frozen `pi_A` on ATTACK slots), then
> [`DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_SPECIALIST_CROSSOVER_EVAL_RESULT.json`](../artifacts/strategic_demand/sppo/DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_SPECIALIST_CROSSOVER_EVAL_RESULT.json)
> sealed **joint gate PASS** on fresh n=128 (seeds 21800001–128):
> Δ_A = +0.156 [+0.039, +0.273], Δ_B = +0.406 [+0.297, +0.508]. Practical paper status
> matches 2v2. **Do not train another 4v4 closure run.** All attention goes to 6v6
> (current closest-split viability in flight).
>
> **FROZEN 4v4 CLOSURE RULE (PI):** confirmatory crossover PASS under *strategy learned,
> role assignment heuristic* closes 4v4. Heuristic assignment alone does not keep the lane open.
>
> ---
>
> Prior — 2026-09-23 — This PC is a **25k viability test** only, not the 200k run. After the checkpoint, an n=16 k=1 vs k=3
> screen on both poles decides: promising → full 200k on the other PC; unclear → 50k here and screen again; obvious
> collapse → stop.
>
> The 25k viability screen is complete:
> [`CLOSEST_SPLIT_VIABILITY_6V6_25000.json`](../artifacts/strategic_demand/sppo/CLOSEST_SPLIT_VIABILITY_6V6_25000.json).
> Cell win rates were `k=1`: Pole A `0.8125`, Pole B `1.0`; `k=3`: Pole A
> `0.875`, Pole B `1.0`. Paired contrasts were `Delta_A=-0.0625`
> `[-0.25,+0.125]` and `Delta_B=0.0` `[0,0]`. Neither composition collapsed,
> but the B improvement criterion was not met. Frozen ladder decision:
> **UNCLEAR**, so the one-time 50k continuation is the only remaining
> viability step; this is not a crossover result or 200k authorization.
>
> ---
>
> Prior — 2026-09-23 (later) — **`CLOSEST_COMPOSITION_CONTRAST_6V6` SEALED: `pattern_match = true`. Geometry-assigned
> scripted 5A/1D vs 3A/3D recreates the sealed FIXED_IDENTITY composition signature. Do not auto-train `pi_D`; PI gate on
> confirmatory n / heuristic+learned bridge.**
>
> Sealed
> [`CLOSEST_COMPOSITION_CONTRAST_6V6_RESULT.json`](../artifacts/strategic_demand/sppo/CLOSEST_COMPOSITION_CONTRAST_6V6_RESULT.json)
> (block `22100001-064`, 256 episodes, scripted macros, `CLOSEST_DEFENDS` `k_D=1` vs `k_D=3`):
>
> | contrast | mean | LCB95 | UCB95 |
> |---|---|---|---|
> | Delta_B (5A/1D − 3A/3D, Pole B) | **+0.594** | +0.453 | +0.719 |
> | Delta_A (5A/1D − 3A/3D, Pole A) | **+0.109** | −0.031 | +0.250 |
>
> Cell WR: Pole B `0.75` vs `0.16`; Pole A `0.83` vs `0.72`. Matches the direction of
> `FIXED_ATTACK_HEAVY_6V6_CONFIRMED` (identity assignment; B +0.69, A +0.125 at n=192). Point estimates only at n=64 —
> not a confirmatory label. **Reading:** the successful asymmetric compositions survive heuristic role assignment; the
> prior failure was specifically “add one defender to generalist `pi_A` vs unchanged `pi_B`,” not “5A/1D vs 3A/3D under
> CLOSEST_DEFENDS.” Next is a PI decision (confirmatory n, or strategy-learned + heuristic allocation bridge) — still no
> auto-`pi_D`.
>
> ---
>
> Prior — 2026-09-23 — **`CLOSEST_DEFENDS_SCREEN_6V6` SEALED: `pattern_match = false`. Closest-to-home fixed 5A/1D on
> frozen `pi_A` is safer than seed-rotated +1D, but it is not enough to create 6v6 crossover. Do not train `pi_D` from that
> scaffold. Next targeted test: geometry-assigned scripted 5A/1D vs 3A/3D (`CLOSEST_COMPOSITION_CONTRAST_6V6`), not a k-sweep
> and not `pi_D`.**
>
> Sealed result
> [`CLOSEST_DEFENDS_SCREEN_6V6_RESULT.json`](../artifacts/strategic_demand/sppo/CLOSEST_DEFENDS_SCREEN_6V6_RESULT.json)
> (block `22000001-064`, 384 episodes, frozen existing 6v6 specialists, `k_D=1`):
>
> | contrast | mean | LCB95 | UCB95 |
> |---|---|---|---|
> | Delta_A_closest (`A_closest` vs `pi_B`, Pole A) | **-0.016** | -0.141 | +0.109 |
> | Delta_B_closest (`pi_B` vs `A_closest`, Pole B) | +0.094 | +0.016 | +0.188 |
> | I_B (`A_closest` vs native `pi_A`, Pole B) | **-0.078** | -0.172 | +0.016 |
>
> Cell means: `A_closest` 0.86/0.88 vs native `pi_A` 0.89/0.95 (Poles A/B). Selection rule mattered relative to
> `DEFENDER_INJECTION_CAUSAL_BRIDGE` (`ONE_DEFENDER_HARM_ONLY`, Pole-B resolved harm much harsher). But Pole A stays a
> near-tie / slight loss vs generalist `pi_B`, so the composite does not recreate crossover. **Frozen branch rule:** do
> not auto-start 6v6 `pi_D` training from this scaffold — there is not enough causal evidence a learned one-defender
> policy would solve it.
>
> **Recorded reading (PI, 2026-09-23):** closest-to-home selection removes much of the prior +1D harm but does not produce
> 6v6 crossover when only the A-side composition is changed. Next experiment is **not** random `k=2`/`k=3` on learned
> specialists; it is the geometry-assigned version of the sealed scripted contrast that *did* separate
> (`FIXED_ATTACK_HEAVY_6V6_CONFIRMED`: 5A/1D vs 3A/3D): `CLOSEST_COMPOSITION_CONTRAST_6V6` — scripted macros, `CLOSEST_DEFENDS`
> `k_D=1` vs `k_D=3`, exploratory `n=64`, block `22100001-064`. If that recreates the signature, bridge is
> strategy-learned + role-assignment heuristic. If it fails, stop rescuing old 6v6 checkpoints with allocation tricks.
>
> ---
>
> Prior — 2026-09-22 (later) — **`CLOSEST_DEFENDS_SCREEN_4V4` COMPLETE: `pattern_match = true`. Both load-bearing pieces
> of the target 4v4 training architecture now have empirical support: GO_TO is sufficient enough (`GOTO_ONLY_DEFEND_SUBSTITUTION_4V4`,
> below) and a fixed distance-based greedy allocator preserves the useful 2A/2D effect (this experiment). Research question shifts
> from "can this architecture work?" to "can PPO learn competent role-conditioned behavior under this fixed allocator?"**
>
> `CLOSEST_DEFENDS_SCREEN_4V4` was an EXPLORATORY DESCRIPTIVE screen, not a confirmatory crossover test -- no LCB95 gate, no
> terminal pass/fail label, by design. It reproduced the sealed scaffold bridge's signature with `pattern_match: true`: Delta_A and
> Delta_B were positive, I_B was strongly negative, Pole A performance was preserved, and all six possible defender pairs were
> selected across seeds. Pre-run package (spec, runner, 13/13 contract record, seed reservation) committed as `f4f4ac6f` before any
> seed in the fresh exploratory block 21300001-21300064 was touched; one self-referential contract bug (a banned-string check
> matching its own declaration, not a real call site) was found and fixed before that freeze. 64 fresh exploratory seeds x 2 poles
> x 3 arms = 384 episodes, all recorded and analyzed cleanly (two shards needed one retry after an unrelated GPU-contention crash
> from launching alongside `GOTO_ONLY_DEFEND_SUBSTITUTION_4V4`'s still-running shards; no data was lost, only redone).
>
> Mechanism: `A_closest` = `pi_A` + the two ACTIVE agents closest to `own_flag_home` at the first decision tick (state-dependent,
> fixed for the whole episode) forced to DEFEND via the SAME mechanism as the sealed `A'` -- the only variable changed is the
> pair-selection rule, from seed rotation to distance-to-home.
>
> | | sealed bridge (seed-rotated pair) | `CLOSEST_DEFENDS` (distance-to-home pair) |
> |---|---|---|
> | Delta_A' | +0.352 [+0.234, +0.461] | +0.266 [+0.125, +0.406] |
> | Delta_B' | +0.359 [+0.250, +0.469] | **+0.422** [+0.297, +0.547] |
> | I_A (A's own-pole win rate) | -0.094 [-0.203, +0.016] | **0.000** [-0.141, +0.141] -- exact tie with native `pi_A` |
> | I_B (A's Pole-B competence) | -0.492 [-0.594, -0.391] | **-0.625** [-0.734, -0.500] |
>
> Cell means: `A_closest` wins Pole A exactly as often as native `pi_A` (0.781 = 0.781) while collapsing to 0.047 on Pole B (even
> lower than the bridge's `A'` at 0.133). Blue-goal and margin currencies move the same direction with equal or larger magnitude.
> Diagnostic: all 6 possible unordered agent-pairs were selected across the 64 seeds (12-30 occurrences each), confirming the rule
> is genuinely state-dependent, not degenerate. Point estimates only -- n=64, no confirmatory claim -- but the direction and
> magnitude are unambiguous and consistent across win rate, goals, and margin.
>
> **What this does and does not support.** Supports using a fixed distance-based greedy allocator as the role-assignment layer:
> `greedy_role_allocator(N, k_D)` with `k_D(4)=2`, closest-to-base -> DEFEND, computed once per episode, kept active at evaluation.
> Does NOT itself authorize PPO or any training -- that remains the next, separate PI decision. Says nothing about `pi_B`, a dynamic
> per-tick reassignment variant (explicitly out of scope here), or any team size other than 4v4. Per the PI: after this commit, stop
> running diagnostic screens and move to the actual training architecture -- `CLOSEST_DEFENDS` allocator -> ATTACK/DEFEND role label
> -> role-conditioned PPO, allocator kept at evaluation -- rather than a further vocabulary or allocation-rule test.
>
> **Earlier 2026-09-22 — `GOTO_ONLY_DEFEND_SUBSTITUTION_4V4` SEALED: `GOTO_ONLY_CROSSOVER_CONFIRMED`, with two caveats attached
> beside the headline, not buried later: Pole A crossover is confirmed but THIN (LCB95 = +0.016), and `N'` did NOT satisfy the
> predeclared path-fidelity gate on Pole A, so faithful scaffold-path reproduction is NOT claimed. Strongest supported conclusion:
> instantaneous heading fidelity is not necessary for 4v4 crossover -- a GO_TO-only defender with substantially worse heading
> agreement than the scaffold still preserved crossover. Clean next implication: we do not need a new DEFEND primitive before
> training. This is the last diagnostic in this line per its own frozen commitment -- no further metric exploration, no re-run.**
>
> Pre-run package (spec, runner, 15/15 contract record, seed reservation) committed as `541e9bab` before any seed in block
> 21200001-21200128 was touched; two real contract bugs were found and fixed before that freeze (`GPUCTFVecEnv.step_wait()`
> auto-resets core buffers on episode end, so 7 apparent boundary/commit-state mismatches were reading next-episode buffers, not
> violating the commit lock; a wording self-test checked for "necessary" against text that correctly said "necessity"). Result
> sealed 2026-09-22 (0/48 gating checks failed, 1024/1024 rows, attestation clean: 256 `N'` cells, 85,861 verified boundary
> events, 396 correctly excluded as unverifiable at episode-end resets).
>
> `N'` = `pi_A` + a causal, oracle-informed controller restricted to the native `(GO_TO, legal waypoint)` surface on the same two
> agents the sealed scaffold `A'` force-injects DEFEND onto; no PPO, no new vocabulary, `pi_B` untouched.
>
> | primary win-rate contrast | mean | LCB95 | UCB95 |
> |---|---|---|---|
> | Delta_A_N (`N'` vs `pi_B`, Pole A) | +0.141 | **+0.016** | +0.258 |
> | Delta_B_N (`pi_B` vs `N'`, Pole B) | +0.438 | +0.336 | +0.539 |
>
> Both lower bounds clear zero -> `GOTO_ONLY_CROSSOVER_CONFIRMED` by the frozen precedence. **Frozen interpretation text, verbatim:**
> "Instantaneous DIRECTION is not necessary for crossover: a defender restricted to native (GO_TO, legal waypoint) actions, which
> removes per-tick heading fidelity relative to the scaffold, preserved the crossover. This does not mean the DIRECTION metric was
> wrong; it measured a real mismatch that was not behaviorally necessary for the endpoint. `N'` also did not meet the pre-declared
> in-env path criterion, so it is not claimed to reproduce the scaffold's path."
>
> **Why the path-fidelity caveat fired (`check_pass_n=False`):** the interaction-free manipulation gate (RMSE <= 2.5 cells,
> LCB95 >= 0.90, same population definition as the representability audit) passed clearly on Pole B (0.980) but came back
> BORDERLINE on Pole A (0.883 [0.834, 0.923] -- LCB just under 0.90). The positive control `A'` passed comfortably on both poles
> (0.948 / 0.962), so the instrument itself is not in question; `N'`'s own path simply did not clear the bar on Pole A.
>
> **Direction was robustly, not marginally, manipulated:** realized per-tick heading agreement dropped from `A'`'s 0.42 / 0.29 to
> `N'`'s 0.19 / 0.11 (Pole A / B); the paired `A'` minus `N'` contrast is clearly positive on both poles (LCB95 0.207 and 0.148,
> n=128 pairs each) -- a large, unambiguous gap, licensing the "not necessary" statement rather than the weaker "no statement made"
> branch of the frozen tree.
>
> **Other honesty notes (non-gating):** the automated split-half check flagged a sign flip on a purely descriptive margin contrast
> (`N'` minus `A'` on Pole A: +0.047 -> -0.016 between halves) -- does not touch the primary win-rate gate, but is consistent with
> Pole A being the noisier axis in this comparison, alongside its thin primary margin.
>
> **What this does and does not authorize.** Per the frozen `POST_RUN_COMMITMENT_FROZEN`: this experiment is the LAST diagnostic
> in the vocabulary-necessity line -- no further metric exploration, no second manipulation-check definition, no re-run of this
> question, regardless of the caveats above. It licenses moving to a training decision (teacher-guided PPO warm-started from
> `pi_A`, see the PROPOSED item below and [[teacher-guided-ppo-successor-plan]] in memory) without first building a new DEFEND
> primitive. It does NOT itself authorize PPO -- that remains a separate PI decision -- and it says nothing about `pi_B`, 6v6, or
> whether PPO will actually learn the behavior the greedy oracle can merely express.
>
> **Earlier 2026-09-21 (later) — 4v4 SCAFFOLD-TO-NATIVE REPRESENTABILITY AUDIT COMPLETE: frozen label `NOT_REPRESENTABLE_VOCABULARY` on a valid audit, driven entirely by the DIRECTION sub-gate; TARGET and TRAJECTORY both pass at high coverage. READ-ONLY DIAGNOSTIC: no PPO, no new seed, `pi_B` not loaded. The label authorizes nothing.**
> Spec [`SCAFFOLD_TO_NATIVE_REPRESENTABILITY_AUDIT_4V4_V1_SPEC.json`](../artifacts/strategic_demand/sppo/SCAFFOLD_TO_NATIVE_REPRESENTABILITY_AUDIT_4V4_V1_SPEC.json)
> (amendment trail A1-A5 from the PI review), runner `experiments/audit_scaffold_to_native_representability_4v4.py` and the passing
> 17/17 contract record were committed together as `59bb5073` before the replay started. The replay re-ran the 256 sealed A' cells
> (block 21100001-128, registry entry untouched and byte-identical, no outcome inference drawn); every cell reproduced its sealed row
> and the `pi_A` parameter digest was unchanged. Question: could native `pi_A`'s existing 4v4 interface (5 macros, a 50-waypoint grid,
> the real action mask, real commit lengths) have produced the defender targets the scaffold injected? Witness set = legal `GO_TO`
> waypoints only (no semantic macro used as a steering primitive); G6 thresholds unchanged (2.5 cells / cos 0.99 / 16 ticks); PASS iff
> episode-clustered LCB95 >= 0.90 (this audit's own pre-declared level). Result
> [`..._RESULT.json`](../artifacts/strategic_demand/sppo/SCAFFOLD_TO_NATIVE_REPRESENTABILITY_AUDIT_4V4_RESULT.json), per-cell
> [`..._CELLS.csv`](../artifacts/strategic_demand/sppo/SCAFFOLD_TO_NATIVE_REPRESENTABILITY_AUDIT_4V4_CELLS.csv).
>
> | gate (unit) | Pole A | Pole B |
> |---|---|---|
> | TARGET (ACTIVE tick: some legal waypoint within 2.5 cells of the DEFEND target) | 0.981 [0.979, 0.983] PASS | 0.984 [0.982, 0.985] PASS |
> | DIRECTION (ACTIVE tick: existence over legal waypoints of one aimed within 8 deg; TARGET-satisfying preferred) | **0.375 [0.343, 0.409] FAIL** | **0.281 [0.255, 0.308] FAIL** |
> | TRAJECTORY, native commit (16-tick window, RMSE <= 2.5 vs the isolated DEFEND path) | 0.964 [0.959, 0.969] PASS | 0.965 [0.961, 0.969] PASS |
> | TRAJECTORY, interruptible (commit 1; localization diagnostic) | 0.995 [0.993, 0.997] PASS | 0.995 [0.994, 0.997] PASS |
>
> Units: 48,068 / 56,576 ACTIVE ticks and 17,781 / 23,702 windows (A / B). Tagged ticks (14.7% / 7.9%) are excluded because the env forces
> them home regardless of macro. Legality violations 0, runtime-parameter mismatches 0, independent re-derivation clean.
>
> **Reading.** The interface can name the DEFEND target to within 2.5 cells on ~98% of ticks and, with hindsight, reproduce the
> defender's position PATH within 2.5 cells RMSE on ~96% of 16-tick windows under its real 4-tick commit lock (99.5% if interruptible,
> so the commit lock is a minor factor; 99.0% / 99.7% on the interaction-free windows). It cannot, with a coarse waypoint grid,
> reproduce the per-tick HEADING toward the DEFEND target on most ticks. Those are different questions; the frozen precedence made the
> per-tick heading decisive, hence the label. **Representability is not learnability:** the path-oracle is greedy and sees the reference
> future, so TRAJECTORY says the path CAN be expressed, not that PPO WILL find it, and a NOT_REPRESENTABLE label is not a proof of impossibility.
>
> **Frozen-gate limitation that only became visible on the population (author's miss):** the inherited G6 `direction_cosine` returns -1
> when the desired vector is zero and the candidate's is not. The defender is exactly at its DEFEND target on 18.1% / 23.0% of ACTIVE
> ticks, and within 1.5 cells of it on 39% / 49% (OUTWARD ticks, target = a boundary point; the own flag is 2 cells from the west wall).
> **Exploratory sensitivity (post hoc, same recorded states, does NOT replace the frozen verdict):** excluding exactly-zero ticks the frozen gate is
> 0.458 / 0.365 and pure existence over any legal waypoint is 0.681 / 0.596 -- still below 0.90, so the DIRECTION failure is not just that
> convention. Other measured drivers: OUTWARD ticks with the target < 1.5 cells away pass 0.4% / 0.3%; INWARD ticks with the flag at home
> pass 27.6% / 26.5% under the TARGET-satisfying-preferred rule (only waypoint idx 5, behind the flag, is within 2.5 cells) but 90.8% / 89.9% under pure
> existence; the own flag is away from home on 55% / 40% of INWARD ticks. Independent re-check of the pipeline's DIRECTION flags with the
> unchanged `direction_cosine`: 425/425 sampled ticks identical.
>
> **Other diagnostics (non-gating):** the exact semantic macro for INWARD, `GO_HOME`, is masked for a no-payload defender on 47% / 62% of
> INWARD ticks (legal substitution costs 0.43 / 0.44 cells on average; admitting every legal macro would change TARGET by 0.000 because
> GO_HOME is legal on only 101 / 562 ACTIVE ticks). G6-style committed-candidate comparator, all four checks jointly: 0.05% / 0.04% (the
> old flip-tracking failure, visible on real states). Isolated-vs-live position RMSE 0.66 / 0.58 cells (the live step's 0.5-cell
> avoid-collision shove; every one-step miss verified to have a neighbour in shove reach). **Open discrepancy, carried not edited:**
> `HOME_LEGALITY_CONFIRMED_INTERFACE_REPAIR.json` says the engine never blocks `GO_HOME` for a non-carrier -- true of the engine decode,
> false of the policy-facing mask that `pi_A` actually sees.
>
> **Narrow reading of the frozen result (PI wording, adopted as the record's summary):** under the preregistered representability
> criteria, the existing GO_TO vocabulary fails because it cannot reproduce scaffold-relative instantaneous direction at the required
> 90% coverage. Target and short-horizon trajectory representability remain high. **Whether instantaneous direction is behaviorally
> necessary for crossover is unresolved.** "DIRECTION fails" is established; "DIRECTION is necessary" is not, and the stronger claim
> "GO_TO cannot express the behavior needed for crossover" is NOT established by this audit. The frozen label stays exactly as computed.
>
> **Routing per the frozen rule:** PPO stays parked; the pre-declared repair question is the minimum primitive that names the desired
> defender target. That routing is a recommendation the label carries by pre-declaration; because necessity is unresolved, no new
> primitive (e.g. a learned-selectable DEFEND) is proposed on the strength of this audit alone. Any re-scoring with a different DIRECTION
> definition, or an env-level test with a native-legal defender controller, is a NEW pre-registered analysis, not a relabeling of this one.
>
> **Earlier 2026-09-21:** **4v4 SCAFFOLDED-A CROSSOVER BRIDGE SEALED: `SCAFFOLDED_CROSSOVER_CONFIRMED` (fresh sealed_confirmatory seeds 21100001-21100128, 30/30 gating checks, run from the pre-run commit `ef5ee95f`). IMPOSING 2A/2D ON `pi_A` RECOVERS THE A/B PAYOFF SEPARATION AGAINST THE EXISTING `pi_B`. THIS IS A SCAFFOLDED-CONTROLLER RESULT, NOT A LEARNED CROSSOVER.**
> Spec [`SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_V1_SPEC.json`](../artifacts/strategic_demand/sppo/SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_V1_SPEC.json),
> committed together with the runner, the seed reservation and the passing 12/12 contract record (including
> exact reproduction of 16 sealed episodes) before any seed in the block was touched; the sealed result's
> own `git_sha` field is that commit. Arms on both certified 4v4 poles (Pole B the B3-3 genome, never
> canonical OP7), the same 128 paired seeds, 768 full episodes, deterministic, PPO off: native `pi_A3`;
> `A'` = `pi_A3` with two of the four agents' resolved targets forced to DEFEND's (pair rotated over the six
> agent pairs by seed); native corrected `pi_B3`. Motivation: every earlier 4v4 learned repair passed
> `Delta_A` and failed `Delta_B` because `pi_A3` is a strong generalist (0.74 on Pole B).
>
> | win rate | Pole A | Pole B |
> |---|---|---|
> | native `pi_A` | 0.813 | 0.625 |
> | `A'` = `pi_A` + 2D | 0.719 | 0.133 |
> | native `pi_B` | 0.367 | 0.492 |
>
> **Primary gate:** `Delta_A'` = `WR(A',A) - WR(pi_B,A)` = **+0.352 [+0.234, +0.461]**; `Delta_B'` =
> `WR(pi_B,B) - WR(A',B)` = **+0.359 [+0.250, +0.469]**; both lower bounds above zero, so the frozen label is
> `SCAFFOLDED_CROSSOVER_CONFIRMED`. **Mechanism diagnostics (not gates):** `I_A` = -0.094 [-0.203, +0.016]
> (A's win rate on its own pole is roughly preserved), `I_B` = -0.492 [-0.594, -0.391] (A's competence on Pole
> B is suppressed) -- the pattern the design hoped for. The native crossover on the fresh block reproduces the
> earlier failure (`Delta_A` +0.445 [+0.336, +0.547], `Delta_B` -0.133 [-0.242, -0.023]; sealed baseline
> +0.273 / -0.156), so the fresh block is not unusual. No sign flip between split halves in any of the 18
> re-derived claims.
>
> **Descriptive, non-gating:** Blue goals `Delta_A'` +1.18, `Delta_B'` +0.59, `I_A` +0.63 [+0.38, +0.86],
> `I_B` -1.23 [-1.48, -0.98]; margin `Delta_A'` +0.82, `Delta_B'` +1.16, `I_A` +0.16 [-0.10, +0.42], `I_B`
> -2.07. From the cell means (Red goals = Blue goals - margin), `A'` concedes far more than native `pi_A`: about
> 0.09 -> 0.55 Red goals per episode on Pole A and 0.01 -> 0.84 on Pole B.
>
> **What this establishes:** imposing the balanced 2A/2D structure on `pi_A` is sufficient to recover the 4v4
> A/B payoff separation against the existing `pi_B`, at the scaffolded-controller level. **What it does not:**
> `pi_B` is still weak (0.367 on Pole A, 0.492 on its own Pole B, below one half); the separation is created by
> weakening A on B, not by B becoming a strong specialist. `A'`'s payoffs resemble the scripted 2A/2D records
> (about 0.11-0.15 on Pole B, 0.74-0.75 on Pole A) but that is a resemblance across different seed blocks and
> attackers, not a test. It is not a fully learned crossover and must not be reported as one. Per the frozen
> branch rules nothing further is run from this experiment; PPO to internalize the structure into `pi_A`,
> removal of the scaffold, and any fully learned crossover are a separate decision, not started.
>
> Prior — 2026-09-21 — **ONE-DEFENDER GOAL-VOLUME CONFIRMATION SEALED: `OPPONENT_CONDITIONED_GOAL_INTERACTION_CONFIRMED` (fresh sealed_confirmatory seeds 21000001-21000096, 20/20 gating checks, independent re-derivation exact). THE POLE-A GAIN AND POLE-B LOSS IN GOAL PRODUCTION REPLICATE. WORDED "OPPONENT-CONDITIONED ACROSS THE TWO CERTIFIED 6v6 POLES"; IT DOES NOT SEPARATE THAT FROM COMPRESSION TOWARD A COMMON GOAL LEVEL.**
> Spec [`ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_V1_SPEC.json`](../artifacts/strategic_demand/sppo/ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_V1_SPEC.json)
> (frozen before any seed was spent; result and audit beside it). The hypothesis was generated post hoc
> from the exploratory records on seeds 20900001-096, which contribute nothing here. Same injection
> mechanism as the causal bridge (source sha256 pinned and re-verified), 768 full episodes, Blue goals
> per episode the sole primary endpoint, margin a consistency check only, contracts 8/8 before the
> run, no interim outcome inspection. Primary criterion: `LCB95(J_p) > 0` for both learned policies,
> `J_p = dG_{p,A} - dG_{p,B}`.
>
> | | native -> +1D mean Blue goals | +1D - native, paired, mean [95% CI] |
> |---|---|---|
> | `pi_A` Pole A | 1.21 -> 2.14 | **+0.93 [+0.66, +1.19]** |
> | `pi_A` Pole B | 2.47 -> 1.85 | **-0.61 [-0.89, -0.34]** |
> | `pi_B` Pole A | 1.07 -> 1.99 | **+0.92 [+0.67, +1.16]** |
> | `pi_B` Pole B | 2.24 -> 1.95 | **-0.29 [-0.55, -0.03]** |
> | **J, `pi_A`** | | **+1.54 [+1.16, +1.93]** |
> | **J, `pi_B`** | | **+1.21 [+0.88, +1.54]** |
>
> Both J lower bounds exceed zero, and all four cell point estimates reproduce the frozen signs, so the
> frozen label is `OPPONENT_CONDITIONED_GOAL_INTERACTION_CONFIRMED`. (All four cell intervals also
> exclude zero, though the frozen rules did not require that; `pi_B` Pole B is the narrow one, upper
> bound -0.03.) Margin agrees (cells +0.90, -0.68, +0.93, -0.34; J +1.57 [+1.20, +1.95] and +1.27
> [+0.94, +1.60]) and is not a second vote. The split-half check shows no sign flip in any of the eight
> re-derived claims. The exploratory estimates did not shrink on fresh seeds (exploratory J was +1.35 for
> both policies), so no winner's-curse signature is visible.
>
> **What this establishes:** a replicated differential response of Blue goal production to a forced
> defender between the two certified 6v6 opponents. **What it does not:** it says nothing about win
> probability (the sealed win-rate endpoint and its `ONE_DEFENDER_HARM_ONLY` label are untouched); no
> mechanism; no "regime" law; and per the frozen guard it does not distinguish opponent-specific
> strategic sensitivity from regression/compression toward a common goal level. That last caveat is
> not hypothetical: natively the four cells span 1.07-2.47 goals, while under +1D all four land between
> 1.85 and 2.14 -- the fresh block reproduces the convergence seen in the exploratory data. Two poles
> cannot separate the readings. A third opponent (none is certified for 6v6 in the repository) or a
> design that varies the native goal level is the discrimination experiment; none is started.
>
> **Third-opponent inventory (read-only, 2026-09-21; nothing was run, no seed touched).** Against the
> three conditions set by the PI: **(c) fails for every candidate on existing evidence** -- no record in
> the repository scores `pi_A`, `pi_B`, or any 6v6 learned policy against an opponent other than Pole A or
> B (`artifacts/6v6_results/robustness/` is empty and the export manifest is marked incomplete; the
> specialists were trained only against their own pole; other-opponent tests use 2-4 agents). **(a) and
> (b) are plausible but unverified:** the opponents are a canonical registry `OP5`-`OP12` of documented
> strategic niches built through the same shared code path as the poles (`canonical_parent` plus the
> size-normalized defender gate, which resolves `min_alive_for_defender=6` for `OP8`/`OP10`); `OP5` is NOT
> eligible (the genome path rejects it as an illegal dispatch key); `OP11`/`OP12` are `adaptive_enabled`,
> so they would adapt to the injected defender and confound the manipulation; nothing in the repository has
> ever instantiated any of them at 6v6, and none has a certification record or config hash. Eligible,
> non-adaptive candidates: `OP8` (escort), `OP9` (split-lane feint), `OP10` (aggressive interceptor). By the
> PI's rule as stated no opponent currently qualifies. Two design points for the decision: a cheap
> native-only measurement (~96 episodes on disposable smoke seeds, ~30 min sharded) would supply (a) and
> (c); and selecting for a middle native level alone is an asymmetric test (compression predicts change
> ~0 there, so a clearly non-zero change refutes compression but ~0 does not confirm it), whereas a
> candidate that DISSOCIATES native level from opponent identity discriminates in both directions.
> Open PI decision: run that small measurement, or freeze the claim as it stands.
>
> **Resolved (PI, 2026-09-21): the third-opponent detour is deferred, not pursued now.** The confirmed
> result stands as worded (an opponent-dependent goal-volume response across the two certified 6v6
> poles; no claim about why). Main line: 4v4 coordination known -> learned repair -> crossover, then 6v6
> A/B coordination effect known -> learned repair -> crossover. Third-opponent work is an
> external-validity / mechanism follow-up if the paper needs it. **Corrections to the inventory above,
> for whoever revisits it:** `OP8`-`OP10` are not part of the certified A/B methodology and were proposed
> only by elimination (genome-path-legal, non-adaptive). But they are not new to the repo: the 2v2
> Strategic Demand Searcher lists `OP6, OP8-OP12` as legal A-pole bases against the fixed `OP7` anchor
> (`experiments/sds_genome.py`; search archives under `artifacts/strategic_demand/searcher*/`, NOT read
> here), and the 4v4 draft `HELDOUT_REGIME_GENERALIZATION_V1_SPEC.json` (status
> `DRAFT_HELD_OUT_SET_UNDECIDED`) already names `OP5` and `OP8`-`OP12` as held-out opponents. That draft
> (i) categorizes by `defender_zone_frac` at the pole midpoint 0.20 (A-like: `OP5`, `OP9`; B-like: `OP8`,
> `OP10`, `OP11`, `OP12`), which supersedes the ad hoc structural ordering above; (ii) records a PI
> decision to run canonical opponents UNMODIFIED with no size-normalized defender gate, so the
> size-normalization applied to `OP8`/`OP10` in the inventory departs from project precedent; and (iii)
> notes the registry also contains `C6A`/`C6B`, which lie outside the pole span and were not examined.
> Read that spec and the searcher archives first if this is ever revived.
>
> Prior — 2026-09-21 — **EXPLORATORY ONE-DEFENDER FAILURE LOCALIZATION DONE (DIAGNOSTIC, NON-GATING, NO VERDICT COMPUTED): THE POLE-B HARM LOCALIZES TO CARRIER CONVERSION AFTER ACQUISITION. TWO CAVEATS QUALIFY HOW THE SEALED CAUSAL RESULT SHOULD BE READ.**
> Spec [`EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_V1_SPEC.json`](../artifacts/strategic_demand/sppo/EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_V1_SPEC.json),
> result `..._RESULT.json` and `..._ROWS.csv` beside it. The sealed causal rows held only outcomes
> (no per-tick telemetry), so the exact 768 sealed cells were replayed (same seeds, sha-pinned
> checkpoints, same code paths, deterministic; no new seed). **Parity 768/768 exact** against the
> sealed causal record, checked per episode as written and again at merge; the merge refuses to run
> unless all 768 cells are present. The first attempt (serial) was lost at 260/768 to a Windows
> Update reboot because it had no resume file; the rerun is 6 resumable shards.
>
> **Pole B (primary), both policies, +1D minus native, paired by seed:** offensive opportunity is
> NOT reduced (carry starts +2.4 all six agents, +3.1/+3.2 for the five non-overridden agents; first
> carry ~2-3 ticks later). Conversion after acquisition collapses: total carry time -49/-39 ticks,
> carriers tagged-while-carrying roughly double (`pi_A` 3.1 -> 6.2, `pi_B` 2.4 -> 5.4), clean
> releases -0.34/-0.19. Defense does improve on the frozen proxy (opponent carry time 62 -> 14 and
> 67 -> 26 ticks) but buys nothing: red scores ~0.01-0.02 goals/episode natively and is never ahead
> in any Pole-B episode under +1D. The +1D non-wins on Pole B are 0-0 timeouts (13/96, 10/96), i.e.
> blue fails to convert, not red wins. Attrition: tagged fraction +0.08, active agents -0.47.
> **Pole A (non-gating contrast):** the attrition (+0.08-0.09 tagged fraction, -0.5 active agents) and
> defensive-proxy (opponent carry time -81/-89) signatures are present there too, so they are generic
> side effects of the injection, not the Pole-B mechanism. What differs is conversion direction:
> clean releases go UP at Pole A (+0.34/+0.45) and DOWN at Pole B, and tag-while-carrying rises far
> less (+1.3/+0.9 vs +3.1/+3.0). Read against the PI's mapping this is closest to "opportunities
> stay similar, conversion collapses"; it does not separate "lost screening from the removed
> attacker" from "disrupted learned coordination" -- that needs a targeted intervention, not more
> telemetry. Full 14-metric x 4-cell table with paired bootstrap intervals is in the RESULT.
>
> **Instrument check:** "clean release" is a proxy for a capture and structurally cannot see an
> episode's terminal capture (the trace is recorded pre-step and stops at `done`). Accounting for
> that, clean releases + terminal captures reproduces the sealed mean blue score in every cell
> checked (e.g. `pi_A`@B native 1.74 + 0.60 = 2.34; +1D 1.40 + 0.36 = 1.76), which supports the proxy.
>
> **Caveat 1 -- Pole B is a ceiling pole.** Native `pi_A` won 96/96 and `pi_B` 95/96 on Pole B, so
> that pole could never show a benefit; its "harm" is a drop from perfect. The sealed causal test
> could only have detected a benefit on Pole A (native 0.83/0.885), where it was INCONCLUSIVE
> (`+0.031 [-0.063,+0.125]`, `-0.010 [-0.094,+0.073]`), not negative. "One defender does not help"
> is therefore trivially true at the ceiling pole and unresolved at the other; "falsified" overstates
> it, and the sealed label `ONE_DEFENDER_HARM_ONLY` should be read as a statement about that
> frozen metric on these seeds, not as closing the role-count hypothesis.
> **Caveat 2 -- the frozen win metric saturates at one goal.** On Pole A native, 90/96 and 92/96
> episodes run the full 240 ticks at ~1.1 goals/episode; under +1D that falls to 49/96 and 47/96
> with mean blue score 1.20 -> 1.97 and 1.04 -> 1.97, while win rate stays flat (win only needs one
> goal). The frozen rules did not include a goal-volume or margin endpoint (the superseded proposal
> listed margin as secondary; it was never computed), so a tiny post-hoc DESCRIPTIVE addendum was
> frozen BEFORE computation and run on the sealed rows (no simulation; rows sha256 bound to the sealed
> audit; margin verified = blue - red on all 768 rows): see the addendum block below.
>
> **DESCRIPTIVE_ONE_DEFENDER_GOAL_VOLUME_ADDENDUM** ([spec](../artifacts/strategic_demand/sppo/DESCRIPTIVE_ONE_DEFENDER_GOAL_VOLUME_ADDENDUM_SPEC.json),
> result beside it; POST-HOC motivation, non-gating, no pass/fail, no terminal label; it does not
> replace the sealed win-rate verdict). Paired by seed over the same 96 seeds, +1D minus native,
> mean [95% CI], native -> +1D means:
>
> | cell | Blue goals / episode | score margin |
> |---|---|---|
> | `pi_A` Pole A | 1.20 -> 1.97, **+0.77 [+0.49, +1.05]** | +0.77 [+0.49, +1.05] |
> | `pi_B` Pole A | 1.04 -> 1.97, **+0.93 [+0.68, +1.18]** | +0.95 [+0.71, +1.20] |
> | `pi_A` Pole B | 2.34 -> 1.76, **-0.58 [-0.87, -0.30]** | -0.64 [-0.93, -0.34] |
> | `pi_B` Pole B | 2.33 -> 1.91, **-0.43 [-0.69, -0.17]** | -0.44 [-0.70, -0.18] |
>
> The Pole-A goal-volume increase survives the paired analysis in both policies on both endpoints
> (intervals exclude zero) while win rate was unresolved: the binary win endpoint was hiding a real
> change in goal production there. It is consistent with the localization (clean releases +0.34/+0.45
> plus ~+0.5 more terminal captures per episode ~ the measured +0.8 goals). Pole B goal production
> falls, in line with its win-rate harm. What this does NOT establish: that +1D raises the
> probability of winning; a mechanism; "efficiency" (goals per tick or per carry were not measured --
> episodes are also shorter, 233 -> 180 ticks on `pi_A` Pole A, but that was not a frozen endpoint);
> or any confirmatory claim. Goals are capped at 3 by the score limit, so Pole B (native ~2.3) had
> little room to rise and Pole A (native ~1.1) had a lot. Any confirmatory version needs a fresh
> seed block and goal volume / margin frozen as an endpoint in advance; none is started.
>
> **CORRECTION NOTE (2026-09-21; supersedes the interpretation, not the record).** The earlier
> `ONE_DEFENDER_HARM_ONLY` label remains correct under its frozen binary win-rate criterion, and
> commit `e0a16c17` is left exactly as it is. The broader interpretation that the causal bridge
> "falsifies the defender-count hypothesis" is superseded: Pole B was ceiling-limited for benefit
> detection (native 96/96 and 95/96 wins), and Pole A effects were unresolved by win rate. The
> exploratory localization indicates the Pole-B harm arises primarily from impaired carrier
> conversion after acquisition (carry starts rise, tagged-while-carrying roughly doubles, clean
> releases fall) despite improved defense, and the goal-volume addendum shows +1D raised goal
> production on Pole A. Net: the role-count question is open, not closed.
>
> Prior — 2026-09-20 — **DEFENDER-INJECTION CAUSAL BRIDGE SEALED: `ONE_DEFENDER_HARM_ONLY`. FORCING ONE REAL DEFENDER NEVER HELPS, AND RESOLVED-HARMS pi_A AND pi_B ON POLE B SPECIFICALLY.**
> PI confirmed the decision rules (per-cell resolved improvement/harm via LCB95/UCB95, a
> per-pole interaction test for B-disproportionate support, and two non-gating secondary
> diagnostics -- crossover-under-condition and its paired change from native) and authorized
> the run: 768 full episodes, `pi_A`/`pi_B` x Pole A/B x {native, +1 defender} x 96 paired
> seeds `20900001-20900096`, cuda, deterministic. **Sealed, 18/18 gating checks, independent
> re-derivation of the interaction/paired-change quantities exact (max diff 0.0), read-only
> guarantee held.**
>
> **No cell shows a resolved improvement.** Two of four resolve HARM, both on Pole B:
> `pi_A@B` `-0.177 [-0.260,-0.104]`, `pi_B@B` `-0.094 [-0.156,-0.042]`. Pole A is unresolved
> for both policies (`pi_A@A +0.031 [-0.063,+0.125]`, `pi_B@A -0.010 [-0.094,+0.073]`). The
> per-pole interaction `I_r` is unresolved on both poles (`I_B +0.083 [-0.010,+0.177]` comes
> closest but does not clear the bar), so `B_DISPROPORTIONATE_ONE_DEFENDER_SUPPORT` is
> `false` on both poles and `TRANSFERABLE_ONE_DEFENDER_SCAFFOLD` is `false`. **Verdict:
> `ONE_DEFENDER_HARM_ONLY`** (>=1 resolved harm, zero resolved improvements anywhere).
>
> **Split-half check (report-only, non-gating) strengthens the harm reading**: the two
> resolved-harm cells (`pi_A@B`, `pi_B@B`) show NO sign flip between the first and second
> half of the seed block; three unresolved quantities (`delta_pi_A_A`, `delta_pi_B_A`,
> `crossover_B_plus1D`) DO flip sign between halves, consistent with those being genuinely
> noisy/near-zero rather than a hidden signal being averaged away.
>
> **Secondary diagnostics (non-gating):** the native-condition specialist crossover on fresh
> seeds is small and unresolved on both poles (`Delta_A_native -0.052 [-0.135,+0.021]`,
> `Delta_B_native -0.010 [-0.031,0.000]`), consistent with the original SHARE0 near-zero
> crossover finding. Under +1D, `Delta_B_plus1D` shifts to `+0.073 [-0.021,+0.167]` (still
> unresolved) -- the paired change `C_B = I_B` exactly (an algebraic identity, confirmed by
> the independent re-derivation), so this is not new information beyond the interaction test.
>
> **Bottom line:** the missing defender is not merely correlated with the near-zero
> `pi_A`/`pi_B` crossover -- forcing one in causally HURTS both policies on Pole B and helps
> neither policy on either pole. The scripted 6v6 composition result (5A/1D beats 3A/3D) does
> **not** transfer causally to these frozen checkpoints via a naive rollout-time defender
> injection. Per the frozen spec's own launch conditions (no PPO, no extra arm, no mid-run
> inspection, no top-up), this experiment is complete; full numbers in
> [`DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json`](../artifacts/strategic_demand/sppo/DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json).
> Whether/how to follow up (e.g. a training-time constraint, or investigating why Pole B
> specifically is harmed) is an open PI decision; nothing further is started.
>
> Prior — 2026-09-20 — **DEFENDER-INJECTION CAUSAL BRIDGE: MECHANISM FROZEN AND CONTRACT-VERIFIED (7/7); DECISION THRESHOLDS PROPOSED, NOT YET PI-CONFIRMED; NO SEED SPENT.**
> Follow-up to the Stage 2 finding below: PI direction 2026-09-20 (no PPO yet) asked to
> freeze a rollout-time test of whether forcing exactly one real defender onto the frozen
> `pi_A`/`pi_B` checkpoints changes their outcomes. Spec:
> [`DEFENDER_INJECTION_CAUSAL_BRIDGE_V1_SPEC.json`](../artifacts/strategic_demand/sppo/DEFENDER_INJECTION_CAUSAL_BRIDGE_V1_SPEC.json).
>
> **A first design (write `MacroAction.DEFEND=7` directly into `blue_commit_macro`,
> bypassing the discrete action head) crashes CUDA at n_macros=5**: the observation's
> action-mask builder (`gpu_env/_core/_observations.py::_build_action_mask`) scatters
> `commit_macro` into a one-hot dimension sized `cfg.n_macros`, and index 7 has no
> representation in a 5-wide mask. The frozen mechanism instead patches one layer
> downstream, at `_build_targets_from_action`'s continuous `(tx, ty)` OUTPUT (a per-instance
> monkey-patch, `experiments.probe_learned_composition.install_forced_defend_target`): the
> real, untouched target-resolution code is called twice per tick, once for the agent's own
> (always small, always-valid) committed macro and once more with a synthetic all-DEFEND
> macro tensor purely to read the DEFEND target, splicing only that value in. `commit_macro`
> itself is never written, so the action-mask and the (DEFEND-blind) reward bookkeeping are
> undisturbed.
>
> **Contracts 7/7 PASS**, including a real equivalence run: the injection reproduces a
> natural scripted DEFEND's full trajectory (position, alive, tagged, carrying, intent,
> outcome) bit-for-bit, at both n_macros=5 and n_macros=8, across 4 (pole, non-trivial
> defender-id) cells. Rotation (`seed % 6`) is exactly uniform (16/16/16/16/16/16) over the
> freshly reserved, disjoint seed block `20900001-20900096` (96 = 16x6, unspent). Checkpoint
> hashes match the pinned values.
>
> **Design frozen, not yet run:** `pi_A`/`pi_B` x Pole A/B x {native, +1 defender} x 96
> paired seeds = 768 FULL episodes (not truncated -- Stage 2's 20-tick truncation reads
> win/margin as ~always 0-0 that early, so an outcome question needs full episodes).
>
> **Open, PI decision — decision thresholds proposed but not frozen:** primary metric
> = paired win-rate delta (native vs +1-defender), same "Delta" convention as
> everywhere else in this line. Proposed: *improves* = delta > 0, CI excludes zero;
> *transferable scaffold* = all 4 cells improve; *B disproportionate* = pi_B improves AND
> the difference-of-deltas CI excludes zero on >=1 pole; *closes the intervention* = no
> resolved improvement anywhere, or any resolved harm. The `run` stage is deliberately
> unwritten until these are confirmed and the PI explicitly authorizes spending the block.
>
> Prior — 2026-09-20 — **LEARNED-COMPOSITION PROBE, STAGE 2 SEALED: BOTH pi_A AND pi_B OPEN FAR MORE ATTACK-HEAVY THAN THE WINNING SCRIPTED 5A/1D STRUCTURE, AND pi_B IS SIGNIFICANTLY MORE SO THAN pi_A.**
> PI direction 2026-09-20 authorized Stage 2 as a dual-instrument diagnostic with no further
> instrument selection: both confirmed survivors (position `[5,15)`, intent `[0,10)`) used as
> co-equal, separately-reported measurements, no averaging, no post-hoc instrument choice, the dead
> primary excluded even as a sanity check, the D_hat inversion using ONLY the original sealed
> calibration's class means (sha256-pinned, never refit on confirmatory or Stage-2 data). Spec:
> [`LEARNED_COMPOSITION_PROBE_STAGE2_V1_SPEC.json`](../artifacts/strategic_demand/sppo/LEARNED_COMPOSITION_PROBE_STAGE2_V1_SPEC.json).
>
> **Sealed** (`LEARNED_COMPOSITION_PROBE_RESULT.json`, 512 learned-policy episodes, `pi_A`/`pi_B` x
> Pole A/B x 128 paired seeds `20700101-20700228`, cuda, deterministic, 7/7 audit, independent
> re-derivation exact; read-only guarantee held: parameter digests and checkpoint sha256 unchanged
> before/after; a real 16-episode parity check against the spent SHARE0 crossover record passed exactly
> before any probe-block seed was spent):
>
> - **Concordance holds in all 4 (policy, pole) cells** -- position and intent instruments agree
>   (`attack_heavy`, D_hat <= 1.5) everywhere, so per the frozen interpretation rule a role-composition
>   claim is licensed for every cell.
> - **Neither policy resembles the confirmed 5A/1D reference (D=1).** Mean |D_hat - 1| is 0.82-1.00
>   across cells -- both read far closer to a fully-committed 6A/0D opening than to the composition that
>   actually beat balanced play in the scripted world.
> - **pi_B is significantly MORE attack-committed than pi_A, not closer to the target** -- the paired
>   contrast D_hat(pi_B) - D_hat(pi_A) is negative with a CI excluding zero in all 4 instrument x pole
>   combinations (position: -0.12 both poles; intent: -0.03 both poles). pi_B's intent-window reading is
>   EXACTLY 0.0 with a [0,0] CI in both poles -- zero of 256 episodes ever showed a home-directed agent
>   in that window.
> - **Interpretation (not itself a further gated claim):** composition does not explain the near-zero
>   `pi_A`/`pi_B` crossover delta because neither policy plays anything resembling the discovered
>   winning structure to begin with -- both are near-unconditional full-attack in the opening, and the
>   one real, concordant difference between them runs away from, not toward, the scripted target.
>
> Full per-cell bootstrap intervals, the paired contrasts, and the concordance table are in the sealed
> result. This closes the opening-composition read of `pi_A`/`pi_B`; any further step (e.g. a
> coordination intervention, or reading sustained-episode behaviour) needs its own PI-frozen spec.
>
> Prior — 2026-09-20 — **LEARNED-COMPOSITION PROBE, CONFIRMATORY CALIBRATION SEALED: `BOTH_CONFIRMED`. NO LEARNED POLICY HAS BEEN READ.**
> PI direction 2026-09-20 chose Option B from `LEARNED_COMPOSITION_PROBE_CALIBRATION_READING.json`:
> re-validate the two pre-declared calibration survivors on a fresh, never-before-spent seed block
> before treating either as a real instrument. Spec:
> [`LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_V1_SPEC.json`](../artifacts/strategic_demand/sppo/LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_V1_SPEC.json)
> (amends the V1 spec; unchanged kill rule, unchanged R_pos/R_intent/windows/tolerance/bar).
>
> **Sealed** (`..._CONFIRMATORY_CALIBRATION_RESULT.json`, 384 fresh truncated scripted episodes, seeds
> `20800001-20800016`, cuda, 7/7 audit, independent re-derivation exact, contracts 6/6 including a
> mechanical check that the two candidates are exactly the two the original calibration recorded as
> passing): **both candidates independently clear the identical V1+V2+V3 rule (recovery >= 0.90, +-1
> defender) on both poles, on seeds that had no part in selecting them.**
>
> - Position, ticks `[5,15)`, R=4.5: pole A `V2 0.991 / V3 0.988`, pole B `V2 0.991 / V3 0.963` ->
>   `pole_valid` true on both.
> - Resolved-target intent, ticks `[0,10)`, R=4.5: pole A `V2 1.000 / V3 1.000`, pole B `V2 0.991 /
>   V3 1.000` -> `pole_valid` true on both.
> - FYI, not gating: the original dead primary (position `[0,10)`) was re-scored on this fresh block too,
>   as a sanity check, and reproduces its earlier failure (`V2 0.875 / V3 0.80-0.85` vs `0.90`,
>   `pole_valid` false on both) -- consistent with the original verdict, not reopened.
>
> **What this does and does not authorize:** per the amendment spec, this verdict gates only which
> instrument(s) a Stage-2 spec addendum is permitted to declare; it does not itself authorize Stage 2 to
> run. No learned-policy checkpoint has been loaded and no learned telemetry has been read under this
> probe line. **Open, PI decision:** whether to now declare and freeze a Stage-2 addendum that reads
> `pi_A`/`pi_B` on the already-reserved seed block `20700101-20700228` (RESERVED, unspent) using BOTH
> confirmed instruments as convergent measurements, per the original spec's read-only guarantees
> (checkpoint hash pinning, parameter-digest equality, `torch.no_grad()`, eval mode, no optimizer).
>
> Prior — 2026-09-19 — **LEARNED-COMPOSITION PROBE, STAGE 1 SEALED: THE FROZEN PRIMARY INSTRUMENT IS `NOT_VALID`; STAGE 2 REFUSED. TWO PRE-DECLARED ALTERNATIVES PASS ON THE SAME DATA. NO LEARNED POLICY HAS BEEN READ.**
> Read-only probe of the existing learned 6v6 specialists (`final_pi_A/B_specialist_6v6`,
> sha256-pinned to the record behind `Delta_A = Delta_B = +0.0078`; spec
> [`LEARNED_COMPOSITION_OPENING_PROBE_V1_SPEC.json`](../artifacts/strategic_demand/sppo/LEARNED_COMPOSITION_OPENING_PROBE_V1_SPEC.json)).
> Because learned policies cannot select `DEFEND` (production `n_macros` is 5), their
> role has to be inferred by an instrument calibrated on scripted ground truth.
> **Whole-episode position-near-home failed its known-answer calibration**
> (scripted attackers sit near home 67-77% of the time; tagged agents walk home and
> the adapter sends all attackers home when any teammate carries; at R=6 scripted
> 6A_0D reads 0.70-0.82 defend-like after tick 100, essentially the same as 0A_6D).
> **Whole-episode resolved-target intent failed its
> pre-declared criterion** (76% / 67% within +-1 against 0.90). Neither was rescued.
>
> **Sealed calibration** (`..._CALIBRATION_RESULT.json`, 384 truncated scripted
> episodes, fresh seeds `20700001-016`, cuda, 7/7 audit, independent re-derivation
> exact): the frozen primary (position, `R=4.5`, ticks `[0,10)`) fails, V2 `0.786`
> and V3 `0.81 / 0.80` against `0.90`, because the class means are evenly spaced
> (~0.09) but within-class SD is 0.10-0.12. The prototype's 95% was
> selection-inflated exactly as warned (fresh exact recovery 35%). **Stage 2 is
> refused; no learned telemetry was read.**
>
> **But two readings declared in the same spec before the run pass the identical
> rule**: position over ticks `[5,15)` (V2 `1.00 / 0.98`, V3 `0.99 / 0.99`) and the
> resolved-target intent reading over `[0,10)` (V2 `1.00 / 1.00`, V3 `1.00 / 1.00`,
> exact recovery ~87%). The cross-assignment control (defenders on the last ids)
> passes, so the id-geometry confound is not the cause. They gate nothing under the
> frozen rule, and promoting one now would select it for having passed on the data
> that validated it. **Open, PI decision** (see
> [`..._CALIBRATION_READING.json`](../artifacts/strategic_demand/sppo/LEARNED_COMPOSITION_PROBE_CALIBRATION_READING.json)):
> stop here, or amend so BOTH survivors are re-validated on a fresh calibration block
> (~32 min) before any learned reading. Nothing further is started. Training and
> PPO remain off.
>
> Prior — 2026-09-19 — **6v6 FIXED 5A/1D vs 3A/3D CONFIRMED (SEALED, n=192, fresh seeds): B +0.69, AND A IMPROVED +0.125.**
> The pre-registered confirmation
> ([`FIXED_ATTACK_HEAVY_6V6_CONFIRMATORY_V1_SPEC.json`](../artifacts/strategic_demand/sppo/FIXED_ATTACK_HEAVY_6V6_CONFIRMATORY_V1_SPEC.json);
> two arms, both certified 6v6 poles, block `20600001-20600192`, 768 episodes,
> PPO off, no router) returned **`FIXED_ATTACK_HEAVY_6V6_CONFIRMED`** on a
> **`SEALED`** record: 13/13 audit checks, 9/9 contracts, no arm-identity
> violations. The composition was chosen by a mechanical maximin rule from the
> sealed sweep before any fresh seed, and re-derived by contract C1. Result:
> [`..._OUTCOME_RESULT.json`](../artifacts/strategic_demand/sppo/FIXED_ATTACK_HEAVY_6V6_OUTCOME_RESULT.json),
> reading: [`..._OUTCOME_READING.json`](../artifacts/strategic_demand/sppo/FIXED_ATTACK_HEAVY_6V6_OUTCOME_READING.json).
>
> **Gates (unchanged):** Pole-B gain `WR_B(5A1D) - WR_B(3A3D) = +0.6875
> [+0.6146, +0.7604]` (**PASS**; 136 seeds won by 5A/1D only, 4 the reverse).
> Pole-A harm `-0.125 [-0.2083, -0.0417]` against `tau 0.10` (**PASS**; the
> harm is *negative* with the whole interval below zero, so 5A/1D beat the
> balanced baseline on A by `+0.125`; 45 seeds to 21, 34% discordant). Win
> rates: B `5A1D 0.755 / 3A3D 0.068`; A `5A1D 0.854 / 3A3D 0.729`. Both halves
> agree on both poles. The refined A-failure interpretation was never
> exercised.
>
> **Correction to the sweep's Pole-A reading:** the sweep called Pole A a
> plateau with no distinguishable winner. On fresh seeds 5A/1D is *better* than
> 3A/3D on A with an interval excluding zero (sweep: `+0.078 [-0.078, +0.234]`,
> same sign, underpowered at n=64). Whether A distinguishes among attack-heavy
> compositions was not tested; only two arms ran.
>
> **Not established:** that routing could not add value beyond 5A/1D (per-pole
> compositions not run; sweep headroom on B `+0.078 [-0.031, +0.203]`);
> anything about learned specialization. This is a *scripted* confirmation.
> The learned-specialist failure (`Delta_A = Delta_B = +0.0078`) and the
> hypothesis that learned policies already play attack-heavy on B remain
> untested. Cross-scale statements are descriptive only: 6v6 Pole B is
> canonical `SDS_PARENT_OP7`, not the 4v4 B3-3 construction, and the
> size-normalized defender gate also differs. **Next step is a PI decision;
> nothing further is started.** PPO remains off.
>
> Prior — 2026-09-19 — **6v6 COMPOSITION SWEEP SEALED (exploratory): B CLEARLY ATTACK-HEAVY, A A TIED PLATEAU; A FIXED ATTACK-HEAVY COMPOSITION IS TIED WITH THE BEST ON BOTH POLES.**
> The frozen 6v6 sweep
> ([`PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_SPEC.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_SPEC.json);
> 7 compositions x 2 certified 6v6 poles, block `20500001-20500064`, 896
> episodes, PPO off) is **`SEALED`** (23/23 audit checks, 9/9 contracts incl.
> 480 live 4v4 ticks of adapter parity at n=4) with label
> **`COMPOSITION_DESCRIPTIVE_SIGNAL`**. Result:
> [`..._SWEEP_RESULT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_RESULT.json),
> reading: [`..._SWEEP_READING.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_READING.json).
>
> **Per-pole argmax (the frozen deliverable):** A `4A_2D` (0.844), B `6A_0D`
> (0.891). **Pole B** has a clear, monotone attack-heavy landscape: `6A_0D 0.891`
> and `5A_1D 0.812` (tied) far above `4A_2D 0.359` and the balanced baseline
> `3A_3D 0.094`; attack-heavy beats the baseline by `+0.80 / +0.72`, intervals
> clear of zero, stable across seed halves. **Pole A** has no distinguishable
> winner: `4A_2D` is tied with `5A_1D`, `6A_0D` and `3A_3D` (indistinguishable
> set 3 of 6), a plateau across 3-6 attackers (0-3 defenders) that falls
> off at 4 defenders (`2A_4D` 0.594) and collapses at `1A_5D` / `0A_6D`; Pole-A
> contrasts flip sign between seed halves. B is more
> attack-heavy than A (0 vs 2 defenders), the 4v4 direction.
>
> **Bearing on the predeclared branch:** the compositions statistically as good
> as each pole's best, intersected, are **`6A_0D` and `5A_1D`**, so a single fixed
> attack-heavy composition is tied with the per-pole best on both poles. The
> 4v4 router exists because all-attack hurt Pole A; at 6v6 that trade-off is not
> visible. By the letter of the branch a clear B-heavy pattern licenses a routed
> freeze, but a router's added value over a fixed `5A_1D` is undemonstrated and
> small on point estimates (upward-biased). **Which reading governs is a PI
> decision; nothing further is started.** Cross-scale statements carry the
> frozen clause: 6v6 Pole B is canonical `SDS_PARENT_OP7`, not the 4v4 B3-3
> construction, so differences cannot be attributed to team size alone.
>
> Prior — 2026-09-19 — **GUARDED ROUTED COMPOSITION CONFIRMED (SEALED, n=128, fresh seeds).**
> The pre-registered confirmatory run of the full-window startup guard
> (`GUARDED_ROUTED_COMPOSITION_OUTCOME_V1`, block `20300001-20300128`, 768
> episodes, PPO off, spec
> [`GUARDED_ROUTED_COMPOSITION_CONFIRMATORY_V1_SPEC.json`](../artifacts/strategic_demand/sppo/GUARDED_ROUTED_COMPOSITION_CONFIRMATORY_V1_SPEC.json))
> returned **`GUARDED_ROUTED_COMPOSITION_CONFIRMED`** on a **`SEALED`** record:
> 13 of 13 gating audit checks passed, 15 of 15 contracts passed, guard
> invariant held (0 violations, earliest first departure tick 39 on both
> poles). This is the first sealed record through `run_state.seal()` and it
> validates the owner-aware `seed_class` fix in use. Result:
> [`..._OUTCOME_RESULT.json`](../artifacts/strategic_demand/sppo/GUARDED_ROUTED_COMPOSITION_OUTCOME_RESULT.json),
> reading: [`..._OUTCOME_READING.json`](../artifacts/strategic_demand/sppo/GUARDED_ROUTED_COMPOSITION_OUTCOME_READING.json).
>
> **Gates (unchanged from the unguarded run):** Pole-B gain
> `+0.6094 [+0.5234, +0.6953]`, LCB above zero (**PASS**; 79 seeds won by the
> guarded router and not by 2A/2D, 1 the reverse). Pole-A harm
> `+0.0156 [-0.0156, +0.0469]` against `tau_A_harm = 0.10` (**PASS**; 3 harm
> flips, 1 gain flip). Win rates: B `guarded 0.758 / 2A2D 0.148 / 4A0D 0.688`;
> A `guarded 0.727 / 2A2D 0.742 / 4A0D 0.547`.
>
> **Sizing agreed closely** (A harm `0 -> +0.016`, B gain `+0.625 -> +0.609`),
> so the optimism disclosed in advance turned out small on these metrics. The
> Pole-A trigger rate did double on fresh seeds (28/128 against 7/64), the
> pessimistic case in the planning table, and harm stayed small.
>
> **What it does not show:** the guardrail is preserved largely because the
> router acts on Pole A only 2.2% of ticks (never in 100 of 128 episodes),
> which is not evidence of *safe switching* on A. Only two opponents, both of
> which shaped the selector, so external validity is untested. No PPO, no
> learned coordination, no 6v6. The unguarded router's verdict is unchanged:
> `B_GAIN_WITH_EXCESS_A_HARM` on an `AUDIT_FAILED` record. **Next step is a PI
> decision; nothing further is started.** Router and guard stay frozen.
>
> Prior — 2026-09-19 — **ROUTED OUTCOME: `B_GAIN_WITH_EXCESS_A_HARM` ON AN `AUDIT_FAILED` RECORD. HARM LOCALIZED TO TICK-0 SWITCHING. STARTUP GUARD FROZEN AND SIZED; CONFIRMATORY FREEZE AWAITS PI.** *(the open confirmatory decision below is now resolved by the entry above)*
> The calibration-side "RESOLVED POSITIVE" entry below is **superseded**: the
> router passed its held-out calibration and then failed its outcome guardrail.
>
> **Outcome run** (`ROUTED_COMPOSITION_OUTCOME_V1`, block `20200001-20200064`,
> 384 episodes, PPO off, three arms, paired): Pole-B gain
> `WR_B(STATE) - WR_B(2A2D) = +0.6719 [+0.5156, +0.8125]` (**PASS**), Pole-A
> harm `+0.1719 [+0.0469, +0.2969]` against `tau_A_harm = 0.10` (**FAIL**,
> harm's LCB above zero). Label **`B_GAIN_WITH_EXCESS_A_HARM`**, not a
> promotion. Win rates: B `STATE 0.875 / 2A2D 0.203 / 4A0D 0.656`; A
> `STATE 0.516 / 2A2D 0.688 / 4A0D 0.422`. The router beat fixed 4A/0D on B
> (`+0.2188 [+0.0938, +0.3438]`, post-hoc). The record is
> **`AUDIT_FAILED`**, not sealed: `run_state.py` called `check_block()` without
> an `experiment_id` and read the run's own Rule-9 reservation as reuse. The
> other 12 gating checks passed and both claims re-derive at `0.00e+00`. The
> terminal record is preserved untouched; fix, 5 regression tests and a replay
> (13/13) are in
> [`ROUTED_COMPOSITION_OUTCOME_AUDIT_CORRECTION.json`](../artifacts/strategic_demand/sppo/ROUTED_COMPOSITION_OUTCOME_AUDIT_CORRECTION.json).
> Rules 7 and 9 had been mutually unsatisfiable in practice
> ([`RESEARCH_RUN_STANDARDS.md`](RESEARCH_RUN_STANDARDS.md)).
>
> **Why the calibration safety proxy failed** (protocol
> [§5.6](experiment-and-evaluation-protocol.md)): the spec projected harm from
> tick exposure at about `0.03`; observed harm was `0.172`, about 6x. Harm sits
> entirely in triggered episodes, and **82% of it in the 27 episodes whose first
> trigger fires at tick 0**
> ([`..._A_HARM_LOCALIZATION.json`](../artifacts/strategic_demand/sppo/ROUTED_COMPOSITION_A_HARM_LOCALIZATION.json)).
> Every exposure measure failed to separate harmful episodes. At tick 0 the
> window holds one sample and the dwell sentinel exempts the first switch, and
> `D_0` is **bit-identical across poles on 64/64 seeds**
> ([`..._SWITCH_SAFETY_CONTRAST.json`](../artifacts/strategic_demand/sppo/ROUTED_COMPOSITION_SWITCH_SAFETY_CONTRAST.json)),
> so a tick-0 decision carries zero regime information. Pass 2 found no
> state-based separator (home pressure and `red_alive` are constant across
> harmful and other switches) and B's bursts are 4x longer than A's, ruling out
> a state veto and a burst cap. Pass 1 localized the harm to early
> first-departure timing; Pass 2 independently established the tick-0 evidence
> is regime-free but did **not** reproduce the per-switch onset contrast.
>
> **Startup guard**
> ([`..._STARTUP_GUARD_V1_SPEC.json`](../artifacts/strategic_demand/sppo/ROUTED_COMPOSITION_STARTUP_GUARD_V1_SPEC.json),
> frozen before sizing): no departure from 2A/2D until the selector holds a full
> evidence window. Regime-blind by construction (reads only buffer length) and
> inert from tick 39 onward. **Descriptive sizing, same spent block, NOT
> evidence** ([reading](../artifacts/strategic_demand/sppo/ROUTED_COMPOSITION_STARTUP_GUARD_SIZING_READING.json)):
> Pole-A harm `+0.172 -> 0` (0 win flips of 64), Pole-B gain `+0.672 -> +0.625`
> (about 93% retained; paired B cost `-0.047 [-0.156, +0.063]`, spans zero).
> Blocked first departures re-fire on B 47 of 49 and on A 3 of 35, so the
> discrimination comes from the router's own statistic once given valid
> evidence, not from the guard. Optimistic by an unknown amount: the mechanism
> was found on these seeds.
>
> **Open, PI decision:** the predeclared "continue" condition is met on its
> face, but no confirmatory experiment is frozen and no seed is allocated. The
> sizing reading carries planning arithmetic for `n` (n=64 vs 128; the A gate
> is a UCB gate and is lenient when discordant pairs are few). Router
> operating point stays frozen. PPO / GPU / 6v6 remain off.
>
> Prior, **superseded** — 2026-09-19 — **CONSERVATIVE B-TRIGGER CALIBRATED (V2). HANDCRAFTED-SELECTOR BRANCH RESOLVED POSITIVE.**
> The PI authorized one final selector attempt (Option B) with an
> uncertainty-aware eligibility rule, frozen in
> [`COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_SPEC.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_SPEC.json).
> **One thing changed:** eligibility became `UCB95_episode(A_FP) <= 0.10`
> (bootstrap over *episodes*, n=16, not over autocorrelated ticks) instead of
> the point estimate. Statistic, grid, dwell, hysteresis, windows, threshold
> rule, tie-break order and state machine were imported unchanged, and the
> **held-out pass rule was deliberately left as the same point rule** so the
> two runs stay comparable. Result:
> [`..._V2_RESULT.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_RESULT.json),
> reading: [`..._V2_READING.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_READING.json).
>
> **Decision: `CONSERVATIVE_B_TRIGGER_CALIBRATED_V2`.** Both frozen held-out
> gates passed on the fresh block `99900501-516`: Pole-A false-positive tick
> rate `0.0674` against the `0.10` operational limit, and Pole-B dominant
> composition `0.9375` against the `0.50` bar. Operating point, now frozen:
> `D_t = P_blue(4.0) - P_red(4.0)`, `W=40`, hysteresis `0.2`, dwell `10`,
> threshold `-0.8333333333333334`, default `2A/2D`, triggered `4A/0D`.
>
> **The repair is measurable.** Eligibility became strictly stricter (3137 of
> 9380 vs 3773 under the point rule). Boundary-hugging is gone: V1's selected
> config sat at point `A_FP = 0.0964` (margin `0.0036`); V2's sits at `0.0651`
> (margin `0.0349`), with the *bound* (`0.0987`) against the cap instead.
> Calibration→held-out A drift fell from `+0.0156` (which crossed the cap) to
> `+0.0023` (which did not). The margin was **derived from the principle, not
> chosen off the frontier** — no raw cap like 0.05 was ever picked. Cost:
> calibration B_TP `0.783 → 0.711`; held-out B dominance *rose* to `0.9375`.
>
> **Honest caveat, recorded in the reading:** applying the same uncertainty
> standard to the held-out block gives a one-sided `UCB95` of `0.1026`, just
> *above* `0.10` — the evidence places the true A rate under the limit with
> roughly **94%** confidence, not "proven safe". The held-out gate is a point
> rule by design and is not reinterpreted. Quote `0.067` with that interval,
> never alone. 7 of 16 Pole-A episodes never triggered; the mean is carried by
> a 3-episode tail. Closing that gap needs more episodes, not another knob.
>
> **Disclosed:** calibration block `99900101-116` informed selection twice
> (V1 point rule, V2 UCB rule), stated in the freeze before execution.
> Calibration-side numbers are not independent evidence; the fresh held-out
> block is. The V1 sealed negative
> ([`..._ASYMMETRIC_RESULT.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_RESULT.json))
> stands unrewritten and is now correctly readable as a falsification of the
> *point-estimate selection rule*, not of the routing architecture.
>
> **Next, NOT authorized by any agent:** contracts C1–C6 then the outcome arms
> (`STATE_B_TRIGGER` / `FIXED_2A2D` / `FIXED_4A0D`, n=64, cpu, PPO off, B
> improvement LCB95 > 0, A harm UCB95 ≤ `tau_A_harm = 0.10`) under
> [`COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_V1_SPEC.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_V1_SPEC.json).
> That is the first non-outcome-blind work in this branch and needs a fresh PI
> go-ahead. Next free smoke pair `99900601-616`. PPO / GPU / 6v6 remain off.
>
> Prior — 2026-09-19 — **TWO-FEATURE SELECTOR REVIVAL: FROZEN GATE FAILED, BUT A SAFE NONZERO REGION EXISTS**
> The two-feature amendment was revived under the asymmetric criterion by
> [`COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_REVIVAL_SPEC.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_REVIVAL_SPEC.json),
> which is the "later freeze" the asymmetric spec required. Exactly one thing
> changed: the online statistic became `D_t = P_blue(4.0) - P_red(r_red*)`.
> The `A_FP <= 0.10` budget, the `B >= 0.50` dominance bar, the dwell grid, the
> window/hysteresis grid, the tie-break order and the state machine were
> inherited verbatim (imported, not re-implemented). Result:
> [`COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_RESULT.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_RESULT.json),
> reading: [`COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_READING.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_READING.json).
>
> **Decision: `NO_CONSERVATIVE_B_TRIGGER_TWO_FEATURE`** — held-out Pole-A
> false-positive tick rate `0.11197916666666667` exceeds the frozen `0.10` cap.
> The cap was not relaxed. The B side passed decisively and was never binding
> (held-out B dominant-composition `0.875` against the `0.50` bar, B TP tick
> rate `0.8122395833333333`). No outcome seeds spent or authorized.
>
> **The failure mode is the opposite of the single-feature run and the
> predeclared FAIL narrative is retracted as factually wrong for this run.**
> Single feature: 816 configs inside the A budget, every one with `B_TP = 0.0`
> — the budget was satisfiable only by a never-firing selector, flat to a cap
> of `0.14`. Two features: 3773 configs inside the budget, best `B_TP = 0.783`
> with B dominance `0.9375` on calibration, and at a cap of **exactly `0.00`**
> the best config still reaches `B_TP = 0.3466`; at `0.02` it reaches `0.5437`
> with B dominance `0.6875` — above the B bar at one fifth of the A budget.
> A safe nonzero region exists. What failed is the **selection rule**: "maximize
> `B_TP` subject to `A_FP <= 0.10`" hugs the constraint boundary by
> construction, and it picked a config whose calibration `A_FP` was `0.0964`
> — 96.4% of the budget, margin `0.0036`. Every frontier number above is
> calibration-side and **not** held-out-verified.
>
> **Feature fact worth keeping:** `r_red* = 4.0`, and `P_red(4.0)` separates the
> poles *perfectly* on full-episode means (A `0.5138`, B `1.5687`, pooled
> misclassification `0.0`) against `P_blue(4.0)`'s `0.15625`. The feature added
> second is the stronger regime discriminator.
>
> **Open PI decision** (see the reading's `THE_OPEN_DECISION_FOR_THE_PI`): the
> run landed between the two preregistered branches — a safe region appeared,
> but the point the frozen rule produced from it failed confirmation for a
> procedural reason unrelated to the feature or the architecture. Option A:
> honour the terminal clause and stop hand-tuned selector work. Option B: one
> margin-aware re-selection under a fresh freeze and a fresh held-out block
> (`99900501-516`; `99900301-316` is now SPENT). **Not decided by any agent.**
> PPO / GPU / 6v6 remain off.
>
> Prior — 2026-09-18 — **ASYMMETRIC B-TRIGGER SUCCESSOR FROZEN; SYMMETRIC SELECTOR PATH CLOSED**
> Pole A is the already-good guardrail; Pole B is the repair target. The
> symmetric online-calibration RESULT stands historically
> ([`COMPOSITION_SELECTOR_ONLINE_CALIBRATION_RESULT.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ONLINE_CALIBRATION_RESULT.json);
> held-out tick error ~37%, dominant-composition accuracy 65.6%), sealed by
> [`COMPOSITION_SELECTOR_SYMMETRIC_CALIBRATION_READING.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_SYMMETRIC_CALIBRATION_READING.json).
> The n≈300 dual positive-gain outcome sizing is **not authorized**. The
> two-feature amendment is **SUSPENDED**. Successor:
> [`COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_V1_SPEC.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_V1_SPEC.json)
> — default `2A/2D`, trigger `4A/0D` only under a conservative B-like
> threshold; calibrate by max A FP tick rate ≤ 0.10 then maximize B detection;
> outcome gates = B improvement required + A regression UCB95 ≤ 0.10. No
> outcome seeds until asymmetric calibration PASS. PPO/GPU/6v6 still off.
> The asymmetric calibration has now completed:
> [`COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_CALIBRATION_RESULT.json`](../artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_CALIBRATION_RESULT.json).
> The frozen grid selected `W=20`, hysteresis `0.0`, dwell `4`, threshold `0.0`.
> Held-out metrics were A false-positive tick rate `0.0`, B true-positive tick
> rate `0.0`, A switches/episode `0.0`, B switches/episode `0.0`, A episodes
> with any trigger `0.0`, and B episodes with dominant `4A/0D` `0.0`.
> The A guardrail passed, but the frozen B dominant-composition requirement
> (`>=0.50`) failed. Decision: **NO_CONSERVATIVE_B_TRIGGER**. No outcome seeds
> were spent or authorized.
>
> Prior — **PYQUATICUS 4v4 TARGETED ROLE-COMPOSITION CONFIRMATION COMPLETE**
> Follow-up — **ORACLE REGIME-CONDITIONED COMPOSITION EVALUATION COMPLETE**.
> The frozen three-arm test ran all 384 CPU episodes on paired seeds
> `20100001–20100064`: oracle dispatch (`A -> 2A/2D`, `B -> 4A/0D`), fixed
> `2A/2D`, and fixed `4A/0D`. Result:
> [`PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT.json).
> Oracle parity passed exactly with zero mismatches. The oracle matched the
> selected fixed composition: Pole A win rate `0.703125` versus fixed `4A/0D`
> `0.53125`, and Pole B win rate `0.734375` versus fixed `2A/2D` `0.125`.
> The paired win-rate gains were positive (`A +0.171875 [0,0.34375]`,
> `B +0.609375 [0.484375,0.734375]`). The B-side gate passed, while the
> A-side lower confidence bound was exactly `0`, so the frozen strict gate
> requiring both LCBs to be greater than zero failed.
> Supporting score-difference intervals were positive, but they do not replace
> the frozen win-rate gate. The decision is **NO_ORACLE_COMPOSITION_CONFIRMATION**.
> An analysis-integrity amendment corrected the B-side wrong-control lookup
> from matched `4A/0D` to frozen wrong `2A/2D`, using the retained rows without
> rerunning episodes or changing the rule. The pre-amendment artifact is
> preserved as
> [`PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT_PREAMENDMENT_BUGGY.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT_PREAMENDMENT_BUGGY.json).
> This demonstrates exact dispatch and descriptive improvement over the wrong
> fixed composition in these samples, but it does not establish selector
> sufficiency. The seed block is marked SPENT. No PPO, learned selector, or 6v6
> promotion is authorized from this result alone.
>
> Follow-up — **ORACLE REGIME-CONDITIONED COMPOSITION TEST FROZEN / PREFLIGHT PASS**.
> The frozen spec
> [`PYQUATICUS_4V4_ORACLE_COMPOSITION_SPEC.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ORACLE_COMPOSITION_SPEC.json)
> defines three paired CPU arms: `ORACLE_REGIME` maps Pole A to `2A/2D` and
> Pole B to `4A/0D`, alongside fixed `2A/2D` and fixed `4A/0D` controls.
> Preflight passed the upstream provenance, V2 semantic status, prior-result
> boundary, certified poles, macro projections, and oracle dispatch checks:
> [`PYQUATICUS_4V4_ORACLE_COMPOSITION_PREFLIGHT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ORACLE_COMPOSITION_PREFLIGHT.json).
> The fresh exploratory block `20100001–20100064` is only the next available
> range; it has not been allocated or spent. The 384-episode CPU evaluation
> requires explicit follow-up authorization. PPO, GPU, and 6v6 remain off.
>
> Follow-up — **TARGETED ROLE-COMPOSITION CONFIRMATION COMPLETE**. The frozen
> confirmation artifact
> [`PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_SPEC.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_SPEC.json)
> compared only `4A/0D` with `2A/2D` on both certified poles using paired
> fresh seeds `20000001–20000064` (256 CPU episodes; PPO/GPU/training off).
> The result is
> [`PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_RESULT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_RESULT.json).
> Pole B improved by `Delta_B = +0.6875` `[+0.5625,+0.8125]`, from win rate
> `0.109375` to `0.796875`. Pole A moved in the opposite direction,
> `Delta_A = -0.3125` `[-0.46875,-0.140625]`, from `0.75` to `0.4375`.
> The paired cross-pole interaction was positive, `Delta_B - Delta_A = +1.0`
> `[+0.796875,+1.203125]`, but the preregistered full confirmation gate also
> required `Delta_A UCB95 >= 0`; that condition failed. The frozen decision is
> **B_COMPOSITION_EFFECT_ONLY**, not `REGIME_COMPOSITION_CONFIRMATION_PASS`.
> This strengthens the Pole-B attack-heavy composition clue but does not show
> that one composition rule solves both poles. The seed block is marked SPENT.
> No PPO launch, composition-selector implementation, or 6v6 promotion is
> authorized from this descriptive scripted result alone.
>
> V2 unified-DEFEND contracts passed the semantic subgate: exact target parity
> and near-perfect direction parity across inward/outward state transitions.
> The historical aggregate G6 remains false because it includes obsolete
> waypoint candidates and the known cross-dynamics speed/trajectory mismatch;
> it is not silently rewritten.
>
> The authorized follow-up was frozen in
> [`PYQUATICUS_4V4_TEAM_EVALUATION_SPEC.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_TEAM_EVALUATION_SPEC.json).
> CPU preflight passed with both certified opponent poles, the evaluation-only
> `n_macros=8` direct reference path, the unified `DEFEND=7` adapter, and all
> three deterministic 2A/2D assignment mappings. The paired block
> `19800001–19800064` was then spent across 2 poles × 3 assignments × 64
> episodes (384 episodes; PPO/GPU/training off). Results:
> [`PYQUATICUS_4V4_TEAM_EVALUATION_RESULT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_TEAM_EVALUATION_RESULT.json),
> episode rows [`PYQUATICUS_4V4_TEAM_EVALUATION_EPISODES.csv`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_TEAM_EVALUATION_EPISODES.csv),
> and mapping audit [`PYQUATICUS_4V4_TEAM_EVALUATION_MAPPING_AUDIT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_TEAM_EVALUATION_MAPPING_AUDIT.json).
> Pole-A Blue win rates were `0.625–0.6875`; Pole-B rates were `0.046875–0.125`.
> Every paired assignment contrast interval included zero, so the frozen
> decision is **NO_DEMONSTRATED_ASSIGNMENT_EFFECT**. This does not show that
> explicit roles are useless; it shows no assignment effect under this exact
> scripted family, opponent pair, and local dynamics. No 6v6 promotion, PPO
> design, or macro change is authorized from this result alone.
>
> Follow-up — **ROLE COMPOSITION SWEEP COMPLETE**. The frozen CPU-only
> sweep [`PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_SPEC.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_SPEC.json)
> tested `4A/0D`, `3A/1D`, `2A/2D`, `1A/3D`, and `0A/4D` under one fixed
> identity ordering, both poles, and fresh paired seeds `19900001–19900064`.
> All 640 episodes completed with PPO/GPU/training off. Result:
> [`PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_RESULT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_RESULT.json).
> Pole B is strongly composition-sensitive: `4A/0D` wins `0.765625`
> `[0.65625,0.859375]` and `3A/1D` wins `0.53125` `[0.40625,0.65625]`,
> versus the fresh `2A/2D` baseline `0.125` `[0.046875,0.21875]`.
> `1A/3D` and `0A/4D` are `0.0` on B. Pole A instead peaks around
> `2A/2D` (`0.640625`) and sharply rejects the defensive-heavy arms:
> `1A/3D` `0.296875` and `0A/4D` `0.015625`.
> The preregistered conservative label is **COMPOSITION_DESCRIPTIVE_SIGNAL**,
> not the stricter `REGIME_DEPENDENT_COMPOSITION_SIGNAL`, because the A-side
> `4A/0D` versus `2A/2D` win-rate interval still includes zero. The practical
> finding is nevertheless clear: **Pole B requires an attack-heavy
> composition under these scripted semantics, while identity assignment was
> not the lever.** This is a composition clue, not a learned-policy repair;
> no PPO launch or 6v6 promotion is authorized yet.
>
> Prior status — **COMMITMENT MECHANISM SUPPORT PASS / REPAIR SPEC NEXT**
> The CPU-only surgical probe passed all contracts, including G6 outcome-blind
> schema, over 64 source traces and seeds `19600001–16`:
> [`ACTION_INTERFACE_COMMITMENT_MECHANISM_READING.json`](../artifacts/strategic_demand/sppo/ACTION_INTERFACE_COMMITMENT_MECHANISM_READING.json).
> In 4v4 BREACH, stale-switch burden is `0.49` versus `0.32` in 2v2, with
> paired scale contrast `+0.167502` `[+0.084054,+0.251855]`. Blocked-switch
> burden is `0.54` versus `0.37`, contrast `+0.171654`
> `[+0.115528,+0.233837]`. Both clear the frozen mechanism gate. Decision-lag
> support was unavailable for one seed pair, but it was not needed because the
> stale and blocked gates passed. Commitment is now a **mechanistically
> supported intervention target**, not yet a demonstrated 4v4 fix. Freeze the
> smallest commitment-only repair and validate its stale/blocked reduction
> before any PPO or team-outcome run. Credit branch closed:
> [`POLE_B_CREDIT_ADVANTAGE_DIAGNOSTIC_READING.json`](../artifacts/strategic_demand/sppo/POLE_B_CREDIT_ADVANTAGE_DIAGNOSTIC_READING.json)
> (anomaly present / not failure-localizing / no credit-correction training).
> Spec: [`ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json`](../artifacts/strategic_demand/sppo/ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json).
> Board: [`4V4_REPAIR_BOARD_20260917.json`](../artifacts/strategic_demand/sppo/4V4_REPAIR_BOARD_20260917.json).
>
> Prior — Pole-B credit diagnostic CLOSED (A3 counterexample on \(G_{\mathrm{diff}}\)).
> Action-interface scale diagnostic SPEC was frozen with no execution until
> contracts; contracts passed and the 64-trace run completed. A mechanism-
> support replay is the only authorized follow-up before any intervention.
>
> Prior same day - Pyquaticus behavioral-role port
> **G1-G5 TRUE PASS / G6 PROJECTED FAIL / TEAM EVALUATION BLOCKED**
> ([`PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json)).
> Official upstream `main` is pinned at
> `72b50e067ab311929390ecd4e59131452be15c6d`; BaseAttacker and
> BaseDefender `mode="easy"` only; Heuristic_CTF_Agent excluded. TRUE_NATIVE
> preserves the global-state contract and direct physical targets. PROJECTED
> failed its frozen outcome-blind representability contracts, so no adapter was
> frozen. Result:
> [`PYQUATICUS_PORT_CONTRACT_RESULT.json`](../artifacts/strategic_demand/sppo/PYQUATICUS_PORT_CONTRACT_RESULT.json).
> Provenance and G1-G5 native semantics passed. G6 failed in the no-carrier
> attack, non-carrier friendly-carrier, defender-outside, and defender-inside
> cases. Exact GET_FLAG targeting still exceeded the 2.5-cell trajectory-RMSE
> ceiling because the macro interface cannot express the upstream 0.5-speed
> intent. In the friendly-carrier/non-carrier case, GO_HOME was structurally
> unsatisfiable while GO_TO failed direction and temporal-motion checks. This is
> **ACTION-INTERFACE DISTORTION**, not an assignment result. The planned first
> assignment rung remains episode-static 2A/2D with
> CLOSEST_DEFENDS, FARTHEST_DEFENDS, and FIXED_IDENTITY. **No implementation,
> team evaluation, training, seed allocation, or old-branch merge beyond the
> completed CPU contract milestone is authorized. No GPU was used.**
>
> Prior same day - Pole-B credit/advantage diagnostic **COMPLETE**
> ([`POLE_B_CREDIT_ADVANTAGE_DIAGNOSTIC_RESULT.json`](../artifacts/strategic_demand/sppo/POLE_B_CREDIT_ADVANTAGE_DIAGNOSTIC_RESULT.json)).
> Additive useful-minus-idle advantage contrasts were negative for all five
> policies: B_t500k `-1.040`, B_final `-0.611`, GETFLAG `-1.678`,
> Assignment-v1 `-1.222`, A3 `-2.540`. Because successful frozen A3 is the
> most negative row, this diagnostic does **not** localize simple credit
> misalignment as the 4v4 failure mechanism. The preregistered ratio CI is
> invalid because its implementation clamps negative denominators with
> `maximum(denominator, 1e-9)`; raw means and additive differences are retained,
> and an explicit analysis-integrity amendment is still required. No PPO repair
> is authorized from this result.
>
> Prior — **GETFLAG sealed MISS**
> ([`GETFLAG_PRESERVE_4V4_READING.json`](../artifacts/strategic_demand/sppo/GETFLAG_PRESERVE_4V4_READING.json)):
> manipulation confirmed, specialization not demonstrated
> (Δ_A=**−0.046875**, Δ_B=+0.015625 point / LCB≤0; integrity on Δ_A).
> Contingency **ACTIVATED** → Pole-B credit/advantage diagnostic.
>
> Prior 2026-09-17 - Assignment-v1 200k COMPLETE; crossover reading
> [`ASSIGNMENT_V1_4V4_READING.json`](../artifacts/strategic_demand/sppo/ASSIGNMENT_V1_4V4_READING.json)
> is **NOT SPECIALIZED / NO DEMONSTRATED RECOVERY**. Delta_A=`+0.125`
> `[-0.03125,+0.28125]`; Delta_B=`-0.015625`
> `[-0.140625,+0.109375]`; joint gate FAIL. The privileged assignment signal
> passed its manipulation contracts but was insufficient to recover 4v4
> specialization. No Assignment-v2, 6v6 promotion, or GETFLAG combination.
>
> Prior same day — Assignment diagnostic PASS (G1=1.0, G2=9/9, G3=PASS);
> [`ASSIGNMENT_CONDITIONING_V1_SPEC.json`](../artifacts/strategic_demand/sppo/ASSIGNMENT_CONDITIONING_V1_SPEC.json)
> frozen and train wiring landed.
>
> Prior same day — Board + Assignment diagnostic SPEC frozen
> ([`ASSIGNMENT_INFORMATION_DIAGNOSTIC_SPEC.json`](../artifacts/strategic_demand/sppo/ASSIGNMENT_INFORMATION_DIAGNOSTIC_SPEC.json)).
> CPU-only G1∧G2∧G3 before any Assignment PPO. GETFLAG stays PAUSED.
> No active GPU experiment.
>
> Prior same day — Board:
> [`4V4_REPAIR_BOARD_20260917.json`](../artifacts/strategic_demand/sppo/4V4_REPAIR_BOARD_20260917.json)
> + [`4V4_RESCUE_TREE_AFTER_ROLE_AMENDMENT_PAUSE_GETFLAG.json`](../artifacts/strategic_demand/sppo/4V4_RESCUE_TREE_AFTER_ROLE_AMENDMENT_PAUSE_GETFLAG.json).
>
> * **ROLE:** finished — `RULE_ROLE_4V4_READING.json`. Δ_A=+0.203
>   [+0.047,+0.359], Δ_B=**−0.046875** [−0.203,+0.109] (= clean `B_t500k`
>   point estimate). Integrity flag kept; **no 6v6 ROLE**.
> * **GETFLAG:** **PAUSED** —
>   `B_GETFLAG_PRESERVE_CROSSOVER_PAUSED.json`. Boxed:
>   **MANIPULATION CONFIRMED / OUTCOME UNEVALUATED** (0.191→0.0013;
>   Δ_B=UNKNOWN). Premature `GETFLAG_PRESERVE_4V4` launch aborted;
>   no rows. Branch intact; n=64 crossover later. **Do not
>   combine with ASSIGNMENT.**
> * **ASSIGNMENT:** **diagnostic frozen** — teacher greedy+`taken`
>   who↔what signal; must PASS G1 agreement / G2 resolve ARM1 TEST_0
>   / G3 no macro leak before any train.
>
> Prior same day — Rescue tree frozen (`4V4_RESCUE_TREE_AFTER_ROLE.json`)
> had sequenced GETFLAG crossover next; that step is superseded by the
> pause amendment above.
>
> Prior same day — **ROLE 4v4 crossover FLAGGED / did not help.**
> `RULE_ROLE_4V4`: Δ_A=+0.203 [+0.047,+0.359], Δ_B=**−0.046875**
> [−0.203,+0.109] — identical point estimate to clean `B_t500k` screen.
> Integrity flag (δ_B≤0); no PASS RESULT. **Do not** 400k ROLE / retune
> H_r / 3–1 roles / port to 6v6.
>
> Prior same day — Parallel routing for 4v4 speed: freeze
> [`RULE_BASED_ROLE_CONDITIONING_SPEC.json`](../artifacts/strategic_demand/sppo/RULE_BASED_ROLE_CONDITIONING_SPEC.json)
> (`π(a|o,r)`, geometric N/2 DEFEND/ATTACK, `H_r=8`, no macro hard-code, no
> GETFLAG mix, short 200k after contracts). **Do not cancel** in-flight
> GETFLAG manipulation; record it when done — role arm does **not** wait on
> GETFLAG Δ_B / kill-table.
>
> Prior same day — Frozen `B_GETFLAG_PRESERVE_SPEC.json`: one short B-only
> continuation from `B_t500k`, 200k steps, gated non-carrying GET_FLAG macro
> preservation. 2×2 kill table frozen; A remains frozen (`A_FREEZE_RULE.json`).
>
> Prior 2026-09-13 — Action-interface diagnosis sealed:
> `ACTION_INTERFACE_DECOMP_SEALED_READING.json` (Rule-12 PASS). Binder is
> **stale GO_TO commitment** (\(L_C^{4v4}=+0.71\)), not W50 projection
> (\(L_S^{4v4}=0\)). Next: `REPAIRED_GO_TO_H1_PROJECTED_SPEC.json`
> (`macro_commit_go_to_ticks` 4→1) for 4v4 specialization-gate recovery.
> **Privileged BC / PPO / SNR remain PAUSED.**
>
> Prior 2026-09-11 — Sealed role crossover **restarted with tqdm**
> (`4V4_B3_ROLE_PRESERVATION_SEALED_CROSSOVER_TQDM_RESTART.json`; no RESULT
> from first attempt). Single worker; bar on stderr (`0/512…`). Shared
> `experiments/tqdm_loop.py` wired into crossover/certify/diagnostic collectors;
> PPO already has SB3-style bar (`enable_progress_bar=True`).
>
> Prior same day — 3v3 strategic-demand Step 1 completed:
> `STRATEGIC_DEMAND_3v3_GUARD_DISTRIBUTED_V2_N192_CERTIFICATION.json` =
> **NOT_CERTIFIED** (`delta_A` +0.0104 [−0.0833, +0.1042], `delta_B` +0.5625
> [+0.4948, +0.6354], n=192, seeds `12321001..12321192`). Reading:
> [`3V3_STRATEGIC_DEMAND_N192_RESULT_READING.json`](../artifacts/strategic_demand/sppo/3V3_STRATEGIC_DEMAND_N192_RESULT_READING.json).
> **No 3v3 specialist training is authorized.** This is demand failure, not
> latent-strategy verification.

### 3v3 strategic-demand ladder (Step 1 only) — `NOT_CERTIFIED`

Preregistered in
[`3V3_STRATEGIC_DEMAND_N192_AMENDMENT.json`](../artifacts/strategic_demand/sppo/3V3_STRATEGIC_DEMAND_N192_AMENDMENT.json)
(seed retirement:
[`3V3_STRATEGIC_DEMAND_N192_SEED_RETIREMENT_AMENDMENT.json`](../artifacts/strategic_demand/sppo/3V3_STRATEGIC_DEMAND_N192_SEED_RETIREMENT_AMENDMENT.json)).
Protocol: size-normalized Pole A/B, GUARD_DISTRIBUTED_V2, paired n=192,
same bootstrap/gate as 4v4 n=192 demand certs. Tool:
`experiments/certify_strategic_demand_scaled.py --team-size 3`.

| Cell | Win rate |
|------|----------|
| GUARD @ A | 0.4323 |
| BREACH @ A | 0.4219 |
| GUARD @ B | 0.0000 |
| BREACH @ B | 0.5625 |

Pole B demand is strong; Pole A GUARD advantage is null at powered n.
Same qualitative pattern as 4v4 GUARD_DISTRIBUTED_V2. Concealment N/A on
this scaled row. Steps 2–4 (specialists / crossover / Share-*) **not started**.

### Current non-latent campaign (V3 M1)

| Item | Status |
|------|--------|
| `RULESET_V3_M1` (`own_flag_home_required_to_score=True`) | **FROZEN** |
| M1 post-block recovery test | **PASS** (`experiments/test_m1_own_flag_home_scoring.py` [5]) |
| M1 2v2 Gate B (block `2300001`) | **SCIENTIFIC FAIL** — OP7 BREACH−GUARD +0.531 PASS; OP6 GUARD−BREACH +0.094 FAIL |
| Strategic Demand Searcher | **COMPLETED THROUGH V3.** Strategic demand validated; frozen poles are A = OP6 + `min_alive_for_defender=2`, B = canonical OP7. |
| PPO / specialists / latent | **SAPPO PASS / EXP2 FAIL / EXP2B FAIL / EXP2C FAIL / SPPPO V1 FAIL (CLOSED).** Automatic latent variants and the SPPPO V1 branch are closed. Phase 0 Gates 0A/0B passed; SPPPO V1 completed its frozen 1 M run and final 192-seed evaluation. |

### H-OG-PSP V4 — `VALID / TRAJECTORY IDENTITY PARTIAL / PAYOFF EVAL AUTHORIZED`

V4 changed one treatment axis from V3: the shared scalar critic output was
replaced by two latent-private scalar heads. Training completed validly at
1,003,520 steps. The frozen terminal checkpoint SHA256 is
`e65d701bee2d10cae98220630b62d9a3bfe539bc1630eaadaedadc009574c2f0`.

The one-shot mechanism diagnostic passed all integrity checks over 128 CALIB
episodes. Pole A retained confident trajectory ordering (`delta_tau_A =
+0.9809`, LCB95 `+0.5302`), while Pole B did not (`delta_tau_B = +0.1117`,
LCB95 `-0.3611`). The permanent reading is `TRAJECTORY_IDENTITY_PARTIAL /
IDENTITY_DEGRADED`, committed at `af03130b`. It is non-gating.

The payoff protocol is
[`HOG_PSP_V4_EVAL_SPEC.json`](../artifacts/strategic_demand/sppo/HOG_PSP_V4_EVAL_SPEC.json).
It scores only the frozen terminal checkpoint on untouched seeds
`11400101..11400132`. The six cells, paired seed-level bootstrap (`n=20000`,
alpha `0.05`, RNG `7`), and requirement that both crossover LCB95s exceed zero
are unchanged from V3. Retention and mechanism identity remain context only.
The one-shot 192-episode evaluation is running as PID `47992`; launch provenance
is recorded in
[`HOG_PSP_V4_EVAL_RUNNING_PROVENANCE.json`](../artifacts/strategic_demand/sppo/HOG_PSP_V4_EVAL_RUNNING_PROVENANCE.json).

### H-OG-PSP V3 — `TRAJECTORY IDENTITY CONFIRMED / CROSSOVER NOT CONFIRMED`

The valid 1 M run terminated at 1,003,520 steps with terminal checkpoint SHA256
`9f705eaed43e83ee48662dd95449819d0f239b5733e64184fa00cabd12885a69`.
All 12 terminal-integrity checks passed. The mechanism result was frozen before
payoff data and committed at `df144084`: paired trajectory-identity LCB95s are
`+5.6553` on Pole A and `+5.1032` on Pole B, so
`TRAJECTORY_IDENTITY_CONFIRMED` is permanent regardless of the payoff result.

The completed one-shot payoff protocol is
[`HOG_PSP_V3_EVAL_SPEC.json`](../artifacts/strategic_demand/sppo/HOG_PSP_V3_EVAL_SPEC.json).
It scores only the terminal checkpoint on untouched seeds
`11300101..11300132`. The six-cell design and paired bootstrap are unchanged
from OG-PSP. Strategic crossover requires both `LCB95(delta_A) > 0` and
`LCB95(delta_B) > 0`; mechanism identity and retention are not payoff gates.
The final payoff result is `HOG_PSP_V3_CROSSOVER_NOT_CONFIRMED`: `delta_A =
-0.0312` with LCB95 `-0.2188`, and `delta_B = -0.1562` with LCB95 `-0.4062`.
The reversal was verified genuine and frozen in commit `9e1684a6`.

### SPPPO V1 — `CLOSED / NOT CONFIRMED`

The frozen protocol is
[`SPPPO_V1_PROTOCOL.json`](../artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json).
Phase 0 collection completed; Gate 0A and Gate 0B passed. The development
lambda sweep selected `lambda_R = 1.0` on the out-of-sample `10200001..10200032`
block. The single 1 M production run maintained persistent assigned-pole training
(`z0 <-> OP6`, `z1 <-> OP7`, zero consistency violations) with strategic ranking
attached. The untouched final evaluation on `10300001..10300192` is complete.

Authoritative verdict:
[`SPPPO_V1_STRATEGIC_PAYOFF_PRESERVING_PPO_NOT_CONFIRMED.json`](../artifacts/strategic_demand/sppo/SPPPO_V1_STRATEGIC_PAYOFF_PRESERVING_PPO_NOT_CONFIRMED.json).
Branch closure:
[`SPPPO_V1_CLOSED.json`](../artifacts/strategic_demand/sppo/SPPPO_V1_CLOSED.json).

Both crossover contrasts were correctly signed at the mean (`delta_A = +0.0365`,
`delta_B = +0.0521`) but neither LCB95 cleared zero. Retention passed strongly
(`rho = 1.221`, `LCB95 = 1.107`). JSD and Pole-A identity passed; Pole-B identity
failed (`margin_B = -0.0309`, `LCB95 = -0.0364`). Classification is **FAIL, not
INVALID**. No rerun, grid extension, seed expansion, or SPPPO V2 without a fresh
prospective PI decision.

Postmortem D0 on the already-scored `10300001..10300192` block found an
`OPTIMISATION_INTERFERENCE` fork at the worst-quartile of `margin_B`, with a
spectacular localized Q_psi inversion on `own_flag_stolen` states
(~24% correct-rate vs ~83% on `own_flag_home`). D1 is authorized and frozen
before computation in
[`D1_PROTOCOL_FROZEN.json`](../artifacts/strategic_demand/sppo/D1_PROTOCOL_FROZEN.json):
**D1A** audits Phase-0 train support for the four frozen categories vs D0 `z1|B`
proportions; **D1B** recomputes worst-quartile Q_psi correct-rate after removing
`own_flag_stolen` with seed-level bootstrap and replicate-internal quartile cuts.
D1B is labeled a postmortem follow-up diagnostic, not a gate. No training until
D1 resolves.

**D1 COMPLETE** (`d54034b7`; closeout
[`D1_COMPLETE.json`](../artifacts/strategic_demand/sppo/D1_COMPLETE.json)).
Coverage is ruled out: Phase-0 raw \(P(\mathrm{stolen})=0.310\),
\(P(\mathrm{carrying})=0.355\) vs lower D0 `z1|B` rates. The stronger diagnosis
is a regime-specific \(Q_\psi\) ranking failure around `own_flag_stolen`, which
materially depresses worst-quartile correctness (`0.709 → 0.823` after exclusion).
Future targets (scorer regime representation; latent optimization interference)
are identified but **not authorized**. Do not jump to collecting more data.

### RASR-PPO successor causal ladder — `STRATIFIED COLLECTION (AMENDMENT_2) RUNNING`

DEV qualification on the original RASR block froze as **INVALID** (five
pole×regime cells below the 32-distinct-seed floor). That stopped the S0–R3
1 M arms. A prospective stratified 16-cell repair was designed; Attempt 1 used
source-only eligibility and was **STOPPED BEFORE FULL COLLECTION / FEASIBILITY
DEFECT** (not FAIL, not INVALID) after a one-seed smoke showed the plan could
not meet its own support floor. Seed `10700001` is quarantined under
`stratified_quarantine_preamendment/`.

**AMENDMENT_2 (Option B)** is frozen: all four `(policy, pole)` trajectories are
eligible; cell identity remains pole×regime×horizon; rarest-first, 160×12, and
the 32-seed floor are unchanged; source provenance is stored; secondary
source-balance tie-break is frozen with mandatory counter rebuild on resume.
Amended full-block collection is running now on `10700001..10700160`. PPO launch
and FINAL remain blocked until that collection completes and the support audit
passes.

### Phase 0 action-conditioned scorer feasibility — `COMPLETE + PASS`

The frozen scorer block is `6500001..6500256`, split by seed into `160` train
and `96` held out. The corrected first-interval evidence covers all four source
paths with seeds `6500001` and `6500002`; replay-to-state reproduction and
teacher-consistent continuation passed exactly. Environment reuse was tested
over 24 comparisons spanning both poles, both teachers, and early/mid/late
branches. It failed 20/24 comparisons, including restored observations, masks,
RNG-sensitive state, telemetry, actions, returns, and outcomes, with only
`1.007x` measured speedup. Reuse is permanently rejected.

The authoritative full collector therefore uses a fresh seeded environment for
every teacher continuation at every branch point. Frozen allocation remains
`1,024` paired plain episodes plus `1,536` matched branch points, with no data
growth after failure. Gate 0A must reproduce the SAPPO crossover; Gate 0B must
recover the held-out matched-state action ordering on both poles. Either failure
stops before SPPPO. If both pass, the SPPPO protocol may be frozen with the
already prospective `1,000,000`-step terminal-only budget.

The full collector completed all `256` seeds on rebuild-per-branch semantics.
Gate 0A reproduced the SAPPO crossover on all `1,024` plain episodes. Gate 0B
passed on the untouched `96`-seed held-out split after the win-margin target
amendment. Collection artifacts remain under
`phase0_scorer_data/full_collection_rebuild_per_branch/`.

### Experiment 2: K=2 supervised latent compression — `VALID / NOT CONFIRMED / FROZEN`

The prospective protocol is frozen in
[`EXP2_K2_LATENT_COMPRESSION_PROTOCOL.json`](../artifacts/strategic_demand/EXP2_K2_LATENT_COMPRESSION_PROTOCOL.json).
It tests whether one shared, episode-persistent `K=2` policy can compress the
confirmed SAPPO repertoire. The student has no router, no `q_phi`, and no
opponent-identity input. The assigned mapping is `z0 <- pi_A` and
`z1 <- pi_B`, so this is a supervised compression experiment classified as
`DIAGNOSTIC_SUPERVISED_COMPRESSION`, not paper-faithful or label-free latent
discovery.

The single frozen attempt uses 2 M total environment steps, a static balanced
8/8/8/8 split over `{z0,z1} x {A,B}`, and one extra objective only: masked
teacher-policy KL with `lambda=0.1` at the established one-per-four PPO actor
cadence. The terminal checkpoint alone is scored. Passing requires both
forced-z crossover LCBs above zero, a paired-bootstrap 90% SAPPO matched-value
retention LCB, and matched-state action identity aligned with both teachers.

The additive implementation lives in
`experiments/run_exp2_k2_latent_compression.py` and is fail-closed by default:
without `--launch` it emits only the resolved config contract; with `--launch`
it additionally requires a passing implementation-gate artifact. The online
teacher path reads the runner at updater-use time, records unconditional
cadence/KL/agreement/cell telemetry, persists cadence and RNG state in the
checkpoint, and is structurally absent when disabled. The synthetic smoke in
`EXP2_K2_TEACHER_KL_SMOKE.json` loaded both frozen checkpoint hashes and
measured one real Adam teacher update after four PPO completions. It reduced
mapped KL for both modes (`z0: 2.5586 -> 1.6565`, `z1: 2.6986 -> 1.2898`)
without constructing an environment or consuming an experiment seed.

Production launch provenance was committed at `0d131a18`, then the launcher was
invoked once with `--launch`. The first permanent runtime interval at 8,192
steps reports 64 PPO actor updates, 16 teacher updates, exact ratio `0.25`,
`lambda=0.1`, K=2 with `q_phi` absent, and realized cell counts/steps of
8/8/8/8 and 2,048 each. Both mapped teacher KLs are live (`z0=1.6350`,
`z1=1.4573`) and no treatment invariant fired. The owning evidence is
`EXP2_K2_PRODUCTION_LAUNCH_VALIDATED.json`. These values are health diagnostics
only and cannot select a checkpoint, alter the treatment, or declare success.

Seed reservations passed the full worktree and Git-history audit before the
protocol was written: training `8100001..8100320`, development
`8200001..8200192`, and evaluation `8300001..8300192`. Production training
completed at 2,002,944 steps with final-checkpoint SHA256
`a4a34c8310b50abb779f7fff8f3921a272c3ec9e23e3cec4f4e96076a84cd44f`.
The measured terminal teacher:PPO ratio was `2843/11374 = 0.249956`, and all
four cells ended at 500,736 realized steps. The terminal evaluator and tests
are committed at `943c9055`; 14 focused tests and the permitted development
smoke passed. The single formal evaluation launched on `8300001..8300192` at
2026-08-21 11:17:31Z. Its evaluator SHA256 is
`393e3479bfe170498772583ceebe76324a3e914907e2e5950b0aa5f7b9934c2c`.
The worker completed all 1,536 episodes with empty stderr. The permanent result
record is `EXP2_K2_LATENT_COMPRESSION_NOT_CONFIRMED.json`. Both crossover gates
failed: `delta_A=+.0208`, LCB95 `-.0729`; `delta_B=-.0313`, LCB95 `-.1354`.
Retention passed at `rho=1.0940`, LCB95 `.9912`, and all three action-identity
gates passed. EXP2 therefore preserved performance and teacher-aligned action
differences without preserving complementary payoff structure.

The leading post-result mechanism is prospective, not established: balanced
cross-pole PPO optimization may have pushed both modes toward generalist payoff
behavior. EXP2B will test that mechanism by changing only the live training-cell
assignment from `8/8/8/8` over `z0|A,z0|B,z1|A,z1|B` to `16/0/0/16`, while
keeping the crossed cells in terminal evaluation. EXP2 alone does not prove
causality. EXP3 and EXP4 remain locked.

### Experiment 2B: specialization-preserving latent compression — `VALID / NOT CONFIRMED / FROZEN`

The prospective causal-ablation protocol is
`EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_PROTOCOL.json`. It freezes
the single scientific delta before implementation: PPO training assignment is
`16 x z0|A + 16 x z1|B`; crossed cells are evaluation-only. Architecture,
teachers, hashes, `lambda=.1`, 1:4 cadence, PPO settings, 2 M budget, persistent
z, no-router contract, terminal-only scoring, bootstrap, retention, identity,
and crossover gates remain unchanged from EXP2.

Fresh blocks passed worktree plus all-branch history audits with 5,000-seed
margins: training `8400001..8400320`, development `8500001..8500192`, and
evaluation `8600001..8600192`. Observational actor-gradient cosine telemetry is
non-gating and must prove zero parameter, optimizer, `.grad`, and RNG mutation.
No EXP2B implementation or training step existed when the protocol was frozen.
Implementation commit `a6c28a56` now realizes only the assigned-cell delta and
adds non-gating actor-gradient-cosine telemetry. The live zero-step proof and a
4,096-step development-seed trainer smoke both resolved `16/0/0/16`; the smoke
executed 32 PPO actor updates and 8 teacher updates (ratio `.25`) with M1 and
the frozen teacher mapping intact. The immutable launch gate is
`EXP2B_IMPLEMENTATION_GATE.json`. No production step had been spent when that
gate was written.

Production launched exactly once from commit `048aa480` with seed `8400001`.
The immutable first-interval record is
`EXP2B_FIRST_INTERVAL_TREATMENT_EVIDENCE.json`. At 4,096 steps it records
`16 x z0|A`, `16 x z1|B`, zero crossed cells, 32 PPO actor updates, 8 teacher
updates, ratio `.25`, active teacher KL, and no treatment invariant violation.
These are treatment-health facts only. The frozen terminal gates retain sole
authority over the EXP2B scientific verdict.

Training terminated cleanly at 2,002,944 environment steps. The final
checkpoint SHA256 is `4b0a0e08051be6abba8b34c11d34c5b7dd82616903fa87f39f859e65a7115a0b`
and matches the evaluation manifest. The single frozen terminal evaluation was
then launched from evaluator commit `ec61be42` on `8600001..8600192`; complete
wrapper plus shared-scoring provenance is frozen in
`EXP2B_EVALUATION_LAUNCH_PROVENANCE.json`. No diagnostic can declare success
before all 1,536 rows and the unchanged gates are complete.

The evaluation completed all 1,536 episode rows and 192 matched-state identity
rows with empty stderr. Both crossover gates failed: `delta_A=-.0469`, LCB95
`-.1354`; `delta_B=-.0052`, LCB95 `-.0990`. Retention passed at `rho=1.1101`,
LCB95 `1.0042`, and every preregistered action-identity gate passed. The
permanent record is
`EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_NOT_CONFIRMED.json`.
Under preregistered branch case B4.1, assigned-pole optimization was
insufficient and the simple cross-pole-pressure explanation is weakened.

### Experiment 2C: mode-specific actor compression — `TRAINING COMPLETE + VALID / TERMINAL EVALUATION RUNNING`

EXP2C is unlocked only by the frozen B4.1 signature: crossover FAIL, retention
PASS, identity PASS. Its prospective question is whether destructive
interference in the fully shared actor prevents payoff specialization. The
only authorized scientific change is a minimal mode-specific actor component
after a shared observation encoder/trunk; EXP2C remains one model and one
checkpoint, not two independent PPO policies. EXP2C protocol, fresh seed
blocks, architecture choice, resolved-config delta, and implementation gate
must be frozen and committed before any training step. EXP3 and EXP4 remain
locked. EXP2C is the last automatic latent architecture variant; a valid FAIL
stops latent architecture experimentation and forbids EXP2D.

The prospective protocol is
`EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_PROTOCOL.json`. It freezes exactly two
private final `Linear(hidden_dim, action_dim)` projections selected by z after
the unchanged shared CNN and actor body. Both heads start as exact copies of
the ordinary shared head. Private adapters, private deep trunks, private
critics, and independent PPO models are forbidden. All other EXP2B treatment
and terminal gates remain unchanged. Fresh audited blocks are training
`8700001..8700320`, development `8800001..8800192`, and evaluation
`8900001..8900192`. No EXP2C implementation, environment construction, or
training step existed when this protocol was frozen.

Implementation commit `06711331` realizes only the frozen private-final-head
delta. The focused gate passed 40 tests; the broader latent/PPO regression gate
passed 221 tests and 1,721 subtests. A real frozen-teacher smoke proved each
mapped latent updates only its own private head while reducing its mapped KL,
and a 4,096-step integrated trainer smoke resolved `16/0/0/16`, executed 32 PPO
actor updates and 8 teacher updates, and moved both private heads. The immutable
launch gate is `EXP2C_IMPLEMENTATION_GATE.json`. No production step had been
spent when that gate was written.

All three reserved blocks passed the full-history audit with a live positive
control. Production launched exactly once from commit `7c9c720c` with seed
`8700001`. The immutable first-interval record is
`EXP2C_FIRST_INTERVAL_TREATMENT_EVIDENCE.json`. At 4,096 steps it records two
active private final actor heads, exact `16/0/0/16` assigned cells, 32 PPO actor
updates, 8 teacher updates, ratio `.25`, `lambda=.1`, mapped gradients on both
private heads, and empty stderr. These are treatment-health facts only. The
2 M terminal checkpoint and untouched `8900001..8900192` evaluation retain
sole authority over crossover, retention, identity, and the scientific verdict.

Training completed at 2,002,944 steps with terminal checkpoint SHA256
`0a763759d49e6ba0ef1d78b25aba04cbc99a4cbcf50df8c19b201e941f646c07`.
Evaluation attempt 1 failed during checkpoint construction before scoring any
episode because the loader omitted the default-off EXP2C architecture flag; a
secondary compatibility error path also referenced an uninitialized variable.
The typed invalid record is `EXP2C_EVALUATION_INVALID_ATTEMPT_1.json`. Fix
commit `0559d0ed` is pinned by loading the real terminal checkpoint with two
private heads. The identical evaluation relaunched with zero prior result rows,
passed exact checkpoint behavioral-equivalence checks, and is scoring all eight
cells. Launch provenance is frozen in
`EXP2C_EVALUATION_LAUNCH_PROVENANCE.json`.

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

### RESOLVED — GO_TO-only env-level DEFEND substitution test + CLOSEST_DEFENDS screen (2026-09-21/22)

**Status: COMPLETE.** Both experiments this item proposed are sealed; see the "Last updated" entries above for full results
(`GOTO_ONLY_DEFEND_SUBSTITUTION_4V4` -> `GOTO_ONLY_CROSSOVER_CONFIRMED` with caveats, commits `541e9bab`/`9d14717d`;
`CLOSEST_DEFENDS_SCREEN_4V4` -> `pattern_match: true`, commit `f4f4ac6f`). The original proposal text below is kept for the
design trail (three PI-approved defaults, the interpretation tree, the fences) rather than rewritten; nothing past this note
is still open. **Next step per the PI: stop running diagnostic screens; move to the training architecture** -- `CLOSEST_DEFENDS`
allocator -> ATTACK/DEFEND role label -> role-conditioned PPO, allocator kept at evaluation. That is a PPO/training change and
is NOT authorized by this entry; it requires its own pre-change checklist per AGENTS.md before any code is written.

<details>
<summary>Original proposal text (2026-09-21, kept for the design trail)</summary>

**Status (at proposal time):** PROPOSED only. No seed block allocated, no code written, nothing launched. Not a preset (no `PPOConfig`, preset, actor,
critic or loss change), so no Proposed Preset Review applies. Follows from the frozen representability audit (`f2e051de`).

**Question.** Is instantaneous DEFEND heading *necessary* for the scaffold's crossover effect, or does GO_TO-only defense -- the coarse
native vocabulary, made to behave as much like the scaffold's defenders as GO_TO permits -- preserve it? The audit shows DIRECTION fails
while TARGET (~98%) and native-commit TRAJECTORY (~96%) pass; it never tested necessity. This experiment intervenes on exactly that
ambiguity, which correlation between low DIRECTION and a non-crossing learned policy cannot do.

**Hierarchy.** forced 2A/2D scaffold `A'` (known crossover, sealed) -> GO_TO-only approximation of 2A/2D `N'` (?).

**Design as proposed -- every choice below is the author's default, for the PI to veto before freeze:**

* Fresh `sealed_confirmatory` block; both certified 4v4 poles (Pole B = the B3-3 genome, never canonical OP7); the same pair rotation
  (`PAIRS[seed % 6]`). Arms: native `pi_A`, `A'` (the unchanged target-injection scaffold), `N'`, native `pi_B`; n = 128 paired seeds
  -> 1,024 episodes (~5 h on 6 shards). `A'` and `pi_B` are re-run on the fresh seeds so `N'` vs `A'` is paired.
* `N'`: the same two agents per episode as in `A'` are driven by a controller whose ONLY outputs are native actions (macro `GO_TO` plus
  a legal waypoint index), fed through the real action interface and taking effect at native commit boundaries. No
  `install_forced_defend_target`, no resolved-target override, no new macro (`n_macros` stays 5), PPO off.
* Controller = a CAUSAL version of the audit's greedy path-oracle: at each commit boundary it recomputes the isolated DEFEND rollout
  from the agent's CURRENT live state (a function of state only, no future information) and commits the legal `GO_TO` waypoint whose
  commit-horizon path best matches it. It uses the teacher law as information, so this tests interface capacity in the env, not learnability.
* Manipulation check (controller-quality gate): in the env, `N'`'s defender path must meet the audit's TRAJECTORY criterion against the
  isolated DEFEND reference (RMSE <= 2.5 cells, coverage >= 0.90 per pole) before a crossover failure may be read as vocabulary necessity.
* Primary: crossover on `N'`: LCB95(WR(N',A) - WR(pi_B,A)) > 0 AND LCB95(WR(pi_B,B) - WR(N',B)) > 0 (paired percentile bootstrap,
  20000 resamples, rng seed 7, as in the bridge). Reported, non-gating: `N'` vs `A'` paired contrasts per pole (how much of the
  scaffold's suppression of A on B is retained), Blue goals, margin.

**Informative outcomes (pre-stated).** (1) GO_TO-only still crosses over and the manipulation check passes -> DIRECTION was not necessary
for this question; the audit's routing was conservative; no DEFEND macro is needed; proceed toward teaching `pi_A` with the existing
vocabulary. (2) GO_TO-only loses crossover with the manipulation check passing -> something absent from the coarse reproduction
matters; DIRECTION becomes a strong candidate and a learned-selectable DEFEND primitive test becomes justified rather than speculative.
(3) The manipulation check fails -> `INCONCLUSIVE_CONTROLLER`: fix the controller before concluding anything; never read as "DIRECTION necessary".

**Fences (as for the bridge).** No PPO; no new vocabulary; native and `A'` arms unchanged; contracts before any seed is spent (including
exact reproduction of sealed episodes through the new runner for the unchanged arms, and a legal-action contract that every `N'` action
is `(GO_TO, legal idx)`); commit the frozen pre-run package before launch; no interim outcome reading; no top-up. Interpretation guard:
`N'` is an oracle-informed GO_TO controller, NOT a learned policy; a positive result says the vocabulary suffices for the effect, not that PPO will find it.

**Choices APPROVED by the PI (2026-09-21):** (1) 4 arms x n=128 (a paired contemporary baseline beats a smaller run); (2) the causal
path-oracle as the `N'` controller (a ring-hold rule would add a second question, "was the controller good enough?"); (3) the
manipulation-check level unchanged at 2.5 cells / 0.90 coverage, never loosened after outcomes.

**Guardrails added at approval.** The interpretation tree is frozen explicitly (`N'` crossover -> "instantaneous DIRECTION is not
necessary for crossover", which does NOT mean the DIRECTION metric was wrong; `N'` loses crossover with TRAJECTORY < 0.90 ->
`INCONCLUSIVE_CONTROLLER` and no claim about direction or vocabulary; `N'` loses crossover with TRAJECTORY >= 0.90 -> "high path
fidelity under native GO_TO was insufficient to preserve crossover, strengthening evidence that behavior omitted by the GO_TO
approximation, including instantaneous directional control, may be causally important", NOT "DIRECTION proven necessary", because `N'`
could still differ from `A'` in another unmeasured way; it would authorize the next directional/interface experiment only). The
controller and analysis must be outcome-blind and deterministic given state/seed, and `N'` carries a per-episode attestation that every
emitted action is on the existing native `(GO_TO, legal waypoint)` surface -- no hidden helper, no direct target injection, no scaffold
action through a side door. Author's additions, flagged for veto at freeze: an `A'` positive control for the in-env trajectory
instrument, and a paired heading-agreement diagnostic showing `N'` actually removes per-tick heading fidelity relative to `A'`
(otherwise the crossover result carries no statement about DIRECTION).

**Conditional successor (PI plan, NOT authorized; gated on this experiment showing GO_TO suffices):** teacher-guided PPO warm-started
from `pi_A` (`L = L_PPO + lambda L_teacher`, lambda annealed 1.0 -> 0.5 -> 0.2 -> 0), an auxiliary ATTACK/DEFEND role head, DAgger-style
on-policy teacher queries, training-only defend shaping annealed to zero, forced roles decayed in stages, and the success criterion that
native `pi_A` ALONE passes crossover with no forced defenders, no scaffold, no teacher at evaluation. Flags for that freeze: the
scaffold's defenders are chosen by seed rotation, not by state, so a state-dependent teacher assignment would need its own bridge
test; and a prior teacher-distillation attempt failed a Compression Crossover (see memory) and GUARD's assignment machinery needs
information absent from the student observation.

**Next (at proposal time):** freeze spec -> contract-test -> commit the pre-run package (spec, runner, seed reservation, passing
contract record) -> reserve the fresh seed block -> await the PI's launch go (the run spends the confirmatory block once).

</details>

### C2 fresh confirmation — `C2_REJECTED` (2026-08-06)

**Candidate:** `none_forward_frac` (fraction of decisions during carrying
where zero blue mates are past the midline).

**Result:** `C2_REJECTED` — all 3 policies failed headroom + actionability.

```text
natural support     PASS    (606 / 624 / 584 failure onsets, prevalence ~87%)
headroom            FAIL    (11.8% / 12.7% / 13.0%)
actionability       FAIL    (0.0 / 0.0 / 0.0)

policy passes       0 / 3
required            >= 2 / 3
verdict             C2_REJECTED
O2                  DO NOT TRAIN
```

**Artifact:** `artifacts/c2_confirmation/C2_CONFIRMATION_FROZEN_RESULT.json`

**Scientific interpretation:** C1 and C2 both found **correlates of bad
outcomes** rather than **genuine strategic decision forks**. C1 was
predictive but had 0.9% headroom. C2 had adequate natural support and
strong discovery deltas, but on fresh data the feature was not reliably
actionable — changing the team response from the same state did not alter
the carrier's fate.

**Decision:** Pivot from aggregate-fraction features to decision-proximal
geometry with counterfactual actionability gating. See
[`c3-decision-proximal-preregistration.md`](c3-decision-proximal-preregistration.md).

### C3 commitment-proximal strategic-fork discovery — `FROZEN` + scan running (2026-08-06)

**Status:** checklist items 1–10 CLOSED; contract FROZEN; execution AUTHORIZED;
tiny smoke PASS; **full discovery scan launched**.
Machine-readable: `artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json`.
Authorization: `artifacts/c3_discovery/C3_EXECUTION_AUTHORIZATION.json`.

```text
C1                     CLOSED / NOT RETAINED
C2                     CLOSED / REJECTED
C3 methodology          FROZEN
C3 code                 PATCHED TO CONTRACT (+7ch CF adapt)
C3 execution            FULL SCAN RUNNING
C3 runtime cells        T_trace=40 H_response=30 delta=0.10
                        minimum_fork_rate=0.20
                        U=carrier_survival doomed_at_or_below=0.0
environment-demand gate PREREGISTERED STRUCTURE
demand-gate execution   LOCKED UNTIL O3 EXISTS
O3                      AUTHORIZED (carry-phase scope; see erratum)
latent birth            LOCKED
router                  LOCKED
```

**O3 precursor audit (2026-08-09, run once):**
`artifacts/o3_preregistration/C_FORK_PRECURSOR_AUDIT.json` — 157 forks,
coverage **0.7006**, `precursor_after_fork_only=47`, `no_precursor_found=0`.
Predicate untouched. Erratum
`artifacts/o3_preregistration/O3_CFORK_RECALL_ERRATUM.json` corrects the
false ce7949f claim that the C3 backward trace stays inside a carry phase;
O3 claim scope is frozen to **carry-phase** specialization (~70% of C3
forks). Pre-pickup forks are unresolved, not a C3 defect. Device rule:
diagnostics must match the artifact device (CPU ≠ CUDA trajectories).

**Provenance (authorized HEAD `5a797ec`):**
- freeze commit (prereg + runner contract wiring): `de93d290b8813ac1b10b1f723a674e6f1f5a409b`
- channel-adapt commit: `71bf2d64a23472fc69127ef863ee5df30de6a26c`
- authorized `c3_prereg_commit` / `runner_commit` (HEAD): `5a797eccb81496929baaac670d363eab3cd00ad4`
- `c3_contract_hash` (SHA256 of `C3_DISCOVERY_PREREG_FROZEN.json`): `c40eb5e8bca4f9a1d927b4779a312281a2c2fbee38bf7c4afa7fb94c35769ae0`
- `c3_prereg_sha256` (SHA256 of `docs/c3-decision-proximal-preregistration.md`): `a70600d96d5bbc5dd1339131ac00a7886d47832555463673f18e9a7aacfc47ab`

**Tiny smoke (PASS, wall≈844s):** `--seeds 3200001 --opponents OP6 --episodes 2 --stage 3`.
Verified natural G0, pressure anchors, backward trace, legal team responses,
doomed reject / `NO_COMMITMENT_FORK`, `H_response` branches, earliest fork
selection, provenance hashes, `CONTROLLABILITY_SCREEN_ONLY`. Result (non-
scientific): 6 anchors, 3 qualified / 3 `NO_COMMITMENT_FORK`, fork_rate=0.50.

**Full scan (ABORTED):** pid 12820 classified
`ABORTED_OPERATIONAL_SCALE / NO_SCIENTIFIC_VERDICT` — see
`artifacts/c3_discovery/C3_ABORTED_OPERATIONAL_SCALE.json`. Does not count
against C3. Cause: Stage-3 combinatorial cost without durable Stage-1 /
per-anchor resume / short-circuit.

**Ops patch (no science-cell change):** Stage-1 persists
`C3_STAGE1_ANCHORS.jsonl` + `C3_STAGE1_MANIFEST.json`; Stage-3 appends
`C3_STAGE3_ANCHOR_RESULTS.jsonl` with resume; existential δ / utility-ceiling
short-circuit. Benchmark then relaunch required before treating any new scan
as scientific.

**Preregistration (frozen):** [`c3-decision-proximal-preregistration.md`](c3-decision-proximal-preregistration.md)

**Purpose (limited):** C3 is a **cheap candidate-fork detector**. It cannot
establish latent necessity, policy complementarity, routing value, or
distinct strategy families. A C3 pass advances a candidate only to
independent task-reward response-oracle (O3) training. Latent eligibility
is decided by the separate Environment-Demand Gate (below), never by C3.

**Pipeline:**
```text
C2 REJECTED → commitment-proximal fork discovery (replay)
→ event-anchored temporal qualification
→ counterfactual controllability screen
→ fresh confirmation (natural + counterfactual)
→ train independent task-reward O3
→ payoff matrix M[c, π] on fresh demand-evaluation seeds
→ ENVIRONMENT-DEMAND GATE (D1–D4)
→ only if PASS → latent birth
```

**Demand gate prereg (`AUTHORITATIVE INTENDED STRUCTURE, executed post-O3`):**
[`environment-demand-gate-preregistration.md`](environment-demand-gate-preregistration.md)
Criteria structure is preregistered before C3 results; numeric cells marked
`TBD-freeze` close at O3 protocol freeze. Execution remains LOCKED until O3
exists. Do not treat this as authorization to run a demand-gate evaluation.

**Implementation checkpoint:** `experiments/run_c3_decision_proximal_discovery.py`,
`rl/analysis/decision_proximal_features.py`, and
`rl/analysis/counterfactual_actionability.py` are patched to the item-10 audit:
pressure anchor only, earliest bounded backward trace, authoritative legal team
responses, doomed-state rejection, improvement-only expected utility through
`H_response`, explicit `NO_COMMITMENT_FORK`, and
`CONTROLLABILITY_SCREEN_ONLY` output semantics. Numeric cells remain sourced
only from the future frozen contract. No scan artifacts exist.

**Stopping rule:** If no candidate clears all gates, record
`C3_NO_QUALIFIED_STRATEGIC_FORK.json` and STOP. If the demand gate fails:
NO LATENT BIRTH, NO ROUTER.

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

**OP6 BLUE_PROBES_V2 revisit (2026-07-28) — TURTLE niche still flipped.**

Joint-matrix flip: SPLIT +1.13 vs TURTLE −1.06
(`artifacts/op6_op10_br_diversity_acceptance_16seed`). Fresh BLUE_PROBES_V2
baseline `artifacts/op6_blue_v2_baseline_dev1_8seed` (8 paired seeds, base
701001, map_b_split_lane):

```text
BLUE_SPLIT   WR=6/8  mean_margin=+0.500
BLUE_TURTLE  WR=0/8  mean_margin=-1.000
BLUE_RUSH    WR=0/8  mean_margin=-2.375
BLUE_ESCORT  WR=0/8  mean_margin=-2.625
```

Cause: OP7 route-lock made SPLIT convert on OP6's empty rear
(`min_alive_for_defender=3` / Nr=2 → never peels), while TURTLE only delays
red scoring. Structural lesson from today's attempts: **any response that
abandons both dual-rushers when blue invades or carries softens red scoring
and helps empty-home SPLIT**.

```text
dev14 peel-ONE dual-invasion:   SPLIT +1.875  TURTLE -1.500  REJECT
dev15 peel-BOTH dual-invasion:  interim SPLIT +2; abandoned
dev16 dual carrier-deny:        SPLIT +0.750  TURTLE -0.500  REJECT as niche
dev17 TURTLE dual-agent counter: REJECT (tags temporary → empty-home score)
dev19 peel-ONE carrier-deny:    SPLIT +0.625  TURTLE -0.500  KEEP partial
dev20 true dual-assault (no peel, opposite lanes): SPLIT +2.250  TURTLE +0.000  REJECT
  (RUSH/ESCORT also improve; this over-punishes "both leave home" generally and
   does not create TURTLE uniqueness)
```

Current code (2026-07-28 Contract B defend-then-counter): OP6 keeps dual-lane
assault (`enable_intercept=False`, `min_alive_for_defender=3`) plus an OP6-only
failed-assault recovery state. Canonical map is **`map_a`** (locked four-niche
contract). Legal triggers: committed dual-rush tag on blue's half, carrier
stop, or flag loss, gated on a non-carrier blue near-home anchor. Recovering
agent stays ATTACKER, takes a midfield redeploy route, recovery ticks pause
while tagged, and team FLAG_RETR is suppressed while the window is active.
No blue-style ID. Unit tests in `tests/test_op6_failed_assault_recovery.py`.
Micro-gates: `experiments/diagnose_op6_recovery_microgates.py` (default
`--map map_a`).

```text
dev21–dev25 on map_b_split_lane: historical only (wrong map for niche lock)
dev26 map_a (`artifacts/op6_recovery_microgates_dev26_map_a`): micro_gates FAIL
  TURTLE recovery arms most (4.4 act, 98 steps; 3/8 recovery-score seeds)
  SPLIT recovery rarer (0.25) — anchor gate works directionally
  but payoff inverted: RUSH +2.88, ESCORT +2.50, SPLIT +2.00, TURTLE +1.00
  red first-score rate 0 across styles; TURTLE−SPLIT = −1.0
```

**Decision (2026-07-28, map_a): PAUSE recovery tuning.** Selectivity is
promising (TURTLE≫SPLIT activations) but OP6 never creates a dangerous
assault (`red first-score = 0`). Trapdoor works; tiger never enters.

Next: **OP6 offensive competence audit** on `map_a` (not more recovery).
Target micro-gates:

```text
OP6 red first-score: RUSH/SPLIT/ESCORT ≥5/8 each; TURTLE ≤2/8
TURTLE counter-score after stopped assault: ≥5/8
```

Track red pickups, failed returns, blue home occupancy, TURTLE anchor
present at assault-stop. Diagnostic:
`experiments/diagnose_op6_offense_competence.py`.

Secondary finding: on map_a SPLIT is no longer best — RUSH leads
(+0.875 vs SPLIT, +0.375 vs ESCORT). Do **not** lock OP6→RUSH yet
(ESCORT gap < ~0.5; need paired CIs). Reassignment is a live option
after offense is competent. Four distinct PPO niches matter more than
preserving the OP6→TURTLE label.

**Offense competence audit (dev27, map_a):**
`artifacts/op6_offense_competence_dev27_map_a` after `enable_flag_retr=False`
+ earlier flag converge. Micro-gates **FAIL**, but mechanism moved:

```text
red pickups ~3/ep (was effectively non-scoring)
SPLIT red-first:  8/8  PASS
TURTLE red-first: 2/8  PASS
TURTLE counter after stop: 8/8  PASS (anchor present at stops)
RUSH  red-first:  1/8  FAIL  (pickups 3.1, failed returns 2.1)
ESCORT red-first: 3/8  FAIL  (pickups 3.0, failed returns 2.0)
margins compressed: RUSH +0.5, ESCORT +0.25, SPLIT/TURTLE 0.0
```

Diagnosis: dual assault now reaches the flag; conversion dies on the
**return** vs V3 RUSH/ESCORT screening. Next offense lever: return-path /
failed-return rate — still not recovery tuning.

**Failed-return trace (dev28, map_a, partial→dominant modes):**
RUSH/ESCORT losses dominated by `tagged_from_behind` and
`red_agents_crossing_blocking` (plus some `carrier_route_detour`). Matches
empty-home screening pressure + carrier evasion churn with partner still
dual-rushing.

**OP6 extraction support (dev29 extract=on, map_a):** micro-gates **FAIL**.
Preserved SPLIT 8/8 and TURTLE gates; RUSH still 1/8 first-score; ESCORT 4/8;
failed returns not reduced (RUSH 2.25 vs ~2.13 pre-extract). Likely cause:
extraction canceled when a blue re-entered home mid-return. Latched arm-at-
abandon fix in flight (dev30); recovery still untouched.

**dev30 extract latch:** ESCORT red-first **6/8 PASS** (offense-audit fix only —
OP11 still owns the ESCORT latent niche); SPLIT/TURTLE gates hold; RUSH still
**1/8 FAIL** (failed returns 2.125). Remaining leak is V3 RUSH return corridor
blocking. Latch + recovery **frozen**.

**dev31 screen-break:** FAIL — RUSH still 1/8, failed returns 2.125 unchanged.
Preserved SPLIT 8/8, ESCORT 7/8, TURTLE 2/8 + counter 7/8. Narrow segment
blocker miss vs V3 RUSH charging red home ahead of the carrier. Widened to
on-segment OR ahead-near-home (dev32); latch/recovery/`lane_amplitude_frac`
still frozen.

**dev32 ahead screen-break:** FAIL — RUSH still **1/8**, failed returns **2.0**
(~6% drop, not ≥50%). SPLIT/ESCORT/TURTLE gates still hold. Corridor peel is
not the RUSH lever. Next needs a fresh RUSH-return instrument (does extract
even arm at pickup when both blues are at the flag?) before more route tweaks.
Do not run payoff matrix yet.

**dev33 extraction activation (RUSH-only, map_a, 8 seeds):**
`artifacts/op6_extraction_activation_dev33_rush_map_a`. Extraction is **alive**,
not asleep.

```text
armed_any_frac:              1.00
armed_at_pickup_frac:        0.00  (1-step BT-before-pickup lag)
never_armed_frac:            0.00
mean pickup→arm delay:       ~1.04
blue_home_anchor block:      1/24 pickups
both blues on return path:   24/24
true failed returns:         1.0 / ep  (offense script overcounts by ~1/score)
fail classes:                C=6/8, D=2/8, A=B=E=0
mean fail carry duration:    ~16 steps (not immediate death)
```

Decision-tree verdict: **C — targets wrong blue** (dual on-path threat; screener
engages one, the other tags). Do **not** retune corridor peel yet. Next lever:
blocker / dual-threat assignment under extraction (geometry-only, no style ID).
Preserve SPLIT/ESCORT/TURTLE/recovery/`lane_amplitude_frac=0.35`. No matrix yet.

**dev34b dual-threat assignment (map_a):** complementary lock at extract arm —
carrier owns evasion-nearest; screener owns the other blue (projected-danger
rank + hard 2v2 distinctness). Carrier peel route unchanged. Offense audit now
uses transition-based true failed returns.

```text
distinct assignments (dual):  24/24 (100%)     PASS ≥90%
wrong-threat fails (C1+C2):   1  (−83% vs 6)  PASS ≥75%↓
fail classes:                 C3=7, C2=1       (C1=0)
true failed returns (offense): 0.5 / ep        PASS ≤0.5
activation failed returns:     1.0 / ep        (carry-window count; see note)
RUSH red-first:                1/8             FAIL ≥5/8
SPLIT/ESCORT/TURTLE gates:     8/6/2 + ctr 8   PASS
```

Verdict: assignment coordination works; remaining leak is **C3 (screener too
late)**. Do not peel-route again. Next lever: earlier/faster screener engage on
the locked complementary threat (or pre-pickup positioning), without touching
recovery / `lane_amplitude_frac` / TURTLE. No payoff matrix yet.

**dev35 / 35b pre-pickup screener (timing-only, map_a):** freeze peel + dual-threat
locks + recovery + `lane_amplitude_frac=0.35`. Start screener before pickup when
attacker is imminent + dual blues threaten return corridor; preengage routes to
**projected intercept** on flag→home (not blue xy). TURTLE home-anchor still
suppresses.

```text
dev35 (radius 2.5):  lead≈0.5  RUSH 1st=1/8  fails=0.375  C3=7
dev35b (radius 6.5): lead≈4.25 RUSH 1st=2/8  fails=0.125 C3=6
                     TURTLE preengage 2/8 (still rare)  safety PASS
                     SPLIT 8 / ESCORT 7 / TURTLE 2+ctr 8  PASS
                     distinct ≥90% PASS
                     RUSH red-first ≥5/8 FAIL; C3≤2 FAIL
```

Race timing: screener predicted arrival still loses to blue-on-path ETA
(`screener_won_predicted_race=0`). Score race is tight — blue often first by
2–4 steps while OP6 pickup→score ≈16. True failed returns are already low, so
protected returns convert; they just convert after RUSH V3 has scored.

Verdict: extraction targeting/timing is largely solved; the remaining gate is
the **overall scoring race** (carrier convert tempo vs V3 RUSH), not more peel
or threat assignment. Kept `_OP6_PREPICKUP_RADIUS=6.5`. Freeze peel +
dual-threat locks + recovery + `lane_amplitude_frac=0.35`. No payoff matrix yet.

**dev34 dual-threat extract assignment (2026-07-28):**
OP9 ESCORT held-out failed; sequence advances here. Contract:

```text
carrier handles one blue threat
screener handles the other
no duplicated targeting
```

Preserve recovery + extract latch. Root cause of C1 duplicates: lock used
`~blue_carrying`, so when one blue held red's flag at arm time only one
"live" threat remained and both reds marked it. Fix: count all
alive/untagged blues; hard-complementary when ≥2. Unit tests pin
complementary lock + carrying-blue-counts
(`tests/test_op6_extraction.py`).

`artifacts/op6_extraction_activation_dev34c_alive_threats_rush_map_a`
(same seeds `601001`, RUSH-only):

```text
distinct_assignment_frac_dual: 1.00  PASS (≥0.90)
C1 duplicates:                 0     (was 3/8)
fail classes:                  C3=6/8 (screener too late), C2=2/8
true failed returns:           1.0 / ep  FAIL (target ≤0.5)
```

Assignment contract **PASS**. Conversion still fails — dominant residual is
**C3 screener latency**, not wrong-target. Do not retune latch/recovery/
`lane_amplitude_frac`. Next OP6 lever: screener engagement timing/path
aggressiveness (bounded), then offense-competence four-style screen before
any payoff matrix.

**dev34c offense competence (2026-07-28):**
`artifacts/op6_offense_competence_dev34c_dual_threat_map_a`, 8 seeds
`601001`, extract=on. Exit code 2 = micro_gates **FAIL** (not a crash):

```text
RUSH  red-first: 0/8  FAIL  (failed_returns mean 0.5; pickups 3.0)
SPLIT red-first: 8/8  PASS
ESCORT red-first: 6/8 PASS
TURTLE red-first: 4/8 FAIL (want ≤2/8)
TURTLE counter after stop: 8/8 PASS
margins (blue): RUSH +0.125, TURTLE −0.125, SPLIT −0.625, ESCORT −0.75
```

Dual-threat cut RUSH failed-return count vs pre-fix (~2.0→0.5) but red
still never converts first vs RUSH (0/8). Matches activation residual **C3**.
Do not run payoff matrix yet. Next bounded lever: earlier/more aggressive
screener engage on the locked complementary threat.

**dev35 pre-pickup screener (2026-07-28):** narrow gate only — arm when
attacker within `_OP6_PREPICKUP_RADIUS=2.5` of available enemy flag, blue home
abandoned, and both blues threaten the return corridor. Sticky until
pickup/expire/anchor-return. Partner is ESCORT toward locked complementary
threat (flag as virtual carrier). Geometry only (no style ID). Unit tests in
`tests/test_op6_extraction.py::TestOP6PrePickupScreener`.

Race diagnostic (`artifacts/op6_preengage_race_dev35_on_map_a`, seed
`611001`, RUSH+TURTLE):

```text
RUSH  preengage: 8/8 frequent   PASS
TURTLE preengage: 0/8 rare       PASS
RUSH  red-first (race): 2/8
mean screener travel @ pickup: ~10.4
mean pickup→score: ~20
typical first-score margin: blue by 1–3 steps
```

Offense competence (`artifacts/op6_offense_competence_dev35_preengage_map_a`,
same seed, extract+preengage on). Exit 2 = micro_gates **FAIL**:

```text
RUSH  red-first: 3/8  FAIL (≥5/8)   failed_returns 0.625
SPLIT red-first: 8/8  PASS
ESCORT red-first: 6/8 PASS
TURTLE red-first: 1/8 PASS (≤2/8)   ← recovered vs dev34c 4/8
TURTLE counter:   8/8 PASS
```

Verdict: **selectivity OK, conversion still loses the RUSH race.** Preengage
removes the assignment lag but leaves ~10 units of screener travel at pickup —
not enough to beat RUSH V3’s parallel score. Do not accept OP6→RUSH. Preengage instrumentation stays in code for
ablations/telemetry but is **rejected as the accepted race mechanism**
(selectivity PASS, timing gain insufficient). Latch/recovery/
`lane_amplitude_frac` still frozen. No payoff matrix.

**dev36 mutual-carry denial (2026-07-28):** diagnose-before-implement.

Feasibility on rejected-dev35 seeds (`611001`),
`artifacts/op6_mutual_carry_feasibility_dev36_map_a`:

```text
RUSH  mutual+abandoned: 8/8 eps (~28 steps/ep)
RUSH  interceptor ETA < blue score ETA: 98% of mutual steps
RUSH  mean ETA slack @ first mutual: ~+5.1 steps
TURTLE mutual+abandoned: 0/8
viable_to_implement: TRUE
```

Implemented OP6-only race mode (legal state only — no style ID):
`red_carry ∧ blue_carry ∧ abandoned → carrier direct home; non-carrier
ROLE_INTERCEPTOR chasing blue carrier`. Dual-threat locks / extract
carrier route / recovery / `lane_amplitude_frac` frozen. Preengage
instrumentation kept. Tests: `TestOP6MutualCarryRaceDenial`.

Race denial (`artifacts/op6_race_denial_dev36_on_map_a`):

```text
RUSH  race fired:        8/8
RUSH  blue interrupted:  8/8  PASS (≥5)
RUSH  red-first:         0/8  FAIL (≥5)
TURTLE race fired:       0/8  PASS (≤1)
mean blue_delay vs ETA:  ~7.6
intercept ETA @ arm: often 0 (in range) but blue still scores first by 1–3
```

Offense competence (`artifacts/op6_offense_competence_dev36_race_map_a`):

```text
RUSH   red-first: 3/8  FAIL
SPLIT  red-first: 8/8  PASS
ESCORT red-first: 8/8  PASS
TURTLE red-first: 2/8  PASS
TURTLE counter:   8/8  PASS
```

Verdict: **selectivity + geometric delay PASS; first-score conversion
FAIL.** Blue is often already ~5 steps from score (near/on home half) when
mutual carry arms — contact does not flip the race. Do **not** accept.
No payoff matrix.

**dev36b/c race denial follow-ups (seed `701001`, map_a):**
- **36b** suppress peel during race + tighter cut: interrupt 8/8, TURTLE
  race 0/8, red-first still **1/8**; offense RUSH 1/8, fails 0.375,
  SPLIT/ESCORT/TURTLE preserved.
- **36c** arm race on blue_carry ∧ abandoned ∧ (red_carry ∨ imminent)
  so denial starts before red finishes pickup: same pattern —
  interrupt **8/8**, TURTLE race **0/8**, red-first still **1/8**;
  offense RUSH **1/8**, fails **0.5**, SPLIT 8 / ESCORT 6 / TURTLE 2+ctr 8.
  Artifacts: `op6_race_denial_dev36c_imminent_map_a`,
  `op6_offense_competence_dev36c_race_denial_map_a`.

**dev36 CLOSED (2026-07-28) — return to payoff tooth.** Reject extraction /
preengage / race-denial as gameplay (toggles default **OFF**;
instrumentation retained; `_OP6_PREPICKUP_RADIUS` stays **6.5** for ablations).
Delaying blue does not flip the first-score race. `RUSH red-first ≥5/8` is
**not** a Summer OP6 acceptance gate — diagnostic proxy only. Real
requirement: `BLUE_TURTLE` uniquely highest final payoff vs OP6.

Frozen landscape OP6 config for held-out:
`artifacts/map_a_v3_landscape_op6_op12_8seed` (TURTLE uniquely best,
gap +0.88 vs ESCORT). Keep dual-assault + recovery +
`lane_amplitude_frac=0.35`. No more scoring-race / peel / threat-assignment
engineering.

**Next: OP6 TURTLE held-out 16-seed** (`map_a`, `BLUE_PROBES_V3`, fresh
seed `621001`, frozen toggles OFF). Artifact:
`artifacts/op6_turtle_heldout16_mapa_seed621001`. Accept iff TURTLE
uniquely best and paired CIs clear vs RUSH/SPLIT/ESCORT plus pooled
best-other LCB > 0. Pass → lock OP6→TURTLE. Fail → close OP6 as unproven
TURTLE host; pick another opponent — do **not** reopen the race rabbit hole.

**OP6 TURTLE held-out — RECONFIRM_FAIL (2026-07-28):**
`artifacts/op6_turtle_heldout16_mapa_seed621001` (seed `621001`,
landscape freeze: extract/preengage/race OFF, dual-assault + recovery +
`lane_amplitude_frac=0.35`).

```text
means:  TURTLE +0.875  RUSH +0.750  SPLIT 0.000  ESCORT −0.062
uniquely best: TURTLE (gap vs RUSH only +0.125)
TURTLE−ESCORT CI: clear PASS
TURTLE−SPLIT  CI: clear PASS
TURTLE−RUSH   CI: [-0.50, +0.69] FAIL (not >0)
pooled vs best-other LCB: 0.00 FAIL
VERDICT: RECONFIRM_FAIL
```

**Decision (locked):** OP6 is **UNPROVEN / NOT LOCKED** as the TURTLE
host. Seed `621001` retired. Do **not** resume extraction / race-denial
engineering. Next TURTLE work = select another OP6–OP12 opponent from
the landscape board (not OP8 frozen RUSH). Extraction/race branch remains
closed.

Do not buff BLUE_TURTLE / change OP9.

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

**Direction (2026-07-28, locked — HARD RESEARCH FORK):** Stop infinite
OP6–OP12 retunes on the same two maps. Arena geometry currently shapes
strategy more than opponent identity. Path:

```text
1. Finish the current multi-map landscape ONCE (no restart, no second scan).
2. If existing maps robustly yield only map_a→RUSH and map_b→SPLIT:
   STOP opponent engineering on those maps.
3. Prove LRO with K=2 complementary specialists first (professor-approved).
4. Add at most ONE carefully designed map that supplies TURTLE + ESCORT
   affordances; use different OPs on that same map for the two jobs.
5. Only then birth K=4 branches and train the router last.
```

Honest scientific fallback if the new-map budget is exhausted:

```text
K=2 LRO: demonstrated
K=4 extension: not yet demonstrated
```

That is stronger than manufacturing four fragile latents via special-case BT.

**Anti-loop budget (LOCKED — do not exceed):**

```text
Current multi-map scan:            one final broad scan (in flight)
New map designs:                   maximum 2 versions (Map C = version 2 in use)
Opponent redesigns / missing niche: maximum 2 rounds
No full matrix before micro-gates pass
No held-out before untouched validation passes
No further OP6 extraction / race-branch engineering
```

**Map C `map_c_home_corridor` (new-map budget, 2026-07-28/29):**

Version 1 wall (`y∈[0.28,0.70]`) left two open bypasses on a 20×20 field
(above ~y=5.3 and below ~y=13.3, each ~28%+ of height). Live traffic went
around, not through — TURTLE chase failures and ESCORT route confusion both
stemmed from the wall not forcing a single choke. Version 1 payoff screen
(`artifacts/mapc_dev1_op6_op11_4seed`) had RUSH/SPLIT best; TURTLE/ESCORT
worst — consistent with missing affordance.

**Version 2 geometry — FROZEN (no third wall version):**

```text
wall: x∈[0.15,0.24], y∈[0.18,1.0]  (flush to bottom edge)
gaps: exactly ONE near the top (~3.4 cells); bottom sealed
tests: tests/test_map_c_home_corridor.py
```

Contract: single mandatory top gap; bottom sealed; gap wide enough for
shared corner routing. Do not shrink below ~0.15 norm.

**Next — two separate micro-gate suites on the same frozen Map C V2
(no four-style payoff matrix until causal effects pass):**

```text
TURTLE → OP6 | map_c   diagnose_mapc_turtle_microgates.py
ESCORT → OP11| map_c   diagnose_mapc_escort_microgates.py
```

TURTLE gates: red-first frequent vs RUSH/SPLIT/ESCORT; rare vs TURTLE;
stop-at-gap + counter-after-stop frequent; 0 bottom bypasses.

ESCORT gates: supported return success clearly exceeds RUSH/SPLIT
unsupported; ESCORT closer + interposes at gap; RUSH brief screen must
not get the same advantage.

Target board:

```text
RUSH   → existing map_a context
SPLIT  → existing map_b_split_lane context
TURTLE → OPx | map_c
ESCORT → OPy | map_c
```

**Map C V2 micro-gate first pass (2026-07-28, wall FROZEN — do not retune):**

Geometry verify **PASS** on both suites: 0 bottom bypasses, 0 router-stall
sums, aggressive styles use the top gap 8/8.

TURTLE / OP6 (`artifacts/mapc_v2_turtle_microgates_op6_8seed`, seed 801001):

```text
RUSH   red-first 3/8  FAIL (≥5)   margin +0.25
SPLIT  red-first 8/8  PASS
ESCORT red-first 8/8  PASS
TURTLE red-first 7/8  FAIL (≤2)   ← red still converts first
TURTLE stop@gap  7/8  PASS
TURTLE counter   5/8  PASS
gates_pass=False
```

Stop/counter fire, but the home-anchor does not prevent red first-score.
This is **not** a map-shape problem — it would require redesigning OP6’s
gap assault / recovery (paused).

ESCORT / OP11 (`artifacts/mapc_v2_escort_microgates_op11_8seed`, seed 811001):

```text
RUSH   return 8/8  prot_dist≈4.1  interpose@gap≈3.5
SPLIT  return 8/8  prot_dist≈12.9 interpose@gap≈0.1
ESCORT return 0/8  prot_dist≈1.7  interpose@gap≈0.4  top_gap_use=0/8
gates_pass=False
```

Formation exists (protector ≈1.7) but ESCORT **never reaches the gap**, so
the map cannot test whether protection helps during extraction. RUSH/SPLIT
succeed via better routing, not strategic unsupported superiority. Would
require blue ESCORT routing (and possibly another OP) — paused.

**Map C CLOSED FOR CURRENT BUDGET (2026-07-28, locked):**

```text
Map C V2 geometry:          PASS
TURTLE causal affordance:   FAIL
ESCORT causal affordance:   FAIL
K=4 environment extension:  CLOSED FOR CURRENT BUDGET
Future work (not now):
  - gap-assault dynamics
  - coordination-aware ESCORT routing
```

Two-version map budget exhausted. Do **not** start OP6 gap-assault tuning,
ESCORT routing changes, a third wall, or any Map C four-style matrix.
Stop Map C and opponent engineering for the K=4 extension.

**Pivot — K=2 LRO proof (professor-approved minimum):**

```text
C_RUSH  = OP6_IMMEDIATE_DUAL_RUSH | map_a     FROZEN CONFIRM_PASS
C_SPLIT = OP9_SPLIT_LANE_FEINT    | map_b_split_lane  FROZEN CONFIRM_PASS
```

Sequence (no skipping):

```text
1. Confirm both contexts on fresh 16-seed blocks → freeze   DONE
2. Train independent RUSH and SPLIT PPO specialists          NEXT
3. Cross-evaluate; each must own its context
4. Require LCB95(Δ_pool) > 0
5. Birth two LRO latent branches
6. Verify forced-latent crossover + policy distinction
7. Train the router last
```

Two genuine learned latent strategies beat an unfinished K=4 extension
swallowing the first demonstrable Summer result.

**K=2 LRO proof (next after landscape finishes) — required before router:**

Plausible anchors (exact OP may shift from landscape; structure fixed):

```text
RUSH  → an OP|map_a context   (default candidate: OP8|map_a)
SPLIT → OP9|map_b_split_lane  (historical held-out anchor)
```

Train two independent PPO specialists. Accept only if:

```text
RUSH policy best on RUSH context
SPLIT policy best on SPLIT context
LCB95(delta_pool) > 0
matched-observation policy distinction
different trajectory fingerprints
```

Then birth them into two latent branches. Do **not** hold the project
hostage to K=4 before this passes.

**K=4 extension (after K=2 proof) — one map, two missing jobs:**

Design one new map with both affordances (not one map per strategy):

```text
TURTLE affordance:
  abandon home → reliable red score route
  keep one blue anchor → stops it
  successful defense → counterattack opening

ESCORT affordance:
  return path has a narrow interception corridor
  unsupported carrier gets caught
  nearby protector blocks / redirects interceptor
```

Target structure (OPs may change; two jobs per map must hold):

```text
RUSH   → OP8|map_a
SPLIT  → OP9|map_b_split_lane
TURTLE → OP6|map_c_home_corridor   (V2 single-choke; micro-gates pending)
ESCORT → OP11|map_c_home_corridor  (same map, different OP job)
```

Same new map must require two different blue responses depending on the
opponent — so the router cannot memorize `map→one style`. Map C V2 seals the
bottom bypass so traffic has one mandatory passage.

**Honest board (anchors + open jobs):**

```text
RUSH   → many gap≥0.5 cells; OP8|map_a is ESCORT-thin (+0.25) in this scan
         (frozen RUSH held-out evidence remains; resolve at selection)
SPLIT  → OP9 | map_b_split_lane   (gap +1.12; K=2 anchor confirmed)
TURTLE → no gap≥0.5 on existing maps (thin: OP12|map_a / OP12|map_b +0.12)
ESCORT → no gap≥0.5 on existing maps (thin: OP8|map_a +0.25)
```

**Paused:** OP6/OP7/OP11 opponent redesign on map_a / map_b for niche
manufacture. `BLUE_PROBES_V3` frozen. Latent / router training remains
stopped until K=2 specialist crossover passes.

**OP8 freeze:** no redesign and no additional held-out reruns of
`op8_rush_heldout16_v3_map_a_seed582001`. Multi-map 8-seed scan conflict
(ESCORT-thin on OP8|map_a) is discovery-only — do not auto-retune OP8.

**Map naming:** `map_a` ≡ `map_a_open`. `map_b_split_lane` is a usable
K=2 SPLIT anchor under this fork (not “historical-only trash”).

### Multi-map landscape scan — COMPLETED

Artifact: `artifacts/multimap_v3_landscape_op6_op12_8seed`
Protocol: OP6–OP12 × {map_a, map_b_split_lane, map_b_split_lane_v2} ×
{RUSH,SPLIT,TURTLE,ESCORT} × 8 paired seeds = **672 episodes**,
`BLUE_PROBES_V3`, base-seed 620001. Resumed once after mid-run kill
(~141/672). Summary: `context_summary.json`.

**Post-scan verdict (LOCKED rule applied):**

```text
Existing maps robustly produce:
  map_a / map_b* → RUSH-dominated cells (many gap≥0.5)
  map_b_split_lane* → SPLIT on OP9 only (gap +1.12)
  TURTLE / ESCORT → no gap≥0.5 tooth on any OP6–OP12 × map cell
→ STOP opponent engineering on map_a / map_b for missing niches
→ K=2 anchors usable: RUSH (pick among gap≥0.5) + SPLIT OP9|map_b
→ TURTLE+ESCORT → Map C path (≤2 versions; V2 frozen)
```

Candidates gap≥0.5: RUSH many (clearest OP10|map_b_v2 +1.38); SPLIT
OP9|map_b and OP9|map_b_v2 (+1.12 each); TURTLE none; ESCORT none.
OP6|map_a is RUSH-best (+0.75), not TURTLE. OP11|map_b is RUSH-best
(+0.88), not ESCORT.

### Full map_a V3 landscape (OP6–OP12) — COMPLETED (superseded as sole board)

Artifact: `artifacts/map_a_v3_landscape_op6_op12_8seed` (8 paired seeds,
base-seed 590001, `BLUE_PROBES_V3`, `map_a` on every row/manifest, no
experimental response flags). Pool gates: **not admissible** (expected for
a discovery scan). `delta_pool` LCB ≤ 0; SPLIT unprotected. Superseded by
the multi-map context scan above for niche selection.

Mean win-margin uniquely-best blue (margin vs 2nd):

| Red | Best | Mean | vs 2nd |
|-----|------|------|--------|
| OP6 | TURTLE | +0.12 | +0.88 |
| OP7 | RUSH | +2.25 | +0.62 |
| OP8 | RUSH | +2.38 | +0.12 |
| OP9 | ESCORT | +1.75 | +0.25 |
| OP10 | RUSH | +2.25 | +1.12 |
| OP11 | RUSH | +3.00 | +0.50 |
| OP12 | ESCORT | +1.88 | +0.12 |

Firm readings (not a locked four-niche board):

* **RUSH** clearly exists across several opponents; OP8 remains the frozen
  candidate (not the clearest landscape margin — do not retune).
* **SPLIT** definitely missing — never uniquely best.
* **TURTLE** promising only on OP6.
* **ESCORT** weak candidates on OP9/OP12 — hypotheses, not recovered niches.
* **OP10 is no longer a SPLIT candidate** — it is the strongest RUSH column
  (`RUSH − SPLIT = +1.125`).

### SPLIT host selection (locked step 1–2)

`split_deficit = best_style_margin − SPLIT_margin` from the same artifact.
Excluded: OP8 (frozen RUSH), OP6 (current TURTLE candidate).

| Rank | Red | Best | Best mean | SPLIT mean | **split_deficit** | Eligible |
|------|-----|------|-----------|------------|-------------------|----------|
| — | OP7 | RUSH | +2.250 | +1.625 | **+0.625** | yes |
| — | OP8 | RUSH | +2.375 | +1.625 | +0.750 | no (frozen RUSH) |
| — | OP11 | RUSH | +3.000 | +2.125 | +0.875 | yes |
| — | OP9 | ESCORT | +1.750 | +0.750 | +1.000 | yes |
| — | OP10 | RUSH | +2.250 | +1.125 | +1.125 | yes (expired SPLIT pick) |
| — | OP6 | TURTLE | +0.125 | −1.000 | +1.125 | no (TURTLE candidate) |
| — | OP12 | ESCORT | +1.875 | −0.250 | +2.125 | yes |

**Selected SPLIT host: `OP7_DEEP_FORTRESS` → CLOSED (failed host)**

Locked execution order:

```text
1. Rank SPLIT deficits          ← DONE
2. Select SPLIT host = OP7      ← DONE
3. Diagnose why SPLIT loses on OP7 ← DONE
4. Add one opponent-specific causal lever ← DONE (2 redesign rounds)
5. 8-seed development           ← DONE — FAIL (tie with RUSH)
6. Explicit final compact lever ← DONE — FAIL (hard stop)
7. Close OP7 as SPLIT host      ← DONE
8. Move SPLIT work to OP11      ← NEXT
9. Validate OP6 → TURTLE
10. Harden OP9 or OP12 → ESCORT
11. Reconfirm final RUSH host
12. Four-host pool gate
```

### OP7 SPLIT lever (separated-threat overcommit) — round 2 COMPLETE

**Contract:** both blues offensively committed + wide separation + opposite
corridors + persistence → OP7 locks both reds onto the first breached
corridor for a bounded window. Legal geometry only; no `BLUE_SPLIT` ID.
OP6/OP8 paths untouched. Implementation: `_bt_update_op7_split_latch` /
`_bt_apply_op7_split_*` in `gpu_env/_core/_bt_red.py`. Tests:
`tests/test_op7_split_overcommit.py`. Diagnostic:
`experiments/diagnose_op7_split_latch.py`.

**Round 1** (`op7_split_latch_microgates_dev8`): selective but weak —
SPLIT latch 4/8; RUSH/ESCORT/TURTLE 0/8. Tactical effect present when armed.

**Round 2** (softer persistence / lateral / teammate / commit-grace):
`artifacts/op7_split_latch_microgates_r2_dev8` — **MICROGATES PASS**

```text
SPLIT latch 8/8; RUSH/ESCORT/TURTLE false latch 0/8
SPLIT TTFS ~40 (was ~130); uncovered-after-latch > 0; same-corridor > 0
RUSH/ESCORT/TURTLE margins unchanged vs round-1 diagnostic
```

**Development matrix** (`artifacts/op7_split_dev8_r2_map_a`, 8 paired seeds,
base-seed 611001, map_a, BLUE_PROBES_V3): **FAIL — not uniquely best**

```text
RUSH   +1.75  TTFS ~75
SPLIT  +1.75  TTFS ~38   (tied; SPLIT−RUSH = 0, need ≈+0.5)
ESCORT +0.75
TURTLE +0.125
```

SPLIT conversion sped up as intended, but RUSH remains co-best.

### OP7 final compact-containment lever — HARD STOP FAIL

Authorized one-shot exception after the 2-round budget. **Did not retune
the SPLIT latch.** Added compact same-corridor detector + deep-defender /
return-interceptor response (`_bt_update_op7_compact_latch` /
`_bt_apply_op7_compact_*`).

`artifacts/op7_compact_microgates_final_dev8` (base-seed 612001):

```text
SPLIT latch:   6/8  PASS
compact RUSH:  8/8  (wanted ≥6) — but also:
compact SPLIT: 8/8  FAIL (want ≤1)
compact TURTLE:8/8  FAIL (want ≤2)
compact ESCORT:8/8
RUSH margin:   +2.875  (worse than prior tie +1.75)
ESCORT margin: +3.000  (worse)
```

Compact geometry was not selective (early-episode closeness fires for all
styles) and the interceptor peel **raised** concentrated-style blue
payoffs instead of containing them. No development matrix run.

**Hard stop executed:**
* `_OP7_COMPACT_LEVER_ENABLED = False` (code retained for audit)
* SPLIT latch remains enabled (near-miss / robustness)
* **OP7 closed as SPLIT host**
* Next SPLIT host: **OP11** (next eligible deficit +0.875)

No fourth lever. No threshold ladder. No held-out retuning.

---

### Contract A — OP8 RUSH host (formation opening)

OP8-only redesign. Early game: red carrier/protector formation is still
deploying; home defense is temporarily incomplete. After a fixed **legal**
trigger (sim-step wall-clock, or blue already carrying — **never** hidden
blue style ID): red gains stable escort, interception, and counter pressure.

```text
RUSH:   shortest route; scores before formation is ready
SPLIT:  lane setup takes too long
ESCORT: coordination is too slow
TURTLE: ignores the opening
```

Implementation (round 2): `_OP8_FORMATION_OPENING_STEPS=18` in
`gpu_env/_core/_bt_adaptive.py`; gate in `_bt_assign_roles` is
`bt_level==8` only; opening routes stage at midfield rally (not dual-rush).
Round-1 (`steps=28`, dual-ATTACKER) failed micro-gates: SPLIT/ESCORT also
picked up before T; mutual race → all margins 0
(`artifacts/op8_formation_microgates_dev8`). Round-2 (`steps=18`, midfield
staging): RUSH/SPLIT/TURTLE pickup timing improved
(`artifacts/op8_formation_microgates_r2_dev4`), but paired 8-seed matrix
`artifacts/op8_rush_dev8_r2` **FAIL** — SPLIT uniquely best (+2.25 WR 8/8),
RUSH 0.00, ESCORT −0.25. Two redesign rounds spent.
Tests: `tests/test_op8_formation_opening.py`. Micro-gates:
`experiments/diagnose_op8_formation_microgates.py`.

**Micro-gates (before full matrix):** RUSH first pickup usually before
activation; SPLIT/ESCORT usually after; TURTLE rarely threatens in opening.

**Promotion:** RUSH uniquely best; `RUSH − best_other ≥ +0.5` mean margin
(practical floor unless protocol says otherwise); paired CI LCB > 0.
Then freeze → fresh 16-seed held-out → lock only when all paired CIs clear.
Max **two** bounded redesign rounds.

**Pivot (2026-07-28):** `BLUE_PROBES_V3` repairs RUSH. Competence gate PASS.
OP8 under V3 on map_a (`op8_rush_dev8_v3_map_a`): **DEVELOPMENT PASS /
NOT LOCKED** — RUSH +2.75 uniquely best, `RUSH−ESCORT=+0.75`, paired CIs
clear; pooled LCB=0. **OP8 frozen** (no redesign). Next: fresh 16-seed
held-out on map_a / V3 / unchanged OP8. OP9 on map_a/V3: RECONFIRM_FAIL
(ESCORT best); map_b OP9 lock is historical only.

### Contract B — OP6 TURTLE (dual-assault)

Keep OP6’s identity (immediate dual red rush). Do **not** make OP6 “generally
stronger” or add generic anti-SPLIT detection. Punish blue teams that send
**both** agents away from home:

* two independent red attack lanes;
* stable offensive assignments (not reactive role churn);
* rapid resume after tags/resets;
* do not collapse both reds onto one nearby blue;
* minimal defense so the match stays offense-vs-defense.

```text
RUSH:   abandons home → loses the race
SPLIT:  two threats but home insufficiently defended vs dual assault
ESCORT: concentrates on one carrier → cannot cover both red lanes
TURTLE: anchor + patrol deny both; counter scores
```

**Micro-gates:** time both reds enter blue territory; % time two independent
red threats active; red first-score rate when both blues leave home vs when
one anchor remains; role/lane persistence.

**Promotion:** TURTLE uniquely best; critical **TURTLE > SPLIT**; paired CIs
clear. Max **two** bounded redesign rounds.

### Contract C — OP11 ESCORT (isolation pressure)

Punish **separated** blue attackers, not “more aggression.” Observable
geometry only:

```text
pair separated beyond threshold + both committed
  → stable red pressure on each isolated blue
carrier with close same-corridor support
  → normal defender/interceptor structure
```

Do not directly punish the protector — the protector must have real value.

```text
SPLIT:  each isolated attacker contained
RUSH:   fast but weak post-pickup protection
TURTLE: not enough offense
ESCORT: protector disrupts interceptor; carrier returns
```

**Micro-gates:** higher tag rate on isolated blues; lower return success for
separated carriers; higher return success for supported carriers; stable red
assignments in isolation mode.

**Promotion:** decisive **ESCORT > SPLIT** (also > RUSH, > TURTLE); paired CIs
clear. Instrumentation / smaller gaps without unique-best do **not** count.
Max **two** bounded redesign rounds. No shared behavior with OP6 / RUSH host.

---

**Per-host execution loop:**

```text
1. Define one causal contract
2. Micro-tests proving the mechanism
3. Paired 8-seed development matrix
4. At most two bounded redesign rounds
5. If uniquely best → freeze
6. Fresh paired 16-seed held-out
7. Lock only when all paired CIs clear zero
```

**Recommended execution order:**

```text
A. OP8 RUSH formation-opening redesign
B. OP6 TURTLE dual-assault redesign
C. OP11 ESCORT isolation-pressure redesign
```

OP11 may continue in a separate branch, but **no shared changes** across hosts.

**After all four niches locked:** full paired matrix → `LCB95(delta_pool)>0`
+ `all_blues_protected`; niche-balanced sampling 25% each; then independent
PPO specialists → learned oracle; then K=4 LRO / forced-z / router.

**Immediate focus:** Both K=2 contexts **FROZEN**.
`C_RUSH=OP6|map_a` (CONFIRM_PASS, gap +0.25) and
`C_SPLIT=OP9|map_b_split_lane` (CONFIRM_PASS, gap +1.25).
Freeze manifests in each artifact dir. Next: matched-budget independent
PPO specialists (`no_latent_baseline`, FIXED_OPPONENT, 2v2). Learned-policy
gates remain strict despite thin C_RUSH scripted gap. Router stopped.
No Map C / OP retune.

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
style: ESCORT. Progress requires payoff inversion — ESCORT uniquely best with
paired CI clear vs SPLIT (critical), RUSH, and TURTLE. Instrumentation or
margin-gap reductions without that inversion do not count. If SPLIT remains
best, do not accept OP11 as another SPLIT matchup; tune OP11 to punish split
play and expose an escort-compatible weakness before held-out confirmation.
Do not change shared paths that would disturb frozen OP9.

**OP11 ESCORT held-out confirmation (2026-07-28) — FAIL:**
`artifacts/op11_escort_heldout16_seed561001`, 16 disjoint paired seeds
(base-seed 561001; no overlap with dev `541001` or the earlier premature
held-out `551001`), frozen lane-containment OP11 + restored ESCORT
evade-follow. Device: cpu (CUDA allocator unstable that session).

Dev10 CPU screen had looked like an ESCORT crossover
(`artifacts/op11_dev10_lane_containment_8seed_cpu`): ESCORT +0.625 uniquely
best over SPLIT +0.500 / RUSH 0 / TURTLE 0. Held-out did **not** confirm:

```text
BLUE_ESCORT  WR=11/16  mean_margin=+0.8750
BLUE_SPLIT   WR=12/16  mean_margin=+1.5000
BLUE_RUSH    WR= 8/16  mean_margin=+0.3750
BLUE_TURTLE  WR= 2/16  mean_margin=+0.0000
```

Paired ESCORT margin advantage on the 16 matched held-out seeds:

```text
ESCORT - SPLIT   mean -0.6250, bootstrap 95% CI [ -1.4375, +0.0641 ]
ESCORT - RUSH    mean +0.5000, bootstrap 95% CI [ -0.0625, +1.1250 ]
ESCORT - TURTLE  mean +0.8750, bootstrap 95% CI [ +0.3125, +1.4375 ]
```

Detector selectivity on held-out: SPLIT trigger 15/16, ESCORT 0/16, RUSH
7/16 (false latch still too high), TURTLE 2/16. Best response remains
BLUE_SPLIT; ESCORT is second, not unique. Only ESCORT-vs-TURTLE clears a
positive paired CI.

Decision: OP11 is **NOT ACCEPTED** as an ESCORT niche. Dev crossover was
seed-local. Do not freeze. Next OP11 work must further punish SPLIT (and
reduce RUSH false latch) until ESCORT is uniquely best on a *new* held-out
seed set; do not reuse 561001 for tuning.

**Map confound (2026-07-28):** All OP11 ESCORT screens above
(dev1–dev10, held-out `561001`) used `--maps map_b_split_lane`. That layout
is a corridor / dual-lane geometry that structurally favors `BLUE_SPLIT`.
Under the locked four-niche map contract (`map = map_a` for every niche
host; see §6.0), those rows are historical only and do **not** count toward
ESCORT acceptance. OP11 rescue continues on `map_a` only.

**OP11 salvage interpretation (2026-07-28) — locked:**
Held-out `561001` is a salvageable FAIL (ESCORT +0.875 vs SPLIT +1.500;
paired ESCORT−SPLIT CI upper edge +0.06), not a collapse. Detector /
instrumentation remains useful; Dev10 crossover was development-only;
`561001` is permanently retired and must never be reused for tuning.
OP11→ESCORT remains unproven.

**OP11 rescue plan (map_a, Phase-1 first):**
1. Diagnose on fresh Dev-A seeds (`571001`, `map_a`) via
   `experiments/diagnose_op11_isolation_failure.py` — rank failure modes
   A–E before any redesign.
2. Build OP11-only stable isolation state (two-threat coverage, locked
   roles) only after Phase-1 names the dominant mode; preserve ESCORT
   (no dual collapse onto convoy / no protector chase).
3. Micro-gates (selectivity + OFF/ON) before another payoff matrix.
4. Evidence ladder: Dev-A (8) → freeze → Dev-B (8 new) → held-out 16
   (new seed; never `561001`). Max two redesign rounds. OP9 path
   untouched; BLUE_PROBES_V3 frozen.

**OP11 map_a baseline (2026-07-28) — locked interpretation:**
`artifacts/op11_dev11_mapa_baseline_8seed`, OP11/map_a, 8 paired seeds
(base-seed 541001, pre-rescue sanity block). `map_a` removed the fake
“SPLIT wins everything” pattern from `map_b_split_lane`. OP11 is **not** an
ESCORT niche yet for a different reason: **saturation** — too easy for every
aggressive style.

```text
BLUE_ESCORT  +2.875  WR 8/8
BLUE_RUSH    +2.875  WR 8/8
BLUE_SPLIT   +2.750  WR 8/8
BLUE_TURTLE  +0.500  WR 4/8
```

Single-red matrix: `delta_pool=0` is automatic (one column). Meaningful
evidence is paired gaps on matched seeds:

```text
ESCORT − RUSH    = 0.000
ESCORT − SPLIT   = +0.125
ESCORT − TURTLE  = +2.375
```

First two gaps are nowhere near niche lock. ESCORT is not losing to SPLIT;
it is tied with RUSH and almost tied with SPLIT. Do **not** globally
strengthen OP11 (would lower ESCORT with everything else). Target:
distinguish **supported offense** (ESCORT) from **unsupported offense**
(RUSH) and **separated offense** (SPLIT):

```text
SPLIT  → isolated attackers punished
RUSH   → unsupported carrier return punished
ESCORT → protector disrupts interception; preserve conversion edge
TURTLE → already weak; no special work
```

Next micro-gates (Phase-1 diagnostic + future OFF/ON ablation):

```text
SPLIT separation/isolation trigger:  frequent (target ≥6/8)
RUSH unsupported-carrier trigger:    frequent (target ≥6/8)
ESCORT false isolation trigger:      rare (target ≤2/8)
TURTLE triggers:                     rare / irrelevant
```

Then OP11-only stable response: cover both SPLIT threats without role
churn; intercept unsupported RUSH carrier aggressively; do not chase or
directly punish ESCORT protector.

Phase-1 diagnostic running: `artifacts/op11_phase1_isolation_diag_mapa_seed571001`
(fresh `571001`, map_a). No redesign until failure modes ranked.

**OP11 Phase-1 diagnostic complete (2026-07-28) — locked:**

```text
SPLIT detector:       PASS (7/8 latch; 0/24 false latches on ESCORT/RUSH/TURTLE)
Current response:     FAIL (~68% latch steps chasing “protector” target — mode D)
Payoff on 571001:     inconclusive (all styles margin 0; no crossover)
```

Geometry recognition works (SPLIT teammate dist ~11.1 vs ESCORT ~2.1). Once
latched, OP11 overcommits to one blue agent instead of controlling two
separated threats.

**OP11 → ESCORT: NOT ACCEPTED (2026-07-28, post map_a matrix) — locked:**

On `map_b_split_lane` the problem was `SPLIT > ESCORT`. On `map_a` it is
`ESCORT = RUSH` (both ≈ +3.0, gap 0). PPO would have no reason to learn an
ESCORT branch. Critical OP11 comparison is now ESCORT vs RUSH, not SPLIT —
but OP11 is **parked** as ESCORT host; OP9 is the natural ESCORT tooth.

```text
OP11 → ESCORT: NOT ACCEPTED
Reason: ESCORT and RUSH saturated tie on map_a
Recommended action: park; potentially repurpose for SPLIT later
```

Keep / freeze (do not delete):

```text
SPLIT geometry detector:   KEEP as valid telemetry (thresholds not retuned)
Protector-chase response:  REJECTED / disabled
OP11 map_a baseline:       FROZEN saturated evidence (dev11 + 581001 column)
Unsupported-carrier work:  DEFERRED (RUSH support-near ~96%; proximity fails)
```

Do **not** force OP11 to remain the ESCORT host. After OP8 / OP9 / OP6
resolve, consider OP11 alongside OP7 / OP10 / OP12 for the missing SPLIT
niche (separated two-lane pressure).

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

**Canonical map (LOCKED 2026-07-28):** `map = map_a` for every niche
experiment — OP6 TURTLE, OP8 RUSH, OP9 ESCORT confirmation, SPLIT-host
search, full payoff matrices, development / validation / held-out runs.
Record `map_a` in every CSV row and manifest. Do not mix other maps into
main niche acceptance evidence; other layouts are robustness-only and
reported separately. Protocol owner:
[`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md) §1.1.
Collector default: `experiments/run_scripted_style_payoff_matrix.py`
`DEFAULT_MAPS = ("map_a",)`.

**Working board on `map_a` (LOCKED 2026-07-28):**

```text
RUSH   → OP8: frozen strong candidate, not formally locked
ESCORT → OP9: mean-leading reserve, held-out FAILED (UNPROVEN / NOT LOCKED)
TURTLE → OP6: UNPROVEN / NOT LOCKED (held-out RECONFIRM_FAIL seed 621001; race branch CLOSED)
SPLIT  → host search later
OP11   → parked
```

Intended four jobs once hosts clear:

```text
OP8 → fast concentrated attack (RUSH)
OP9 → supported carrier attack (ESCORT) — unproven until CI lock
OP6 → defense and counterattack (TURTLE)
OP11/OP7/OP10/OP12 → separated two-lane pressure (SPLIT) — TBD
```

Prior `map_b_split_lane` locks (e.g. OP9→SPLIT) are **historical only** and
do not transfer. OP11→ESCORT is **NOT ACCEPTED** on map_a.

**Immediate sequence (LOCKED 2026-07-28):**

1. **OP9 → ESCORT** 16-seed held-out — **DONE / RECONFIRM_FAIL**
   (`artifacts/op9_escort_heldout16_mapa_seed591001`). Official status:

```text
OP9 → ESCORT: UNPROVEN / NOT LOCKED
Seed block 591001: retired held-out evidence
OP9 behavior: unchanged
Further tuning using 591001: prohibited
```

ESCORT highest mean is encouraging but not a protected niche
(ESCORT−RUSH / ESCORT−SPLIT / pooled LCB all fail to clear 0). OP9 is not
dead; it has not earned the ESCORT tooth yet. Return after OP6.

2. **OP6 → TURTLE** — **DONE / RECONFIRM_FAIL**
   (`artifacts/op6_turtle_heldout16_mapa_seed621001`). Official status:

```text
OP6 → TURTLE: UNPROVEN / NOT LOCKED
Seed block 621001: retired held-out evidence
Extraction / preengage / race-denial: CLOSED (defaults OFF)
Further race-branch engineering: prohibited
```

TURTLE uniquely best by mean (+0.875) but TURTLE−RUSH CI and pooled
best-other LCB do not clear 0. Do **not** reopen the scoring-race rabbit
hole. Next TURTLE work = another OP from the landscape / multi-map scan.
3. **OP8** leave frozen — no redesign; revisit strict pooled confirm later
   under a predeclared protocol after other niches.
4. **SPLIT host** search last among OP7/OP10/OP11/OP12 from map_a evidence;
   one bounded opponent-local mechanism after picking the closest candidate.

Active milestone: TURTLE host search after OP6 close-out (multi-map / other OP).

**Running (2026-07-28):** `artifacts/scripted_style_payoff_matrix_mapa_baseline_8seed`,
OP6–OP12 × four blues × map_a, 8 paired seeds (`581001`), no experimental
flags. Per-red report via `experiments/analyze_mapa_payoff_baseline.py`.

**Canonical map_a baseline complete (2026-07-28):**
`artifacts/scripted_style_payoff_matrix_mapa_baseline_8seed`, 224 episodes
(7 reds × 4 blues × 8 seeds), `581001`, map_a. Per-red summary:
`mapa_per_red_summary.txt`.

Uniquely best blue by red (paired gap vs runner-up):

```text
OP6  → RUSH    (+0.250 vs ESCORT)   margins RUSH +3.00 / ESCORT +2.75
OP7  → RUSH    (+0.750 vs ESCORT)
OP8  → RUSH    (+0.625 vs ESCORT)   RUSH-host still not locked
OP9  → ESCORT  (+0.750 vs RUSH)     SPLIT +0.875 — SPLIT niche NOT on map_a
OP10 → RUSH    (+1.000 vs ESCORT)
OP11 → ESCORT  (+0.000 vs RUSH)     saturated tie at +3.00
OP12 → RUSH    (+0.500 vs ESCORT)
```

Matrix verdict (scenario **D** leaning): 2/7 distinct best responses
(RUSH×5, ESCORT×2). **SPLIT and TURTLE never uniquely best.** Pool
`delta_pool=+0.039`, LCB **−0.179** — not admissible. No global blue
dominator; most columns still too easy (high WR, small aggressive spreads).

Implications (locked):
- OP9 is the healthier natural **ESCORT** tooth (+0.75); confirm on fresh
  16-seed held-out before lock. Do not treat OP9 as SPLIT on map_a.
- OP11→ESCORT **NOT ACCEPTED** (saturated ESCORT=RUSH); park OP11; later
  consider for SPLIT with OP7/OP10/OP12.
- OP8 remains RUSH candidate; OP6 remains TURTLE engineering.
- SPLIT host still open — no column uniquely rewards SPLIT on this matrix.

**Execution order (map_a):** A OP8 RUSH freeze/held-out → B OP9 ESCORT
16-seed held-out confirm → C OP6 TURTLE engineering → D SPLIT-host search
among OP11/OP7/OP10/OP12. Floors: uniquely best; practical gap ≈ +0.5 vs
runner-up; critical paired CIs clear for niche claim.

**Current blocker evidence:** OP6–OP10 joint acceptance
(`artifacts/op6_op10_br_diversity_acceptance_16seed`) FAIL — SPLIT uniquely
best on all five; `LCB(delta_pool) ≤ 0`. OP9 alone is LOCKED under
BLUE_PROBES_V2 (`artifacts/op9_split_heldout16_blue_probes_v2_seed521001`,
`RECONFIRM_PASS`) but that is one corner, not pool crossover. OP6 TURTLE
revisit (2026-07-28): baseline SPLIT +0.50 / TURTLE −1.00; best partial is
single carrier-deny (`op6_dev19_single_carrier_deny_8seed`) SPLIT +0.625 /
TURTLE −0.500 — still flipped. Dual-rush abandon responses REJECTED.

**Now:** OP8 = STRONG FROZEN CANDIDATE (not locked; pooled LCB≤0). Running
unchanged OP6–OP12 × four-style V3 landscape on map_a
(`artifacts/map_a_v3_landscape_op6_op12_8seed`) to nominate SPLIT / ESCORT /
TURTLE hosts and check whether OP8 remains the clearest RUSH host in-pool.

**Exact question for the four-column pool:**

> Do the four niche hosts force blue to need different strategies, or can one
> strategy beat almost all of them?

Judge by blue best-response diversity + `LCB(delta_pool)>0`, not red BT
fingerprints. Sample future training 25% per niche.

* Module: `experiments/payoff_matrix_analysis.py`
* Collector: `experiments/run_scripted_style_payoff_matrix.py`
* Niche CI helper: `experiments/analyze_niche_heldout_reconfirm.py`
  (invoke with `AICTFProject\.venv\Scripts\python.exe`, not bare `python`)
* Blue styles: `gpu_env/_core/_scripted_blue_styles.py`
* Gates: `all_blues_protected` + `delta_pool_lcb_positive` (+ support gates);
  `--min-br-diversity 4` for the four-style claim.
* Pinned by `tests/test_payoff_matrix_analysis.py`.
* Do not retrain latents until four niches + steps 5–7 of the Direction lock clear.

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

## OP6 unmodified development screen -- RUSH-candidate check (2026-07-28)

Structural read of the BT profiles (before any run) flagged OP6 as the
strongest RUSH candidate among OP6-OP10: lowest `intercept_block_base`
(0.40), lowest `intercept_feasibility_ratio` (0.45), smallest
`threat_radius` (2.0), and long, slow-to-adapt locks (`lock_intercept=18`,
`lock_defender=16`) -- weak/late initial reaction, opposite of OP9's
near-instant 3-4 step reassignment that makes it such a tight SPLIT niche.
Per explicit instruction, this was **not** acted on directly -- the
unmodified OP6 8-seed development screen was run first, same protocol
shape as the locked OP9 confirmation (`map_b_split_lane`, `BLUE_PROBES_V2`,
paired seeds, `max_decision_steps=240`): `artifacts/op6_dev1_8seed`,
base-seed 561001, `OP6_IMMEDIATE_DUAL_RUSH`, no OP6 tuning.

```text
             mean margin   win rate
BLUE_ESCORT     -2.250       0/8
BLUE_RUSH       -2.000       0/8
BLUE_SPLIT      +1.625       8/8
BLUE_TURTLE     -1.250       1/8
```

Paired per-seed (RUSH margin vs the others, same 8 seeds 561001-561008):

```text
ep  seed     RUSH  TURTLE  SPLIT   R-T   R-S
0   561001    -2     -1      1     -1    -3
1   561002    -1     -3      2      2    -3
2   561003    -2     -3      2      1    -4
3   561004    -2     -1      2     -1    -4
4   561005    -2      1      2     -3    -4
5   561006    -3     -1      2     -2    -5
6   561007    -2     -1      1     -1    -3
7   561008    -2     -1      1     -1    -3

mean(RUSH-TURTLE) = -0.750   (RUSH is worse than TURTLE, not better)
mean(RUSH-SPLIT)  = -3.625   (RUSH is far worse than SPLIT)
```

No exact ties for any style (0/8 each). `time_to_first_score` for SPLIT is
consistently fast (38-56 steps, every episode); RUSH's is inconsistent and
often later (39-145 steps) despite RUSH being nominally the "fast" style --
OP6's own immediate dual-rush identity apparently trades evenly or wins
against a mirrored blue rush (two aggressive teams racing head-on, not the
one-sided opening RUSH needs), while SPLIT's two-lane approach exploits
OP6's narrow `lane_amplitude_frac` (0.08) and tiny `threat_radius` (2.0) --
OP6's own agents barely react/adapt to a stretched two-front threat.

**Result: the structural hypothesis was wrong.** SPLIT is not just
competitive, it is a clean sweep (8/8, +1.625, best `time_to_first_score`
every episode); RUSH is statistically tied for OP6's *worst* matchup with
ESCORT, and is on average worse than even TURTLE (which itself only wins
1/8). Per the locked decision tree: **SPLIT is uniquely best -> OP6
supports neither the RUSH niche nor the TURTLE niche under the current
probes.** The tracker's earlier "OP6: provisional TURTLE niche" label is
superseded by this result and should not be relied on going forward -- it
was never run through the full 4-style screen before now.

**Cross-opponent pattern, now six opponents deep.** OP6, OP7, OP8, OP9,
OP10 are all SPLIT-dominant by direct measurement, and OP12 (RUSH-niche
redesign, closed above) also had BLUE_SPLIT as its best response even
after two full redesign rounds. Only OP11 (ESCORT redesign, paused
mid-investigation) is not currently a confirmed SPLIT niche. This is no
longer a per-opponent tuning question -- SPLIT is winning across
essentially the entire OP6-OP12 family regardless of each opponent's
distinct BT identity (fortress, interceptor, escort, feint, dual-rush,
late-converter). Worth flagging explicitly before spending further budget
on a seventh single-opponent redesign attempt for RUSH: either (a) the
shared BT framework has some general property SPLIT reliably exploits
(e.g. `_bt_assign_roles`'s single-nearest-threat framing throughout, the
same shape as the ESCORT-geometry bug found for OP12), or (b) the RUSH
blue-style script itself (`_blue_rush_targets`) is comparatively weak/naive
relative to `_blue_split_targets`, independent of which red opponent it
faces. Recommend investigating that shared-mechanism question before
picking another individual opponent to redesign for RUSH.

## OP7 RUSH-host redesign attempt (2026-07-28) -- three attempts, all rejected, OP7 left unmodified

Per explicit direction to stop searching outside OP6-OP12 and instead pick
a RUSH host from OP7/OP8/OP10, ranked by existing held-out data:

```text
             RUSH margin   best style (SPLIT)   gap = best - RUSH
OP7             -0.2500          +0.9375              1.1875   <- smallest, chosen
OP8              0.0000          +2.4375              2.4375
OP10            -0.8750          +1.6250              2.5000
```

**Diagnosis (event trace, `scratchpad/trace_op7_vs_rush.py`, seeds
461001-461004, OP7's own frozen base_seed):** OP7's DEFENDER commits within
9-10 steps of episode start and both agents sit at
`['DEFENDER','DEFENDER']` for nearly the whole episode, insta-tagging
RUSH's carrier every pickup (e.g. pickup t=15 -> tagged t=21, repeated
3-5x/episode). Root cause is deeper than "camps the flag":
`_bt_route_target`'s DEFENDER branch only uses the zone/orbit position
BEFORE `any_intruder` is first true; the instant an intruder is detected
(which is also what triggers DEFENDER's Priority-5 assignment in
`_bt_assign_roles`), its target becomes the intruder's own current
position directly -- a direct chase, not a camped zone. RUSH's first
action (entering red's territory) trips this immediately.

**Attempt 1:** reused the shared `opening_active` mechanism (OP8/OP12's
existing pattern) to force both OP7 agents to `ROLE_ATTACKER` for the
first 20 steps. Isolated 8-seed dev screen
(`artifacts/op7_rush_opening_dev1_8seed`, base_seed 461001): RUSH -0.25 ->
**-1.00, worse**. TURTLE also worse (-0.75). SPLIT/ESCORT roughly flat.

**Attempt 2:** gated only the Priority-5 DEFENDER condition directly
(`need_def & late_or_ready_op7`) instead of the blanket force+unlock, on
the theory that avoiding the explicit force/relock would avoid an
unintended side effect. Isolated 8-seed screen
(`artifacts/op7_rush_opening_dev2_8seed`): **bit-identical to attempt 1**
across all four styles. A follow-up trace explained why: with no other
priority capable of firing that early (FLAG_RETR/ESCORT/INTERCEPTOR/COUNTER
all require preconditions that can't be true yet), gating DEFENDER alone
is behaviorally identical to force-forced ATTACKER -- there was no real
difference between the two mechanisms to begin with. The trace also showed
the actual failure mode: `ROLE_ATTACKER` for OP7 is not idle, it is
"actively rush blue's flag" -- both attempts accidentally handed OP7 a
genuine early-offense option it normally never takes at all (OP7 is a
pure defense-first fortress with no attacker identity). RUSH did start
scoring for the first time (0/4 -> 2/4 traced episodes had at least one
blue score, vs 0/4 in the unmodified baseline trace), but OP7's own attack
converted just as fast or faster (red scored in 3/4 traced episodes,
twice in one) -- matched-speed mutual aggression favors whichever side
converts faster, the same reason OP6's own "immediate dual rush" identity
hurts blue's RUSH rather than helping it. **Both attempts reverted.**

**Attempt 3 (reasoned through, not executed):** widen DEFENDER's
`defender_zone_frac`/`defender_orbit_radius` during the opening instead of
suppressing the role, so OP7 never gets an offensive alternative. Not run:
per the diagnosis above, the zone/orbit position is only consulted BEFORE
`any_intruder` is first true, and RUSH trips that condition on its very
first action -- DEFENDER never spends any time patrolling the zone once
RUSH exists to chase instead. Widening the zone would not change the
direct-chase behavior that actually stops RUSH. Implementing and running
this would have been testing a change already shown not to touch the
relevant mechanism.

**Decision: OP7 is left UNMODIFIED**, restored to its original frozen
SPLIT-niche state (verified via syntax check + grep that no OP7-specific
logic remains in `_bt_red.py`, only explanatory comments). OP7's
"fortress" identity -- chase any detected intruder directly the instant
it's detected, with no attacker alternative at all -- is fundamentally
harder to carve a real RUSH opening out of than OP12's identity was,
because every mechanism tried either (a) hands OP7 a symmetric offensive
option that wins the resulting mutual-aggression race, or (b) doesn't
actually touch the direct-chase behavior at all.

**Running tally: RUSH has now failed to become uniquely best against
THREE separate hosts** (OP6, unmodified; OP12, two full redesign rounds;
OP7, three redesign attempts). Combined with the six-opponent SPLIT
dominance pattern noted above, this is a second data point suggesting the
difficulty may not be fully explained by "wrong opponent" -- worth
weighing against the shared-BT-framework / RUSH-blue-script-competence
question already flagged, rather than proceeding straight to OP8 on the
same per-opponent-tuning assumption that failed three times running.

## map_a canonical-map switch + RUSH_PROBE_ROOT_CAUSE_AUDIT (2026-07-28)

**Canonical rule (locked):** all niche experiments (OP6 TURTLE work, OP9
SPLIT/ESCORT confirmation, OP11 ESCORT work, RUSH-host experiments, full
payoff matrices, development/validation/held-out runs) now use `map_a`
(resolves to `MAP_A_OPEN`), not `map_b_split_lane`. Record `map` explicitly
in every manifest going forward. Everything above this point in the
tracker (OP6-OP12 SPLIT/RUSH/ESCORT work, the six-opponent SPLIT-dominance
pattern, all three OP7 RUSH attempts) was built on `map_b_split_lane` and
is NOT combined with map_a niche-acceptance evidence -- kept for
historical reference only.

Separately, the RUSH blue-probe controller has been substantially
redesigned since that work (now "BLUE_PROBES_V3" in
`_scripted_blue_styles.py`: carrier returns directly home post-pickup, no
evasion detour; non-carrier becomes a "screening blocker" that interposes
between carrier and the nearest live threat, then pushes ahead toward
home).

**8-seed dev matrix, OP6-OP12 x 4 styles, map_a**
(`artifacts/full_matrix_mapa_dev1_8seed`, base-seed 601001): appeared to
show three distinct niches -- RUSH best on OP6/OP7/OP8, ESCORT best on
OP9/OP10/OP11, TURTLE best on OP12, SPLIT never uniquely best anywhere.

**16/24-seed held-out confirmations (fresh disjoint seeds, map_a)
COLLAPSED that apparent diversity -- RUSH won all four:**

```text
                RUSH      2nd place           3rd            4th
OP12 (n=16)    1.6875   TURTLE 1.5000    ESCORT 1.0625   SPLIT  0.1250
OP9  (n=16)    1.4375   SPLIT  1.3125    ESCORT 1.0625   TURTLE -0.0625
OP7  (n=16)    2.0000   ESCORT 0.9375    SPLIT  0.6875   TURTLE 0.2500
OP10 (n=24)    2.1667   SPLIT  1.2083    ESCORT 0.9167   TURTLE 0.4167
```

OP12's TURTLE lead (dev: 2.5 vs 1.5) and OP9's ESCORT lead (dev: 2.0 vs
1.375) both flipped to RUSH at n=16. OP10's SPLIT gap (dev: 0.25, closest
of any candidate) widened against SPLIT at n=24 rather than closing. The
8-seed dev matrix's apparent three-niche structure was very likely
small-sample noise, not a real signal.

**Decision: opponent-specific niche engineering (OP6 TURTLE, OP9 ESCORT,
OP10 SPLIT) is PAUSED.** Per explicit instruction, do not resume until a
root-cause audit determines whether `BLUE_RUSH_V3` is simply overbuilt,
`map_a` intrinsically rewards direct tempo, or RUSH has a real but
narrower niche that the other three opponents' designs could still
reclaim. No probe or opponent edits until diagnosis completes.

```text
Opponent niche engineering: PAUSED (until landscape ranks TURTLE candidate)
RUSH V3 root-cause audit: COMPLETE — keep RUSH V3 unchanged
map_a: canonical
OP6-OP12 only (no opponent outside the permitted pool)
BLUE_RUSH_V3: accepted competent probe (do not weaken direct-home)
```

Audit plan (locked, three phases, OP7/OP9/OP10/OP12, unchanged red
opponents throughout):
1. Decompose V3 into R0 (old pre-V3 RUSH, no direct-home/no screen), R1
   (direct-home return only), R2 (screening blocker only), R3 (full V3) --
   diagnostic-only monkey-patched variants, `_scripted_blue_styles.py`
   itself untouched.
2. Compare RUSH's post-pickup behavior against ESCORT's (carrier-teammate
   distance, interposition rate, support duration) -- check whether V3 has
   absorbed ESCORT's core mechanism rather than remaining a distinct style.
3. Decompose each confirmed RUSH win into pickup-timing advantage +
   conversion advantage + defensive/counter-score cost.

**Tooling correctness note (important, found mid-audit):** the first
decomposition run (`experiments/diagnose_rush_v3_decomposition.py`) showed
absurdly small margins (0.00-0.38) for all four variants including R3
(nominally full, unmodified V3) against OP7/OP9/OP10/OP12 -- inconsistent
with the official held-out confirmations' RUSH margins (1.4-2.2) for the
same opponents. Root-caused to two bugs in the custom script, NOT in the
engine or RUSH controller itself: (1) `set_phase`/`set_next_opponent`/
`blue_scripted`/`set_blue_style` were only applied BEFORE `env.reset()`;
the official tool (`run_scripted_style_payoff_matrix.py::_run_one_episode`)
applies them a second time immediately AFTER `env.reset()` too, and
skipping that second call apparently lets reset revert the intended
opponent/style configuration; (2) final scores were read directly from
`core.blue_score`/`core.red_score` tensors, which get cleared as part of
automatic done-triggered reset -- the official tool instead reads
`infos[0]["episode_result"]` captured at the exact `done` step. Fixed both;
verified the fix reproduces the official OP7/BLUE_RUSH/seed-631001 episode
exactly (margin=2, steps=90, matching `episode_results.csv` bit for bit).
The SAME two bugs were present in the earlier `diagnose_rush_probe_root_cause.py`
Phase-1 baseline script from this session -- its specific numbers (RUSH
converting 4/4 everywhere, path-length/target-change comparisons) are
UNRELIABLE and should not be cited going forward. This does not affect the
held-out confirmations or the full 7x4 matrix that triggered this audit --
those were run through the official tool, which already applies both
patterns correctly.

**Phase 1+2 results (corrected, 8 seeds/variant/opponent, base-seed
651001):**

```text
                        OP7    OP9    OP10   OP12   MEAN
R0 (old pre-V3 RUSH)    0.38   1.00   0.50   0.50   0.60
R1 (direct-home only)   2.38   1.38   2.25   1.62   1.91
R2 (screen only)        0.62   1.75   1.00   1.00   1.09
R3 (full V3)            2.25   1.88   1.88   1.62   1.91
```

R1 (direct-home carrier return, no screening blocker) matches R3 (full V3)
almost exactly -- 1.91 mean margin either way, R1 even edges out R3 on OP7
(2.38 vs 2.25). R2 (screening blocker alone, carrier still uses generic
multi-threat evasion) is much weaker, barely ahead of R0. Win rates make
this starker: R1 hits 8/8 in all four opponents (vs R0's 4/8-7/8), the
single biggest jump in the whole table.

Phase 2 (ESCORT-overlap) reads directly off the same data: R1's
teammate-near-carrier fraction (0.16-0.29) and interposition fraction
(0.04-0.10) are LOW -- its non-carrier is doing its own independent thing,
not escorting -- yet R1 performs as well as full V3. R2, which IS
escort-like (teammate-near 0.74-0.87, interposition 0.25-0.56), performs
worse. **RUSH has not absorbed ESCORT's mechanism as its real advantage;
the screening blocker is the weaker component, not the driver.**

Phase 3 (margin decomposition): pickup timing is identical across variants
by construction (~13.8-13.9 steps, pre-pickup route held fixed). The
decisive lever is return RELIABILITY (WR) and, in 3/4 opponents, return
SPEED (R1 return-time 14.9-46.6 vs R0's 16.8-99.3; OP7 is the exception,
R1 slightly slower than R0 at 42.8 vs 39.2 but still wins far more often).
Own-flag-loss stays high regardless of variant (OP12 is 8/8 in every
variant) -- that vulnerability traces to the pre-pickup dual-attack
opening structure (both agents committing forward from the start), not to
which post-pickup mechanism is used. The trade-off is real and orthogonal
to the R0-R3 axis, not something direct-home-return specifically causes or
fixes.

**Decision-tree verdict: "Direct-home return alone creates most of the
gain."** The old RUSH probe was genuinely underbuilt (matching this
session's earlier pre-map_a audit finding); V3's direct-home return is the
competent, not overbuilt, version of the same style -- a real, legitimate
competence fix, not scope creep into ESCORT's territory. Combined with the
"real but orthogonal home-defense trade-off" finding: RUSH wins through
faster, more reliable conversion and pays for it with a consistently
undefended base, which is a legitimate strategic cost, just not one that
OP7/OP9/OP10/OP12 currently punish hard enough to overcome RUSH's tempo
advantage.

**Recommendation per the locked decision tree: keep RUSH V3 unchanged (do
not build a bounded V4, do not weaken direct-home return or basic carrier
competence).** The productive next step is finding or engineering an
opponent whose identity specifically and decisively punishes an abandoned
home base during the return window, since none of the four opponents
tested currently do that enough to make RUSH non-dominant. This is a
different, more targeted question than "redesign OP6/OP9/OP10 for their
old assumed niches" -- those redesigns remain PAUSED pending explicit
direction on which opponent (existing or newly scoped within OP6-OP12) to
target for a home-defense-punishing identity.

**LOCKED (2026-07-28): RUSH V3 accepted as competent probe — do not "fix RUSH."**

Audit decomposition (direct-home ≈ full V3; screening modest; home abandonment
real):

```text
Strength:  fast, reliable flag conversion (direct-home return)
Weakness:  both agents attack → blue own base exposed
```

Keep `BLUE_RUSH_V3` unchanged. Do **not** weaken direct-home return.

Updated strategic picture:

```text
RUSH probe:    competent and accepted
RUSH weakness: abandoned home defense
SPLIT anchor:  likely map_b_split_lane context
TURTLE task:   exploit RUSH’s home-defense sacrifice
ESCORT task:   reward persistent carrier support
```

**Next (after multi-map landscape finishes — do not restart):** rank every
`(OP6–OP12, map)` with:

```text
red score rate while both blue agents away from home
blue own-flag loss before first blue score
red pickup-to-score conversion
simultaneous-carry frequency
RUSH payoff
TURTLE payoff
```

TURTLE-engineering candidate = context where RUSH’s abandoned base is
punished most **and** TURTLE’s home anchor prevents that punishment.

Focused micro-gates for that candidate:

```text
vs RUSH:   red first-score ≥6/8; blue own-flag lost frequently
vs TURTLE: red first-score ≤2/8; TURTLE counter-after-stop ≥5/8
payoff:    TURTLE uniquely best; TURTLE−RUSH ≳ +0.5
```

Tooling: `experiments/diagnose_rush_home_defense_gap.py` (extend to all
landscape maps after scan completes). Landscape artifact:
`artifacts/multimap_v3_landscape_op6_op12_8seed` (672 eps). One resume only.

`diagnose_rush_home_defense_gap.py` results (map_a, 8 seeds/opponent,
base-seed 661001, RUSH vs TURTLE across all 7 opponents): **RUSH beats
TURTLE in every single one.** Margins range +0.375 to +2.875 for RUSH,
-0.500 to +1.625 for TURTLE; RUSH's WR is 6-8/8 everywhere, TURTLE's tops
out at 7/8 (OP11) and is often 2-3/8. Own-flag-loss is high for TURTLE too
in most opponents (OP7 6/8, OP9 4/8, OP10 7/8, OP11 8/8) -- comparable to
or worse than RUSH's own rate in several cases. This confirms the
map-level diagnosis directly: on map_a's fully open geometry, a stationary
anchor has nothing defensible to guard, so it protects the flag no better
than not having one. No existing (OP, map_a) context is a TURTLE
candidate; a geometric affordance is required, matching the locked
K=4 plan's premise.

## New-map design for TURTLE+ESCORT affordances -- BOTH attempts failed, budget exhausted (2026-07-29)

Per the locked K=4 plan and hard anti-loop budget ("New map designs:
maximum 2 versions"), built `map_c_home_corridor` (`gpu_env/_maps.py`,
`gpu_env/state/map_state.py`): reuses the existing single-obstacle/
corner-routing mechanism Map B already has, but positions the wall near
BLUE's home (x=0.15-0.24 normalized, just past blue's flag at x=2 in a
20-wide field) instead of centered, intending a chokepoint on blue's
flag-return leg that TURTLE could anchor-defend and ESCORT could protect
an unescorted carrier through.

**V1** (y=0.28-0.70, open gaps at both top and bottom, ~28% of field
height each): 4-episode sanity screen (OP6, OP11, all 4 styles,
`artifacts/mapc_dev1_op6_op11_4seed`) showed TURTLE and ESCORT as the
WORST performers (TURTLE -1.25 to -2.25, ESCORT -1.00 to 0.00), not the
best -- RUSH and SPLIT both did well instead. A mechanistic trace
(`scratchpad/trace_mapc_turtle_escort.py`) explained why: with two open
bypasses, attackers/carriers simply routed around whichever end was open
without ever passing through anything a defender could guard. Not a
chokepoint at all in practice, just an irrelevant piece of terrain most of
the time.

**V2** (revised: wall pinned flush to the bottom edge, y=0.18-1.0, exactly
one gap near the top): re-traced the same two matchups. Worse failure
mode this time -- both TURTLE-vs-OP6 and ESCORT-vs-OP11 showed agents
genuinely STUCK (position changing <0.05 units for multiple consecutive
steps) with targets oscillating between conflicting waypoints near the
wall, a navigation breakdown in the shared corner-router, not just a
missed affordance. Quantitatively
(`artifacts/mapc_v2_dev1_op6_op11_4seed`, same 4-episode screen): TURTLE
and ESCORT both went NEGATIVE against both opponents (ESCORT -0.25 to
-2.00, TURTLE -0.25 to -1.25) while RUSH remained dominant or
least-bad. Pinning the wall to force a single mandatory gap made the
outcome worse, not better -- the corridor being close to the map edge and
close to blue's spawn/flag appears to leave the router too little room to
resolve a valid path, and RUSH (direct beelines, no coordination
dependency) is comparatively unaffected by that breakdown while
TURTLE/ESCORT (which need to loiter, patrol, or synchronize near exactly
that zone) are hurt the most.

**Decision: the 2-version map-design budget is exhausted. Do not attempt a
third map-c version.** Status per the locked decision rule (corrected
2026-07-29 -- an earlier draft of this entry wrongly said "K=2 LRO:
demonstrated"; environment/context construction is NOT the LRO proof, which
additionally requires trained PPO specialists, latent births, and the
policy-distinction gates):

```text
Map C V2 geometry:          PASS
TURTLE causal affordance:   FAIL
ESCORT causal affordance:   FAIL
K=4 environment extension:  CLOSED FOR CURRENT BUDGET
Map C V1/V2:                CLOSED_FAILED (2-version budget exhausted)
K=2 context construction:
    C_RUSH  = OP6 | map_a              FROZEN (CONFIRM_PASS, gap +0.25)
    C_SPLIT = OP9 | map_b_split_lane   FROZEN (CONFIRM_PASS, gap +1.25)
K=2 LRO proof:              ACTIVE — specialists NEXT
```

The two failed Map C versions remain useful evidence in their own right: a
single rectangular choke does not automatically create escort or defense
niches when the shared waypoint router cannot coordinate reliably around
it.

This is not a dead end for K=4 in principle -- it means the SPECIFIC
single-rectangular-obstacle mechanism, positioned near a team's home, is
not obviously sufficient on its own, and further attempts would need
either a different geometric mechanism (not just repositioning the same
wall) or engine changes to the router (out of scope for a diagnostic-only
budget). Recorded honestly rather than spending a third attempt outside
the locked budget.

## K=2 LRO proof -- locked plan (2026-07-29), NEXT UP

Environment engineering STOPS once the two contexts below are confirmed.
No further opponent redesigns, no third map version. This is the approved
milestone ("validate at least two distinct complementary strategies before
router training") and does not require K=4 first.

### Rules complementarity ladder (LOCKED 2026-07-31)

Before expensive training or rule redesign, force **tradeoffs** (being good
at one decision costs somewhere else). Baseline stays faithful:

```text
1. Keep RULESET_V2_AQUATICUS_10S as the faithful baseline.
2. Finish exact Gate 1 (opponent admissibility under V2 / map_a).
3. Run Gate 2 episode-level affordance scenarios.
4. Check whether one defender improves defense while reducing offense.
5. Check whether decoy/escort play exploits cooldown.
6. Train a short fresh PPO pilot on map_a.
7. Test whether one generalist still dominates every situation.
8. Only then test one rule adjustment at a time.
```

Accept pattern **before** expensive training / K>2 birth:

```text
aggressive policy wins some contexts
defensive policy wins other contexts
escort/decoy policy wins carrier situations
no single policy wins everywhere
```

Key: not “more rules,” but **complementary costs**. That is the soil
distinct latent strategies need. Do not skip to step 8 until 2–7 show the
pattern (or honestly fail under the V2 baseline).

**Parked (2026-07-31) — clean boundary.** Gate 1 / V2 baseline work stopped
at a verified handoff. **Next session, one task:** build **Gate 2B** —
isolation test first; read **authoritative environment state** throughout
(not post-step reconstructed proxies). Build order and downstream locks
unchanged; nothing else in scope until Gate 2B lands.

### Formal run identity — production wiring (2026-07-31)

Passport helpers (`RunIdentity`, `stamp_*`, `validate_bundle`) already existed.
Production travelers are now stamped from one frozen object resolved in
`orchestrate_training_run` immediately after `build_training_env` and before
any artifact write:

```text
env = build_training_env(...)
run_identity = build_formal_run_identity(env, run_id=cfg.run_tag)
write_startup_formal_artifacts(cfg, run_identity=...)  # run_config + training_manifest
build_trainer(..., run_identity=...)                   # mandatory
# episode CSV rows stamped at write time
# checkpoint save/load share the same identity universe
# in-run evaluation_manifest + result_summary stamped from the same object
```

**Smoke unlock condition — unlocked (2026-07-31).** Items 1–6 are PASS.
The 50k–100k stamped PPO smoke is **AUTHORIZED**. Unlock criteria were:

```text
1. production call sites use RunIdentity
2. identity is mandatory throughout (missing fails before first rollout)
3. real production artifacts validate as one bundle
4. checkpoint save/load uses the same identity universe
5. standalone evaluation verifies checkpoint compatibility
6. diagnostic override cannot produce a formal result
```

Status vs unlock checklist (2026-07-31):

```text
1–4  PASS — orchestrator + writers + formal-bundle integration test
5    PASS — standalone eval resolves live identity, verifies checkpoint
             before inference (map + full ruleset fields), stamps
             evaluation_manifest / episode_rows / result_summary, and
             diagnostic override cannot produce a formal result
             (tests/test_standalone_eval_identity.py)
6    PASS — allow_override forces formal_result_eligible=False
```

```text
1–6 production identity gates   PASS
50k–100k stamped PPO smoke      AUTHORIZED
tag_telemetry production path   PASS (2026-07-31)
G0-v2                           AUTHORIZED — seeds 2500001 / 2500002 / 2500003
latent birth                    locked
router                          locked
```

Integration gates:
- `tests/test_production_formal_bundle.py` (training artifact-bundle-only)
- `tests/test_standalone_eval_identity.py` (standalone eval identity border)
- `tests/test_tag_telemetry_config_wiring.py` (PPOConfig → live env → artifacts;
  formal G0-shaped config refuses omitted/false telemetry before rollout)

**Tag telemetry production path closed (2026-07-31).** `tag_telemetry_enabled`
now travels through `PPOConfig` → `build_training_env` → live `GPUCTFVecEnv`
→ `run_config.json` / `training_manifest.json`. Formal runs with telemetry
false or omitted are rejected in `_validate_config_gates` before env build.
The formal smoke no longer monkeypatches `GPUFieldConfig`; it sets
`cfg.formal_run=True` and `cfg.tag_telemetry_enabled=True`. The 4,096-step
`SMOKE_DRY_RUN=1` plumbing check passed with tag_success>0, cooldown
denials>0, zero hard legality violations, both artifacts recording
`tag_telemetry_enabled=true`, and bundle validation PASS. G0-v2 may launch
with seeds `2500001`, `2500002`, `2500003`. Latent birth / router remain
locked until their own unlock criteria.

The stamped 50k–100k PPO smoke remains authorized as the longer formal
rehearsal; it is not a blocker for the three G0-v2 launches above.

**Gate 2B meaning (LOCKED):** first check that the corrected game has
**strategic soil** on `map_a` — not that latents have been learned.

```text
BOTH_ATTACK  → more offensive progress, less home protection
ONE_DEFENDER → less offensive progress, better home protection
```

Under V1, one defender was nearly useless, so everyone-forward was the
master key and PPO had no reason for a defensive branch. A Gate 2B
**PASS** means both effects are real: two genuinely competing choices
exist on the same map. It does **not** mean distinct latent strategies
exist yet.

Path after Gate 2B (do not skip):

```text
Gate 2B proves trade-off exists
  → train fresh G0-v2 on map_a
  → find situations where G0-v2 is weak
  → train response oracle O1 against one weakness
  → prove G0-v2 and O1 each win different situations
  → prove selecting between them beats either alone
  → create z0 / z1 as distinct latent branches
```

Learned-policy evidence required before naming latents:

```text
O1 > G0-v2 on the weakness context
G0-v2 > O1 on an anchor context
combined repertoire > either fixed policy
behavioral difference is meaningful
```

Only then: `z0` = offensive/balanced, `z1` = defensive/intercept.
Grow K the same way (pool weakness → next oracle), not by forcing four
branches up front:

```text
K=1 → weakness → O1 → K=2
K=2 → pool weakness → O2 → K=3
K=3 → pool weakness → O3 → K=4
```

Context selection (after `artifacts/multimap_v3_landscape_op6_op12_8seed`
completes):

```text
C_RUSH  = strongest NON-DEGENERATE OPx | map_a context
C_SPLIT = strongest NON-DEGENERATE OPy | map_b_split_lane context
```

"Non-degenerate" matters: the payoff-matrix tool flags degenerate red
presets (saturated/too-easy columns), and several map_a columns were
flagged in the 7x4 matrix. Pick from unflagged columns with a real
best-vs-runner-up gap, not merely the largest raw margin.

Proof sequence (in order, no skipping):

```text
0. Confirm C_RUSH and C_SPLIT on fresh disjoint 16-seed blocks; freeze.
1. Train an independent PPO specialist on C_RUSH.
2. Train an independent PPO specialist on C_SPLIT.
3. Cross-evaluate both policies on BOTH contexts.
4. Gate: each policy must be best in its OWN context.
        target matrix:
                            C_RUSH     C_SPLIT
        RUSH specialist      BEST
        SPLIT specialist                 BEST
5. Gate: positive, statistically supported repertoire gain,
        LCB95(delta_pool) > 0.
6. Birth the RUSH response into one latent branch.
7. Freeze it; birth the SPLIT response into a second branch.
8. Verify forced-z payoff crossover, matched-observation policy
   distinction, and distinct trajectory fingerprints.
9. Train a two-strategy legal persistent router (LAST).
```

**16-seed context confirms — BOTH FROZEN (2026-07-28/29):**

`C_RUSH` `artifacts/k2_c_rush_op6_mapa_heldout16_seed821001` — **CONFIRM_PASS / FROZEN**
```text
OP6 | map_a | BLUE_PROBES_V3 | n=16 | seed 821001
RUSH +0.875 WR 15/16 | TURTLE +0.625 | gap +0.25
env commit: 458a57a0989ace3e7b17d820a01a719334cc47b9
analysis: experiments/run_scripted_style_payoff_matrix.py
freeze: context_freeze.json
```
Thin gap: valid scripted opportunity only — does **not** guarantee PPO
crossover. Do not retune OP6/map_a/V3 for a prettier number.

`C_SPLIT` `artifacts/k2_c_split_op9_mapb_heldout16_seed831001` — **CONFIRM_PASS / FROZEN**
```text
OP9 | map_b_split_lane | BLUE_PROBES_V3 | n=16 | seed 831001
SPLIT +2.875 WR 16/16 | RUSH +1.625 | gap +1.25
env commit: 458a57a0989ace3e7b17d820a01a719334cc47b9
analysis: experiments/run_scripted_style_payoff_matrix.py
freeze: context_freeze.json
```

```text
C_RUSH:  CONFIRMED / FROZEN
C_SPLIT: CONFIRMED / FROZEN
PPO specialists: LAUNCHED (matched 1M no_latent_baseline FIXED_OPPONENT 2v2)
  πR: checkpoints/k2_pi_rush  seed 821001  OP6|map_a_open
  πS: checkpoints/k2_pi_split seed 831001  OP9|map_b_split_lane
LRO branches: not started
Router: stopped
```

Learned-policy acceptance (strict; scripted gaps are not enough):
```text
πR > πS on C_RUSH
πS > πR on C_SPLIT
LCB95(Δ_pool) > 0
```

Only after that does K=4 become a justified extension rather than a
prerequisite blocking the whole Summer result.

### Landscape scan COMPLETE (672 eps) -- context selection (2026-07-29)

`artifacts/multimap_v3_landscape_op6_op12_8seed`, 7 opponents x 3 maps x 4
styles x 8 seeds, BLUE_PROBES_V3, base-seed 620001. Full non-degenerate
context inventory, ranked by best-vs-runner-up gap:

```text
BLUE_RUSH best (8 non-degenerate contexts)
  OP10|map_b_split_lane_v2   +2.125  2nd SPLIT +0.750  gap +1.375  WR 1.000
  OP10|map_b_split_lane      +1.875  2nd SPLIT +1.000  gap +0.875  WR 1.000
  OP11|map_b_split_lane      +2.500  2nd SPLIT +1.625  gap +0.875  WR 1.000
  OP8|map_b_split_lane_v2    +2.500  2nd SPLIT +1.625  gap +0.875  WR 1.000
  OP6|map_a                  +0.750  2nd TURTLE +0.000 gap +0.750  WR 0.875
  OP7|map_b_split_lane_v2    +2.250  2nd SPLIT +1.625  gap +0.625  WR 1.000
  OP6|map_b_split_lane       +1.000  2nd SPLIT +0.875  gap +0.125  WR 1.000
  OP8|map_b_split_lane       +2.000  2nd SPLIT +2.000  gap +0.000  WR 1.000

BLUE_SPLIT best (4 non-degenerate contexts)
  OP9|map_b_split_lane       +2.625  2nd RUSH  +1.500  gap +1.125  WR 1.000
  OP9|map_b_split_lane_v2    +2.625  2nd RUSH  +1.500  gap +1.125  WR 1.000
  OP6|map_b_split_lane_v2    +0.750  2nd RUSH  +0.500  gap +0.250  WR 0.875
  OP7|map_b_split_lane       +1.750  2nd RUSH  +1.500  gap +0.250  WR 1.000

BLUE_ESCORT best (1 non-degenerate context)
  OP9|map_a                  +1.125  2nd SPLIT +1.125  gap +0.000  WR 0.750  <- TIE, not a clean win

BLUE_TURTLE best (0 non-degenerate contexts)
  (only OP12|map_a and OP12|map_b_split_lane, BOTH degenerate, gaps +0.125)
```

**SELECTED for the K=2 proof:**

```text
C_RUSH  = OP6_IMMEDIATE_DUAL_RUSH | map_a
          RUSH +0.750, runner-up TURTLE +0.000, gap +0.750, WR 7/8
          (the ONLY non-degenerate map_a context where RUSH is best --
           OP7|map_a, OP10|map_a, OP11|map_a all RUSH-best but DEGENERATE)

C_SPLIT = OP9_SPLIT_LANE_FEINT | map_b_split_lane
          SPLIT +2.625, runner-up RUSH +1.500, gap +1.125, WR 8/8
          (strongest non-degenerate SPLIT context; confirms the long-held
           OP9|map_b SPLIT-anchor hypothesis at n=8 on the full landscape)
```

**Caveat to weigh before training (flagged, not decided):** these two
contexts differ in BOTH opponent and map, so a router could in principle
separate them from map features alone without any opponent modeling. That
is acceptable for a K=2 existence proof but is the weaker version of the
test. The same-map alternative would be `OP6|map_b_split_lane` (RUSH,
gap +0.125) vs `OP9|map_b_split_lane` (SPLIT, gap +1.125) -- identical
map, different opponent, so the router MUST use opponent behavior -- but
the RUSH side's gap there is very thin (+0.125) and likely not separable
at n=8. Recommendation: run the primary selection above, and treat the
same-map pair as a follow-up robustness test rather than the main proof.

**Independent K=4 confirmation from this scan:** TURTLE has ZERO
non-degenerate contexts anywhere in the 21-context landscape, and ESCORT's
single non-degenerate context is a tie rather than a clean win. This
reproduces the "K=4 NOT YET demonstrated" verdict from the full existing
map family, entirely independently of the two failed map_c attempts --
the missing TURTLE/ESCORT affordances are a real, measured gap, not an
artifact of one bad map design.

Pool-level gates on the full 21-context landscape:
`no_dominating_blue_style` PASS, `best_response_diversity` PASS (4/21),
`all_blues_protected` PASS (but only by counting degenerate columns),
`tie_rate_under_threshold` PASS; `delta_pool_lcb_positive` FAIL
(delta_pool +0.0176, CI95 [-0.0418, +0.2381], LCB -0.0418 -- much closer
to zero than any earlier pool), `best_blue_wr_in_band` FAIL,
`no_degenerate_red_styles` FAIL (8/21 degenerate).

### Steps 1-2: fresh 16-seed context confirmations (2026-07-29)

Tooling: `experiments/paired_bootstrap_ci.py` (new; validated by exactly
reproducing the frozen OP7|map_a held-out numbers before use).

**C_SPLIT = OP9_SPLIT_LANE_FEINT | map_b_split_lane -- STRICT PASS,
FREEZE-READY.** `artifacts/c_split_op9_mapb_heldout16_seed691001`,
base-seed 691001 (disjoint from the 620001 discovery block):

```text
BLUE_SPLIT   +2.7500  WR 16/16   <- best
BLUE_RUSH    +1.9375  WR 16/16
BLUE_TURTLE  -0.1875  WR  0/16
BLUE_ESCORT  -0.2500  WR  2/16

SPLIT - RUSH    +0.8125  CI95 [+0.3750, +1.2500]  PASS
SPLIT - TURTLE  +2.9375  CI95 [+2.6875, +3.1875]  PASS
SPLIT - ESCORT  +3.0000  CI95 [+2.5000, +3.5625]  PASS
non-degenerate: PASS (TURTLE and ESCORT genuinely LOSE)
```

Margin held up vs discovery (+2.625 -> +2.750): no discovery-scan
overfitting. This is a real niche, not a gradient.

**C_RUSH = OP6_IMMEDIATE_DUAL_RUSH | map_a -- CROSSOVER PASS,
NON-DEGENERACY FAIL -> demoted to validated fallback/pilot.**
`artifacts/c_rush_op6_mapa_heldout16_seed681001`, base-seed 681001:

```text
BLUE_RUSH    +0.8750  WR 15/16   <- best
BLUE_ESCORT  +0.1875  WR  8/16
BLUE_SPLIT   +0.0625  WR  8/16
BLUE_TURTLE  +0.0625  WR  8/16

RUSH - ESCORT  +0.6875  CI95 [+0.3125, +1.1250]  PASS
RUSH - SPLIT   +0.8125  CI95 [+0.2500, +1.3750]  PASS
RUSH - TURTLE  +0.8125  CI95 [+0.2500, +1.4375]  PASS
non-degenerate: FAIL -- ALL FOUR styles have positive mean margin
```

The degeneracy test (`payoff_matrix_analysis.py:267`) flags a red preset
when every blue style's mean margin is same-signed. At n=8 discovery,
ESCORT (-0.75) and SPLIT (-0.125) were negative so it read
non-degenerate; on fresh seeds they crept positive (+0.1875, +0.0625).
OP6|map_a rewards RUSH but does not PUNISH alternatives -- payoff spread
0.81 vs C_SPLIT's 3.0. Risk for the LRO proof: if the context is easy
enough that any competent policy wins, a SPLIT-trained specialist may also
solve it and the learned crossover collapses for reasons about the
context, not about LRO. Retained as pilot/fallback (checkpoints and
curves preserved; not cited as the primary K=2 proof).

### PREDECLARED C_RUSH selection rule (locked BEFORE candidate results)

Same-map candidates under confirmation: `OP10|map_b_split_lane` and
`OP11|map_b_split_lane` (both RUSH-best with gap +0.875 at n=8 discovery,
both non-degenerate there). A same-map primary pair is strictly stronger
than the cross-map pair because it removes the trivial explanation
"map_a -> RUSH, map_b -> SPLIT" and forces opponent-conditioned selection.

Rule, implemented in `experiments/select_c_rush_context.py` and validated
against OP6|map_a before the candidates finished (it correctly reported
gates 1/2/4 pass, gate 3 fail):

```text
Hard gates (all required):
  1. RUSH uniquely best at 16 seeds
  2. All paired RUSH-vs-other CIs clear zero
  3. Non-degenerate
  4. Pooled best-other LCB > 0
Ranking among survivors:
  5. Larger LCB for (RUSH - runner-up)
  6. Lower saturation, then lower tie rate
```

Outcome policy (also predeclared): both pass -> script ranks mechanically;
one passes -> that one; NEITHER passes -> **pause and decide explicitly**,
do NOT auto-promote OP6|map_a, because the cross-map fallback changes what
the claim proves:

```text
same-map pair  -> opponent-conditioned strategy selection
cross-map pair -> broader context-conditioned selection, with map
                  identity as a possible shortcut
```

### Router information contract (locked, applies at step 11)

```text
ALLOWED:   legal global state/history during centralized training;
           observable opponent behavior; map geometry
FORBIDDEN: hidden opponent preset ID; scripted-style label;
           hindsight episode outcome
```

### C_RUSH SELECTED -- selector run unchanged, output accepted (2026-07-29)

Both same-map candidates confirmed at 16 fresh seeds and **both passed all
four hard gates**; the predeclared ranking then chose between them.

```text
OP10_AGGRESSIVE_INTERCEPTOR | map_b_split_lane   (seed 701001)
  RUSH   +2.0000   SPLIT  +1.2500
  ESCORT -0.2500   TURTLE -0.7500
  RUSH-SPLIT  +0.7500  CI95 [+0.0625, +1.3750]
  RUSH-pooled +1.9167  CI95 [+1.4375, +2.3333]
  degenerate=False  saturation=0.938  tie_rate=0.328   HARD GATES PASS

OP11_ADAPTIVE_EXPLOITER | map_b_split_lane        (seed 711001)   <- SELECTED
  RUSH   +2.4375   SPLIT  +1.2500
  ESCORT +1.0000   TURTLE +0.0000
  RUSH-SPLIT  +1.1875  CI95 [+0.5625, +1.7500]
  RUSH-pooled +1.6875  CI95 [+1.3125, +2.0417]
  degenerate=False  saturation=1.000  tie_rate=0.234   HARD GATES PASS

Criterion 5 (larger RUSH-runner_up LCB): OP11 +0.5625 vs OP10 +0.0625
  -> OP11 wins outright; criterion 6 never invoked (no tie on 5).
```

**Why this is also the right pick on the merits (not just by rule):** the
runner-up in BOTH contexts is BLUE_SPLIT, which is precisely the behavior
the rival specialist pi_S will embody. The crossover gate needs a context
where RUSH-like play beats SPLIT-like play decisively. OP11 separates them
by +1.1875 [+0.5625, +1.7500]; OP10 by only +0.7500 [+0.0625, +1.3750] --
an LCB barely off zero, a risky foundation for the proof.

**Honest caveats recorded, not hidden:**
- OP11's non-degeneracy is MARGINAL: it survives only because TURTLE lands
  at exactly +0.0000. Had TURTLE been +0.0625, OP11 would have been flagged
  degenerate. OP10's non-degeneracy is robust by comparison (ESCORT -0.25
  and TURTLE -0.75 both clearly losing).
- OP11 saturation is 1.000 (RUSH wins 16/16); both candidates FAIL the
  pool-level `best_blue_wr_in_band` gate, which is NOT among the six
  predeclared criteria but is worth noting as a context-difficulty signal.
- If the specialist crossover later fails on OP11 for saturation-related
  reasons, OP10|map_b_split_lane is the designated first alternative
  (robustly non-degenerate, lower saturation), and OP6|map_a remains the
  cross-map fallback/pilot.

### FROZEN K=2 CONTEXT PAIR (step 3 complete)

```text
C_RUSH  = OP11_ADAPTIVE_EXPLOITER | map_b_split_lane
C_SPLIT = OP9_SPLIT_LANE_FEINT    | map_b_split_lane
```

**Same map for both** -- the stronger primary proof. Map geometry is held
constant, so specialists (and later the router) must respond to opponent
behavior, not terrain. This upgrades the intended claim from
context-conditioned to opponent-conditioned strategy selection, and makes
the step-12 same-map follow-up redundant (it is now the primary test).

Freeze record:
```text
env commit:        458a57a0989ace3e7b17d820a01a719334cc47b9
uncommitted:       gpu_env/_maps.py (ADDITIVE map_c_home_corridor only;
                   verified via git diff to touch NO map_a/map_b geometry,
                   so both frozen contexts are unaffected)
blue probes:       BLUE_PROBES_V3 (_scripted_blue_styles.py, unmodified)
map:               map_b_split_lane (both contexts)
max_decision_steps 240
opponents:         OP11_ADAPTIVE_EXPLOITER, OP9_SPLIT_LANE_FEINT (untuned)
confirm artifacts: c_rush_alt_op11_mapb_heldout16_seed711001
                   c_split_op9_mapb_heldout16_seed691001
```

Do not tune either opponent from this point forward.

### K=2v2 specialist stage — FORMAL GATE FAIL @ 1M (2026-07-30)

Same-map pair (above). Specialists: `no_latent_baseline`, 2v2, 1M steps,
3 seeds/family (`901001–3` πR on OP11; `902001–3` πS on OP9). Artifacts:
`artifacts/k2v2_*_train_s*`, `checkpoints/k2v2_piR|piS/`.

Cross-eval: `experiments/run_k2_specialist_cross_eval.py` +
`experiments/analyze_k2_specialist_crossover.py`.
32 fresh paired seeds/context (`1010001–32` / `1020001–32`), margin-based
gates, hierarchical clustered bootstrap for `Δ_pool`. `s902002` retained
in the formal result (training instability; does not cause the FAIL alone).

**Formal @ 1M (sole gate):**

```text
              C_RUSH    C_SPLIT
πR              2.45       2.33
πS              1.13       1.18

πR > πS on C_RUSH:   PASS  Δ=+1.32 CI95 [+1.01, +1.64]
πS > πR on C_SPLIT:  FAIL  Δ=-1.16 CI95 [-1.31, -1.00]
Δ_pool LCB95 > 0:    FAIL  Δ_pool=0.00 (V_sel = V_fixed = πR)

Specialist-stage verdict: FAIL
Latent branch birth:      BLOCKED
Router:                   BLOCKED
0/9 seed pairings show two-direction crossover
```

**Scientific reading:** scripted contexts differed (OP11→RUSH, OP9→SPLIT)
and both passed frozen-context gates, but that difference did **not**
survive PPO to 1M. The OP11-trained policy became a dominant generalist on
the shared map. No complementary repertoire for a router to select.

**Trajectory diagnostics (200k/300k/500k; not formal):**

```text
              C_RUSH(πR-πS)   C_SPLIT(πS-πR)   Δ_pool      pairings
300k          PASS +0.42      PASS +0.25       +0.125 LCB=0  8/9 ok
500k          PASS +0.23      FAIL -0.17       0.00          0/9
1M (formal)   PASS +1.32      FAIL -1.16       0.00          0/9
200k          RUNNING
```

Precise conclusion on the 300k signal:

> At 300k, directional crossover appeared across the specialist families.
> However, statistically supported repertoire gain was not established
> because `LCB95(Δ_pool) = 0`. By 500k, πR had already generalized onto
> C_SPLIT, and by 1M it dominated both contexts.

This is **candidate transient specialization**, not a successful LRO birth.
The 1M formal FAIL is unchanged by any trajectory or behavior-audit result.

Working hypothesis (pending 200k + behavior audit):

```text
200k: specialist responses forming
300k: temporary complementary window
500k: πR begins solving both contexts
1M:   πR becomes dominant generalist
```

Behavior audit harness: `experiments/audit_k2_specialist_behavior.py`
(asks whether πR/πS occupied different policy regions at 300k).

```text
Context niche demonstration:       PASS
Independent PPO competence:        PASS
Learned specialist crossover @1M:  FAIL
Complementary repertoire gain:     FAIL
K=2 LRO specialist proof @1M:      FAIL
300k discovery signal:             promising but unconfirmed
Latent birth and router:           not justified
```

### LOCKED NEXT: 300k confirmatory replication — CANCELLED_PRELAUNCH (2026-07-30)

Preregistration Rev 4 remains on disk as a properly frozen, never-launched
design. **Do not delete** the document, hashes, or commits.

```text
status: CANCELLED_PRELAUNCH

reason:
Completed trajectory analysis showed that the 300k crossover was
isolated, non-monotonic, and unsupported under hierarchical
training-seed resampling. No replication training or evaluation
data were generated.
```

```text
200k: πR ahead on both contexts
300k: isolated πS advantage on C_SPLIT (hierarchical: unconfirmed)
500k: πR ahead on both contexts
1M:   πR decisively ahead on both contexts
```

Record: `artifacts/k2v3_300k_replication/CANCELLED_PRELAUNCH.json`
Freeze commit (Rev 4): `993839b` — preserved.

Discovery behavior audit: finish for postmortem diagnosis only; does **not**
justify reviving the OP11/OP9 specialist pair.

```text
300k replication: CANCELLED_PRELAUNCH
Discovery audit:  finish for diagnosis
Latent birth:     blocked
Router:           blocked
```

### NEXT: G0 learned-incumbent weakness sweep (2026-07-30)

Promote the three completed 1M πR policies to the frozen incumbent family:

```text
G0 = {s901001, s901002, s901003}
checkpoints: checkpoints/k2v2_piR/final_k2v2_piR_op11_mapb_s{seed}_2v2.zip
             (or ckpt_*_1000000.zip)
```

**Important:** G0 was trained on `map_b_split_lane`. The running sweep tests
whether it **transfers** to the primary map (`map_a`). That decides Step 1
of the locked sequential plan below.

Sweep for contexts that defeat the **learned** incumbent, not scripted probes:

```text
Opponents:     OP6–OP12 only
Map:           map_a  (recorded explicitly in every row)
Policies:      all three G0 seeds
Horizon:       240
Evaluation:    deterministic, no DR, n_envs=1
Discovery:     32 fresh episodes per (policy × opponent) cell
Launcher:      experiments/run_g0_weakness_sweep.py  (RUNNING)
```

Select a context only if it challenges the **entire** incumbent family,
ideally:

```text
all three G0 policies have negative mean margins
family-level UCB95 < 0   (strict)
```

**Competence (three-way; locked before reading the real sweep):**

```text
0–2 opponents with negative family mean  → COMPETENT
exactly 3                                → AMBIGUOUS (no C1)
4–7                                      → INCOMPETENT (train G0_map_a)
```

AMBIGUOUS / INCOMPETENT never select a response-oracle context. Pooled
map-wide mean/CI are diagnostics only.

The 32-seed sweep is **discovery**. Any qualifier must survive confirmation
(`docs/g0-c1-confirmation-preregistration.md`: 64 fresh paired seeds, same
strict gate) before freezing C1. Do **not** train O1 from the discovery winner.

Do **not** name the weakness RUSH / SPLIT / TURTLE / ESCORT until the
response-oracle behavior is observed.

Analyzer: `experiments/analyze_g0_weakness.py`
Fixtures: `tests/test_analyze_g0_weakness.py`

```text
1M K=2 proof:             FAIL
300k replication:         CANCELLED_PRELAUNCH
G0 map_a BASE sweep:      COMPLETE — COMPETENT, no C1
G0 map_a variant tags:    RUNNING (21 synonym labels → 7 niches; confirmatory)
Same-map scenario bank:   NEXT if variants find no C1
Rule fidelity checklist:  PASS (hard probes; magnitude not tuned)
Latent birth:             blocked
Router:                   blocked
```

**Discovery result (672 eps, map_a, BASE OP6–OP12):**

```text
competence:  COMPETENT  (0/7 opponents with negative family mean)
pooled mean: +1.96  CI95 [+1.90, +2.02]  (diagnostic)
weakness gate: NO discovery candidate
nearest (NOT selected): OP6 W=+1.13 family=+0.96; OP12; OP9
```

G0 transfers to `map_a` as a strong incumbent against all seven base
opponents. No isolated learned weakness under the strict gate.

**Framing lock (2026-07-30):** primary latent acceptance stays on **one map
(`map_a`)** with several strategically incompatible *situations*, not one
expert per map. See [`same-map-tactical-regimes.md`](same-map-tactical-regimes.md).

**Do not** launch a long-name “variant” re-run of OP6–OP12: short tags already
resolve to the seven distinct LRO niches; `OP6_TURTLE` / `OP7_SWITCHER` etc.
are synonyms of the same BT profiles. That would duplicate this sweep.

**Next:** same-map scenario bank S1–S6 on `map_a` (legal state / mid-episode
opponent switches / commitment costs), plus an Aquaticus rule-fidelity
checklist. Confirmation @ 64 seeds still required before any O1 training.

### LOCKED PLAN: sequential weakness → oracle → latent birth (2026-07-30)

Core rule:

> First prove two policies are genuinely complementary. Then assign them
> different latent IDs. A latent label by itself does nothing.

**Step 1 — competent `map_a` incumbent**

```text
G0 transfers well on map_a vs most OP6–OP12
  → keep as incumbent

G0 fails most opponents on map_a
  → train G0_map_a: map_a, OP6–OP12 mixture, multi-seed,
    task reward only, no latent
```

**Step 2 — find C1 that defeats the learned incumbent**

All G0 seeds struggle; strongest seed still struggles; loss is
opponent-specific, not general map incompetence.

**Step 3 — train full independent response oracle O1 against C1**

Freeze G0. Fresh PPO family, multi-seed, task reward only. No tiny latent
adapter / shared head (those collapsed before). Extra parameters buy
experimental clarity.

**Step 4 — retain O1 only if complementary**

```text
O1 beats G0 on C1
G0 beats O1 on ≥1 incumbent context
selective repertoire > best fixed policy
```

Outcomes: retain as z1 / O1 replaces G0 (still K=1) / discard / behavior-
only (no branch).

**Step 5 — birth z0/z1 as frozen experts**

Separate full policy params per z; z fixed per episode; freeze during
integration; verify KL≈0 / argmax agree / same payoffs vs source ckpts.
Do not jointly fine-tune immediately.

**Step 6 — grow K=2→K=4 sequentially**

```text
C_{k+1} = argmin_c max_{π ∈ P_k} payoff(π,c)
train O_{k+1} against that weakness
retain iff pool with O_{k+1} beats pool without it
```

**Router last** — only after ≥2 retained branches. May use legal state/history;
never hidden opponent ID, scripted label, or future outcome.

```text
K=1: competent map_a incumbent
K=2: first confirmed complementary oracle
K=3: solves a weakness of the two-policy pool
K=4: solves a weakness of the three-policy pool
Router: after branches exist
```

Immediate sequence from here:

```text
1. BASE G0 map_a sweep                         DONE — COMPETENT, no C1
2. Skip long-name alias re-run (duplicate niches)
3. Aquaticus rule-fidelity checklist on map_a   NEXT (diagnostic)
4. Same-map scenario bank S1–S6 on map_a        NEXT (implementation)
5. Evaluate G0 on scenarios; seek C1 under locked competence + gate
6. Confirm C1 @ 64 fresh seeds
7. Train full independent O1; retain only if complementary
8. Encode z0/z1; grow pool → O2 → O3 → K=4; router last
```

Framing: [`same-map-tactical-regimes.md`](same-map-tactical-regimes.md)
Confirmation prereg: [`g0-c1-confirmation-preregistration.md`](g0-c1-confirmation-preregistration.md)

---

## C2 fresh confirmation REJECTED → C3 draft pivot (2026-08-06)

**C2 confirmation complete.** Artifacts:
`artifacts/c2_confirmation/C2_CONFIRMATION_FROZEN_RESULT.json`.

```text
verdict                 C2_REJECTED
fresh seed block        9800001+ (SPENT — do not reuse / retune on this block)
policy passes           0 / 3  (required >= 2 / 3)
natural support         PASS (~87% onset prevalence; hundreds of onsets)
headroom                FAIL (~0.12)
actionability           FAIL (0.0)
O2 training             DO NOT TRAIN
```

Scientific read: Stage 2 discovery was reproducible enough to look interesting,
but the niche did not survive fresh confirmation as an **intervention** target.
Support was not the failure mode. Same class of trap as C1 (correlate without
usable strategic fork), with C2's twist of abundant natural carrier-failure
support and discovery deltas that still failed headroom/actionability.

**Discipline locked:** no reinterpretation of `none_forward_frac`, no lag-band
retune, no threshold relax, no runner-up promotion on `9800001+`.

**Next direction (approved, not frozen):** decision-proximal /
counterfactual-actionability discovery (C3). Draft only:
[`c3-decision-proximal-preregistration.md`](c3-decision-proximal-preregistration.md).

Key draft corrections vs an earlier lag-band proposal:

- event-anchored `CARRIER_PRESSURE_ONSET` (not flag-pickup ∪ pressure mix)
- Stage 2 is **not** C2 `[-30,-20)` bands; features at \(t_0\) + min lead time
- \(A(s)\) = best-legal **improvement** over G0, not absolute outcome shift
- force alternative at fork only; H=30 is evaluation horizon
- exhaustive **legal** macros; seeds `9400000+` for Stages 1–3; `9810000+`
  for Stage 4 natural **and** fresh counterfactual replication
- else record `C3_NO_QUALIFIED_STRATEGIC_FORK` and stop

Do **not** implement Stage 1 or write `C3_DISCOVERY_PREREG_FROZEN.json` until
the draft's open freeze checklist is closed and status flips to FROZEN.

**Item 9 SETTLED (2026-08-06):** commitment-fork definition locked in
[`c3-decision-proximal-preregistration.md`](c3-decision-proximal-preregistration.md)
§"Commitment fork definition — item 9 SETTLED": natural G0/`map_a` state;
≥2 legal team responses; upstream of pressure/failure; measurable utility
divergence over `H_response`; backward trace selects the **earliest** such
state. Pressure onset is an anchor only. Numeric cells (`T_trace`,
`H_response`, \(\delta\), \(U\)) still close with items 2–6.

**Implementation checkpoint:** [`c3-item10-code-audit.md`](c3-item10-code-audit.md)
records `ITEM 10 IMPLEMENTATION PATCHED`; focused tests pass and no C3 scan was
run. Next, close items 1–8 and freeze `T_trace`, `H_response`, `delta`, and `U`.

```text
item 9 SETTLED → item 10 AUDIT COMPLETE → PATCH COMPLETE → close items 1–8
→ freeze final C3 contract → authorization artifact → smoke → full scan
```

Do **not** run C3, write `C3_EXECUTION_AUTHORIZATION.json`, train O3, or touch
latent birth until that path completes.

**Scope limit added (2026-08-06, second revision):** C3 is a candidate
commitment-fork detector only; it cannot establish latent necessity,
complementarity, routing value, or strategy families. Latent eligibility is
owned by the preregistered **Environment-Demand Gate**
([`environment-demand-gate-preregistration.md`](environment-demand-gate-preregistration.md)):
D1 `LCB95(G_available) > 0`, D2 preference reversal (O3 > G0 in C3 context AND
G0 > O3 in anchor context), D3 per-context competence floor, D4 matched-state
behavioral nonredundancy — all on fresh demand-evaluation seeds with frozen
context frequencies, executed only after an independent task-reward O3
exists. For every later birth, D5 additionally requires
`LCB95(Delta V_repertoire) > 0` beyond the existing selective pool. Any
required failure → NO LATENT BIRTH, NO ROUTER.

---

## 7. Cross-references

| Need                                              | Where to look                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Mandatory agent behavior                          | [`AGENTS.md`](../../AGENTS.md)                                                      |
| Scientific definition of the paper method         | [`summer-method-spec.md`](summer-method-spec.md)                                    |
| Fidelity rules / classification / proposal form   | [`summer-fidelity-rules.md`](summer-fidelity-rules.md)                              |
| Per-preset facts, aliases, deltas, run tags       | [`latent-preset-registry.md`](latent-preset-registry.md)                            |
| Launch / eval / statistical protocols             | [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)    |
| C2 confirmation (REJECTED) / spent block 9800001+ | [`c2-qualification-preregistration.md`](c2-qualification-preregistration.md); `artifacts/c2_confirmation/` |
| C3 commitment-proximal draft (NOT frozen)         | [`c3-decision-proximal-preregistration.md`](c3-decision-proximal-preregistration.md) |
| Environment-demand gate (latent eligibility)      | [`environment-demand-gate-preregistration.md`](environment-demand-gate-preregistration.md) |
| v6i2 frozen gate thresholds (pre-confirmatory)  | [`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md)                    |
| Codeâ†”manuscript trace                             | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)                    |
| Algorithm sketch                                  | [`../../docs/algorithm.md`](../../docs/algorithm.md)                                |
