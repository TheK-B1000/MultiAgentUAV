# Summer 2026 Audit — Phase 0 (read-only)

Generated from live repository state, not from memory. Commands and outputs
are the source of truth; nothing here is inferred from prior session notes
without re-verification.

## Snapshot at audit time

```
commit         5e549e3a5cf078224680caacf25e6300b71fb2b1
branch         vgc
working tree   clean except untracked Phase-1 outputs (listed below, uncommitted)
python procs   0
GPU            9% util, 2067/12227 MiB (idle; residual driver allocation, not a job)
disk free      ~94 GB on K:
```

Untracked (not yet committed — this session's live Phase-1 output):
```
AICTFProject/artifacts/vgc_diversity/policies_primary.json
AICTFProject/artifacts/vgc_diversity/crossplay/            (d1_vs_d7_summary.json, d1_vs_d7_matrix.md)
AICTFProject/artifacts/vgc_diversity/vgc_d1_seed3600001/   (final checkpoint + full training artifacts)
AICTFProject/artifacts/vgc_diversity/vgc_d1_seed3600002/
AICTFProject/artifacts/vgc_diversity/vgc_d1_seed3600003/
```
Also present, unrelated to this experiment (C7, from the prior work in this
session — not touched, not part of this audit's scope): a modified
`c7_stage2/STAGE2_STATUS_BOARD.txt` and a follower script.

## REUSABLE (verified present and working)

| Asset | Path | Verified |
|---|---|---|
| PPO trainer | `rl/custom_ppo/trainer.py` | in continuous use, no changes needed |
| Frozen training entrypoint | `experiments/run_g0_v5_long.py` | reused wholesale by D1/D3/D7 runner |
| Mixed-opponent sampling | `mode=OPPONENT_POOL`, `cfg.opponent_pool` | pre-existing; D1 verified restricted to 1 opponent across 64 episodes |
| **D1 diversity runner** | `experiments/run_vgc_diversity.py` | ran 3/3 seeds to 1M steps, all healthy |
| **D3 preflight (frozen + implemented, not yet run)** | `experiments/run_d3_pool_preflight.py`, `artifacts/vgc_diversity/D3_POOL_PREFLIGHT_FROZEN.json` | 7 criteria frozen before any D3 datum exists |
| **Unified cross-play evaluator** | `experiments/run_crossplay_eval.py` | ran to completion: 6 policies × 7 opponents × 30 episodes = 1,260 games |
| PPO-as-opponent (SNAPSHOT) seam | `gpu_env/state/snapshots.py`, `_core/_step.py` | checkpoint loads as `CustomPPOInferencePolicy`, drives red distinctly from scripted (hash-verified) |
| **FP loadability guard + SNAPSHOT-pool rotator** | `experiments/run_fictitious_play.py`, `rl/custom_ppo/curriculum_runtime.py::_sample_snapshot_opponents` | wired additively; scripted-path compatibility verified at the RNG level (separate stream, seed+902 vs seed+901) |
| **FP smoke gate (frozen, not yet run)** | `artifacts/vgc_fp/FP_SMOKE_FROZEN.json` | 9 criteria, two-checkpoint pool to avoid the singleton-pool flaw |
| Explicit population trainer | `rl/population/population_trainer.py` | audited read-only; inherits map/team/reward/budget from base_cfg; `map_pool` field is inert for training but pollutes manifests — flagged, not yet fixed |
| Checkpoint save/load, identity verification | `rl/custom_ppo/checkpoints/`, `rl/evaluation/checkpoint.py` | unchanged, in continuous use |
| OP6–OP12 opponent registry | `gpu_env/_core/_bt_profiles.py` | canonical only; no OP13 created |
| map_a / RULESET_V2 | `experiments/run_g0_v2_evaluation.CANONICAL_MAP`, `V2_RULES` | inherited by every runner above; never overridden |
| Diagnostic response-crossover tooling | `experiments/run_c4_opportunity_cost.py`, `analyze_c5_discovery.py`, `srctf/artifacts.py` | built for C4–C7; directly reusable for Phase 5's crossover test at the RESPONSE level (see Incompatible/Gap notes) |
| Forced-z / latent eval | `experiments/run_forced_z_eval.py`, `eval_v6i9_repertoire_complementarity.py` | present, untouched; Phase 11 candidate |

## D1/D3/D7 status (already Phase 1–4 of the new spec)

```
D1  seeds 3600001-3   COMPLETE   1M steps each, 33 health panels, 0 failures
D7  seeds 3200001-3   COMPLETE   reused verbatim from G0-V5 (no retraining)
D3  seeds 3700001-3   NOT STARTED   frozen pool {OP7,OP9,OP12}; preflight gate frozen, not run
```

**Cross-play evaluation already ran**: `artifacts/vgc_diversity/crossplay/d1_vs_d7_summary.json`
(1,260 episodes, raw episode-level rows preserved, not aggregate-only).

Headline (full detail in that file, not restated here to avoid a second
source of truth):

```
D1 overall 0.803   D7 overall 0.798   (statistically indistinguishable)
D1 on OP9 (its only training opponent): 0.733
D7 on OP9 (never trained on it specifically): 0.689   diff +0.044, within SE
=> no measurable specialization from single-opponent training on this board
```

This is directly Phase 4 + half of Phase 6 of the new spec's ladder, already done.

## MISSING (does not exist yet, must be built)

| Gap | Needed for | Notes |
|---|---|---|
| `experiments/summer_2026/manifest.yaml` | Phase 1 (new spec's naming convention) | data already exists in `VGC_DIVERSITY_SETS_FROZEN.json` + per-run `vgc_condition.json` sidecars; this is a rollup, not new data |
| `scripts/run_summer_2026.py` state machine | Phase 12 | orchestration wrapper; must call existing runners, not reimplement them |
| `artifacts/summer_2026/state.json`, lock file | Phase 12 | new |
| Explicit single-opponent specialists beyond D1 (`S_OP7`, `S_OP12`) | Phase 5 crossover test | **D1 (=S_OP9) is the only single-opponent specialist that exists.** Phase 5 as specified (policy A/B × opponent X/Y crossover) needs at least 2 differently-trained specialists. This is the actual next scientific decision point, not an engineering gap — see Safe Next Action. |
| Oracle repertoire / selector / GRU selector code | Phases 6, 9, 10 | not started; correctly gated behind Phase 5 |
| FP population loop (init → mixture → best response → repeat) | Phase 8 | the seam and rotator exist; the outer FP loop (beyond `FP_SMOKE`) does not |
| `SUMMER_2026_STATUS.md` / `RESULTS.md` / `PROVENANCE.md` / `FAILURES.md` | Phase 12 output | not yet generated; will be generated as gates complete, not fabricated ahead of data |

## INCOMPATIBLE (exists but needs adaptation, not replacement)

| Item | Issue | Resolution |
|---|---|---|
| `rl/population/population_member.py` `map_pool` default (`map_b_split_lane*`) | inert for training, but written into the member manifest — a Phase-2/7 run would train on map_a while its own artifact claims a different map | fix before first use: assert + record the map actually inherited by `member_cfg`, not the config label |
| C4–C7 crossover tooling operates on **response-level** counterfactuals (30-step branches from one state), not **policy-level** matchup win rates | Phase 5 as specified is framed at the policy level (A vs B win rate under opponents X/Y) | both are valid operationalizations of "crossover"; policy-level is cheaper (reuses `run_crossplay_eval.py` directly) and is the natural next step once ≥2 specialists exist. Response-level remains available as a secondary/stronger test, consistent with the already-frozen C4–C7 machinery. |
| The new spec's `D1/D3/D7` phase names overlap textually with the existing `S_OP7/S_OP9/S_OP12` specialist naming already established in this session's handoff | naming collision risk | preserved distinction, restated below |

## CURRENTLY_RUNNING

**Nothing.** 0 python processes, GPU idle. No conflicting job to serialize against.

## Naming reconciliation (binding, not re-litigated)

```
D1 / D3 / D7    = ONE PPO's training DIVERSITY (fixed meaning, frozen)
                  D1 = OP9-only, D3 = {OP7,OP9,OP12}, D7 = OP6..OP12

S_OP7/S_OP9/S_OP12  = SEPARATE specialist PPOs, one opponent each
                       D1's finished checkpoints ARE S_OP9 (same training
                       condition); S_OP7 and S_OP12 do not exist yet
```
The new spec's Phase 5/7 language ("specialist A", "specialist B") maps onto
the `S_*` namespace, not onto D1/D3/D7. `run_vgc_diversity.py` has no `--pool`
override, so this cannot be violated by accident.

## SAFE_NEXT_ACTION

Three items are unblocked and require no new scientific decision:

1. **`D3_POOL_PREFLIGHT`** — frozen, implemented, never run. Cheap (short smoke),
   GPU idle, no conflict.
2. **`FP_SMOKE`** — frozen, implemented, never run. Same status.
3. **Commit the untracked Phase-1 outputs** (D1 checkpoints, crossplay results,
   registry) so they are not sitting only on disk.

Both gates are required before either D3 training or FP generation-1 training
can start, and both were already next-in-queue before this audit began — this
audit confirms rather than changes that order.

**Deferred, not blocking**: Phase 5 as newly specified needs `S_OP7`/`S_OP12`
to exist. Training those is a real GPU commitment (2 more 1M-step runs) and a
scientific scope decision (train them now vs. wait for D3 to land and pick the
minimum informative next specialist). Not started without that decision.
