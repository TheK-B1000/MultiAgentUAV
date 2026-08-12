# VGC-Style Maritime CTF — Opponent-Diversity Scaling

**Single authoritative handoff.** Do not create competing status documents.

---

## CURRENT PHASE
Phase 12/13 — D1 training live and health-verified. D3 queued behind it (GPU-bound).

## LAST COMPLETED STEP
`TRAINING_STARTED_AND_VERIFIED` for D1 (3 seeds).

## STATUS FLAGS
```
PPO_AS_OPPONENT_REUSE            = PASS   (snapshot seam verified end-to-end)
FP_FOUNDATION                    = PASS
HISTORICAL_SELF_PLAY_TRAINER     = REMOVED
HISTORICAL_SELF_PLAY_CHECKPOINTS = UNAVAILABLE
MIXED_PPO_SMOKE                  = PASS   (only OP9 sampled across 64 episodes)
MANIFESTS                        = PASS   (vgc_condition.json sidecar)
UNIFIED_EVALUATION               = BUILT / NOT YET RUN
FP_SMOKE                         = PENDING
BUILD COMPLETE                   = NOT YET

CORRECTION: an earlier `SELF_PLAY_REUSE = PASS` overstated this. What was
verified is the PPO-as-opponent SEAM, not a self-play trainer. The trainer is
removed and the historical self-play checkpoints are unavailable.
```

## ACTIVE RUNS
| method | cond | seed | budget | log | artifacts | status |
|---|---|---|---|---|---|---|
| Mixed-PPO | D1 | 3600001 | 1M | `artifacts/vgc_diversity/d1_seed3600001.log` | `vgc_d1_seed3600001/` | healthy |
| Mixed-PPO | D1 | 3600002 | 1M | `d1_seed3600002.log` | `vgc_d1_seed3600002/` | healthy |
| Mixed-PPO | D1 | 3600003 | 1M | `d1_seed3600003.log` | `vgc_d1_seed3600003/` | healthy |

git commit at launch: `16703638`. Team size 2v2, pool `['OP9']`, held-out 6.

## GPU HEALTH
~6.8/12.2 GB, 69% util with 3 runs (~2.3 GB each). **Three more would exceed
capacity — D3 must wait for D1**, not run concurrently.

## D1/D3/D7 DEFINITIONS
```
D1 = OP9                      (median offensive_pressure)
D3 = OP7, OP9, OP12           (min, median, max)
D7 = OP6..OP12                (already trained)
```
**PRIMARY D7 = the three 2v2 G0-V5 policies (3200001-3) ONLY.**
The 4v4 C7 policies (3300001-3) are a SECONDARY team-size check and may not
enter the primary D1/D3/D7 comparison -- the frozen artifact requires one team
size across rungs, so mixing 4v4 in would confound team size with diversity.

## NEXT AUTOMATIC STEP
1. Build FP driver on the snapshot seam + FP smoke (code only, no GPU).
2. Extend cross-play evaluator with policy_id / method / diversity fields.
3. When D1 finishes -> launch D3 (3 seeds).
4. Evaluate the THREE primary 2v2 D7 baselines once GPU frees.

---

## Phase 1 — Reuse Matrix

| Capability | Existing location | Reuse directly | Minimal extension | Missing |
|---|---|---|---|---|
| PPO trainer | `rl/custom_ppo/trainer.py` | ✅ | | |
| Training entrypoint (frozen protocol) | `experiments/run_g0_v5_long.py` | ✅ | | |
| 4v4 wrapper | `experiments/run_c7_stage0_4v4.py` | ✅ | | |
| Checkpoint save/load | `rl/custom_ppo/checkpoints/`, `rl/evaluation/checkpoint.py` | ✅ | | |
| Env factory | `rl/training/env_factory.py`, `gpu_env/` | ✅ | | |
| Map + ruleset | `map_a`, `V2_RULES` / `RULESET_V2` | ✅ | | |
| OP6–OP12 definitions | `gpu_env/_core/_bt_profiles.py` | ✅ | | |
| **Mixed-opponent sampling** | `mode=OPPONENT_POOL`, `opponent_pool`, `opponent_randomize` | ✅ **already exists** | | |
| Weighted opponent sampling | `opponent_pool_weights` | ✅ | | |
| **PPO-vs-PPO (self-play seam)** | `gpu_env/state/snapshots.py`, `_core/_step.py:84` | ✅ **live** | | |
| League/snapshot selection logic | `rl/league.py` (deleted; recoverable `6815ad4c`) | | recover if FP needs it | |
| Full-board cross-play eval | `experiments/run_g0_v2_evaluation.py` | ✅ | add policy_id/method fields | |
| Health gates | TASK_HEALTH / SYSTEM_HEALTH probes | ✅ | | |
| Seed handling | `cfg.seed`, frozen blocks | ✅ | | |
| Manifests | `training_manifest.json`, `run_config.json` | ✅ | add `diversity_condition` | |
| Scan/status tooling | `experiments/scan_status.py`, `srctf/artifacts.py` | ✅ | | |
| C4–C7 tooling | `run_c4_opportunity_cost.py`, `analyze_c5_discovery.py` | ✅ (diagnostic only) | | |
| Forced-z / latent eval | `experiments/run_forced_z_eval.py` | ✅ (secondary) | | |
| Behavioral JSD | `rl/analysis/`, latent diagnostics | ✅ (secondary) | | |
| Fictitious Play driver | — | | build on snapshot seam | ⬜ small |

### Key findings

1. **Mixed-opponent PPO already exists.** `mode=OPPONENT_POOL` samples *uniform per completed
   episode* over `cfg.opponent_pool`, deterministic under `cfg.seed`, no mid-episode switching.
   **D1/D3/D7 are just pool sizes — no new sampling code needed.**

2. **D7 runs already exist.** `G0-V5` (seeds 3200001-3, 2v2) and `C7 Stage 0` (seeds 3300001-3,
   4v4) were both trained with `opponent_pool = OP6..OP12` at 1M steps. **These are Mixed-PPO D7.**

3. **Self-play was NOT deleted — it was never SB3-dependent.** `SELF_PLAY` as a *train mode* is
   removed (`config_validation.py:61` downgrades it to FIXED_OPPONENT), and the old SB3-era
   `checkpoints_sb3/` league/self-play zips are gone. But the **PPO-vs-PPO env seam is live** and
   uses the *current* stack.

---

## SELF-PLAY STATUS — seam only

Verified end-to-end, not inferred:

```
_load_snapshot_policy(g0_v5 ckpt) -> CustomPPOInferencePolicy   (not a silent None)
red trajectory SNAPSHOT a531098955b7a096 != SCRIPTED OP7 6927af0a94f03c8f
```

- `core.set_next_opponent("SNAPSHOT", <ckpt path>)` drives red from a checkpoint.
- Loader is `rl.custom_ppo.load_custom_ppo_policy` — current format, with CNN channel
  expansion 7→8 and a behavioral-equivalence PASS.
- **Hazard noted:** `snapshots.py` wraps loading in `except Exception: model = None`, so a
  failed load degrades silently. Any FP/self-play runner must assert the policy is non-None.

## MIXED PPO STATUS
`MIXED_PPO_SMOKE = PASS`. Entrypoint `experiments/run_vgc_diversity.py` reuses
run_g0_v5_long and rebinds only pool/seeds/paths. Verified from episode_rows.csv
that a D1 run samples **only OP9** across all 64 logged episodes.

## FP STATUS
Not built. Will reuse the snapshot seam + historical checkpoint list.

## D1/D3/D7 DEFINITIONS
Pending — see freeze artifact.

## BLOCKERS
None. D3 is queued rather than blocked (GPU capacity).

## OPERATIONAL NOTE
A smoke run (seed 3699999) was left alive after its artifacts were deleted and
contended for GPU with the real runs; killed, and ~1.35 GB reclaimed. Kill the
process before deleting a run's artifacts, not after.

## RECOVERY COMMANDS
```
python experiments/scan_status.py --shard-dir <dir> --arm <2v2|4v4>
git show 6815ad4c^:AICTFProject/rl/league.py      # recover league selection logic
```
