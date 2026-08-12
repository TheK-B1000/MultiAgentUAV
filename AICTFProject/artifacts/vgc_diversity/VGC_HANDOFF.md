# VGC-Style Maritime CTF — Opponent-Diversity Scaling

**Single authoritative handoff.** Do not create competing status documents.

---

## CURRENT PHASE
Phase 5 — freezing D1/D3/D7, then launch.

## LAST COMPLETED STEP
Phase 1 audit + Phase 2 self-play recovery: **`SELF_PLAY_REUSE = PASS`**.

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

## SELF-PLAY STATUS — `SELF_PLAY_REUSE = PASS`

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
Supported by existing trainer. No code change required. Configs pending.

## FP STATUS
Not built. Will reuse the snapshot seam + historical checkpoint list.

## D1/D3/D7 DEFINITIONS
Pending — see freeze artifact.

## ACTIVE RUNS
None.

## NEXT AUTOMATIC STEP
Freeze D1/D3/D7 → Mixed PPO smoke → launch.

## BLOCKERS
None.

## RECOVERY COMMANDS
```
python experiments/scan_status.py --shard-dir <dir> --arm <2v2|4v4>
git show 6815ad4c^:AICTFProject/rl/league.py      # recover league selection logic
```
