"""Diagnostic only: does the distilled student's z0-vs-Pole-A episode length explain the sealed
eval's slow first cell? Logs real step counts on FRESH, non-sealed seeds. Does not touch the
sealed Compression Crossover run (11922001..11922064) or write any protected artifact.

Run:  python experiments/diagnose_compression_student_episode_length.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
DIAG_SEEDS = list(range(11_930_001, 11_930_009))  # 8 episodes, disjoint from every sealed block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    frozen = json.loads((SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json").read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    device = args.device

    probe = R2.build_env(device, DIAG_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)

    print(f"DIAGNOSTIC: student z0 vs Pole A episode length, {len(DIAG_SEEDS)} fresh seeds "
          f"(NOT the sealed block), MAX_STEPS={R2.MAX_STEPS}\n", flush=True)

    lengths, hit_cap, margins, t0 = [], 0, [], time.time()
    for seed in DIAG_SEEDS:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = 0
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = {"OP6": pole_A_genome()}
            install_keyed_opponent_overlays(core, genomes)
            key = P0.POLES["A"]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=f"diag seed {seed}")
            steps, terminal = 0, None
            ep_t0 = time.time()
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                steps += 1
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            ep_dt = time.time() - ep_t0
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
                hit_cap += 1
            lengths.append(steps)
            margins.append(terminal[0] - terminal[1])
            print(f"  seed {seed}: steps={steps:3d}  hit_cap={steps >= R2.MAX_STEPS}  "
                  f"blue={terminal[0]} red={terminal[1]}  wall={ep_dt:.1f}s", flush=True)
        finally:
            env.close()

    total_dt = time.time() - t0
    print(f"\n  mean steps={np.mean(lengths):.1f}  median={np.median(lengths):.1f}  "
          f"hit_cap={hit_cap}/{len(DIAG_SEEDS)}  mean_wall_per_ep={total_dt/len(DIAG_SEEDS):.1f}s")
    print(f"  extrapolated cell (64 eps) at this pace: ~{total_dt/len(DIAG_SEEDS)*64/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
