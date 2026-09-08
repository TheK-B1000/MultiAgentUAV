"""NON-SCIENTIFIC smoke test: can the env + policy instantiate and step at 2v2, 4v4, 6v6?

Readiness audit only. Traces the REAL PPO training env path
(rl.training.env_factory.build_training_env) rather than a bespoke constructor, so a PASS
here means the executable path works, not that a config field exists.

SEEDS ARE NON-SCIENTIFIC (99_9xx_xxx range) and produce no evaluable artifact. Nothing this
script emits is eligible for any experiment. It writes no records under artifacts/.

Run:  python experiments/smoke_team_size_readiness.py [--device cpu] [--sizes 2,4,6]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMOKE_SEED = 99_900_001  # NON-SCIENTIFIC. Never eligible for evaluation.


def probe(team_size: int, device: str) -> dict:
    import numpy as np
    import torch

    from rl.config.ppo_config import PPOConfig
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    from rl.training.env_factory import build_training_env

    out: dict = {"team_size": team_size}
    env = None
    try:
        cfg = PPOConfig()
        cfg.seed = SMOKE_SEED
        cfg.device = device
        cfg.n_envs = 2
        cfg.max_blue_agents = team_size
        cfg.max_decision_steps = 8
        cfg.gpu_native_env = True
        cfg.use_latent_strategy = False
        cfg.train_domain_randomization = False  # nominal only

        env = build_training_env(cfg, initial_phase="phase1", initial_opponent_tag="OP6")
        obs_space, act_space = env.observation_space, env.action_space
        out["obs_space_grid"] = tuple(obs_space.spaces["grid"].shape)
        out["action_space"] = str(act_space)
        out["n_action_heads"] = int(len(act_space.nvec))

        # the grid observation is the tensor that was hard-bound to 2 agents historically
        grid_agents = int(obs_space.spaces["grid"].shape[0])
        out["grid_agent_dim"] = grid_agents
        out["grid_agent_dim_matches_team"] = bool(grid_agents == team_size)
        # MultiDiscrete should be 2 heads per agent (macro, target)
        out["action_heads_match_team"] = bool(len(act_space.nvec) == 2 * team_size)

        policy = SharedActorCentralizedCritic(obs_space, act_space)
        policy.eval()
        out["policy_params"] = int(sum(p.numel() for p in policy.parameters()))

        obs = env.reset()
        out["reset_ok"] = True
        steps = 0
        with torch.no_grad():
            for _ in range(3):
                a = np.stack([act_space.sample() for _ in range(cfg.n_envs)])
                obs, rew, done, info = env.step(a)
                steps += 1
        out["stepped"] = steps
        out["VERDICT"] = "PASS" if (out["grid_agent_dim_matches_team"]
                                    and out["action_heads_match_team"]) else "SHAPE_MISMATCH"
    except Exception as e:  # noqa: BLE001 - readiness audit wants the message, not a raise
        out["VERDICT"] = "FAIL"
        out["error"] = f"{type(e).__name__}: {e}"
        out["trace_tail"] = traceback.format_exc().strip().splitlines()[-1]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", default="2,4,6")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print("=" * 74)
    print(f"TEAM-SIZE READINESS SMOKE  device={args.device}  sizes={sizes}")
    print(f"  seed {SMOKE_SEED} -- NON-SCIENTIFIC, no artifact is written")
    print("=" * 74)

    results = []
    for ts in sizes:
        r = probe(ts, args.device)
        results.append(r)
        print(f"\n  --- team_size={ts} : {r['VERDICT']} ---")
        for k, v in r.items():
            if k in ("team_size", "VERDICT"):
                continue
            print(f"      {k}: {v}")

    print("\n" + "=" * 74)
    for r in results:
        print(f"  {r['team_size']}v{r['team_size']}: {r['VERDICT']}")
    return 0 if all(r["VERDICT"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
