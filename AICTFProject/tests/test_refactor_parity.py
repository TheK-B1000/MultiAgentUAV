from __future__ import annotations

import csv
import io
import unittest

import numpy as np

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig


_EPISODE_COLUMNS = (
    "step",
    "reward",
    "done",
    "blue_score",
    "red_score",
    "decision_steps",
    "red_reward_total",
    "blue_reward_total",
    "winner",
)


def _fixed_rollout_csv(*, seed: int = 123, n_steps: int = 1000) -> str:
    cfg = GPUFieldConfig(
        n_envs=2,
        n_agents_per_team=2,
        device="cpu",
        seed=seed,
        max_decision_steps=25,
    )
    env = GPUCTFVecEnv(cfg)
    try:
        env.seed(seed)
        env.reset()
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for step in range(n_steps):
            actions = np.empty((cfg.n_envs, cfg.max_blue_agents * 2), dtype=np.int64)
            actions[:, 0::2] = rng.integers(0, cfg.n_macros, size=(cfg.n_envs, cfg.max_blue_agents))
            actions[:, 1::2] = rng.integers(0, cfg.n_targets, size=(cfg.n_envs, cfg.max_blue_agents))
            env.step_async(actions)
            _, rewards, dones, infos = env.step_wait()
            for env_i, info in enumerate(infos):
                if not bool(dones[env_i]):
                    continue
                episode = info.get("episode_result", info)
                rows.append(
                    {
                        "step": step,
                        "reward": f"{float(rewards[env_i]):.8f}",
                        "done": int(bool(dones[env_i])),
                        "blue_score": int(episode.get("blue_score", info.get("blue_score", 0))),
                        "red_score": int(episode.get("red_score", info.get("red_score", 0))),
                        "decision_steps": int(episode.get("decision_steps", info.get("decision_steps", 0))),
                        "red_reward_total": f"{float(episode.get('red_reward_total', 0.0)):.8f}",
                        "blue_reward_total": f"{float(episode.get('blue_reward_total', info.get('reward_total', 0.0))):.8f}",
                        "winner": str(episode.get("winner", "")),
                    }
                )
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=_EPISODE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()
    finally:
        env.close()


class RefactorParityTests(unittest.TestCase):
    def test_fixed_seed_rollout_is_byte_identical(self) -> None:
        self.assertEqual(_fixed_rollout_csv(), _fixed_rollout_csv())
