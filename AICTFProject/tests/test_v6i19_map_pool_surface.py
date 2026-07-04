"""Pinning tests for V6I19 per-episode map pool over the v6i18 surface."""

from __future__ import annotations

import dataclasses
import unittest

import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from gpu_env._maps import MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2
from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.env_factory import build_training_env


class V6i19MapPoolPresetTests(unittest.TestCase):
    def test_aliases_resolve_to_map_pool_arm(self) -> None:
        aliases = [
            "v6i19",
            "v6i19_map_pool_surface_diagnostic",
            "v6i19_map_pool_surface",
            "latent_v6i19_map_pool_surface_diagnostic",
            "plan_faithful_latent_v6i19_map_pool_surface_diagnostic",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i19_is_exact_map_pool_diff_over_v6i18(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i18")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i19")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {"experiment_id", "map_pool", "run_tag"},
        )
        self.assertEqual(cfg_obj.experiment_id, "v6i19")
        self.assertEqual(
            tuple(cfg_obj.map_pool),
            (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2),
        )
        self.assertEqual(cfg_obj.map_layout, MAP_B_SPLIT_LANE)

    def test_v6i19_preserves_v6i18_specialist_scaffold(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i19")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(tuple(str(x).upper() for x in cfg.opponent_pool), ("OP8", "OP9", "OP10", "OP11", "OP12"))
        self.assertTrue(cfg.latent_contract_specialist_enabled)
        self.assertEqual(cfg.latent_contract_specialist_variant, "sharp")
        self.assertAlmostEqual(float(cfg.latent_contract_specialist_coef), 0.75)
        self.assertTrue(cfg.enable_latent_z_residual)
        self.assertTrue(cfg.latent_actor_z_adapter_enabled)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")
        self.assertEqual(cfg.max_decision_steps, 240)
        self.assertAlmostEqual(float(cfg.env_surface_score_margin_coef), 0.15)

    def test_env_factory_forwards_map_pool(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i19")
        normalize_and_validate_training_config(cfg)
        env = build_training_env(cfg, initial_phase="OP8", initial_opponent_tag="OP8")
        try:
            self.assertEqual(
                tuple(env.cfg.map_pool),
                (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2),
            )
            layouts = {env.core._map_layout_for_env(i) for i in range(env.core.B)}
            self.assertTrue(layouts.issubset({MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2}))
        finally:
            env.close()


class MapPoolSamplingTests(unittest.TestCase):
    def test_reset_sampling_is_seed_deterministic(self) -> None:
        def _layouts_after_resets(seed: int, n_resets: int) -> list[str]:
            cfg = GPUFieldConfig(
                n_envs=1,
                max_blue_agents=2,
                max_red_agents=2,
                map_layout=MAP_B_SPLIT_LANE,
                map_pool=(MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2),
                max_decision_steps=32,
                device="cpu",
                seed=seed,
            )
            env = GPUCTFVecEnv(cfg)
            try:
                out: list[str] = [env.core._map_layout_for_env(0)]
                mask = torch.tensor([0], dtype=torch.long, device=env.core.device)
                for _ in range(n_resets):
                    env.core.reset_indices(mask)
                    out.append(env.core._map_layout_for_env(0))
                return out
            finally:
                env.close()

        a = _layouts_after_resets(123, 8)
        b = _layouts_after_resets(123, 8)
        self.assertEqual(a, b)
        self.assertGreater(len(set(a)), 1)

    def test_episode_info_includes_map_id(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=2,
            map_layout=MAP_B_SPLIT_LANE,
            map_pool=(MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2),
            max_decision_steps=8,
            device="cpu",
            seed=7,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            obs = env.reset()
            actions = torch.zeros((env.core.B * env.core.Nb * 2,), dtype=torch.int64)
            _, _, done, infos = env.step(actions.detach().cpu().numpy())
            if not done[0]:
                self.skipTest("episode did not terminate in one noop step")
            info = infos[0]
            self.assertIn("map_layout", info)
            self.assertIn("map_id", info)
            self.assertGreaterEqual(int(info["map_id"]), 0)
            er = info.get("episode_result", {})
            self.assertEqual(er.get("map_layout"), info.get("map_layout"))
            self.assertEqual(er.get("map_id"), info.get("map_id"))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
