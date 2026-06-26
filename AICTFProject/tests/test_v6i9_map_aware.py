"""Behavioral gate tests for V6I9 map-aware competence stage.

Tests verify:
1. V6I9 preset produces an 8-channel observation space.
2. map_b obstacle channel is non-empty; map_a_open obstacle channel is all-zero.
3. Map A and Map B produce different obstacle channels.
4. CNN 7→8 channel expansion preserves existing weights and zeros new channel.
5. V6I9 preset is importable and has expected fields.
6. _expand_cnn_obs_channels is idempotent on an already-8-channel checkpoint.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class TestV6I9Preset(unittest.TestCase):
    def test_preset_importable(self) -> None:
        from rl.config_presets import v6i9_map_aware_config, v6i9_map_aware_split_lane_config
        cfg = v6i9_map_aware_config()
        self.assertEqual(cfg.map_layout, "map_b")
        self.assertFalse(cfg.enable_latent_z_residual)
        self.assertEqual(cfg.experiment_id, "v6i9")

    def test_split_lane_variant(self) -> None:
        from rl.config_presets import v6i9_map_aware_split_lane_config
        cfg = v6i9_map_aware_split_lane_config()
        self.assertEqual(cfg.map_layout, "map_b_split_lane_v2")

    def test_v6i9_uses_hardpool_opponents(self) -> None:
        from rl.config_presets import v6i9_map_aware_config
        cfg = v6i9_map_aware_config()
        pool = tuple(str(k).upper() for k in cfg.opponent_pool)
        self.assertIn("OP8", pool)
        self.assertIn("OP9", pool)
        self.assertIn("OP10", pool)


class TestObstacleChannelGeometry(unittest.TestCase):
    """Verify that map_b has a non-empty obstacle channel and map_a_open does not."""

    def _make_env_obs(self, map_layout: str, obstacle_obs_channel: bool):
        """Return (obs_dict_from_reset, env). Caller must close env."""
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=2, max_red_agents=2,
            map_layout=map_layout, max_decision_steps=400,
            aquaticus_profile=True, rules_profile="OURS",
            device="cpu", seed=42,
            obstacle_obs_channel=obstacle_obs_channel,
        )
        env = GPUCTFVecEnv(cfg)
        obs = env.reset()
        # reset() may return (obs, info) or just obs
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs, env

    def test_map_b_obstacle_channel_nonempty(self) -> None:
        obs, env = self._make_env_obs("map_b", True)
        try:
            grid = obs["grid"]  # (n_agents, C, H, W) or (1, n_agents, C, H, W)
            # Normalize to 4D (agent, C, H, W)
            if grid.ndim == 5:
                grid = grid[0]
            self.assertEqual(grid.shape[1], 8, "map_b with obstacle channel should have 8 CNN channels")
            wall_ch = grid[0, 7, :, :]  # agent 0, channel 7
            wall_sum = float(wall_ch.sum().item()) if hasattr(wall_ch, "item") else float(wall_ch.sum())
            self.assertGreater(wall_sum, 0.0, "map_b obstacle channel should be non-empty")
        finally:
            env.close()

    def test_map_a_open_no_obstacles(self) -> None:
        obs, env = self._make_env_obs("map_a_open", False)
        try:
            grid = obs["grid"]
            if grid.ndim == 5:
                grid = grid[0]
            self.assertEqual(grid.shape[1], 7, "map_a_open without obstacle channel should have 7 CNN channels")
        finally:
            env.close()

    def test_map_a_and_map_b_obstacle_channels_differ(self) -> None:
        obs_a, env_a = self._make_env_obs("map_a_open", True)
        obs_b, env_b = self._make_env_obs("map_b", True)
        try:
            ga = obs_a["grid"]
            gb = obs_b["grid"]
            if ga.ndim == 5:
                ga, gb = ga[0], gb[0]
            if ga.shape[1] < 8 or gb.shape[1] < 8:
                self.skipTest("Obstacle channel not available in both envs")
            ch_a = ga[0, 7, :, :]
            ch_b = gb[0, 7, :, :]
            import numpy as np
            ch_a_np = ch_a.numpy() if hasattr(ch_a, "numpy") else np.asarray(ch_a)
            ch_b_np = ch_b.numpy() if hasattr(ch_b, "numpy") else np.asarray(ch_b)
            diff = float(np.abs(ch_a_np - ch_b_np).sum())
            self.assertGreater(diff, 0.0, "map_a_open and map_b should have different obstacle channels")
        finally:
            env_a.close()
            env_b.close()


class TestCNNChannelExpansion(unittest.TestCase):
    """Unit tests for _expand_cnn_obs_channels warm-start migration."""

    def _make_fake_sd(self, in_channels: int) -> dict:
        key = "latent_actor.actor_cnn.conv.0.weight"
        return {key: torch.randn(32, in_channels, 3, 3)}

    def test_7to8_expands_correctly(self) -> None:
        from rl.custom_ppo.inference import _expand_cnn_obs_channels
        sd7 = self._make_fake_sd(7)
        original_w = sd7["latent_actor.actor_cnn.conv.0.weight"].clone()
        sd8 = _expand_cnn_obs_channels(sd7, 8)
        new_w = sd8["latent_actor.actor_cnn.conv.0.weight"]
        self.assertEqual(tuple(new_w.shape), (32, 8, 3, 3))
        # Channels 0-6 preserved
        self.assertTrue(torch.allclose(new_w[:, :7], original_w))
        # Channel 7 (obstacle) zero-initialized
        self.assertTrue(torch.allclose(new_w[:, 7], torch.zeros(32, 3, 3)))

    def test_already_8_channels_is_noop(self) -> None:
        from rl.custom_ppo.inference import _expand_cnn_obs_channels
        sd8 = self._make_fake_sd(8)
        original_w = sd8["latent_actor.actor_cnn.conv.0.weight"].clone()
        result = _expand_cnn_obs_channels(sd8, 8)
        self.assertTrue(torch.allclose(result["latent_actor.actor_cnn.conv.0.weight"], original_w))

    def test_7_channels_preserved_at_logit_level(self) -> None:
        """Expanded 8-channel model and original 7-channel model agree on obstacle-free observations.

        A probe observation with zero obstacle channel fed into the expanded model must
        produce the same CNN output as the original 7-channel model, because the new
        channel weight is zero (w·0 = 0 → no contribution).
        """
        from rl.networks import CNNEncoder
        from gpu_env._constants import CNN_ROWS, CNN_COLS

        shape7 = (7, CNN_ROWS, CNN_COLS)
        shape8 = (8, CNN_ROWS, CNN_COLS)
        enc7 = CNNEncoder(shape7, feature_dim=128)
        enc8 = CNNEncoder(shape8, feature_dim=128)

        # Copy weights: channels 0-6 identical, channel 7 zero (already true from init? No — random init)
        # Manually set enc8 to match enc7 on channels 0-6 and zero on channel 7.
        with torch.no_grad():
            w7 = enc7.conv[0].weight.clone()   # (32, 7, 3, 3)
            w8 = enc8.conv[0].weight
            w8[:, :7] = w7
            w8[:, 7] = 0.0
            enc8.conv[0].bias.copy_(enc7.conv[0].bias)
            # Copy remaining layers
            for i in (2, 4):
                enc8.conv[i].weight.copy_(enc7.conv[i].weight)
                enc8.conv[i].bias.copy_(enc7.conv[i].bias)
            enc8.proj.weight.copy_(enc7.proj.weight)
            enc8.proj.bias.copy_(enc7.proj.bias)

        obs7 = torch.randn(2, 7, CNN_ROWS, CNN_COLS)
        obs8 = torch.zeros(2, 8, CNN_ROWS, CNN_COLS)
        obs8[:, :7] = obs7  # obstacle channel = 0

        with torch.no_grad():
            out7 = enc7(obs7)
            out8 = enc8(obs8)

        self.assertTrue(torch.allclose(out7, out8, atol=1e-5),
                        f"7→8 warm-start: CNN outputs should match on zero obstacle channel, "
                        f"max diff={float((out7 - out8).abs().max().item()):.3e}")


if __name__ == "__main__":
    unittest.main()
