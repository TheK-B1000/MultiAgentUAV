from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch
import numpy as np

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.global_state import GLOBAL_STATE_DIM
from rl.qphi_features import C2_QPHI_CONTEXT_DIM, build_c2_qphi_context_batch
from rl.train_ppo import PPOConfig, TrainMode, _apply_training_preset, train_ppo


_WORKSPACE_TMP = Path(__file__).resolve().parents[1] / ".test_runs" / "c2_qphi_features"


class C2QphiFeatureTests(unittest.TestCase):
    def test_c2_preset_expands_qphi_only(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "latent_c2_router_ce_features")

        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.mode, TrainMode.OPPONENT_POOL.value)
        self.assertEqual(cfg.opponent_pool, ("OP3", "OP5", "OP6", "OP7"))
        self.assertTrue(cfg.freeze_actor_critic)
        self.assertEqual(cfg.qphi_oracle_mode, "none")
        self.assertEqual(cfg.qphi_oracle_dim, 0)
        self.assertEqual(cfg.qphi_context_mode, "c2_temporal")
        self.assertEqual(cfg.qphi_context_dim, C2_QPHI_CONTEXT_DIM)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertAlmostEqual(cfg.router_ce_coef, 1.0)
        self.assertEqual(cfg.router_ce_mode, "soft")

    def test_c2_context_builder_shape_and_finite(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(n_envs=2, n_agents_per_team=4, max_decision_steps=20, device="cpu", seed=710)
        )
        try:
            env.reset()
            action = np.zeros((2, len(env.action_space.nvec)), dtype=np.int64)
            for _ in range(3):
                env.step(action)
            ctx = build_c2_qphi_context_batch(env.core)
            self.assertEqual(tuple(ctx.shape), (2, C2_QPHI_CONTEXT_DIM))
            self.assertTrue(torch.isfinite(ctx).all().item())
            self.assertEqual(env.state().shape[-1], GLOBAL_STATE_DIM)
        finally:
            env.close()

    def test_strategy_router_accepts_c2_context_without_actor_critic_width_change(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(n_envs=2, n_agents_per_team=4, max_decision_steps=20, device="cpu", seed=711)
        )
        try:
            env.reset()
            model = SharedActorCentralizedCritic(
                env.observation_space,
                env.action_space,
                latent_k=4,
                z_embed_dim=16,
                qphi_context_dim=C2_QPHI_CONTEXT_DIM,
            )
            gs = torch.as_tensor(env.state(), dtype=torch.float32)
            ctx = build_c2_qphi_context_batch(env.core)
            logits = model.strategy_logits(gs, context=ctx)
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertEqual(model.qphi_context_dim, C2_QPHI_CONTEXT_DIM)
            self.assertEqual(model.critic.global_state_dim, GLOBAL_STATE_DIM)
            self.assertNotEqual(model._decentralized_actor_in_dim, GLOBAL_STATE_DIM + C2_QPHI_CONTEXT_DIM)
            with self.assertRaisesRegex(ValueError, "context feature width"):
                model.strategy_logits(gs, context=torch.zeros((2, C2_QPHI_CONTEXT_DIM + 1)))
        finally:
            env.close()

    def test_c2_smoke_run_reaches_update_with_context_probe_metrics(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_c2_router_ce_features_2v2"
        labels_path = _WORKSPACE_TMP / "router_labels.json"
        labels_path.write_text(
            '{"k":4,"opponents":{"OP3":{"opponent_id":2,"hard_z":1,"soft":[0.0,1.0,0.0,0.0]}}}',
            encoding="utf-8",
        )
        cfg = _apply_training_preset(PPOConfig(), "latent_c2_router_ce_features")
        cfg.seed = 712
        cfg.total_timesteps = 8
        cfg.n_envs = 1
        cfg.n_steps = 8
        cfg.batch_size = 8
        cfg.n_epochs = 1
        cfg.device = "cpu"
        cfg.max_blue_agents = 2
        cfg.checkpoint_dir = str(_WORKSPACE_TMP)
        cfg.run_tag = tag
        cfg.router_ce_labels_path = str(labels_path)
        cfg.enable_progress_bar = False
        cfg.episode_log_every = 0
        try:
            train_ppo(cfg)
            final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
            self.assertTrue(final_zip.is_file())
            metrics_csv = _WORKSPACE_TMP / f"{tag}_metrics.csv"
            self.assertTrue(metrics_csv.is_file())
            header = metrics_csv.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("qphi_context_probe_acc", header)
            self.assertIn("qphi_context_pairwise_sep_min", header)
        finally:
            for path in (
                labels_path,
                _WORKSPACE_TMP / f"final_{tag}.zip",
                _WORKSPACE_TMP / f"{tag}_metrics.csv",
                _WORKSPACE_TMP / f"{tag}_episodes.csv",
                _WORKSPACE_TMP / f"{tag}_run_config.json",
                _WORKSPACE_TMP / f"{tag}.run.lock",
            ):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
