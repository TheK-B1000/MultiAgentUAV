"""Phase 0 characterization and Phase 1 correctness tests for episode credit refactor."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.distributions import Categorical

from rl.custom_ppo.latent.credit.episode.advantages import compute_fixed_episode_advantages
from rl.custom_ppo.latent.credit.episode.awrd_targets import resolve_awrd_coef
from rl.custom_ppo.latent.credit.episode.manager import EpisodeCreditManager
from rl.custom_ppo.latent.credit.episode.refresh_targets import build_refresh_targets
from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.records import EpisodeStrategyRecorder
from rl.custom_ppo.latent.types import RouterAction, RouterActionSource


class FixedAdvantageTests(unittest.TestCase):
    def test_advantages_identical_across_engine_epochs(self) -> None:
        captured: list[torch.Tensor] = []
        original = __import__("rl.ppo_core", fromlist=["ppo_policy_loss"]).ppo_policy_loss

        def _spy(new_log_prob, old_log_prob, adv, clip_eps):
            captured.append(adv.detach().clone())
            return original(new_log_prob, old_log_prob, adv, clip_eps)

        class _TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.strategy_encoder = torch.nn.Linear(8, 4)
                self.episode_strategy_value_head = torch.nn.Linear(8, 1)

            def strategy_logits(self, state, selector_hidden=None):
                return self.strategy_encoder(state)

            def episode_strategy_value(self, state, z, selector_hidden=None):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer = MagicMock()
        trainer.device = torch.device("cpu")
        trainer.latent_k = 4
        trainer.latent_episode_strategy_ppo = True
        trainer.fixed_latent_strategy = False
        trainer.latent_episode_strategy_coef = 1.0
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_n_epochs = 3
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_q_phi_train_after_steps = 0
        trainer.global_step = 0
        trainer.cfg.max_grad_norm = 0.5
        trainer.cfg.target_kl = None
        trainer.cfg.latent_q_phi_marginal_baseline = False
        trainer.latent_bucket_baseline = None
        trainer.latent_q_phi_bucket_baseline = None
        trainer.router_optimizer = None
        trainer.latent_router_optimizer = None
        trainer.model = _TinyModel()
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

        host = MagicMock()
        host.trainer = trainer
        host._strategy_encoder_params = MagicMock(return_value=[])
        host._value_head_params = MagicMock(return_value=[])
        host.rollout_strategy_episode_records = [
            {
                "global_state_0": torch.zeros(8),
                "z": 1,
                "executed_z": 1,
                "z_logprob_old": -0.5,
                "behavior_log_prob": -0.5,
                "episode_return": 1.0,
                "opponent_id": 4,
                "bucket_id": 2,
                "action_source": "router",
            },
            {
                "global_state_0": torch.ones(8),
                "z": 2,
                "executed_z": 2,
                "z_logprob_old": -0.6,
                "behavior_log_prob": -0.6,
                "episode_return": 0.5,
                "opponent_id": 5,
                "bucket_id": 3,
                "action_source": "router",
            },
        ]
        host.latent_preference_buffer = []
        host.refresh_preference_buffer = []
        host.rollout_refresh_records = []
        host.rollout_forced_z_episode_count_by_z = torch.zeros(4)
        host.rollout_forced_episode_count_by_opp_z = torch.zeros(7, 4)

        mgr = EpisodeCreditManager(host)
        with patch("rl.custom_ppo.latent.optimization.router_ppo.ppo_policy_loss", side_effect=_spy):
            mgr.apply_episode_strategy_ppo(latent_lam_h=0.0)
        self.assertGreaterEqual(len(captured), 2)
        for adv in captured[1:]:
            self.assertTrue(torch.allclose(captured[0], adv))


class RouterActionRecordingTests(unittest.TestCase):
    def test_epsilon_records_proposed_executed_and_probs(self) -> None:
        rec = EpisodeStrategyRecorder()
        rec.record_start(
            env_index=0,
            episode_id=1,
            global_state_0=torch.zeros(4),
            proposed_z=0,
            executed_z=2,
            behavior_log_prob=-1.2,
            router_log_prob=-0.4,
            action_source=RouterActionSource.EPSILON_MIXTURE,
            bucket_id=1,
            q_phi_probs=[0.7, 0.1, 0.1, 0.1],
        )
        out = rec.record_outcome(env_index=0, episode_return=1.0, episode_win=1, opponent_id=4)
        assert out is not None
        self.assertEqual(out["proposed_z"], 0)
        self.assertEqual(out["executed_z"], 2)
        self.assertAlmostEqual(out["behavior_log_prob"], -1.2)
        self.assertAlmostEqual(out["router_log_prob"], -0.4)
        self.assertEqual(out["action_source"], "epsilon_mixture")


class AwrdScheduleTests(unittest.TestCase):
    def test_boost_scales_with_fraction_not_hardcoded_step(self) -> None:
        trainer = MagicMock()
        trainer.cfg.latent_awrd_boost_after_steps = 0
        trainer.cfg.latent_awrd_boost_after_fraction = 0.5
        trainer.cfg.latent_awrd_boost_multiplier = 1.5
        trainer.cfg.curriculum_nominal_timesteps = 200_000
        boosted = resolve_awrd_coef(
            trainer=trainer,
            base_coef=0.1,
            soft_margin=True,
            global_step=100_000,
        )
        not_boosted = resolve_awrd_coef(
            trainer=trainer,
            base_coef=0.1,
            soft_margin=True,
            global_step=50_000,
        )
        self.assertAlmostEqual(boosted, 0.15)
        self.assertAlmostEqual(not_boosted, 0.1)


class RecurrentRefreshTests(unittest.TestCase):
    def test_missing_hidden_marks_refresh_invalid(self) -> None:
        trainer = MagicMock()
        trainer.device = torch.device("cpu")
        trainer.latent_k = 4
        trainer.latent_event_preference_key_mode = "event_flag"
        trainer.latent_v3i3_event_preference_enabled = True
        trainer.latent_v3i3_event_preference_coef = 0.1
        trainer.latent_v3i3_event_preference_warmup_steps = 0
        trainer.latent_v3i3_event_preference_normalize = False
        trainer.global_step = 0
        trainer.model.use_recurrent_selector = True
        host = MagicMock()
        host.trainer = trainer
        host.refresh_preference_buffer = [{"opponent_id": 0, "event_type": 0, "flag_state_bucket": 0, "z": 1, "future_return": 1.0}]
        host.rollout_refresh_records = [
            {
                "opponent_id": 0,
                "reason_id": 0,
                "flag_state_bucket": 0,
                "refresh_state": torch.zeros(8),
            }
        ]
        targets = build_refresh_targets(trainer=trainer, host=host)
        self.assertTrue(targets.active)
        self.assertFalse(targets.valid)


class OptimizerRegistryTests(unittest.TestCase):
    def test_staged_v6_requires_router_optimizer(self) -> None:
        trainer = MagicMock()
        trainer.router_optimizer = None
        trainer.latent_router_optimizer = None
        with patch(
            "rl.custom_ppo.latent.optimization.router_registry.is_staged_v6i1_curriculum",
            return_value=True,
        ):
            with self.assertRaises(RuntimeError):
                LatentOptimizerRegistry.from_trainer(trainer)
