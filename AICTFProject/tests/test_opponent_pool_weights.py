"""Coverage for the opponent_pool_weights lever (Phase-2 contested-signal frequency).

Verifies:
  * Validation auto-normalizes weights to sum 1.0 and rejects mis-aligned/negative entries.
  * TrainerHyperparams plumbing carries weights from PPOConfig to the trainer.
  * The training-opponent sampler hook (curriculum_runtime) calls
    np.random.Generator.choice(p=weights) when weights are present, and falls back
    to uniform when they are not.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.custom_ppo.curriculum_runtime import _hook_sample_training_opponent_before_reset
from rl.training.config_validation import (
    _normalize_opponent_pool_weights,
    normalize_and_validate_training_config,
)


class OpponentPoolWeightsValidationTests(unittest.TestCase):
    def test_empty_weights_is_uniform_noop(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP3", "OP5", "OP6")
        cfg.opponent_pool_weights = ()
        _normalize_opponent_pool_weights(cfg)
        self.assertEqual(cfg.opponent_pool_weights, ())

    def test_normalizes_to_sum_one(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP3", "OP5", "OP6")
        cfg.opponent_pool_weights = (2.0, 5.0, 3.0)
        _normalize_opponent_pool_weights(cfg)
        self.assertAlmostEqual(sum(cfg.opponent_pool_weights), 1.0, places=9)
        self.assertAlmostEqual(cfg.opponent_pool_weights[0], 0.2, places=9)
        self.assertAlmostEqual(cfg.opponent_pool_weights[1], 0.5, places=9)
        self.assertAlmostEqual(cfg.opponent_pool_weights[2], 0.3, places=9)

    def test_rejects_length_mismatch(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP3", "OP5", "OP6")
        cfg.opponent_pool_weights = (0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "length 2 but opponent_pool has length 3"):
            _normalize_opponent_pool_weights(cfg)

    def test_rejects_negative_entry(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP3", "OP5", "OP6")
        cfg.opponent_pool_weights = (0.5, -0.1, 0.6)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _normalize_opponent_pool_weights(cfg)

    def test_rejects_all_zero(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP3", "OP5", "OP6")
        cfg.opponent_pool_weights = (0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "sum must be > 0"):
            _normalize_opponent_pool_weights(cfg)

    def test_full_validate_normalizes_and_strips_op4(self) -> None:
        cfg = PPOConfig()
        cfg.mode = TrainMode.OPPONENT_POOL.value
        cfg.opponent_pool = ("OP3", "OP5", "OP6")
        cfg.opponent_pool_weights = (1.0, 2.0, 1.0)
        cfg = normalize_and_validate_training_config(cfg)
        self.assertEqual(cfg.opponent_pool, ("OP3", "OP5", "OP6"))
        self.assertAlmostEqual(sum(cfg.opponent_pool_weights), 1.0, places=9)
        self.assertAlmostEqual(cfg.opponent_pool_weights[1], 0.5, places=9)


class OpponentPoolWeightsSamplerTests(unittest.TestCase):
    def _make_trainer_stub(self, weights):
        env = mock.MagicMock()
        rng = mock.MagicMock(spec=np.random.Generator)
        rng.choice = mock.MagicMock(return_value="OP5")
        trainer = SimpleNamespace(
            curriculum=None,
            _opponent_randomize_training=True,
            _opponent_pool_tags=["OP3", "OP5", "OP6"],
            _opponent_pool_weights=weights,
            _rng_opponent=rng,
            env=env,
        )
        return trainer, rng, env

    def test_uniform_sampler_omits_p_kwarg(self) -> None:
        trainer, rng, _env = self._make_trainer_stub(weights=None)
        _hook_sample_training_opponent_before_reset(trainer, np.array([True]), [{}])
        rng.choice.assert_called_once()
        _, kwargs = rng.choice.call_args
        self.assertNotIn("p", kwargs)

    def test_weighted_sampler_forwards_p_kwarg(self) -> None:
        weights = [0.2, 0.5, 0.3]
        trainer, rng, _env = self._make_trainer_stub(weights=weights)
        _hook_sample_training_opponent_before_reset(trainer, np.array([True]), [{}])
        rng.choice.assert_called_once()
        _args, kwargs = rng.choice.call_args
        self.assertIn("p", kwargs)
        self.assertEqual(list(kwargs["p"]), weights)

    def test_no_sampling_when_episode_not_done(self) -> None:
        trainer, rng, _env = self._make_trainer_stub(weights=[0.2, 0.5, 0.3])
        _hook_sample_training_opponent_before_reset(trainer, np.array([False, False]), [{}, {}])
        rng.choice.assert_not_called()


if __name__ == "__main__":
    unittest.main()
