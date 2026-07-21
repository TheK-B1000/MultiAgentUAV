"""Pinning tests for V6I13 delayed-commit opening-window router."""
from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.run_v6i13_opening_window_advantage_router import (
    _CONTEXT_DIM,
    _OPENING_STATE_DIM,
    build_opening_context_from_record,
)
from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from tests.test_latent_episode_warmup import LatentStrategyState, _make_trainer


class V6i13PresetContractTests(unittest.TestCase):
    _ALIASES = [
        "v6i13",
        "v6i13_opening_window_advantage_router",
        "v6i13_opening_window",
        "v6i13_advantage_router",
        "latent_v6i13_opening_window_advantage_router",
        "plan_faithful_latent_v6i13_opening_window_advantage_router",
    ]

    def test_aliases_resolve_equal(self) -> None:
        base = dataclasses.asdict(apply_preset(PPOConfig(), self._ALIASES[0]))
        for alias in self._ALIASES[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_diff_vs_v6i12_is_delayed_commit_only(self) -> None:
        parent = dataclasses.asdict(apply_preset(PPOConfig(), "v6i12"))
        cfg = dataclasses.asdict(apply_preset(PPOConfig(), "v6i13"))
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "latent_episode_strategy_warmup_decision_steps",
                "router_arc_post_commit_only",
                "router_opening_context_mode",
                "router_warmup_uniform_z",
                "run_tag",
            },
        )

    def test_runtime_contract(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i13")
        self.assertEqual(cfg.experiment_id, "v6i13")
        self.assertEqual(cfg.latent_episode_strategy_warmup_decision_steps, 32)
        self.assertTrue(cfg.router_warmup_uniform_z)
        self.assertTrue(cfg.router_arc_post_commit_only)
        self.assertEqual(cfg.router_opening_context_mode, "initial_commit_delta")
        self.assertTrue(cfg.latent_arc_credit_enabled)
        self.assertEqual(float(cfg.latent_arc_credit_coef), 0.0)
        self.assertEqual(float(cfg.router_uniform_exploration_prob), 0.5)


class V6i13OpeningContextTests(unittest.TestCase):
    def test_build_opening_context_uses_record_field_and_opponent_onehot(self) -> None:
        opening = torch.arange(_OPENING_STATE_DIM, dtype=torch.float32)
        ctx = build_opening_context_from_record(
            {"opening_context": opening, "global_state_0": torch.zeros(34), "opponent_id": 8}
        )
        self.assertEqual(tuple(ctx.shape), (_CONTEXT_DIM,))
        self.assertTrue(torch.equal(ctx[:_OPENING_STATE_DIM], opening))
        self.assertEqual(ctx[-3:].tolist(), [0.0, 1.0, 0.0])

    def test_fallback_context_is_state_state_zero_delta(self) -> None:
        gs = torch.arange(34, dtype=torch.float32)
        ctx = build_opening_context_from_record({"global_state_0": gs, "opponent_id": -1})
        self.assertTrue(torch.equal(ctx[:34], gs))
        self.assertTrue(torch.equal(ctx[34:68], gs))
        self.assertTrue(torch.equal(ctx[68:102], torch.zeros(34)))
        self.assertEqual(float(ctx[-3:].sum().item()), 0.0)


class V6i13DelayedArcLifecycleTests(unittest.TestCase):
    def _state(self, z_signal: int, dim: int = 4) -> torch.Tensor:
        s = torch.zeros((1, dim), dtype=torch.float32)
        s[0, 0] = float(z_signal)
        s[0, 1:] = torch.tensor([0.1, 0.2, 0.3])[: dim - 1]
        return s

    def test_warmup_uniform_z_and_post_commit_arc_only(self) -> None:
        warmup = 2
        trainer = _make_trainer(
            1,
            warmup=warmup,
            episode_credit=False,
            resample_every_n=0,
            gs_dim=4,
        )
        trainer.cfg.router_warmup_uniform_z = True
        trainer.cfg.router_arc_post_commit_only = True
        trainer.cfg.router_opening_context_mode = "initial_commit_delta"
        trainer.latent_arc_credit_enabled = True
        trainer.latent_arc_credit_min_len = 1
        trainer.latent_arc_credit_baseline = "running_mean"
        trainer.latent_arc_credit_coef = 0.0
        trainer.latent_arc_credit_return_norm = True
        trainer.latent_arc_credit_n_epochs = 1
        trainer.latent_arc_credit_clip_eps = 0.2
        trainer.selector_memory = SimpleNamespace(reset_rows=lambda _mask: None)

        ls = LatentStrategyState(trainer)
        ls.reset()

        z0, _, aux0 = ls.strategy_for_step(self._state(3))
        self.assertEqual(int(z0.item()), 0)  # uniform logits + fake argmax -> z0
        self.assertFalse(bool(aux0["z_resampled"].item()))
        self.assertFalse(bool(ls.arc_has_open.item()), "warmup arc must not open at reset")

        ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._state(1))
        self.assertFalse(bool(ls.arc_has_open.item()))

        ls.mark_strategy_step_done(np.array([False]))
        z_commit, _, aux_commit = ls.strategy_for_step(self._state(2))
        self.assertEqual(int(z_commit.item()), 2)
        self.assertTrue(bool(aux_commit["z_resampled"].item()))
        self.assertTrue(bool(ls.arc_has_open.item()))
        self.assertEqual(int(ls.arc_open_commit_step.item()), warmup)

        ls.arc_accumulate_step(torch.tensor([1.25]))
        pushed = ls.arc_finalize(
            torch.tensor([True]),
            reason="episode_end",
            opponent_ids=torch.tensor([7]),
        )
        self.assertEqual(pushed, 1)
        self.assertEqual(len(ls.rollout_strategy_arc_records), 1)
        rec = ls.rollout_strategy_arc_records[0]
        self.assertEqual(rec["reason"], "episode_end")
        self.assertEqual(rec["commit_step"], warmup)
        self.assertEqual(rec["arc_length"], 1)
        self.assertAlmostEqual(rec["arc_return"], 1.25)
        self.assertEqual(tuple(rec["opening_context"].shape), (12,))
        self.assertAlmostEqual(float(rec["opening_context"][0]), 3.0)
        self.assertAlmostEqual(float(rec["opening_context"][4]), 2.0)
        self.assertAlmostEqual(float(rec["opening_context"][8]), -1.0)


if __name__ == "__main__":
    unittest.main()
