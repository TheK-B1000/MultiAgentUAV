"""Pins the v6i10 episode-router exploration preset."""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.latent.behavior_policy import behavior_log_prob_from_probs, epsilon_behavior_probs
from rl.presets import apply_preset


PARENT = "v6i9_mapaware_router_feedforward_hardpool"
PRESET = "v6i10_episode_router_explore_hardpool"


def _resolved(name: str) -> dict:
    return dataclasses.asdict(apply_preset(PPOConfig(), name))


class V6I10PresetTests(unittest.TestCase):
    def test_aliases_resolve_identically(self) -> None:
        base = _resolved(PRESET)
        for alias in (
            "v6i10",
            "v6i10_episode_router_explore",
            "latent_v6i10_episode_router_explore_hardpool",
            "plan_faithful_latent_v6i10_episode_router_explore_hardpool",
        ):
            self.assertEqual(_resolved(alias), base)

    def test_episode_router_knobs(self) -> None:
        cfg = _resolved(PRESET)
        self.assertEqual(cfg["experiment_id"], "v6i10")
        self.assertEqual(cfg["v6i9_training_stage"], "router")
        self.assertTrue(cfg["router_freeze_actor"])
        self.assertEqual(cfg["recurrent_selector_hidden_dim"], 0)
        self.assertEqual(cfg["latent_resample_every_n"], 0)
        self.assertEqual(cfg["strategy_interval"], 0)
        self.assertEqual(cfg["latent_lam_p"], 0.0)
        self.assertEqual(cfg["latent_strategy_ppo_coef"], 0.0)
        self.assertEqual(cfg["learning_rate"], 1e-4)
        self.assertEqual(cfg["router_uniform_exploration_prob"], 0.20)

    def test_label_free_extension_contract(self) -> None:
        cfg = _resolved(PRESET)
        self.assertEqual(cfg["latent_assignment_mode"], "router")
        self.assertEqual(cfg["latent_forced_z_episode_frac"], 0.0)
        self.assertIsNone(cfg["latent_episode_strategy_lr"])
        self.assertFalse(cfg["latent_strategy_aux_return_head"])
        self.assertEqual(cfg["latent_strategy_aux_predict_phase_coef"], 0.0)
        self.assertFalse(cfg["latent_router_distill_enabled"])
        self.assertFalse(cfg["latent_v3i3_event_preference_enabled"])
        self.assertEqual(cfg["latent_preference_coef"], 0.0)
        self.assertFalse(cfg["latent_awrd_enabled"])

    def test_credit_and_marginal_coverage_contract(self) -> None:
        cfg = _resolved(PRESET)
        self.assertTrue(cfg["latent_arc_credit_enabled"])
        self.assertEqual(cfg["latent_arc_credit_baseline"], "running_mean")
        self.assertEqual(cfg["latent_arc_credit_min_len"], 1)
        self.assertEqual(cfg["latent_entropy_mode"], "marginal")
        self.assertEqual(cfg["latent_entropy_objective"], "maximize")
        self.assertEqual(cfg["latent_lam_h"], 0.015)
        self.assertEqual(cfg["router_ent_coef"], 0.002)

    def test_parent_diff_is_explicit(self) -> None:
        parent = _resolved(PARENT)
        cfg = _resolved(PRESET)
        diff = {key for key in set(parent) | set(cfg) if parent.get(key) != cfg.get(key)}
        self.assertEqual(
            diff,
            {
                "experiment_id",
                "h_mode",
                "latent_arc_credit_baseline",
                "latent_arc_credit_enabled",
                "latent_arc_credit_min_len",
                "latent_entropy_anneal_end",
                "latent_entropy_anneal_start",
                "latent_entropy_mode",
                "latent_entropy_objective",
                "latent_lam_h",
                "latent_lam_h_end",
                "latent_lam_p",
                "latent_resample_every_n",
                "latent_strategy_ppo_coef",
                "learning_rate",
                "router_ent_coef",
                "router_uniform_exploration_prob",
                "run_tag",
                "strategy_interval",
            },
        )


class RouterExplorationMixtureTests(unittest.TestCase):
    def test_behavior_distribution_matches_requested_mixture(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0, -2.0]], dtype=torch.float32)
        router_probs = torch.softmax(logits, dim=-1)
        behavior_probs = epsilon_behavior_probs(router_probs, epsilon=0.20, latent_k=4)
        expected = 0.80 * router_probs + 0.20 * torch.full_like(router_probs, 0.25)
        torch.testing.assert_close(behavior_probs, expected)
        executed = torch.tensor([3])
        torch.testing.assert_close(
            behavior_log_prob_from_probs(behavior_probs, executed),
            torch.log(expected[:, 3]),
        )

    def test_preset_value_is_available_to_runtime_cfg(self) -> None:
        cfg = SimpleNamespace(router_uniform_exploration_prob=_resolved(PRESET)["router_uniform_exploration_prob"])
        eps = max(0.0, min(1.0, float(getattr(cfg, "router_uniform_exploration_prob", 0.0))))
        self.assertEqual(eps, 0.20)


if __name__ == "__main__":
    unittest.main()
