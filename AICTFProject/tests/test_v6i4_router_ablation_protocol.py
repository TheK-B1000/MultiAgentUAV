"""Pins v6i4 as evaluation-only Summer-plan router ablation protocol."""

from __future__ import annotations

import unittest
from dataclasses import asdict

from rl.config.ppo_config import PPOConfig
from rl.evaluation.router_ablation import V6I4_CLASSIFICATION
from rl.presets import PRESET_REGISTRY, apply_preset
from rl.train_ppo import train_ppo


class V6I4RouterAblationProtocolTests(unittest.TestCase):
    def test_v6i4_inherits_v6i2_training_contract_and_only_adds_eval_metadata(self) -> None:
        v6i2 = asdict(apply_preset(PPOConfig(), "v6i2"))
        v6i4 = asdict(apply_preset(PPOConfig(), "v6i4"))
        changed = {k for k in v6i2 if v6i2[k] != v6i4[k]}
        allowed = {
            "experiment_id",
            "run_tag",
            "evaluation_only_preset",
            "evaluation_only_runner",
            "evaluation_only_requires_checkpoint",
            "evaluation_only_checkpoint_family",
            "router_ablation_protocol_version",
            "router_ablation_claim_label",
            "router_ablation_classification",
            "router_ablation_conditions",
            "router_ablation_oracle_conditions",
            "router_ablation_primary_metrics",
            "router_ablation_diagnostic_metrics",
            "router_ablation_opponents",
            "router_ablation_calibration_seed_set",
            "router_ablation_evaluation_seed_set",
        }
        self.assertEqual(changed, allowed)
        for key in (
            "use_latent_strategy",
            "latent_k",
            "latent_z_embed_dim",
            "latent_actor_conditioning",
            "enable_actor_z_film",
            "latent_actor_z_adapter_enabled",
            "latent_strategy_hidden",
            "latent_vf_hidden",
            "reward_scale",
            "reward_dense_weight",
            "opponent_pool",
            "opponent_randomize",
            "map_layout",
            "max_decision_steps",
            "gate_protocol_version",
        ):
            if key in v6i2 and key in v6i4:
                self.assertEqual(v6i4[key], v6i2[key], key)

    def test_v6i4_manifest_locks_conditions_and_oracle_labels(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i4")
        self.assertEqual(cfg.experiment_id, "v6i4")
        self.assertTrue(cfg.evaluation_only_preset)
        self.assertTrue(cfg.evaluation_only_requires_checkpoint)
        self.assertEqual(cfg.evaluation_only_checkpoint_family, "promoted_v6i2")
        self.assertEqual(
            cfg.router_ablation_classification,
            V6I4_CLASSIFICATION,
        )
        self.assertEqual(
            cfg.router_ablation_conditions,
            (
                "learned_qphi_switching",
                "uniform_episode_fixed",
                "uniform_random_at_router_opportunities",
                "preselected_global_fixed_z",
                "fixed_z0",
                "fixed_z1",
                "fixed_z2",
                "fixed_z3",
                "qphi_initial_only_no_switch",
                "shuffled_qphi_outputs",
            ),
        )
        self.assertEqual(
            cfg.router_ablation_oracle_conditions,
            (
                "posthoc_global_fixed_oracle",
                "posthoc_opponent_oracle",
                "posthoc_episode_oracle",
            ),
        )
        self.assertFalse(cfg.router_ablation_episode_oracle_is_deployable)
        self.assertEqual(cfg.router_ablation_opponents, ("OP5", "OP6", "OP7"))

    def test_v6i4_aliases_resolve(self) -> None:
        for alias in (
            "v6i4",
            "v6i4_router_ablation",
            "v6i4_router_ablation_protocol",
            "latent_v6i4_router_ablation_protocol",
            "plan_faithful_latent_v6i4_router_ablation_protocol",
        ):
            self.assertIn(alias, PRESET_REGISTRY)
            self.assertTrue(apply_preset(PPOConfig(), alias).evaluation_only_preset)

    def test_train_ppo_rejects_v6i4_before_training(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i4")
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            train_ppo(cfg)


if __name__ == "__main__":
    unittest.main()
