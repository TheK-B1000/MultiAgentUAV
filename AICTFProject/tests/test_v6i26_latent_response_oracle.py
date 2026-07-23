"""Tests for V6I26 Latent Response-Oracle (LRO-Summer).

Pins:
  - Preset aliases resolve to equal configs
  - Parent is v6i23; contract OFF; deep branches ON; router OFF
  - Payoff / G_available / response-target helpers
  - Distill/route refuses without niche gate

Classification: DIAGNOSTIC (not PAPER-FAITHFUL).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl.config.ppo_config import PPOConfig


def _resolve(name: str) -> PPOConfig:
    from rl.presets import PRESET_REGISTRY

    return PRESET_REGISTRY[name](PPOConfig())


class V6i26PresetResolutionTests(unittest.TestCase):
    ALIASES = [
        "v6i26_latent_response_oracle",
        "v6i26",
        "v6i26_lro",
        "latent_v6i26_latent_response_oracle",
        "plan_faithful_latent_v6i26_latent_response_oracle",
        "v6i26_phase_pod_population",
    ]

    def test_all_aliases_resolve(self) -> None:
        for alias in self.ALIASES:
            with self.subTest(alias=alias):
                self.assertIsInstance(_resolve(alias), PPOConfig)

    def test_alias_equality(self) -> None:
        configs = [asdict(_resolve(a)) for a in self.ALIASES]
        for i, alias_i in enumerate(self.ALIASES):
            for j, alias_j in enumerate(self.ALIASES):
                if i >= j:
                    continue
                self.assertEqual(configs[i], configs[j], f"{alias_i} != {alias_j}")


class V6i26ConfigContractTests(unittest.TestCase):
    def test_lro_capacity_and_birth_flags(self) -> None:
        cfg = _resolve("v6i26")
        self.assertFalse(cfg.latent_contract_specialist_enabled)
        self.assertEqual(cfg.latent_contract_specialist_coef, 0.0)
        self.assertTrue(cfg.latent_lro_deep_branches)
        self.assertTrue(cfg.latent_lro_active_branch_only)
        self.assertTrue(cfg.latent_population_birth_active_z_only)
        self.assertTrue(cfg.latent_population_birth_per_z_action_heads)
        self.assertTrue(cfg.enable_latent_z_residual)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")
        self.assertTrue(cfg.fixed_latent_strategy)
        self.assertEqual(cfg.latent_assignment_mode, "fixed")
        self.assertEqual(cfg.latent_strategy_ppo_coef, 0.0)
        self.assertFalse(cfg.train_router_when_forced)
        self.assertEqual(cfg.experiment_id, "v6i26")
        self.assertIn("v6i26", cfg.run_tag)

    def test_diff_vs_v6i23_is_intentional(self) -> None:
        v26 = asdict(_resolve("v6i26"))
        v23 = asdict(_resolve("v6i23"))
        # Allowed surface: LRO capacity + birth loop controls (contract already OFF
        # on v6i23; repertoire stage / residual / per-z heads already inherited).
        allowed = {
            "experiment_id",
            "run_tag",
            "latent_lro_deep_branches",
            "latent_lro_active_branch_only",
            "fixed_latent_strategy",
            "fixed_latent_strategy_id",
            "latent_assignment_mode",
            "latent_strategy_ppo_coef",
            "latent_episode_strategy_ppo",
            "latent_episode_strategy_coef",
            "train_router_when_forced",
            "train_router_critic_when_forced",
            "recurrent_selector_hidden_dim",
            "v6i9_training_stage",
            "freeze_return_norm_after_load",
            "opponent_randomize",
            "mode",
            "phase_pod_id",
            "latent_contract_specialist_enabled",
            "latent_contract_specialist_coef",
            "latent_z_residual_alpha",
            "enable_latent_z_residual",
            "latent_population_birth_active_z_only",
            "latent_population_birth_per_z_action_heads",
            "opponent_pool",
        }
        actual = {k for k in v26 if v26[k] != v23.get(k)}
        unexpected = actual - allowed
        self.assertFalse(
            unexpected,
            f"Unexpected config keys changed vs v6i23: {sorted(unexpected)}",
        )
        for key in (
            "latent_lro_deep_branches",
            "latent_lro_active_branch_only",
            "fixed_latent_strategy",
            "latent_assignment_mode",
            "experiment_id",
            "run_tag",
        ):
            self.assertIn(key, actual, f"missing expected LRO delta: {key}")

    def test_no_forbidden_credit_channels(self) -> None:
        cfg = _resolve("v6i26")
        self.assertEqual(cfg.latent_episode_strategy_coef, 0.0)
        self.assertFalse(cfg.latent_episode_strategy_ppo)
        self.assertIsNone(getattr(cfg, "latent_episode_strategy_lr", None))


class V6i26PayoffHelperTests(unittest.TestCase):
    def test_smoothed_mixture_caps_single_cell(self) -> None:
        from experiments.v6i26_lro_core import select_response_target

        # One context has huge raw regret; smoothing + cap must keep weight ≤ 0.35.
        payoff = np.array(
            [
                [0.9, 0.2, 0.5, 0.5],
                [0.1, 0.85, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        contexts = ["OP8|m1", "OP8|m2", "OP9|m1", "OP9|m2"]
        target = select_response_target(
            payoff,
            contexts=contexts,
            policy_labels=["a", "b"],
            episodes_per_cell=4,
            prior_strength=4.0,
            max_mixture_weight=0.35,
            aggregate_by_opponent=True,
        )
        self.assertTrue(target["smoothed"])
        for w in target["mixture_weights"].values():
            self.assertLessEqual(w, 0.35 + 1e-6)
        self.assertAlmostEqual(sum(target["mixture_weights"].values()), 1.0, places=5)

    def test_reject_diagnosis_stuck_generalist(self) -> None:
        from experiments.v6i26_lro_core import diagnose_lro_reject

        d = diagnose_lro_reject(
            branch_kl=0.002,
            niche_payoff_improvement=0.0,
            general_competence_change=0.0,
            delta_g=0.0,
        )
        self.assertEqual(d["diagnosis_code"], "STUCK_GENERALIST_BASIN")
        self.assertFalse(d["escalate_to_task_niches"])

    def test_accept_requires_delta_g_nonredundant_competence(self) -> None:
        from experiments.v6i26_lro_core import accept_lro_round

        payoff = np.array(
            [
                [0.80, 0.20, 0.55, 0.40],
                [0.25, 0.85, 0.50, 0.45],
            ],
            dtype=np.float64,
        )
        ok = accept_lro_round(
            g_before=0.0,
            g_after=0.12,
            payoff_after=payoff,
            branch_idx=1,
            competence_floor=0.30,
        )
        self.assertTrue(ok["accepted"])
        self.assertEqual(ok["verdict"], "ACCEPT")

        bad = accept_lro_round(
            g_before=0.10,
            g_after=0.10,
            payoff_after=payoff,
            branch_idx=1,
            competence_floor=0.30,
        )
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["verdict"], "REJECT")

    def test_niche_signal_when_specialists_exist(self) -> None:
        from experiments.v6i26_lro_core import (
            payoff_tensor_summary,
            select_response_target,
        )

        # Two policies; each uniquely best on one context with margin.
        payoff = np.array(
            [
                [0.80, 0.20, 0.55, 0.40],
                [0.25, 0.85, 0.50, 0.45],
            ],
            dtype=np.float64,
        )
        labels = ["pi0", "pi1"]
        contexts = ["OP11|m1", "OP12|m1", "OP11|m2", "OP12|m2"]
        summary = payoff_tensor_summary(
            payoff, policy_labels=labels, contexts=contexts, margin=0.10
        )
        self.assertGreaterEqual(summary["unique_best_count"], 2)
        self.assertGreater(summary["G_available_point"], 0.0)
        self.assertTrue(summary["niche_signal"])
        self.assertFalse(summary["parallel_rows"])
        target = select_response_target(
            payoff, contexts=contexts, policy_labels=labels
        )
        self.assertIn("mixture_weights", target)
        self.assertEqual(len(target["mixture_weights"]), 4)
        self.assertIn(target["branch_to_train_index"], (0, 1))

    def test_parallel_rows_fail_task_distribution(self) -> None:
        from experiments.v6i26_lro_core import payoff_tensor_summary

        payoff = np.array(
            [
                [0.70, 0.65, 0.60, 0.55],
                [0.71, 0.66, 0.61, 0.56],
            ],
            dtype=np.float64,
        )
        summary = payoff_tensor_summary(
            payoff,
            policy_labels=["a", "b"],
            contexts=["c0", "c1", "c2", "c3"],
            margin=0.10,
        )
        self.assertLessEqual(summary["unique_best_count"], 1)
        self.assertFalse(summary["niche_signal"])
        self.assertTrue(summary["parallel_rows"])


class V6i26DistillGateTests(unittest.TestCase):
    def test_distill_refuses_without_niche(self) -> None:
        from experiments.run_v6i26_distill_and_route import _gate_allows

        ok, reason = _gate_allows({"summary": {"niche_signal": False}})
        self.assertFalse(ok)
        self.assertIn("no niche", reason)

    def test_distill_allows_niche_signal(self) -> None:
        from experiments.run_v6i26_distill_and_route import _gate_allows

        ok, reason = _gate_allows({"summary": {"niche_signal": True}})
        self.assertTrue(ok)
        self.assertIn("niche_signal", reason)


class V6i26DeepBranchArchitectureTests(unittest.TestCase):
    def test_latent_branch_trunks_created_when_flag_set(self) -> None:
        import torch
        from rl.latent_marl import LatentConditionedActor

        actor = LatentConditionedActor(
            local_feature_dim=16,
            latent_k=4,
            action_dim=5,
            z_embed_dim=8,
            hidden_dim=32,
            enable_latent_z_residual=True,
            latent_population_birth_per_z_action_heads=True,
            latent_lro_deep_branches=True,
        )
        self.assertIsNotNone(actor.latent_branch_trunks)
        self.assertEqual(len(actor.latent_branch_trunks), 4)
        z = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        obs = torch.randn(4, 16)
        logits = actor(obs, z_idx=z)
        self.assertEqual(tuple(logits.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
