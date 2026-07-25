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
            episodes_per_cell=32,
            ci95_low_delta_g=0.01,
            training_seed_count=3,
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

    def test_four_episode_positive_screen_is_promising_not_accept(self) -> None:
        from experiments.v6i26_lro_core import accept_lro_round

        payoff = np.array(
            [
                [0.80, 0.20, 0.55, 0.40],
                [0.25, 0.85, 0.50, 0.45],
            ],
            dtype=np.float64,
        )
        result = accept_lro_round(
            g_before=0.0,
            g_after=0.12,
            payoff_after=payoff,
            branch_idx=1,
            competence_floor=0.30,
            behavior_distinctness={"branch_behavior_nonredundant": True},
            require_behavior_distinctness=True,
            episodes_per_cell=4,
            ci95_low_delta_g=None,
            training_seed_count=1,
        )
        self.assertTrue(result["screening_pass"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["verdict"], "PROMISING_DIRECTION")
        self.assertFalse(result["confirmation_episode_count_pass"])
        self.assertFalse(result["ci95_delta_G_gt_0"])
        self.assertFalse(result["multi_seed_repetition_pass"])

    def test_current_response_selector_excludes_saturated_contexts(self) -> None:
        from experiments.v6i26_lro_core import select_current_response_target

        payoff = np.array(
            [
                [0.95, 0.40, 0.78, 0.70],
                [0.96, 0.45, 0.80, 0.68],
                [0.94, 0.48, 0.82, 0.72],
                [0.97, 0.49, 0.81, 0.74],
            ],
            dtype=np.float64,
        )
        contexts = ["OP8|m1", "OP9|m1", "OP10|m1", "OP11|m1"]
        target = select_current_response_target(
            payoff,
            contexts=contexts,
            policy_labels=["z0", "z1", "z2", "z3"],
            saturation_cutoff=0.90,
            target_fraction=0.75,
            competence_floor=0.70,
        )

        self.assertEqual(target["selection_basis"], "current_forced_z_payoff")
        self.assertEqual(target["target_context"], "OP9|m1")
        self.assertIn("OP8|m1", {c["context"] for c in target["excluded_saturated_contexts"]})
        self.assertNotEqual(
            target["branch_to_train_index"],
            target["current_best_z_on_target"],
        )
        self.assertIn(target["branch_to_train_index"], target["stable_branch_indices"])
        self.assertAlmostEqual(sum(target["mixture_weights"].values()), 1.0, places=6)
        target_mass = sum(
            target["mixture_weights"][ctx] for ctx in target["target_contexts"]
        )
        anchor_mass = sum(
            target["mixture_weights"][ctx] for ctx in target["anchor_contexts"]
        )
        self.assertAlmostEqual(target_mass, 0.75, places=6)
        self.assertAlmostEqual(anchor_mass, 0.25, places=6)

    def test_current_response_selector_falls_back_when_all_cells_saturated(self) -> None:
        from experiments.v6i26_lro_core import select_current_response_target

        payoff = np.array(
            [
                [0.91, 0.95],
                [0.93, 0.94],
                [0.92, 0.96],
                [0.94, 0.97],
            ],
            dtype=np.float64,
        )
        target = select_current_response_target(
            payoff,
            contexts=["OP8|m1", "OP9|m1"],
            policy_labels=["z0", "z1", "z2", "z3"],
            saturation_cutoff=0.90,
        )

        self.assertEqual(target["target_context"], "OP8|m1")
        self.assertEqual(len(target["excluded_saturated_contexts"]), 1)

    def test_behavior_distinctness_flags_duplicate_branch(self) -> None:
        from experiments.v6i26_lro_core import behavior_distinctness_summary
        from rl.forced_z_behavior_vectors import FORCED_Z_BEHAVIOR_VECTOR_NAMES

        names = FORCED_Z_BEHAVIOR_VECTOR_NAMES

        def vec(values: tuple[float, ...]) -> dict[str, float]:
            return {name: float(values[i]) for i, name in enumerate(names)}

        report = {
            "forced_z_behavior_vector_names": list(names),
            "per_z_behavior_vectors": {
                "z0": vec((0.05, 0.90, 0.10, 0.10, 0.10, 0.10, 0.10)),
                "z1": vec((0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20)),
                "z2": vec((0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20)),
                "z3": vec((0.90, 0.05, 0.90, 0.90, 1.20, 0.90, 0.90)),
            },
        }

        duplicate = behavior_distinctness_summary(
            report,
            branch_idx=2,
            min_branch_distance=0.20,
        )
        self.assertFalse(duplicate["branch_behavior_nonredundant"])
        self.assertEqual(duplicate["branch_nearest_behavior_neighbor"], 1)

        distinct = behavior_distinctness_summary(
            report,
            branch_idx=3,
            min_branch_distance=0.20,
        )
        self.assertTrue(distinct["branch_behavior_nonredundant"])
        self.assertEqual(distinct["verdict"], "BEHAVIOR_DISTINCT_PASS")

    def test_accept_can_require_behavior_distinctness(self) -> None:
        from experiments.v6i26_lro_core import accept_lro_round

        payoff = np.array(
            [
                [0.80, 0.20, 0.55, 0.40],
                [0.25, 0.85, 0.50, 0.45],
            ],
            dtype=np.float64,
        )
        rejected = accept_lro_round(
            g_before=0.0,
            g_after=0.12,
            payoff_after=payoff,
            branch_idx=1,
            competence_floor=0.30,
            behavior_distinctness={"branch_behavior_nonredundant": False},
            require_behavior_distinctness=True,
        )
        self.assertFalse(rejected["accepted"])
        self.assertFalse(rejected["behavior_distinctness_pass"])
        self.assertEqual(rejected["verdict"], "REJECT")

        accepted = accept_lro_round(
            g_before=0.0,
            g_after=0.12,
            payoff_after=payoff,
            branch_idx=1,
            competence_floor=0.30,
            behavior_distinctness={"branch_behavior_nonredundant": True},
            require_behavior_distinctness=True,
            episodes_per_cell=32,
            ci95_low_delta_g=0.01,
            training_seed_count=3,
        )
        self.assertTrue(accepted["accepted"])
        self.assertTrue(accepted["behavior_distinctness_pass"])

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

    def test_distill_requires_strategy_pass_when_present(self) -> None:
        from experiments.run_v6i26_distill_and_route import _gate_allows

        ok, reason = _gate_allows(
            {
                "phase2_strategy_verdict": "PHASE2_STRATEGY_HOLD_OR_FAIL",
                "summary": {"niche_signal": True},
            }
        )
        self.assertFalse(ok)
        self.assertIn("phase2_strategy_verdict", reason)

        ok, reason = _gate_allows(
            {"phase2_strategy_verdict": "PHASE2_STRATEGY_PASS"}
        )
        self.assertTrue(ok)
        self.assertIn("PHASE2_STRATEGY_PASS", reason)


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

    def test_latent_branch_trunks_identity_sync_preserves_logits(self) -> None:
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
        actor.sync_per_z_action_heads_from_shared()
        actor._residual_bypass_for_compat = True
        obs = torch.randn(8, 16)
        z = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        shared_logits = actor(obs, z_idx=z)

        actor._residual_bypass_for_compat = False
        actor.sync_latent_branch_trunks_to_identity()
        branched_logits = actor(obs, z_idx=z)

        self.assertTrue(torch.allclose(branched_logits, shared_logits, atol=1e-6))


class V6i26MarginHeadroomTargetTests(unittest.TestCase):
    def test_selects_recoverable_margin_not_tiny_gap(self) -> None:
        from experiments.v6i26_lro_core import (
            calibrate_margin_headroom_threshold,
            select_margin_response_target,
        )

        # Context0: big recoverable gap for z0 vs best z3
        # Context1: tiny gap only
        winrate = np.array(
            [
                [0.90, 0.95],
                [0.95, 0.95],
                [0.95, 0.95],
                [0.96, 0.96],
            ],
            dtype=np.float64,
        )
        margin = np.array(
            [
                [0.65, 1.38],
                [1.00, 1.40],
                [1.10, 1.42],
                [1.45, 1.45],
            ],
            dtype=np.float64,
        )
        calib = calibrate_margin_headroom_threshold(
            np.full_like(margin, 0.4),
            n_episodes=32,
            se_multiplier=2.0,
            absolute_floor=0.15,
        )
        target = select_margin_response_target(
            winrate,
            margin,
            contexts=["OP_A|map_a", "OP_B|map_a"],
            policy_labels=["z0", "z1", "z2", "z3"],
            min_margin_headroom=float(calib["min_margin_headroom"]),
            wr_competence_floor=0.75,
            branch_wr_floor=0.50,
            target_fraction=0.75,
        )
        self.assertEqual(target["selection_metric"], "win_margin")
        self.assertTrue(target["selection_viable"])
        self.assertEqual(target["target_context"], "OP_A|map_a")
        self.assertEqual(target["branch_to_train_index"], 0)
        self.assertAlmostEqual(target["target_sensitive_headroom"], 0.80, places=5)
        self.assertNotEqual(target["branch_to_train_index"], target["current_best_z_on_target"])
        self.assertGreaterEqual(target["target_best_wr"], 0.75)

    def test_calibration_uses_se_floor(self) -> None:
        from experiments.v6i26_lro_core import calibrate_margin_headroom_threshold

        std = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float64)
        calib = calibrate_margin_headroom_threshold(
            std, n_episodes=32, se_multiplier=2.0, absolute_floor=0.15
        )
        # SE = 0.2/sqrt(32)≈0.035; 2*SE≈0.07 < floor 0.15
        self.assertAlmostEqual(calib["min_margin_headroom"], 0.15, places=5)


class V6i26CurrentPayoffTargetTests(unittest.TestCase):
    def test_excludes_saturated_and_avoids_best_branch(self) -> None:
        from experiments.v6i26_lro_core import select_current_response_target

        # winrate matrix: z3 saturates most cells; uncovered OP11|map_a.
        winrate = np.array(
            [
                [0.50, 0.55, 0.40, 0.95],
                [0.60, 0.58, 0.35, 0.96],
                [0.70, 0.72, 0.30, 0.97],
                [0.95, 0.96, 0.20, 0.98],
            ],
            dtype=np.float64,
        )
        contexts = [
            "OP8|map_a",
            "OP10|map_a",
            "OP11|map_a",
            "OP12|map_a",
        ]
        labels = ["z0", "z1", "z2", "z3"]
        target = select_current_response_target(
            winrate,
            contexts=contexts,
            policy_labels=labels,
            saturation_cutoff=0.90,
            target_fraction=0.75,
        )
        self.assertEqual(target["selection_basis"], "current_forced_z_payoff")
        self.assertEqual(target["target_context"], "OP11|map_a")
        self.assertEqual(target["current_best_z_on_target"], 0)
        self.assertNotEqual(
            target["branch_to_train_index"], target["current_best_z_on_target"]
        )
        excluded = {row["context"] for row in target["excluded_saturated_contexts"]}
        self.assertIn("OP8|map_a", excluded)
        self.assertIn("OP10|map_a", excluded)
        self.assertIn("OP12|map_a", excluded)
        self.assertNotIn("OP11|map_a", excluded)

    def test_learning_signal_flags_tiny_kl(self) -> None:
        from experiments.v6i26_lro_core import (
            summarize_training_learning_signal,
            write_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            path.write_text(
                "approx_kl,clip_fraction,entropy,explained_variance,value_loss,grad_norm,"
                "latent_episode_adv_mean,latent_episode_adv_std\n"
                "1e-6,0.0,2.5,0.2,0.4,1e-5,0.01,0.5\n"
                "2e-6,0.0,2.4,0.3,0.3,2e-5,0.02,0.4\n",
                encoding="utf-8",
            )
            summary = summarize_training_learning_signal(path)
            self.assertEqual(summary["status"], "NO_USABLE_LEARNING_PRESSURE")
            self.assertTrue(summary["flags"]["tiny_approx_kl"])
            write_json(Path(tmp) / "learning_signal.json", summary)


class V6i26BranchKLLogitsTests(unittest.TestCase):
    def test_distribution_logits_handles_multihead_callable_list(self) -> None:
        import torch
        from experiments.run_v6i26_lro_oracle_round import _distribution_logits

        class _Head:
            def __init__(self, logits: torch.Tensor) -> None:
                self.logits = logits

        class _MultiHead:
            def __init__(self) -> None:
                self.heads = [_Head(torch.zeros(2, 3)), _Head(torch.ones(2, 4))]

            def logits(self):
                return [h.logits for h in self.heads]

        out = _distribution_logits(_MultiHead())
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(tuple(out.shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
