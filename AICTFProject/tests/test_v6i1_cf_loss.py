"""Tests for V6I1 competence-weighted counterfactual separation loss."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from rl.custom_ppo.trainer_optimizers import collect_actor_optimizer_parameters
from rl.custom_ppo.v6i1_cf_loss import (
    actor_cf_grad_norm,
    actor_cf_ppo_grad_diagnostics,
    actor_diagnostic_grad_norm,
    extract_forced_z_pair_values,
    forced_z_pairwise_profile_available,
    global_grad_norm,
    v6i1_cf_separation_loss,
)


class _CfModel(nn.Module):
    n_agents = 1
    per_agent_action_dims = (3,)

    def policy_logits(self, obs, z_idx=None, *, detach_local_features=False):
        z = z_idx.long().reshape(-1).clamp(min=0, max=3)
        logits = torch.zeros((int(z.shape[0]), 3), dtype=torch.float32)
        logits[z == 0] = torch.tensor([4.0, -4.0, -4.0])
        logits[z == 1] = torch.tensor([-4.0, 4.0, -4.0])
        logits[z == 2] = torch.tensor([4.0, -4.0, -4.0])
        logits[z == 3] = torch.tensor([4.0, -4.0, -4.0])
        return logits

    @staticmethod
    def _mask_logits(logits, mask):
        return logits


class _TrainableZLogitsModel(nn.Module):
    """Per-z categorical logits as a trainable ``latent_actor`` parameter."""

    n_agents = 1
    per_agent_action_dims = (3,)

    def __init__(self, z_logits: list[list[float]]) -> None:
        super().__init__()
        self.latent_actor = nn.Parameter(torch.tensor(z_logits, dtype=torch.float32))

    def policy_logits(self, obs, z_idx=None, *, detach_local_features=False):
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_actor.shape[0] - 1)
        return self.latent_actor[z]

    @staticmethod
    def _mask_logits(logits, mask):
        return logits


class _IdenticalZLogitsModel(_TrainableZLogitsModel):
    def __init__(self) -> None:
        same = [4.0, -4.0, -4.0]
        super().__init__([same, same, same, same])


class _SlightPerturbZLogitsModel(_TrainableZLogitsModel):
    def __init__(self) -> None:
        base = [4.0, -4.0, -4.0]
        perturbed = [4.05, -4.05, -4.0]
        super().__init__([base, perturbed, base, base])


class _AboveMarginZLogitsModel(_TrainableZLogitsModel):
    def __init__(self) -> None:
        super().__init__(
            [
                [8.0, -8.0, -8.0],
                [-8.0, 8.0, -8.0],
                [-8.0, -8.0, 8.0],
                [-8.0, 8.0, 8.0],
            ]
        )


class V6I1CfLossTests(unittest.TestCase):
    @staticmethod
    def _actor_params(model: nn.Module) -> list[nn.Parameter]:
        return [model.latent_actor]

    def test_pairwise_profile_requires_all_six_keys(self) -> None:
        self.assertFalse(
            forced_z_pairwise_profile_available({"forced_z_macro_jsd_mean": 0.02})
        )
        profile = {f"forced_z_pair_jsd_{i}": 0.0 for i in range(6)}
        self.assertTrue(forced_z_pairwise_profile_available(profile))

    def test_competence_weights_reduce_penalty_on_collapsed_pairs(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        competence = np.array([1.0, 1.0, 0.01, 0.01], dtype=np.float32)
        loss_weighted, _ = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=competence,
            competence_ready=True,
        )
        loss_uniform, _ = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=True,
        )
        self.assertLess(float(loss_weighted.item()), float(loss_uniform.item()))

    def test_partial_collapse_stays_positive(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        loss, stats = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        self.assertGreater(float(loss.item()), 0.0)
        self.assertLess(float(stats["min_jsd"].item()), 0.02)


    def test_cf_loss_exports_per_pair_batch_stats(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        _, stats = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        self.assertEqual(int(stats["pair_jsd"].numel()), 6)
        self.assertGreater(float(stats["pairs_below_margin"].item()), 0.0)
        self.assertEqual(float(stats["cf_hinge_active"].item()), 1.0)

    def test_identical_distributions_hinge_active_grad_may_be_zero(self) -> None:
        obs = {"mask": torch.ones((4, 3), dtype=torch.float32)}
        model = _IdenticalZLogitsModel()
        loss, stats = v6i1_cf_separation_loss(
            model,
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        self.assertEqual(float(stats["cf_hinge_active"].item()), 1.0)
        self.assertGreater(float(stats["pairs_below_margin"].item()), 0.0)
        self.assertAlmostEqual(float(stats["min_jsd"].item()), 0.0, places=5)
        grad_norm = actor_cf_grad_norm(loss, self._actor_params(model))
        self.assertEqual(grad_norm, 0.0)

    def test_slightly_separated_below_margin_has_nonzero_grad(self) -> None:
        obs = {"mask": torch.ones((4, 3), dtype=torch.float32)}
        model = _SlightPerturbZLogitsModel()
        loss, stats = v6i1_cf_separation_loss(
            model,
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        self.assertEqual(float(stats["cf_hinge_active"].item()), 1.0)
        self.assertEqual(float(stats["cf_hinge_effective"].item()), 1.0)
        self.assertGreater(float(loss.item()), 0.0)
        grad_norm = actor_cf_grad_norm(loss, self._actor_params(model))
        self.assertGreater(grad_norm, 0.0)

    def test_all_pairs_above_margin_hinge_inactive_zero_grad(self) -> None:
        obs = {"mask": torch.ones((4, 3), dtype=torch.float32)}
        model = _AboveMarginZLogitsModel()
        loss, stats = v6i1_cf_separation_loss(
            model,
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        self.assertEqual(float(stats["cf_hinge_active"].item()), 0.0)
        self.assertEqual(float(stats["pairs_below_margin"].item()), 0.0)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        grad_norm = actor_cf_grad_norm(loss, self._actor_params(model))
        self.assertEqual(grad_norm, 0.0)

    def test_zero_competence_weights_make_hinge_ineffective(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        competence = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        loss, stats = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=competence,
            competence_ready=True,
        )
        self.assertEqual(float(stats["cf_hinge_active"].item()), 1.0)
        self.assertEqual(float(stats["cf_hinge_effective"].item()), 0.0)
        self.assertAlmostEqual(float(stats["cf_weight_sum"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_cf_diag_exports_requested_telemetry_keys(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        model = _SlightPerturbZLogitsModel()
        loss, stats = v6i1_cf_separation_loss(
            model,
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        for key in (
            "cf_valid_team_groups",
            "cf_weight_sum",
            "cf_effective_pairs",
            "min_jsd",
            "max_jsd",
            "cf_hinge_effective",
        ):
            self.assertIn(key, stats)
        self.assertGreater(float(stats["cf_valid_team_groups"].item()), 0.0)
        self.assertTrue(bool(loss.requires_grad))

    def test_actor_cf_grad_norm_uses_autograd_grad_without_populating_grad(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        model = _SlightPerturbZLogitsModel()
        loss, _ = v6i1_cf_separation_loss(
            model,
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        actor_params = self._actor_params(model)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        scaled_cf_loss = 0.01 * loss
        grad_norm = actor_cf_grad_norm(scaled_cf_loss, actor_params)
        self.assertGreater(grad_norm, 0.0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        scaled_cf_loss.backward(retain_graph=True)
        self.assertIsNotNone(model.latent_actor.grad)
        model.zero_grad(set_to_none=True)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class ActorGradDiagnosticTests(unittest.TestCase):
    def test_ppo_diagnostic_leaves_param_grad_unchanged(self) -> None:
        actor = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        ppo_loss = (actor**2).sum()
        self.assertIsNone(actor.grad)
        norm = actor_diagnostic_grad_norm(ppo_loss, [actor])
        self.assertGreater(norm, 0.0)
        self.assertIsNone(actor.grad)

    def test_cf_norm_uses_coefficient_scaled_loss(self) -> None:
        actor = nn.Parameter(torch.tensor(2.0, requires_grad=True))
        cf_loss = 4.0 * actor
        params = [actor]
        full_norm = actor_cf_grad_norm(cf_loss, params)
        scaled_norm = actor_cf_grad_norm(0.5 * cf_loss, params)
        self.assertAlmostEqual(full_norm, 4.0)
        self.assertAlmostEqual(scaled_norm, 2.0)

    def test_ppo_norm_excludes_cf_loss(self) -> None:
        actor = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        ppo_loss = actor[0]
        cf_loss = actor[1]
        actor_params = [actor]
        ppo_only = actor_diagnostic_grad_norm(ppo_loss, actor_params)
        combined = actor_diagnostic_grad_norm(ppo_loss + cf_loss, actor_params)
        self.assertAlmostEqual(ppo_only, 1.0)
        self.assertAlmostEqual(combined, 2.0**0.5)
        self.assertNotAlmostEqual(ppo_only, combined)

    def test_ppo_norm_excludes_critic_and_router_parameters(self) -> None:
        model = nn.Module()
        model.latent_actor = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        model.critic = nn.Parameter(torch.tensor([3.0, 4.0], requires_grad=True))
        model.strategy_encoder = nn.Parameter(torch.tensor([5.0], requires_grad=True))
        actor_params = [model.latent_actor]
        ppo_loss = model.latent_actor.sum()
        critic_loss = model.critic.sum()
        router_loss = model.strategy_encoder.sum()
        ppo_norm = actor_diagnostic_grad_norm(ppo_loss, actor_params)
        critic_on_actor = actor_diagnostic_grad_norm(critic_loss, actor_params)
        router_on_actor = actor_diagnostic_grad_norm(router_loss, actor_params)
        self.assertAlmostEqual(ppo_norm, float(2.0**0.5))
        self.assertEqual(critic_on_actor, 0.0)
        self.assertEqual(router_on_actor, 0.0)

    def test_ratio_equals_cf_over_ppo_actor_only_norms(self) -> None:
        actor = nn.Parameter(torch.tensor(3.0, requires_grad=True))
        ppo_loss = actor
        cf_loss = 2.0 * actor
        cf_norm, ppo_norm, ratio = actor_cf_ppo_grad_diagnostics(
            scaled_cf_loss=cf_loss,
            ppo_actor_loss=ppo_loss,
            actor_parameters=[actor],
        )
        self.assertAlmostEqual(cf_norm, 2.0)
        self.assertAlmostEqual(ppo_norm, 1.0)
        self.assertAlmostEqual(ratio, 2.0)

    def test_cancellation_opposing_grads_keep_ppo_denominator_nonzero(self) -> None:
        actor = nn.Parameter(torch.tensor(2.0, requires_grad=True))
        ppo_loss = actor.pow(2)
        cf_loss = -actor.pow(2)
        actor_params = [actor]
        _, ppo_norm, ratio = actor_cf_ppo_grad_diagnostics(
            scaled_cf_loss=cf_loss,
            ppo_actor_loss=ppo_loss,
            actor_parameters=actor_params,
        )
        self.assertAlmostEqual(ppo_norm, 4.0)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertIsNone(actor.grad)
        (ppo_loss + cf_loss).backward()
        self.assertAlmostEqual(float(actor.grad.item()), 0.0, places=6)

    def test_combined_backward_happens_once_after_diagnostics(self) -> None:
        actor = nn.Parameter(torch.tensor(1.5, requires_grad=True))
        ppo_loss = actor * 2.0
        cf_loss = actor * 3.0
        actor_params = [actor]
        actor_cf_ppo_grad_diagnostics(
            scaled_cf_loss=cf_loss,
            ppo_actor_loss=ppo_loss,
            actor_parameters=actor_params,
        )
        self.assertIsNone(actor.grad)
        total_loss = ppo_loss + cf_loss
        backward_calls = {"count": 0}
        original_backward = total_loss.backward

        def counted_backward(*args, **kwargs):
            backward_calls["count"] += 1
            return original_backward(*args, **kwargs)

        total_loss.backward = counted_backward  # type: ignore[method-assign]
        total_loss.backward()
        self.assertEqual(backward_calls["count"], 1)
        self.assertIsNotNone(actor.grad)

    def test_collect_actor_optimizer_parameters_includes_all_groups(self) -> None:
        p_fast = nn.Parameter(torch.tensor(1.0))
        p_slow = nn.Parameter(torch.tensor(2.0))
        opt = torch.optim.Adam(
            [
                {"params": [p_fast], "lr": 1e-3},
                {"params": [p_slow], "lr": 1e-4},
            ]
        )
        collected = collect_actor_optimizer_parameters(opt)
        self.assertEqual(len(collected), 2)
        self.assertEqual({id(p) for p in collected}, {id(p_fast), id(p_slow)})

    def test_global_grad_norm_sums_across_tensors(self) -> None:
        g1 = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(global_grad_norm([g1]), 5.0)


class _TwoHeadCfModel(nn.Module):
    n_agents = 1
    per_agent_action_dims = (3, 4)

    def policy_logits(self, obs, z_idx=None, *, detach_local_features=False):
        z = z_idx.long().reshape(-1).clamp(min=0, max=3)
        batch = int(z.shape[0])
        logits = torch.zeros((batch, 7), dtype=torch.float32)
        for zi in range(4):
            mask = z == zi
            if not bool(mask.any()):
                continue
            row = torch.zeros(7)
            row[0] = float(zi)
            row[3] = float(zi) * 2.0
            logits[mask] = row
        return logits

    @staticmethod
    def _mask_logits(logits, mask):
        return logits


class V6i1CfPerHeadDiagnosticsTests(unittest.TestCase):
    def test_cf_batch_macro_and_waypoint_jsd_emitted(self) -> None:
        model = _TwoHeadCfModel()
        obs = {"mask": torch.ones((8, 1))}
        _, stats = v6i1_cf_separation_loss(
            model,
            obs,
            latent_k=4,
            margin=0.01,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=True,
        )
        self.assertIn("cf_batch_macro_jsd", stats)
        self.assertIn("cf_batch_waypoint_jsd", stats)
        self.assertGreater(float(stats["cf_batch_macro_jsd"].item()), 0.0)
        self.assertGreater(float(stats["cf_batch_waypoint_jsd"].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
