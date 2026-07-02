"""Tests for feedforward sparse-router credit assignment and entropy."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributions import Categorical

from rl.custom_ppo.update.entropy_objectives import EntropyObjective
from rl.custom_ppo.update.strategy_credit import (
    encoder_grad_norm_from_loss,
    is_feedforward_sparse_router,
    is_recurrent_router,
    resolve_strategy_advantages,
    router_decision_mask,
)
from rl.custom_ppo.update.strategy_objectives import StrategyObjective
from rl.latent_losses import feedforward_router_entropy_loss, strategy_ppo_loss
from rl.latent_marl import StrategyEncoder


DEVICE = torch.device("cpu")


def _feedforward_cfg(*, router_reward_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        router_reward_enabled=router_reward_enabled,
        recurrent_selector_hidden_dim=0,
        latent_q_phi_option_advantage=False,
        router_ent_coef=0.005,
        latent_lam_p=0.0,
        latent_lam_h=0.0,
    )


def _recurrent_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        router_reward_enabled=True,
        recurrent_selector_hidden_dim=64,
        latent_q_phi_option_advantage=False,
        router_ent_coef=0.005,
    )


class ResolveStrategyAdvantagesTests(unittest.TestCase):
    def test_feedforward_uses_router_advantages(self) -> None:
        cfg = _feedforward_cfg()
        actor_adv = torch.tensor([[-5.0, -5.0, -5.0, -5.0]], dtype=torch.float32)
        router_adv = torch.tensor([[1.0, 0.0, 2.0, 0.0]], dtype=torch.float32)
        batch = {
            "router_advantages": router_adv,
            "router_decision_valid": torch.tensor([[True, False, True, False]]),
            "z_resampled": torch.tensor([[True, False, True, False]]),
        }
        adv, source = resolve_strategy_advantages(
            cfg=cfg, batch=batch, actor_advantages=actor_adv
        )
        self.assertEqual(source, "router")
        self.assertTrue(torch.equal(adv, router_adv))

    def test_missing_router_advantages_raises(self) -> None:
        cfg = _feedforward_cfg()
        batch = {
            "router_decision_valid": torch.tensor([[True, False]]),
            "z_resampled": torch.tensor([[True, False]]),
        }
        with self.assertRaises(RuntimeError):
            resolve_strategy_advantages(
                cfg=cfg,
                batch=batch,
                actor_advantages=torch.zeros(1, 2),
            )

    def test_disabled_router_reward_falls_back_to_actor_gae(self) -> None:
        cfg = _feedforward_cfg(router_reward_enabled=False)
        actor_adv = torch.tensor([[3.0, -1.0]], dtype=torch.float32)
        adv, source = resolve_strategy_advantages(
            cfg=cfg,
            batch={},
            actor_advantages=actor_adv,
        )
        self.assertEqual(source, "actor_gae")
        self.assertTrue(torch.equal(adv, actor_adv))

    def test_recurrent_does_not_require_router_advantages_in_main_loop(self) -> None:
        cfg = _recurrent_cfg()
        actor_adv = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        adv, source = resolve_strategy_advantages(
            cfg=cfg,
            batch={},
            actor_advantages=actor_adv,
        )
        self.assertEqual(source, "actor_gae")
        self.assertTrue(torch.equal(adv, actor_adv))


class FeedforwardRouterEntropyTests(unittest.TestCase):
    def test_entropy_applied_only_on_decision_rows(self) -> None:
        entropy = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([True, False, True, False])
        loss, stats = feedforward_router_entropy_loss(
            entropy, mask, router_ent_coef=0.01, device=DEVICE
        )
        expected = 0.01 * (-entropy[mask].mean())
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)
        loss.backward()
        self.assertAlmostEqual(float(entropy.grad[0].item()), -0.01 / 2.0, places=6)
        self.assertEqual(float(entropy.grad[1].item()), 0.0)
        self.assertAlmostEqual(float(entropy.grad[2].item()), -0.01 / 2.0, places=6)
        self.assertEqual(float(entropy.grad[3].item()), 0.0)
        self.assertLess(stats["feedforward_router_entropy_loss"], 0.0)

    def test_zero_router_ent_coef_removes_entropy_grad(self) -> None:
        entropy = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([True, True])
        loss, _ = feedforward_router_entropy_loss(
            entropy, mask, router_ent_coef=0.0, device=DEVICE
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_feedforward_entropy_component_active_only_for_feedforward(self) -> None:
        cfg = _feedforward_cfg()
        obj = EntropyObjective(model=mock.Mock(), cfg=cfg, hparams=mock.Mock(), device=DEVICE)
        entropy = torch.tensor([1.0, 2.0], dtype=torch.float32)
        mask = torch.tensor([True, False])
        active = obj.feedforward_router_component(
            strategy_entropy=entropy,
            router_decision_mask=mask,
            router_ent_coef=0.01,
            apply=True,
            zero_scalar=torch.zeros((), device=DEVICE),
        )
        inactive = obj.feedforward_router_component(
            strategy_entropy=entropy,
            router_decision_mask=mask,
            router_ent_coef=0.01,
            apply=False,
            zero_scalar=torch.zeros((), device=DEVICE),
        )
        self.assertTrue(active.active)
        self.assertFalse(inactive.active)
        self.assertEqual(inactive.scaled_loss.item(), 0.0)


class RouterModeIsolationTests(unittest.TestCase):
    def test_feedforward_sparse_router_detection(self) -> None:
        self.assertTrue(is_feedforward_sparse_router(_feedforward_cfg()))
        self.assertFalse(is_feedforward_sparse_router(_recurrent_cfg()))
        self.assertFalse(is_feedforward_sparse_router(_feedforward_cfg(router_reward_enabled=False)))

    def test_recurrent_router_detection(self) -> None:
        self.assertTrue(is_recurrent_router(_recurrent_cfg()))
        self.assertFalse(is_recurrent_router(_feedforward_cfg()))


class StrategyPpoAdvantageSourceTests(unittest.TestCase):
    def _run_ppo_delta(
        self,
        *,
        actor_adv_value: float,
        router_adv_value: float,
        use_router: bool,
    ) -> float:
        torch.manual_seed(0)
        encoder = StrategyEncoder(state_dim=35, latent_k=4, hidden=32)
        state = torch.randn(1, 35, dtype=torch.float32)
        logits = encoder(state)
        logits.retain_grad()
        dist = Categorical(logits=logits)
        z = torch.tensor([2], dtype=torch.long)
        old_log_prob = dist.log_prob(z).detach()

        cfg = _feedforward_cfg(router_reward_enabled=use_router)
        batch = {
            "router_decision_valid": torch.tensor([True]),
            "z_resampled": torch.tensor([True]),
            "z_persist_mask": torch.tensor([False]),
            "prev_z": torch.tensor([0], dtype=torch.long),
            "z_logits_prev": logits.detach(),
            "z_kl_prev_valid": torch.tensor([False]),
            "z_log_probs": old_log_prob,
            "z": z,
            "global_state": state,
            "returns": torch.tensor([0.0]),
            "phase_id": torch.tensor([0], dtype=torch.long),
        }
        if use_router:
            batch["router_advantages"] = torch.tensor([router_adv_value], dtype=torch.float32)
        actor_adv = torch.tensor([actor_adv_value], dtype=torch.float32)

        objective = StrategyObjective(
            model=encoder,
            cfg=cfg,
            hparams=SimpleNamespace(
                fixed_latent_strategy=False,
                latent_k=4,
                clip_range=0.2,
                latent_strategy_ppo_coef=1.0,
                latent_kl_consecutive=0.0,
                latent_strategy_aux_return_head=False,
                latent_strategy_aux_return_coef=0.0,
                latent_strategy_aux_predict_phase_coef=0.0,
                latent_resample_every_n=32,
                latent_resample_on_flag=False,
                latent_event_refresh_enabled=False,
                latent_sparse_tactical_refresh_enabled=False,
            ),
            runtime=SimpleNamespace(latent_router_optimizer=None),
            device=DEVICE,
        )
        zero = torch.zeros((), device=DEVICE)
        entropy_component = EntropyObjective(
            model=encoder, cfg=cfg, hparams=mock.Mock(), device=DEVICE
        ).feedforward_router_component(
            strategy_entropy=torch.tensor([1.0]),
            router_decision_mask=torch.tensor([True]),
            router_ent_coef=0.0,
            apply=False,
            zero_scalar=zero,
        )
        bundle = objective.compute(
            batch=batch,
            aux={
                "strategy_log_prob": dist.log_prob(z),
                "strategy_logits": logits,
                "strategy_entropy": dist.entropy(),
            },
            advantages=actor_adv,
            latent_lam_h=0.0,
            apply_main_loop_qphi_loss=True,
            apply_entropy_loss=False,
            apply_persistence_loss=False,
            apply_kl_loss=False,
            entropy_component=entropy_component,
            marginal_component=None,
            epoch_marginal_stats={},
            rollout_marginal_coef=0.0,
            h_mode="conditional",
            zero_scalar=zero,
        )
        self.assertEqual(bundle.credit_telemetry["strategy_advantage_source"], 2.0 if use_router else 0.0)
        bundle.latent_loss.backward()
        grad = logits.grad
        self.assertIsNotNone(grad)
        return float(grad[0, 2].item())

    def test_router_advantage_direction_controls_z2_gradient(self) -> None:
        pos_grad = self._run_ppo_delta(actor_adv_value=-1.0, router_adv_value=1.0, use_router=True)
        neg_grad = self._run_ppo_delta(actor_adv_value=1.0, router_adv_value=-1.0, use_router=True)
        self.assertLess(pos_grad, neg_grad)

    def test_dense_actor_advantage_used_when_router_reward_disabled(self) -> None:
        pos_actor = self._run_ppo_delta(actor_adv_value=1.0, router_adv_value=-1.0, use_router=False)
        neg_actor = self._run_ppo_delta(actor_adv_value=-1.0, router_adv_value=1.0, use_router=False)
        self.assertLess(pos_actor, neg_actor)

    def test_router_advantage_overrides_actor_advantage_sign(self) -> None:
        router_grad = self._run_ppo_delta(actor_adv_value=-1.0, router_adv_value=1.0, use_router=True)
        actor_grad = self._run_ppo_delta(actor_adv_value=-1.0, router_adv_value=1.0, use_router=False)
        self.assertNotAlmostEqual(router_grad, actor_grad, places=5)


class EncoderGradNormTests(unittest.TestCase):
    def test_encoder_grad_norm_from_policy_component(self) -> None:
        encoder = StrategyEncoder(state_dim=35, latent_k=4, hidden=16)
        state = torch.randn(1, 35, dtype=torch.float32)
        logits = encoder(state)
        dist = Categorical(logits=logits)
        z = torch.tensor([1], dtype=torch.long)
        mask = torch.tensor([True])
        loss, _ = strategy_ppo_loss(
            dist.log_prob(z),
            dist.log_prob(z).detach(),
            torch.tensor([1.0]),
            mask,
            clip_range=0.2,
            coef=1.0,
            device=DEVICE,
        )
        norm = encoder_grad_norm_from_loss(loss, encoder)
        self.assertGreater(norm, 0.0)


if __name__ == "__main__":
    unittest.main()
