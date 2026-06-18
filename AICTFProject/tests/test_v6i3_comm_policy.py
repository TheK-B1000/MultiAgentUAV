"""V6I3 Slice 2: policy, rollout buffer, and PPO wiring tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gymnasium import spaces

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.communication import extend_observation_space_if_needed, extra_cnn_channels
from rl.custom_ppo.communication.listener import receiver_macro_jsd_by_message
from rl.custom_ppo.communication.observation import inject_message_grid_channels
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.trainer_optimizers import collect_actor_parameters


def _base_obs_space(*, n_agents: int = 4, c: int = 7) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n_agents, c, 20, 20),
                dtype=np.float32,
            ),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 20), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(n_agents,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(n_agents * 25,), dtype=np.float32),
        }
    )


def _comm_model(*, latent: bool = True) -> SharedActorCentralizedCritic:
    cfg = PPOConfig(communication_enabled=True, use_latent_strategy=latent, latent_k=4)
    base = _base_obs_space()
    obs_space = extend_observation_space_if_needed(base, cfg)
    action_space = spaces.MultiDiscrete([5, 20] * 4)
    return SharedActorCentralizedCritic(
        obs_space,
        action_space,
        latent_k=4 if latent else 0,
        z_embed_dim=16 if latent else 0,
        communication_enabled=True,
        comm_num_symbols=4,
        actor_cnn_feature_dim=32,
        actor_hidden_dim=64,
    )


class CommPolicyTests(unittest.TestCase):
    def test_observation_space_extends_grid_channels(self) -> None:
        cfg = PPOConfig(communication_enabled=True)
        base = _base_obs_space(c=7)
        extended = extend_observation_space_if_needed(base, cfg)
        self.assertEqual(int(extended.spaces["grid"].shape[1]), 7 + int(extra_cnn_channels(cfg)))

    def test_message_injection_matches_policy_expected_channels(self) -> None:
        cfg = PPOConfig(communication_enabled=True)
        bsz, n, h, w = 2, 4, 20, 20
        extra = int(extra_cnn_channels(cfg))
        expected = 12
        base = np.zeros((bsz, n, expected - extra, h, w), dtype=np.float32)
        msg = torch.ones((bsz, n, extra, h, w), dtype=torch.float32)
        appended = inject_message_grid_channels(
            {"grid": base},
            message_channels=msg,
            cfg=cfg,
            expected_grid_channels=expected,
        )
        self.assertEqual(tuple(appended["grid"].shape), (bsz, n, expected, h, w))

        already_extended = np.zeros((bsz, n, expected, h, w), dtype=np.float32)
        replaced = inject_message_grid_channels(
            {"grid": already_extended},
            message_channels=msg,
            cfg=cfg,
            expected_grid_channels=expected,
        )
        self.assertEqual(tuple(replaced["grid"].shape), (bsz, n, expected, h, w))
        self.assertTrue(np.allclose(replaced["grid"][:, :, -extra:, :, :], 1.0))

    def test_received_message_channels_reach_cnn_and_logits(self) -> None:
        torch.manual_seed(7)
        model = _comm_model()
        bsz, n = 2, 4
        obs0 = {
            "grid": torch.zeros((bsz, n, *model.grid_shape), dtype=torch.float32),
            "vec": torch.zeros((bsz, n, model.vec_dim), dtype=torch.float32),
            "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
            "mask": torch.ones((bsz, model.per_agent_logits * n), dtype=torch.float32),
        }
        obs1 = dict(obs0)
        obs1["grid"] = obs0["grid"].clone()
        base_channels = int(model.grid_shape[0]) - 4
        obs1["grid"][:, 0, base_channels + 1, 0, 0] = 1.0

        _, cnn0, _ = model._encode_local_obs(obs0)
        _, cnn1, _ = model._encode_local_obs(obs1)
        self.assertGreater(float((cnn1[:, 0] - cnn0[:, 0]).abs().max().item()), 0.0)

        z = torch.zeros((bsz,), dtype=torch.long)
        logits0 = model.policy_logits(obs0, z_idx=z)
        logits1 = model.policy_logits(obs1, z_idx=z)
        receiver_start = int(model.per_agent_logits) * 0
        receiver_end = receiver_start + int(model.per_agent_logits)
        self.assertEqual(
            tuple(logits0[:, receiver_start:receiver_end].shape),
            (bsz, int(model.per_agent_logits)),
        )
        self.assertGreater(
            float((logits1[:, receiver_start:receiver_end] - logits0[:, receiver_start:receiver_end]).abs().max().item()),
            0.0,
        )

    def test_listener_diagnostic_uses_receiver_macro_distribution(self) -> None:
        model = _comm_model()
        bsz, n = 3, 4
        obs = {
            "grid": torch.zeros((bsz, n, *model.grid_shape), dtype=torch.float32),
            "vec": torch.zeros((bsz, n, model.vec_dim), dtype=torch.float32),
            "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
            "mask": torch.ones((bsz, model.per_agent_logits * n), dtype=torch.float32),
        }
        z = torch.zeros((bsz,), dtype=torch.long)
        stats = receiver_macro_jsd_by_message(
            model,
            obs,
            z_idx=z,
            receiver_agent=0,
            num_symbols=4,
            base_channels=int(model.grid_shape[0]) - 4,
        )
        self.assertEqual(stats["receiver_listener_pairs"], 6.0)
        self.assertGreaterEqual(stats["receiver_action_jsd_by_message_pair_mean"], 0.0)

    def test_message_head_in_actor_optimizer(self) -> None:
        model = _comm_model()
        names = {p for n, p in model.named_parameters() for part in ("message_head",) if part in n}
        actor_params = collect_actor_parameters(model)
        self.assertTrue(any(id(p) in {id(x) for x in actor_params} for p in names))

    def test_message_logprob_zero_off_boundary(self) -> None:
        model = _comm_model()
        bsz, n = 2, 4
        obs = {
            "grid": torch.zeros((bsz, n, *model.grid_shape), dtype=torch.float32),
            "vec": torch.zeros((bsz, n, model.vec_dim), dtype=torch.float32),
            "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
            "mask": torch.ones((bsz, model.per_agent_logits * n), dtype=torch.float32),
        }
        z = torch.zeros((bsz,), dtype=torch.long)
        boundary = torch.zeros((bsz,), dtype=torch.bool)
        aux = model._sample_messages(obs, z_idx=z, comm_boundary_mask=boundary)
        self.assertTrue(torch.all(aux["message_log_probs"] == 0.0))
        self.assertTrue(torch.all(aux["message_entropy"] == 0.0))

    def test_message_logprob_active_on_boundary(self) -> None:
        model = _comm_model()
        bsz, n = 2, 4
        obs = {
            "grid": torch.randn((bsz, n, *model.grid_shape), dtype=torch.float32),
            "vec": torch.randn((bsz, n, model.vec_dim), dtype=torch.float32),
            "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
            "mask": torch.ones((bsz, model.per_agent_logits * n), dtype=torch.float32),
        }
        z = torch.tensor([0, 1], dtype=torch.long)
        boundary = torch.ones((bsz,), dtype=torch.bool)
        aux = model._sample_messages(obs, z_idx=z, comm_boundary_mask=boundary)
        self.assertTrue(bool((aux["message_log_probs"] != 0.0).any()))

    def test_evaluate_actions_replays_message_logprob(self) -> None:
        model = _comm_model()
        bsz, n = 1, 4
        obs = {
            "grid": torch.randn((bsz, n, *model.grid_shape), dtype=torch.float32),
            "vec": torch.randn((bsz, n, model.vec_dim), dtype=torch.float32),
            "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
            "mask": torch.ones((bsz, model.per_agent_logits * n), dtype=torch.float32),
        }
        z = torch.zeros((bsz,), dtype=torch.long)
        gs = torch.zeros((bsz, model.global_state_dim), dtype=torch.float32)
        boundary = torch.ones((bsz,), dtype=torch.bool)
        sampled = model._sample_messages(obs, z_idx=z, comm_boundary_mask=boundary)
        actions, _, action_lp, _ = model.act(obs, gs, z_idx=z)
        _, _, _, aux = model.evaluate_actions(
            obs,
            gs,
            actions,
            z_idx=z,
            message_symbols=sampled["message_symbols"],
            message_boundary_mask=boundary,
        )
        self.assertIn("message_log_probs", aux)
        self.assertAlmostEqual(
            float(aux["message_log_probs"].item()),
            float(sampled["message_log_probs"].item()),
            places=5,
        )
        self.assertAlmostEqual(
            float((action_lp + aux["message_log_probs"]).item()),
            float((action_lp + sampled["message_log_probs"]).item()),
            places=5,
        )

    def test_v6i2_unchanged_when_comm_disabled(self) -> None:
        cfg = PPOConfig(communication_enabled=False, use_latent_strategy=True, latent_k=4)
        base = _base_obs_space(c=7)
        self.assertIs(extend_observation_space_if_needed(base, cfg), base)
        model = SharedActorCentralizedCritic(
            base,
            spaces.MultiDiscrete([5, 20] * 4),
            latent_k=4,
            z_embed_dim=16,
            communication_enabled=False,
        )
        self.assertIsNone(model.message_head)


    def test_one_ppo_decision_per_hold_window(self) -> None:
        """Across 32 steps only one row carries nonzero message PPO log-probs."""
        from rl.custom_ppo.communication import CommConfig, LocalCommTransport

        transport = LocalCommTransport(CommConfig(interval_steps=32))
        transport.reset(batch_size=1, num_agents=4, device=torch.device("cpu"))
        model = _comm_model()
        bsz, n = 1, 4
        logprob_rows: list[float] = []
        boundary_rows: list[bool] = []
        for _ in range(33):
            obs = {
                "grid": torch.randn((bsz, n, *model.grid_shape), dtype=torch.float32),
                "vec": torch.randn((bsz, n, model.vec_dim), dtype=torch.float32),
                "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
            }
            z = torch.zeros((bsz,), dtype=torch.long)
            boundary = transport.is_comm_boundary()
            aux = (
                model._sample_messages(obs, z_idx=z, comm_boundary_mask=torch.tensor([boundary]))
                if boundary
                else {
                    "message_log_probs": torch.zeros((bsz,), dtype=torch.float32),
                    "message_boundary_mask": torch.tensor([boundary]),
                }
            )
            logprob_rows.append(float(aux["message_log_probs"].sum().item()))
            boundary_rows.append(bool(boundary))
            alive = torch.ones((1, 4), dtype=torch.bool)
            x = torch.ones((1, 4), dtype=torch.float32) * 5.0
            y = torch.ones((1, 4), dtype=torch.float32) * 5.0
            if boundary:
                transport.submit_outbound(
                    symbols=aux.get("message_symbols", torch.zeros((1, 4), dtype=torch.long)),
                    sender_x=x,
                    sender_y=y,
                    alive=alive,
                    apply_dropout=False,
                )
            transport.advance_step(
                alive=alive,
                sender_x=x,
                sender_y=y,
                receiver_x=x,
                receiver_y=y,
                cols=20.0,
                rows=20.0,
            )
        self.assertEqual(sum(1 for b in boundary_rows if b), 1)
        self.assertEqual(sum(1 for lp in logprob_rows if lp != 0.0), 1)

    def test_message_head_no_grad_off_boundary_minibatch(self) -> None:
        model = _comm_model()
        bsz, n = 4, 4
        obs = {
            "grid": torch.randn((bsz, n, *model.grid_shape), dtype=torch.float32),
            "vec": torch.randn((bsz, n, model.vec_dim), dtype=torch.float32),
            "agent_mask": torch.ones((bsz, n), dtype=torch.float32),
        }
        z = torch.arange(bsz, dtype=torch.long) % 4
        gs = torch.zeros((bsz, model.global_state_dim), dtype=torch.float32)
        boundary = torch.tensor([False, True, False, False])
        symbols = torch.randint(0, 4, (bsz, n))
        actions = torch.zeros((bsz, len(model.action_dims)), dtype=torch.long)
        for param in model.message_head.parameters():
            param.grad = None
        _, action_lp, _, aux = model.evaluate_actions(
            obs,
            gs,
            actions,
            z_idx=z,
            message_symbols=symbols,
            message_boundary_mask=boundary,
        )
        loss = (action_lp + aux["message_log_probs"]).sum()
        loss.backward()
        assert model.message_head is not None
        grad_norm = sum(
            float(p.grad.norm().item()) for p in model.message_head.parameters() if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

        for param in model.message_head.parameters():
            param.grad = None
        boundary_zero = torch.zeros((bsz,), dtype=torch.bool)
        _, action_lp2, _, aux2 = model.evaluate_actions(
            obs,
            gs,
            actions,
            z_idx=z,
            message_symbols=symbols,
            message_boundary_mask=boundary_zero,
        )
        loss2 = (action_lp2 + aux2["message_log_probs"]).sum()
        loss2.backward()
        grad_norm2 = sum(
            float(p.grad.norm().item()) for p in model.message_head.parameters() if p.grad is not None
        )
        self.assertEqual(grad_norm2, 0.0)


if __name__ == "__main__":
    unittest.main()
