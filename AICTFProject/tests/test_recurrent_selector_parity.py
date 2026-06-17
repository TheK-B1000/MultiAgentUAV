"""Rollout-to-replay parity for the recurrent strategy selector (V6I1)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Categorical

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.custom_ppo.latent_strategy_state import LatentStrategyState, _stack_selector_hidden_records
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import CONTEXT_STATE_DIM


def _obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _recurrent_model(*, strategy_tau: float = 1.0) -> SharedActorCentralizedCritic:
    torch.manual_seed(0)
    return SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=64,
        use_recurrent_selector=True,
        recurrent_selector_hidden_dim=16,
        use_episode_strategy_value_head=True,
        strategy_tau=strategy_tau,
    )


class RecurrentSelectorParityTests(unittest.TestCase):
    def test_rollout_logits_and_log_prob_match_replay(self) -> None:
        model = _recurrent_model()
        context = torch.randn(8, CONTEXT_STATE_DIM)
        hidden_before = torch.randn(8, 16)

        with torch.no_grad():
            z_idx, rollout_lp, _, rollout_logits, _ = model.sample_strategy(
                context, selector_hidden=hidden_before
            )
            stored_hidden = hidden_before.detach().clone()
            stored_z = z_idx.detach().clone()

            replay_logits = model.strategy_logits(context, selector_hidden=stored_hidden)
            replay_lp = Categorical(logits=replay_logits).log_prob(stored_z)

        torch.testing.assert_close(replay_logits, rollout_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(replay_lp, rollout_lp, rtol=0.0, atol=0.0)

    def test_stored_rollout_hidden_is_detached(self) -> None:
        model = _recurrent_model()
        context = torch.randn(4, CONTEXT_STATE_DIM)
        hidden = torch.randn(4, 16, requires_grad=True)
        with torch.no_grad():
            _, _, _, _, _ = model.sample_strategy(context, selector_hidden=hidden)
        stored = hidden.detach().clone()
        aux = {"selector_hidden": stored}
        self.assertFalse(aux["selector_hidden"].requires_grad)

    def test_shuffled_minibatch_replay_matches_rollout(self) -> None:
        model = _recurrent_model()
        batch = 12
        context = torch.randn(batch, CONTEXT_STATE_DIM)
        hidden = torch.randn(batch, 16)

        with torch.no_grad():
            z_idx, rollout_lp, _, rollout_logits, _ = model.sample_strategy(
                context, selector_hidden=hidden
            )
            records = [
                {
                    "global_state_0": context[i].cpu(),
                    "z": int(z_idx[i].item()),
                    "z_logprob_old": float(rollout_lp[i].item()),
                    "selector_hidden_0": hidden[i].detach().cpu(),
                }
                for i in range(batch)
            ]
            perm = torch.randperm(batch)
            shuffled = [records[int(i)] for i in perm]

            states = torch.stack([r["global_state_0"].float() for r in shuffled], dim=0)
            z = torch.as_tensor([r["z"] for r in shuffled], dtype=torch.long)
            hidden_b = _stack_selector_hidden_records(shuffled, device=states.device)
            assert hidden_b is not None

            replay_logits = model.strategy_logits(states, selector_hidden=hidden_b)
            replay_lp = Categorical(logits=replay_logits).log_prob(z)

            expected_logits = rollout_logits[perm]
            expected_lp = rollout_lp[perm]

        torch.testing.assert_close(replay_logits, expected_logits, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(replay_lp, expected_lp, rtol=0.0, atol=1e-6)

    def test_reset_zeros_selector_hidden(self) -> None:
        from types import SimpleNamespace

        from rl.config.ppo_config import PPOConfig

        model = _recurrent_model()
        cfg = PPOConfig()
        trainer = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            env=SimpleNamespace(num_envs=2),
            model=model,
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_k=4,
            latent_episode_strategy_ppo=False,
            latent_kl_consecutive=0.0,
            latent_arc_credit_enabled=False,
            temporal_tracker=None,
            global_step=0,
        )
        state = LatentStrategyState(trainer)
        state.selector_hidden.fill_(4.2)
        state.reset()
        self.assertTrue(torch.allclose(state.selector_hidden, torch.zeros_like(state.selector_hidden)))

    def test_macro_open_record_replays_with_stored_hidden(self) -> None:
        from types import SimpleNamespace

        from rl.config.ppo_config import PPOConfig

        model = _recurrent_model()
        cfg = PPOConfig()
        cfg.use_v6i1_curriculum = True
        cfg.training_mode = "staged_team_intent_curriculum"
        cfg.experiment_id = "v6i1"
        trainer = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            env=SimpleNamespace(num_envs=1),
            model=model,
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_k=4,
            latent_episode_strategy_ppo=False,
            latent_kl_consecutive=0.0,
            latent_arc_credit_enabled=False,
            temporal_tracker=None,
            global_step=500_000,
            v6i1_curriculum=SimpleNamespace(
                phase="B",
                t_A=400_000,
                nominal_steps=1_000_000,
                resolve_phase=lambda _step=None: "B",
            ),
        )
        state = LatentStrategyState(trainer)
        context = torch.randn(1, CONTEXT_STATE_DIM)
        hidden = torch.randn(1, 16)
        z = torch.tensor([2], dtype=torch.long)
        z_lp = torch.tensor([-0.7], dtype=torch.float32)

        state.macro_open(
            torch.tensor([True]),
            global_state=context,
            z_idx=z,
            z_log_prob=z_lp,
            selector_hidden=hidden,
        )
        state.macro_return_accum[0] = 1.0
        state.macro_steps_accum[0] = 2
        state.macro_finalize(torch.tensor([True]), reason="test")

        record = state.rollout_strategy_macro_records[0]
        self.assertIn("selector_hidden_0", record)
        stacked = _stack_selector_hidden_records([record], device=context.device)
        assert stacked is not None
        with torch.no_grad():
            logits = model.strategy_logits(context, selector_hidden=stacked)
            v = model.episode_strategy_value(context, z, selector_hidden=stacked)
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertEqual(tuple(v.shape), (1,))

    def test_strategy_tau_applied_in_sampling_and_reeval(self) -> None:
        cold = _recurrent_model(strategy_tau=1.0)
        hot = _recurrent_model(strategy_tau=2.0)
        hot.load_state_dict(cold.state_dict())
        hot.strategy_tau = 2.0
        context = torch.randn(6, CONTEXT_STATE_DIM)
        hidden = torch.randn(6, 16)

        with torch.no_grad():
            z_cold, lp_cold, _, logits_cold, _ = cold.sample_strategy(
                context, selector_hidden=hidden
            )
            z_hot, lp_hot, _, logits_hot, _ = hot.sample_strategy(
                context, selector_hidden=hidden
            )
            replay_cold = cold.strategy_logits(context, selector_hidden=hidden)
            replay_hot = hot.strategy_logits(context, selector_hidden=hidden)

        torch.testing.assert_close(logits_cold, replay_cold, rtol=0.0, atol=0.0)
        torch.testing.assert_close(logits_hot, replay_hot, rtol=0.0, atol=0.0)
        self.assertFalse(torch.allclose(logits_cold, logits_hot, atol=1e-6))
        replay_lp_cold = Categorical(logits=replay_cold).log_prob(z_cold)
        replay_lp_hot = Categorical(logits=replay_hot).log_prob(z_hot)
        torch.testing.assert_close(replay_lp_cold, lp_cold, rtol=0.0, atol=0.0)
        torch.testing.assert_close(replay_lp_hot, lp_hot, rtol=0.0, atol=0.0)

    def test_episode_strategy_value_matches_router_context(self) -> None:
        model = _recurrent_model()
        context = torch.randn(5, CONTEXT_STATE_DIM)
        hidden = torch.randn(5, 16)
        z = torch.randint(0, 4, (5,))

        with torch.no_grad():
            logits = model.strategy_logits(context, selector_hidden=hidden)
            v = model.episode_strategy_value(context, z, selector_hidden=hidden)
            ctx = model._build_selector_context(context, hidden)
            z_oh = torch.nn.functional.one_hot(z, 4).float()
            expected = model.episode_strategy_value_head(torch.cat([ctx, z_oh], dim=-1)).squeeze(-1)

        self.assertEqual(tuple(v.shape), (5,))
        torch.testing.assert_close(v, expected, rtol=0.0, atol=1e-6)
        self.assertFalse(torch.allclose(logits, torch.zeros_like(logits)))


class RecurrentInferencePolicyTests(unittest.TestCase):
    def test_fixed_z_predict_advances_recurrent_hidden(self) -> None:
        from rl.custom_ppo.inference import CustomPPOInferencePolicy

        model = _recurrent_model()
        model.eval()
        policy = CustomPPOInferencePolicy(model, device="cpu", cfg={"latent_k": 4})
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = 1
        obs_space = _obs_space()
        obs = {
            "grid": np.zeros((2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": np.ones((2,), dtype=np.float32),
            "mask": np.ones((110,), dtype=np.float32),
            "global_state": np.zeros((GLOBAL_STATE_DIM,), dtype=np.float32),
        }

        actions, _ = policy.predict(obs, deterministic=True)
        self.assertEqual(actions.shape, (4,))
        actions2, _ = policy.predict(obs, deterministic=True)
        self.assertEqual(actions2.shape, (4,))
        self.assertIsNotNone(policy._selector_hidden)


if __name__ == "__main__":
    unittest.main()
