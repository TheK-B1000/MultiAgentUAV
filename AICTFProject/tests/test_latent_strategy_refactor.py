"""Characterization tests for LatentStrategyState refactor (Phases 0-1)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch
from torch.distributions import Categorical

from rl.custom_ppo.latent.behavior_policy import (
    behavior_log_prob_from_probs,
    epsilon_behavior_probs,
)
from rl.custom_ppo.latent.intervention_state import InterventionState, PairwiseEMAState
from rl.custom_ppo.latent.lifecycle import EpisodeLifecycleState
from rl.custom_ppo.latent.records import EpisodeStrategyRecorder
from rl.custom_ppo.latent.selector_memory import SelectorMemory
from rl.custom_ppo.latent.types import RouterActionSource


class BehaviorPolicyTests(unittest.TestCase):
    def test_epsilon_mixture_log_prob(self) -> None:
        logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        router_probs = torch.softmax(logits, dim=-1)
        behavior_probs = epsilon_behavior_probs(router_probs, epsilon=0.25, latent_k=4)
        executed = torch.tensor([0])
        lp = behavior_log_prob_from_probs(behavior_probs, executed)
        manual = torch.log(behavior_probs[0, 0])
        self.assertAlmostEqual(float(lp.item()), float(manual.item()), places=5)


class LifecycleTests(unittest.TestCase):
    def test_begin_and_complete(self) -> None:
        life = EpisodeLifecycleState(n_envs=4, device=torch.device("cpu"))
        mask = torch.tensor([True, False, False, False])
        life.begin(mask)
        self.assertTrue(bool(life.active[0].item()))
        life.complete(mask)
        self.assertFalse(bool(life.active[0].item()))


class SelectorMemoryTests(unittest.TestCase):
    def test_reset_rows_zeros_hidden(self) -> None:
        mem = SelectorMemory(n_envs=2, hidden_dim=4, device=torch.device("cpu"))
        assert mem.current() is not None
        mem.current().fill_(1.0)
        mem.reset_rows(torch.tensor([True, False]))
        self.assertEqual(float(mem.current()[0].sum().item()), 0.0)
        self.assertEqual(float(mem.current()[1].sum().item()), 4.0)


class EpisodeRecorderTests(unittest.TestCase):
    def test_record_start_outcome_fields(self) -> None:
        rec = EpisodeStrategyRecorder()
        rec.record_start(
            env_index=0,
            episode_id=7,
            global_state_0=torch.zeros(8),
            proposed_z=1,
            executed_z=2,
            behavior_log_prob=-0.5,
            router_log_prob=-0.4,
            action_source=RouterActionSource.EPSILON_MIXTURE,
            bucket_id=3,
            q_phi_probs=[0.25, 0.25, 0.25, 0.25],
        )
        out = rec.record_outcome(env_index=0, episode_return=1.5, episode_win=1, opponent_id=2)
        assert out is not None
        self.assertEqual(out["executed_z"], 2)
        self.assertEqual(out["z"], 2)
        self.assertAlmostEqual(out["z_logprob_old"], -0.5)
        self.assertEqual(out["action_source"], "epsilon_mixture")


class InterventionEMATests(unittest.TestCase):
    def test_rejects_duplicate_step(self) -> None:
        ema = PairwiseEMAState.zeros(6)
        ok1 = ema.update([0.1] * 6, global_step=10, alpha=0.1, pass_predicate=lambda _: True)
        ok2 = ema.update([0.2] * 6, global_step=10, alpha=0.1, pass_predicate=lambda _: True)
        self.assertTrue(ok1)
        self.assertFalse(ok2)

    def test_pair_count_k4(self) -> None:
        self.assertEqual(InterventionState.pair_count_for_latent_k(4), 6)


class BeginEpisodesOrderingTests(unittest.TestCase):
    def test_reset_hidden_before_logits_invariant(self) -> None:
        """Simulate begin_episodes: logits must not depend on stale hidden."""
        hidden_dim = 8
        gs_dim = 16
        latent_k = 4
        n_envs = 1

        class _Encoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = torch.nn.GRUCell(gs_dim, hidden_dim)
                self.head = torch.nn.Linear(hidden_dim, latent_k)

            def forward(self, gs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                h2 = self.gru(gs, h)
                return self.head(h2), h2

        enc = _Encoder()
        gs = torch.randn(n_envs, gs_dim)
        stale = torch.ones(n_envs, hidden_dim) * 5.0
        fresh = torch.zeros(n_envs, hidden_dim)
        logits_stale, _ = enc(gs, stale)
        logits_fresh, _ = enc(gs, fresh)
        self.assertFalse(torch.allclose(logits_stale, logits_fresh))


if __name__ == "__main__":
    unittest.main()
