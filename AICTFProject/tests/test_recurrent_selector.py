"""V6I7 invariant tests for the recurrent strategy selector.

Coverage
--------
1. ``advance_selector_hidden`` produces shape ``(N_env, hidden_dim)`` — NOT
   ``(N_env * N_agents, hidden_dim)`` — confirming one GRU step per env.
2. Episode boundary resets hidden to zero on terminated/truncated envs;
   other envs retain their hidden state.
3. ``forward_router_sequence`` runs BPTT through the GRU and returns
   logits of the correct shape.
4. ``iter_router_sequence_minibatches`` yields chunks with the correct
   temporal structure and non-overlapping time indices.
5. ``compute_router_returns`` propagates returns correctly across decision
   boundaries and terminates/truncates.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.custom_ppo.option_returns import compute_router_returns
from rl.custom_ppo.update.sequence_minibatch import iter_router_sequence_minibatches
from rl.global_state import GLOBAL_STATE_DIM
from rl.ppo_core import TensorDictRolloutBuffer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


HIDDEN_DIM = 16
LATENT_K = 4

def _recurrent_model() -> SharedActorCentralizedCritic:
    torch.manual_seed(42)
    return SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        latent_k=LATENT_K,
        z_embed_dim=16,
        strategy_hidden_dim=64,
        use_recurrent_selector=True,
        recurrent_selector_hidden_dim=HIDDEN_DIM,
        use_episode_strategy_value_head=True,
        router_context_mode="current",
    )


def _b(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def _f(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. advance_selector_hidden — shape and per-env-step invariant
# ---------------------------------------------------------------------------

class AdvanceSelectorHiddenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _recurrent_model()

    def test_output_shape_is_n_env_not_n_agents(self) -> None:
        """Hidden state shape must be (N_env, hidden_dim), not (N_env*N_agents, hidden_dim)."""
        N_env = 6
        gs = torch.randn(N_env, GLOBAL_STATE_DIM)
        h = torch.zeros(N_env, HIDDEN_DIM)
        episode_boundary = torch.zeros(N_env, dtype=torch.bool)
        h_new = self.model.advance_selector_hidden(gs, h, episode_boundary)
        self.assertEqual(h_new.shape, (N_env, HIDDEN_DIM))

    def test_episode_boundary_resets_to_zero(self) -> None:
        """Envs flagged in episode_boundary get h=0; others keep their hidden."""
        N_env = 4
        torch.manual_seed(1)
        gs = torch.randn(N_env, GLOBAL_STATE_DIM)
        h = torch.ones(N_env, HIDDEN_DIM)
        episode_boundary = torch.tensor([True, False, True, False])
        h_new = self.model.advance_selector_hidden(gs, h, episode_boundary)
        # Reset envs: hidden must be all zeros.
        self.assertTrue(h_new[0].eq(0).all().item(), "env 0 should have been reset to zero")
        self.assertTrue(h_new[2].eq(0).all().item(), "env 2 should have been reset to zero")
        # Non-reset envs: hidden must be nonzero (GRU transform of all-ones input).
        self.assertFalse(h_new[1].eq(0).all().item(), "env 1 should not be zero after GRU")
        self.assertFalse(h_new[3].eq(0).all().item(), "env 3 should not be zero after GRU")

    def test_no_gradients_in_output(self) -> None:
        """advance_selector_hidden is @torch.no_grad and returns a detached tensor."""
        N_env = 2
        gs = torch.randn(N_env, GLOBAL_STATE_DIM, requires_grad=True)
        h = torch.randn(N_env, HIDDEN_DIM)
        episode_boundary = torch.zeros(N_env, dtype=torch.bool)
        h_new = self.model.advance_selector_hidden(gs, h, episode_boundary)
        self.assertFalse(h_new.requires_grad, "advance_selector_hidden must return detached tensor")

    def test_ordinary_step_changes_hidden(self) -> None:
        """A non-boundary step must actually change the hidden state."""
        N_env = 3
        gs = torch.randn(N_env, GLOBAL_STATE_DIM)
        h = torch.zeros(N_env, HIDDEN_DIM)
        episode_boundary = torch.zeros(N_env, dtype=torch.bool)
        h_new = self.model.advance_selector_hidden(gs, h, episode_boundary)
        # GRU applied to zero input with zero hidden produces nonzero output
        # (because of the GRU's bias terms).
        self.assertFalse(h_new.eq(0).all().item(), "GRU step must change hidden state")


# ---------------------------------------------------------------------------
# 2. forward_router_sequence — BPTT forward
# ---------------------------------------------------------------------------

class ForwardRouterSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _recurrent_model()

    def test_output_shapes(self) -> None:
        T, B, K = 40, 3, LATENT_K
        # V6I7 mode: global state is 35-dim (raw 34 + scheduler phase).
        gs = torch.randn(T, B, GLOBAL_STATE_DIM + 1)
        h_start = torch.zeros(B, HIDDEN_DIM)
        done_mask = torch.zeros(T, B, dtype=torch.bool)
        logits, hiddens = self.model.forward_router_sequence(gs, h_start, done_mask)
        self.assertEqual(logits.shape, (T, B, K))
        self.assertEqual(hiddens.shape, (T, B, HIDDEN_DIM))

    def test_done_resets_hidden(self) -> None:
        """A done at step t means the hidden at t+1 should differ from the non-done case."""
        T, B = 10, 2
        gs = torch.randn(T, B, GLOBAL_STATE_DIM + 1)
        h_start = torch.randn(B, HIDDEN_DIM)

        # Baseline: no dones.
        done_none = torch.zeros(T, B, dtype=torch.bool)
        _, hiddens_none = self.model.forward_router_sequence(gs, h_start, done_none)

        # Done at step 5 for env 0 only.
        done_at5 = torch.zeros(T, B, dtype=torch.bool)
        done_at5[5, 0] = True
        _, hiddens_done = self.model.forward_router_sequence(gs, h_start, done_at5)

        # After the done, hidden should differ from the no-done case.
        self.assertFalse(
            torch.allclose(hiddens_done[6, 0], hiddens_none[6, 0]),
            "Done at step 5 must reset hidden, causing divergence at step 6",
        )
        # Env 1 should be unaffected.
        self.assertTrue(
            torch.allclose(hiddens_done[6, 1], hiddens_none[6, 1]),
            "Non-done env 1 should be identical in both runs",
        )

    def test_gradients_flow_through_gru(self) -> None:
        """Loss on final logits must produce nonzero grad on GRU input weight."""
        T, B = 5, 2
        gs = torch.randn(T, B, GLOBAL_STATE_DIM + 1)
        h_start = torch.zeros(B, HIDDEN_DIM)
        done_mask = torch.zeros(T, B, dtype=torch.bool)

        logits, _ = self.model.forward_router_sequence(gs, h_start, done_mask)
        loss = logits.mean()
        loss.backward()

        gru = self.model.selector_gru
        # GRU has input_hh, input_ih etc; check at least one has nonzero grad.
        any_nonzero = any(
            p.grad is not None and p.grad.abs().max().item() > 0.0
            for p in gru.parameters()
        )
        self.assertTrue(any_nonzero, "GRU parameters must receive nonzero gradients")


# ---------------------------------------------------------------------------
# 3. iter_router_sequence_minibatches — chunk structure
# ---------------------------------------------------------------------------

def _make_buffer(T: int, B: int) -> TensorDictRolloutBuffer:
    """Minimal buffer with required V6I7 fields."""
    buf = TensorDictRolloutBuffer(buffer_size=T, n_envs=B, device="cpu")
    buf.register_field("global_state", shape=(GLOBAL_STATE_DIM + 1,))
    buf.register_field("selector_hidden", shape=(HIDDEN_DIM,))
    buf.register_field("z", dtype=torch.long)
    buf.register_field("z_log_probs")
    buf.register_field("router_decision_valid", dtype=torch.bool)
    buf.register_field("advantages")
    buf.register_field("router_advantages")
    buf.register_field("terminated", dtype=torch.bool)
    buf.register_field("truncated", dtype=torch.bool)
    torch.manual_seed(0)
    for field_name, tensor in buf.fields.items():
        if tensor.dtype == torch.bool:
            tensor.fill_(False)
        elif tensor.dtype == torch.long:
            # Long tensors don't support normal_(); use uniform integers in [0, K).
            tensor.random_(0, 4)
        else:
            tensor.normal_()
    buf.pos = T
    return buf


class IterRouterSequenceMinibatchesTests(unittest.TestCase):
    def test_chunk_shape(self) -> None:
        T, B, burn_in, seq_len = 80, 4, 8, 32
        buf = _make_buffer(T, B)
        chunks = list(
            iter_router_sequence_minibatches(buf, burn_in=burn_in, seq_len=seq_len, chunks_per_batch=2)
        )
        self.assertGreater(len(chunks), 0)
        chunk_total = burn_in + seq_len
        for chunk in chunks:
            for field_name, tensor in chunk.items():
                if field_name == "selector_hidden_start":
                    self.assertEqual(tensor.shape[1], HIDDEN_DIM)
                else:
                    self.assertEqual(tensor.shape[0], chunk_total, f"field {field_name} wrong T dim")

    def test_selector_hidden_start_present(self) -> None:
        T, B = 80, 2
        buf = _make_buffer(T, B)
        chunk = next(iter_router_sequence_minibatches(buf, burn_in=8, seq_len=32, chunks_per_batch=1))
        self.assertIn("selector_hidden_start", chunk)
        self.assertEqual(chunk["selector_hidden_start"].shape[-1], HIDDEN_DIM)

    def test_chunks_cover_all_envs(self) -> None:
        """Total chunk count should equal (T // chunk_len) * B."""
        T, B = 80, 3
        burn_in, seq_len = 8, 32
        chunk_total = burn_in + seq_len  # 40
        expected_chunks = (T // chunk_total) * B  # 2 * 3 = 6
        buf = _make_buffer(T, B)
        all_chunks = list(
            iter_router_sequence_minibatches(
                buf, burn_in=burn_in, seq_len=seq_len, chunks_per_batch=999, shuffle=False
            )
        )
        actual_sequences = sum(c["global_state"].shape[1] for c in all_chunks)
        self.assertEqual(actual_sequences, expected_chunks)

    def test_empty_buffer_yields_nothing(self) -> None:
        T, B = 10, 2  # chunk_total=40 > T=10
        buf = _make_buffer(T, B)
        chunks = list(iter_router_sequence_minibatches(buf, burn_in=8, seq_len=32, chunks_per_batch=2))
        self.assertEqual(len(chunks), 0)


# ---------------------------------------------------------------------------
# 4. compute_router_returns — boundary semantics
# ---------------------------------------------------------------------------

class ComputeRouterReturnsTests(unittest.TestCase):
    def test_single_decision_bootstrap_from_next_values(self) -> None:
        """No mid-buffer decisions after t=0 — return bootstraps from next_values at t=T-1."""
        T, N = 4, 1
        rewards = _f([[1.0], [2.0], [3.0], [4.0]])
        values = _f([[10.0], [10.0], [10.0], [10.0]])
        next_values = _f([[10.0], [10.0], [10.0], [50.0]])
        terminated = _b([[0], [0], [0], [0]])
        truncated = _b([[0], [0], [0], [0]])
        # Only t=0 is a decision step; no new decision within the buffer.
        rdv = _b([[1], [0], [0], [0]])
        returns, advantages = compute_router_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            router_decision_valid=rdv,
            gamma=0.5,
        )
        # t=3: r=4 + 0.5*50 = 29
        # t=2: r=3 + 0.5*29 = 17.5
        # t=1: r=2 + 0.5*17.5 = 10.75
        # t=0: r=1 + 0.5*10.75 = 6.375
        self.assertAlmostEqual(float(returns[0, 0]), 6.375, places=3)

    def test_decision_boundary_bootstraps_from_values(self) -> None:
        """Return at opportunity j bootstraps from values[j+1] at next decision step."""
        T, N = 4, 1
        rewards = _f([[1.0], [2.0], [3.0], [4.0]])
        values = _f([[10.0], [99.0], [10.0], [10.0]])  # values[1]=99 is the bootstrap
        next_values = _f([[10.0], [10.0], [10.0], [50.0]])
        terminated = _b([[0], [0], [0], [0]])
        truncated = _b([[0], [0], [0], [0]])
        # Decision at t=0 and t=2; t=1 folds into window starting at t=0.
        rdv = _b([[1], [0], [1], [0]])
        returns, advantages = compute_router_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            router_decision_valid=rdv,
            gamma=0.5,
        )
        # Window 1 (t=0,1): t=1 sees next step t=2 is decision → carry = values[2]=10
        # t=1: r=2 + 0.5*10 = 7; t=0: r=1 + 0.5*7 = 4.5
        self.assertAlmostEqual(float(returns[0, 0]), 4.5, places=3)
        self.assertAlmostEqual(float(returns[1, 0]), 7.0, places=3)

    def test_termination_zeros_future(self) -> None:
        """Episode termination at t must zero bootstrap (no future rewards)."""
        T, N = 3, 1
        rewards = _f([[1.0], [2.0], [3.0]])
        values = _f([[10.0], [10.0], [10.0]])
        next_values = _f([[0.0], [0.0], [100.0]])
        terminated = _b([[0], [1], [0]])  # done at t=1
        truncated = _b([[0], [0], [0]])
        rdv = _b([[1], [0], [1]])
        returns, _ = compute_router_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            router_decision_valid=rdv,
            gamma=0.9,
        )
        # t=1 terminated: next_val = 0 (terminated overrides next_values bootstrap)
        # returns[1] = 2 + 0.9 * 0 = 2
        self.assertAlmostEqual(float(returns[1, 0]), 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
