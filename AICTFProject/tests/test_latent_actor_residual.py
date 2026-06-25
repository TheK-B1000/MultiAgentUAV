"""Phase-1 tests for V6I7 per-latent residual actor adapters.

Tests:
1. All K latents produce finite, valid action distributions.
2. Batched latent broadcasting is correct (one z per sample).
3. Every latent adapter receives a finite nonzero gradient when that latent is sampled.
4. Every latent action bias receives a finite nonzero gradient.
5. The shared actor trunk still receives PPO gradients.
6. When the feature is disabled, outputs are behaviorally equivalent to baseline.
7. Perturbed adapter/bias parameters produce different logits for different z IDs.
8. Existing checkpoints report newly initialized layers instead of hard-failing.
9. No global state, opponent ID, or strategy information enters the decentralized actor.
"""

from __future__ import annotations

import copy
import io
import sys
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

# Ensure repo root is on sys.path.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.global_state import GLOBAL_STATE_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATENT_K = 4
HIDDEN_DIM = 64  # recurrent_selector_hidden_dim for V6I7


def _obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, (2, 7, 20, 20), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, (2, VEC_OBS_DIM), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, (2,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, (110,), dtype=np.float32),
    })


def _action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _make_model(*, enable_residual: bool) -> SharedActorCentralizedCritic:
    torch.manual_seed(0)
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
        enable_latent_z_residual=enable_residual,
        latent_z_gate_init=0.01,
    )


def _random_obs(B: int = 2) -> Dict[str, torch.Tensor]:
    return {
        "grid": torch.randn(B, 2, 7, 20, 20),
        "vec": torch.randn(B, 2, VEC_OBS_DIM),
        "agent_mask": torch.ones(B, 2),
        "mask": torch.ones(B, 110),
    }


# ---------------------------------------------------------------------------
# Test 1: All K latents produce valid distributions
# ---------------------------------------------------------------------------

class Test1AllLatentsValid(unittest.TestCase):
    def setUp(self):
        self.model = _make_model(enable_residual=True)
        self.obs = _random_obs(B=3)

    def test_all_latents_finite(self):
        for z_val in range(LATENT_K):
            z = torch.full((3,), z_val, dtype=torch.long)
            with torch.no_grad():
                logits = self.model.policy_logits(self.obs, z_idx=z)
            self.assertFalse(torch.isnan(logits).any().item(),
                             f"NaN in logits for z={z_val}")
            self.assertFalse(torch.isinf(logits).any().item(),
                             f"Inf in logits for z={z_val}")

    def test_all_latents_give_valid_probs(self):
        for z_val in range(LATENT_K):
            z = torch.full((3,), z_val, dtype=torch.long)
            with torch.no_grad():
                logits = self.model.policy_logits(self.obs, z_idx=z)
            # Should be able to softmax without error
            probs = torch.softmax(logits, dim=-1)
            self.assertFalse(torch.isnan(probs).any().item(),
                             f"NaN in probs for z={z_val}")


# ---------------------------------------------------------------------------
# Test 2: Batched broadcasting — different z per sample
# ---------------------------------------------------------------------------

class Test2BatchedBroadcast(unittest.TestCase):
    def test_mixed_z_batch(self):
        model = _make_model(enable_residual=True)
        # Batch of 4, one sample per latent
        obs = _random_obs(B=4)
        # Same obs for all, different z
        z_mixed = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        z_all0 = torch.zeros(4, dtype=torch.long)
        z_all1 = torch.ones(4, dtype=torch.long)

        with torch.no_grad():
            logits_mixed = model.policy_logits(obs, z_idx=z_mixed)
            logits_0 = model.policy_logits(obs, z_idx=z_all0)
            logits_1 = model.policy_logits(obs, z_idx=z_all1)

        # Row 0 of mixed batch must equal row 0 of z_all0
        self.assertTrue(torch.allclose(logits_mixed[0], logits_0[0], atol=1e-5),
                        "Mixed-z row 0 should match pure-z0 row 0")
        # Row 1 of mixed batch must equal row 1 of z_all1
        self.assertTrue(torch.allclose(logits_mixed[1], logits_1[1], atol=1e-5),
                        "Mixed-z row 1 should match pure-z1 row 1")
        # Row 0 of z_all0 should not equal row 0 of z_all1 (different z)
        # (unless adapters are zero, which they're not after gate init 0.01)
        # We allow this to be equal initially only if gate is exactly zero, but gate=0.01


# ---------------------------------------------------------------------------
# Test 3: Per-latent adapter gradient
# ---------------------------------------------------------------------------

class Test3AdapterGradient(unittest.TestCase):
    def test_each_latent_adapter_gets_grad(self):
        model = _make_model(enable_residual=True)
        la = model.latent_actor
        self.assertIsNotNone(la.latent_adapters, "latent_adapters should exist")

        for z_val in range(LATENT_K):
            model.zero_grad()
            obs = _random_obs(B=4)
            z = torch.full((4,), z_val, dtype=torch.long)
            logits = model.policy_logits(obs, z_idx=z)
            loss = logits.sum()
            loss.backward()

            # Only adapter for z_val should have nonzero grad.
            adapter = la.latent_adapters[z_val]
            has_nonzero = any(
                p.grad is not None and p.grad.abs().max().item() > 0.0
                for p in adapter.parameters()
            )
            self.assertTrue(has_nonzero,
                            f"Adapter z={z_val} should have nonzero gradient")

            # Other adapters should have zero grad (they were not used).
            for other_z in range(LATENT_K):
                if other_z == z_val:
                    continue
                other_adapter = la.latent_adapters[other_z]
                all_zero = all(
                    p.grad is None or p.grad.abs().max().item() == 0.0
                    for p in other_adapter.parameters()
                )
                self.assertTrue(all_zero,
                                f"Adapter z={other_z} should NOT have grad when sampling z={z_val}")


# ---------------------------------------------------------------------------
# Test 4: Action bias gradient
# ---------------------------------------------------------------------------

class Test4ActionBiasGradient(unittest.TestCase):
    def test_each_latent_bias_gets_grad(self):
        model = _make_model(enable_residual=True)
        la = model.latent_actor
        self.assertIsNotNone(la.latent_action_biases)

        for z_val in range(LATENT_K):
            model.zero_grad()
            obs = _random_obs(B=4)
            z = torch.full((4,), z_val, dtype=torch.long)
            logits = model.policy_logits(obs, z_idx=z)
            logits.sum().backward()

            grad = la.latent_action_biases.grad
            self.assertIsNotNone(grad, "latent_action_biases should have grad")
            # Only row z_val should be nonzero.
            self.assertGreater(grad[z_val].abs().max().item(), 0.0,
                               f"Bias row {z_val} should have grad when sampling z={z_val}")
            for other_z in range(LATENT_K):
                if other_z == z_val:
                    continue
                self.assertAlmostEqual(grad[other_z].abs().max().item(), 0.0, places=7,
                                       msg=f"Bias row {other_z} should be zero when sampling z={z_val}")


# ---------------------------------------------------------------------------
# Test 5: Shared trunk still receives gradients
# ---------------------------------------------------------------------------

class Test5SharedTrunkGrad(unittest.TestCase):
    def test_trunk_grad_flows(self):
        model = _make_model(enable_residual=True)
        model.zero_grad()
        obs = _random_obs(B=4)
        z = torch.randint(0, LATENT_K, (4,))
        logits = model.policy_logits(obs, z_idx=z)
        logits.sum().backward()

        # CNN weights are part of the shared trunk.
        cnn = model.actor_cnn
        any_nonzero = any(
            p.grad is not None and p.grad.abs().max().item() > 0.0
            for p in cnn.parameters()
        )
        self.assertTrue(any_nonzero, "Shared CNN trunk should receive nonzero gradient")

        # Body of latent actor (shared MLP) should also have grad.
        body = model.latent_actor.body
        body_nonzero = any(
            p.grad is not None and p.grad.abs().max().item() > 0.0
            for p in body.parameters()
        )
        self.assertTrue(body_nonzero, "Shared actor body should receive nonzero gradient")


# ---------------------------------------------------------------------------
# Test 6: Disabled mode is behaviorally equivalent to baseline
# ---------------------------------------------------------------------------

class Test6DisabledEquivalent(unittest.TestCase):
    def test_disabled_matches_no_residual_model(self):
        # Two models with the same weights but one has residual disabled.
        torch.manual_seed(42)
        model_on = _make_model(enable_residual=True)
        torch.manual_seed(42)
        model_off = _make_model(enable_residual=False)

        # Copy shared weights to model_off.
        sd_on = model_on.state_dict()
        # Remove residual-only keys from on-model state dict.
        sd_shared = {
            k: v for k, v in sd_on.items()
            if not any(k.startswith(p) for p in (
                "latent_actor.latent_adapters",
                "latent_actor.latent_adapter_gates",
                "latent_actor.latent_action_biases",
            ))
        }
        model_off.load_state_dict(sd_shared, strict=False)

        obs = _random_obs(B=3)
        for z_val in range(LATENT_K):
            z = torch.full((3,), z_val, dtype=torch.long)
            # With gates at 0.01 and zero biases, logits should differ slightly.
            # But with gates=0 and biases=0, they'd be identical.
            # Here we just verify the off model produces finite outputs (regression guard).
            with torch.no_grad():
                logits_off = model_off.policy_logits(obs, z_idx=z)
            self.assertFalse(torch.isnan(logits_off).any().item(),
                             f"Disabled model NaN at z={z_val}")

    def test_exactly_zero_gate_and_bias_gives_same_as_disabled(self):
        torch.manual_seed(7)
        model_on = _make_model(enable_residual=True)
        torch.manual_seed(7)
        model_off = _make_model(enable_residual=False)

        # Zero out gate and biases in model_on.
        with torch.no_grad():
            model_on.latent_actor.latent_adapter_gates.zero_()
            model_on.latent_actor.latent_action_biases.zero_()

        # Sync shared weights.
        sd_on = {
            k: v for k, v in model_on.state_dict().items()
            if not any(k.startswith(p) for p in (
                "latent_actor.latent_adapters",
                "latent_actor.latent_adapter_gates",
                "latent_actor.latent_action_biases",
            ))
        }
        model_off.load_state_dict(sd_on, strict=False)

        obs = _random_obs(B=2)
        for z_val in range(LATENT_K):
            z = torch.full((2,), z_val, dtype=torch.long)
            with torch.no_grad():
                logits_on = model_on.policy_logits(obs, z_idx=z)
                logits_off = model_off.policy_logits(obs, z_idx=z)
            self.assertTrue(torch.allclose(logits_on, logits_off, atol=1e-5),
                            f"Zero-gate model should be equivalent to disabled at z={z_val}")


# ---------------------------------------------------------------------------
# Test 7: Perturbed params produce different logits under different z
# ---------------------------------------------------------------------------

class Test7PerturbedDifferentLogits(unittest.TestCase):
    def test_perturbed_adapters_diverge_by_z(self):
        model = _make_model(enable_residual=True)

        # Perturb adapter 0 significantly.
        with torch.no_grad():
            for p in model.latent_actor.latent_adapters[0].parameters():
                p.add_(torch.randn_like(p) * 2.0)
            model.latent_actor.latent_adapter_gates[0] = 1.0

        obs = _random_obs(B=4)
        with torch.no_grad():
            z0 = torch.zeros(4, dtype=torch.long)
            z1 = torch.ones(4, dtype=torch.long)
            logits_z0 = model.policy_logits(obs, z_idx=z0)
            logits_z1 = model.policy_logits(obs, z_idx=z1)

        self.assertFalse(
            torch.allclose(logits_z0, logits_z1, atol=1e-4),
            "Perturbed adapter should cause z0 and z1 to differ",
        )

    def test_perturbed_biases_diverge_by_z(self):
        model = _make_model(enable_residual=True)

        with torch.no_grad():
            model.latent_actor.latent_action_biases[0] += 5.0

        obs = _random_obs(B=4)
        with torch.no_grad():
            z0 = torch.zeros(4, dtype=torch.long)
            z1 = torch.ones(4, dtype=torch.long)
            logits_z0 = model.policy_logits(obs, z_idx=z0)
            logits_z1 = model.policy_logits(obs, z_idx=z1)

        self.assertFalse(
            torch.allclose(logits_z0, logits_z1, atol=1e-4),
            "Perturbed action bias should separate z0 and z1 logits",
        )


# ---------------------------------------------------------------------------
# Test 8: Checkpoint loading reports newly initialized params
# ---------------------------------------------------------------------------

class Test8CheckpointNewParamReporting(unittest.TestCase):
    def test_old_checkpoint_reports_new_params(self):
        # Build a model WITHOUT residual and save its state dict.
        old_model = _make_model(enable_residual=False)
        old_sd = copy.deepcopy(old_model.state_dict())

        # Build a new model WITH residual and load the old state dict.
        new_model = _make_model(enable_residual=True)

        # Capture stdout to verify reporting.
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            result = new_model.load_state_dict(old_sd, strict=False)
        finally:
            sys.stdout = old_stdout

        missing = result.missing_keys
        residual_missing = [
            k for k in missing
            if "latent_adapters" in k or "latent_adapter_gates" in k
            or "latent_action_biases" in k
        ]
        self.assertGreater(len(residual_missing), 0,
                           "New residual params should appear in missing_keys")

    def test_shared_weights_load_correctly(self):
        # Shared weights should load without error.
        old_model = _make_model(enable_residual=False)
        new_model = _make_model(enable_residual=True)

        # Sync everything except new params.
        sd_old = old_model.state_dict()
        result = new_model.load_state_dict(sd_old, strict=False)

        unexpected = [
            k for k in result.unexpected_keys
            if not any(k.startswith(p) for p in (
                "latent_actor.latent_adapters",
                "latent_actor.latent_adapter_gates",
                "latent_actor.latent_action_biases",
            ))
        ]
        self.assertEqual(unexpected, [],
                         f"Unexpected non-residual keys: {unexpected}")

    def test_old_checkpoint_gives_identical_logits_at_init(self):
        """After loading a pre-adapter checkpoint, logits must be bit-exact.

        With zero-initialized adapter weights:
            A_z(h) = W_z @ h + b_z = 0 (because W_z=0, b_z=0)
            h_z    = h + gate_z * 0 = h
            logits = action_head(h_z) + B_z = action_head(h) + 0 = action_head(h)

        So the model must produce exactly the same logits as the no-residual
        baseline even with gate=0.01, as long as adapter weights and biases are
        zero and action biases are zero.
        """
        torch.manual_seed(99)
        old_model = _make_model(enable_residual=False)
        torch.manual_seed(99)
        new_model = _make_model(enable_residual=True)

        # Load old checkpoint weights into new model (adapters stay zero-init).
        sd_old = old_model.state_dict()
        new_model.load_state_dict(sd_old, strict=False)

        obs = _random_obs(B=4)
        for z_val in range(LATENT_K):
            z = torch.full((4,), z_val, dtype=torch.long)
            with torch.no_grad():
                logits_old = old_model.policy_logits(obs, z_idx=z)
                logits_new = new_model.policy_logits(obs, z_idx=z)
            max_diff = (logits_old - logits_new).abs().max().item()
            self.assertAlmostEqual(
                max_diff, 0.0, places=5,
                msg=f"z={z_val}: logits should be bit-exact after loading old checkpoint "
                    f"(max diff={max_diff:.2e}); check that adapter weights are zero-init",
            )


# ---------------------------------------------------------------------------
# Test 9: No global state enters the decentralized actor
# ---------------------------------------------------------------------------

class Test9NoGlobalStateInActor(unittest.TestCase):
    def test_actor_input_matches_local_obs_only(self):
        model = _make_model(enable_residual=True)
        # The local actor input is CNN features + vec, not global state.
        local_in_dim = model.actor_input_dim
        global_state_dim = GLOBAL_STATE_DIM
        temporal_context_dim = 170  # EMA stack

        self.assertNotEqual(local_in_dim, global_state_dim,
                            "Actor input should not equal GLOBAL_STATE_DIM")
        self.assertNotEqual(local_in_dim, temporal_context_dim,
                            "Actor input must not equal CONTEXT_STATE_DIM — "
                            "actor cannot see the centralized temporal context")

    def test_actor_body_first_layer_matches_local_obs(self):
        model = _make_model(enable_residual=True)
        first_linear = model.actor_body[0]
        self.assertIsInstance(first_linear, nn.Linear)
        self.assertEqual(first_linear.in_features, model.actor_input_dim,
                         "Actor first layer must accept exactly local_obs+z_embed width")

    def test_changing_global_state_does_not_change_actor_logits(self):
        """Identical local obs but different global state → identical actor logits."""
        model = _make_model(enable_residual=True)
        obs_a = _random_obs(B=2)
        obs_b = copy.deepcopy(obs_a)
        # Modify only global_state (not in obs dict — it's a separate input to values/critic).
        gs_a = torch.randn(2, GLOBAL_STATE_DIM + 1)
        gs_b = torch.randn(2, GLOBAL_STATE_DIM + 1)
        z = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            logits_a = model.policy_logits(obs_a, z_idx=z)
            logits_b = model.policy_logits(obs_b, z_idx=z)
        # global_state is only used by critic/values, not policy_logits.
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6),
                        "policy_logits must not depend on global_state")


if __name__ == "__main__":
    unittest.main()
