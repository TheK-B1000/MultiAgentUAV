"""V6I3 slices 3–6 integration tests."""

from __future__ import annotations

import unittest

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.communication.corruption import CommCorruptionMode, apply_message_channel_corruption
from rl.custom_ppo.communication.gates import (
    evaluate_communication_usage,
    evaluate_listener_causal_response,
)
from rl.custom_ppo.curriculum.protocols import build_gate_protocol
from rl.custom_ppo.gate_protocol import (
    GATE_FAMILY_NAMES_V6I3,
    GATE_STATUS_PASS,
    V6I3_GATE_PROTOCOL,
    get_gate_family_names,
    validate_protocol_config,
)
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.update.param_registry import ParameterRegistry
from rl.custom_ppo.update.phase_policy import set_model_requires_grad_for_phase
from rl.presets import PRESET_REGISTRY, apply_preset


class V6I3IntegrationTests(unittest.TestCase):
    def test_v6i3_preset_enables_communication(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        self.assertEqual(str(cfg.experiment_id), "v6i3")
        self.assertTrue(bool(cfg.communication_enabled))
        self.assertEqual(str(cfg.gate_protocol_version), V6I3_GATE_PROTOCOL)
        validate_protocol_config(cfg)

    def test_v6i3_gate_protocol_families(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        self.assertEqual(get_gate_family_names(cfg), GATE_FAMILY_NAMES_V6I3)
        protocol = build_gate_protocol(cfg)
        self.assertEqual(protocol.version, V6I3_GATE_PROTOCOL)
        self.assertIn("communication_usage", protocol.required_families())
        self.assertIn("listener_causal_response", protocol.required_families())

    def test_phase_b_freezes_message_head(self) -> None:
        from rl.custom_ppo.communication import extend_observation_space_if_needed
        from gymnasium import spaces
        import numpy as np

        cfg = apply_preset(PPOConfig(), "v6i3")
        base = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (4, 7, 20, 20), dtype=np.float32),
                "vec": spaces.Box(-1.0, 1.0, (4, 20), dtype=np.float32),
                "agent_mask": spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
                "mask": spaces.Box(0.0, 1.0, (100,), dtype=np.float32),
            }
        )
        obs_space = extend_observation_space_if_needed(base, cfg)
        model = SharedActorCentralizedCritic(
            obs_space,
            spaces.MultiDiscrete([5, 20] * 4),
            latent_k=4,
            z_embed_dim=16,
            communication_enabled=True,
            comm_num_symbols=4,
            actor_cnn_feature_dim=32,
            actor_hidden_dim=64,
        )
        set_model_requires_grad_for_phase(model, "A")
        self.assertTrue(any(p.requires_grad for n, p in model.named_parameters() if "message_head" in n))
        set_model_requires_grad_for_phase(model, "B")
        self.assertFalse(any(p.requires_grad for n, p in model.named_parameters() if "message_head" in n))
        set_model_requires_grad_for_phase(model, "C")
        self.assertTrue(any(p.requires_grad for n, p in model.named_parameters() if "message_head" in n))
        ParameterRegistry.from_model(model).validate(model)

    def test_communication_usage_gate_passes_with_activity(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        cfg.comm_min_valid_boundaries = 1
        cfg.comm_min_symbols_used = 2
        result = evaluate_communication_usage(
            cfg,
            {
                "comm_valid_boundaries": 4.0,
                "comm_delivery_count": 10.0,
                "comm_symbols_used": 3.0,
                "comm_symbol_entropy_normalized": 0.5,
                "comm_symbol_dominance": 0.4,
            },
        )
        self.assertEqual(result.status, GATE_STATUS_PASS)

    def test_listener_gate_uses_jsd_margin(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        cfg.comm_listener_jsd_margin = 0.01
        fail = evaluate_listener_causal_response(
            cfg,
            {"receiver_action_jsd_by_message_pair_mean": 0.0, "receiver_listener_pairs": 6.0},
        )
        self.assertNotEqual(fail.status, GATE_STATUS_PASS)
        ok = evaluate_listener_causal_response(
            cfg,
            {
                "receiver_action_jsd_by_message_pair_mean": 0.05,
                "receiver_listener_pairs": 6.0,
                "receiver_argmax_disagreement_frac": 0.2,
                "comm_valid_boundaries": 10.0,
            },
        )
        self.assertEqual(ok.status, GATE_STATUS_PASS)

    def test_silence_corruption_zeros_channels(self) -> None:
        channels = torch.ones((1, 4, 4, 20, 20))
        out = apply_message_channel_corruption(channels, mode=CommCorruptionMode.SILENCE)
        self.assertEqual(float(out.sum().item()), 0.0)

    def test_v6i3_aliases_resolve(self) -> None:
        for alias in ("v6i3", "v6i3_local_comm", "latent_v6i3_strategy_local_comm"):
            self.assertIn(alias, PRESET_REGISTRY)


if __name__ == "__main__":
    unittest.main()
