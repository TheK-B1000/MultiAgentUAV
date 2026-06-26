"""V6I3 slices 3–6 integration tests."""

from __future__ import annotations

import inspect
import unittest

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.communication.corruption import CommCorruptionMode, apply_message_channel_corruption
from rl.custom_ppo.communication.gates import (
    evaluate_communication_usage,
    evaluate_listener_causal_response,
)
from rl.custom_ppo.communication.listener import inject_message_symbol_into_grid
from rl.custom_ppo.communication.telemetry import rollout_comm_usage_telemetry
from rl.custom_ppo.communication.transport import LocalCommTransport
from rl.custom_ppo.curriculum.protocols import build_gate_protocol
from rl.custom_ppo.gate_protocol import (
    GATE_FAMILY_NAMES_V6I3,
    GATE_STATUS_PASS,
    V6I3_GATE_PROTOCOL,
    gate_config_fingerprint,
    get_gate_family_names,
    is_v6i2_dual_evidence_protocol,
    staged_latent_stdout_tag,
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

    def test_v6i3_confirmatory_communication_thresholds_are_nontrivial(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        self.assertEqual(cfg.comm_min_valid_boundaries, 1024)
        self.assertEqual(cfg.comm_min_deliveries, 4096)
        self.assertEqual(cfg.comm_num_symbols, 5)
        self.assertEqual(cfg.comm_silence_symbol, 0)
        self.assertEqual(cfg.comm_message_grid_channels, 4)
        self.assertEqual(cfg.comm_min_symbols_used, 2)
        self.assertEqual(cfg.comm_entropy_floor, 0.0)
        self.assertEqual(cfg.comm_symbol_dominance_ceiling, 1.0)
        self.assertGreater(cfg.comm_listener_jsd_margin, 0.0)
        self.assertEqual(cfg.comm_listener_min_passing_pairs, 3)
        self.assertEqual(cfg.comm_listener_min_states, 64)
        self.assertEqual(cfg.comm_listener_consecutive_updates, 1)

    def test_v6i3_confirmatory_fingerprints_are_frozen(self) -> None:
        self.assertEqual(gate_config_fingerprint(apply_preset(PPOConfig(), "v6i2")), "9c4aa70664495294")
        self.assertEqual(gate_config_fingerprint(apply_preset(PPOConfig(), "v6i3")), "de8e9c142e27acef")

    def test_v6i3_gate_protocol_families(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        self.assertEqual(get_gate_family_names(cfg), GATE_FAMILY_NAMES_V6I3)
        self.assertTrue(is_v6i2_dual_evidence_protocol(cfg))
        protocol = build_gate_protocol(cfg)
        self.assertEqual(protocol.version, V6I3_GATE_PROTOCOL)
        self.assertIn("communication_usage", protocol.required_families())
        self.assertNotIn("listener_causal_response", protocol.required_families())
        self.assertIn("actor_intervention", protocol.required_families())
        self.assertIn("behavioral_realization", protocol.required_families())
        self.assertNotIn("matched_seed_behavior", protocol.required_families())

    def test_v6i3_stdout_tag_is_not_legacy_v6i1(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        self.assertEqual(staged_latent_stdout_tag(str(cfg.gate_protocol_version)), "V6I3")

    def test_phase_a_controller_uses_dual_evidence_branch_for_v6i3(self) -> None:
        from rl.custom_ppo.curriculum.controller import V6I1CurriculumController

        source = inspect.getsource(V6I1CurriculumController.check_and_run_gate)
        self.assertIn("is_v6i2_dual_evidence_protocol(self.cfg)", source)

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
            comm_num_symbols=cfg.comm_num_symbols,
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
                "comm_delivery_count": 5000.0,
                "comm_symbols_used": 3.0,
                "comm_symbol_entropy_normalized": 0.8,
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
                "receiver_listener_pairs_above_margin": 6.0,
                "receiver_listener_states": 64.0,
                "receiver_argmax_disagreement_frac": 0.2,
                "comm_valid_boundaries": 10.0,
            },
        )
        self.assertEqual(ok.status, GATE_STATUS_PASS)

    def test_listener_gate_requires_above_margin_message_pairs(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        fail = evaluate_listener_causal_response(
            cfg,
            {
                "receiver_action_jsd_by_message_pair_mean": 0.01,
                "receiver_listener_pairs": 6.0,
                "receiver_listener_pairs_above_margin": 2.0,
                "receiver_listener_states": 64.0,
                "receiver_argmax_disagreement_frac": 0.1,
            },
        )
        self.assertNotEqual(fail.status, GATE_STATUS_PASS)
        ok = evaluate_listener_causal_response(
            cfg,
            {
                "receiver_action_jsd_by_message_pair_mean": 0.01,
                "receiver_listener_pairs": 6.0,
                "receiver_listener_pairs_above_margin": 3.0,
                "receiver_listener_states": 64.0,
                "receiver_argmax_disagreement_frac": 0.1,
            },
        )
        self.assertEqual(ok.status, GATE_STATUS_PASS)

    def test_silence_symbol_does_not_render_message_channel(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i3")
        grid = torch.ones((1, 4, 11, 20, 20))
        silence = inject_message_symbol_into_grid(
            grid,
            receiver_agent=0,
            symbol=0,
            num_symbols=cfg.comm_num_symbols,
            message_grid_channels=cfg.comm_message_grid_channels,
            silence_symbol=cfg.comm_silence_symbol,
            base_channels=7,
        )
        active = inject_message_symbol_into_grid(
            grid,
            receiver_agent=0,
            symbol=4,
            num_symbols=cfg.comm_num_symbols,
            message_grid_channels=cfg.comm_message_grid_channels,
            silence_symbol=cfg.comm_silence_symbol,
            base_channels=7,
        )
        self.assertEqual(float(silence[:, 0, 7:11].sum().item()), 0.0)
        self.assertEqual(float(active[:, 0, 10, 0, 0].item()), 1.0)

    def test_silence_symbol_clears_transport_without_delivery(self) -> None:
        from rl.custom_ppo.communication.config import resolve_comm_config

        cfg = apply_preset(PPOConfig(), "v6i3")
        comm = resolve_comm_config(cfg)
        transport = LocalCommTransport(comm)
        transport.reset(batch_size=1, num_agents=2, device=torch.device("cpu"))
        alive = torch.ones((1, 2), dtype=torch.bool)
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[1.0, 1.0]])
        transport.submit_outbound(
            symbols=torch.tensor([[0, 1]]),
            sender_x=x,
            sender_y=y,
            alive=alive,
            apply_dropout=False,
        )
        stats = transport.telemetry.to_dict(
            num_symbols=cfg.comm_num_symbols,
            silence_symbol=cfg.comm_silence_symbol,
            message_grid_channels=cfg.comm_message_grid_channels,
        )
        self.assertEqual(stats["comm_silence_count"], 1.0)
        self.assertEqual(stats["comm_send_count"], 1.0)
        self.assertEqual(stats["comm_symbols_used"], 1.0)
        transport.advance_step(
            alive=alive,
            sender_x=x,
            sender_y=y,
            receiver_x=x,
            receiver_y=y,
            cols=20,
            rows=20,
        )
        self.assertEqual(float(transport.active_signal[0, 0, 1].item()), 0.0)
        self.assertEqual(float(transport.active_signal[0, 1, 0].item()), -1.0)

    def test_rollout_usage_counts_active_symbols_separately_from_silence(self) -> None:
        class Buffer:
            pos = 2
            n_envs = 1
            device = torch.device("cpu")
            fields = {
                "message_boundary_mask": torch.tensor([[True], [True]]),
                "message_symbols": torch.tensor([[[0, 1, 1, 4]], [[0, 0, 2, 2]]]),
            }

        telemetry = rollout_comm_usage_telemetry(Buffer(), num_symbols=5, silence_symbol=0)
        self.assertEqual(telemetry["comm_silence_count"], 3.0)
        self.assertEqual(telemetry["comm_active_send_count"], 5.0)
        self.assertEqual(telemetry["comm_symbols_used"], 3.0)

    def test_silence_corruption_zeros_channels(self) -> None:
        channels = torch.ones((1, 4, 4, 20, 20))
        out = apply_message_channel_corruption(channels, mode=CommCorruptionMode.SILENCE)
        self.assertEqual(float(out.sum().item()), 0.0)

    def test_v6i3_uses_dual_evidence_actor_intervention_path(self) -> None:
        from rl.custom_ppo.gate_protocol import is_v6i2_dual_evidence_protocol
        from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
        from rl.custom_ppo.update.loss_result import measurement_from_pair_tensor

        cfg = apply_preset(PPOConfig(), "v6i3")
        self.assertTrue(is_v6i2_dual_evidence_protocol(cfg))
        updater = ActorInterventionEvidenceUpdater()
        latent_state = type("LS", (), {"update_cf_pair_jsd_ema": lambda self, vals, step: True})()
        measurement = measurement_from_pair_tensor(
            torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
            active_fraction=1.0,
            valid_groups=8,
        )
        result = updater.update(latent_state, measurement, cfg=cfg, global_step=1000)
        self.assertTrue(result.measurement_valid)

    def test_v6i3_aliases_resolve(self) -> None:
        for alias in ("v6i3", "v6i3_local_comm", "latent_v6i3_strategy_local_comm"):
            self.assertIn(alias, PRESET_REGISTRY)


if __name__ == "__main__":
    unittest.main()
