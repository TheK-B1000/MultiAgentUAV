"""Tests for router rollout dump packaging and integrity checks."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from types import SimpleNamespace

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.diagnostics.router_credit_audit import audit_router_rollout_dump
from rl.custom_ppo.diagnostics.router_rollout_dump import (
    assert_rollout_integrity,
    package_rollout_tensors,
    save_router_rollout_audit,
)
from rl.global_state import GLOBAL_STATE_V6I7_DIM
from rl.ppo_core import TensorDictRolloutBuffer


def _synthetic_buffer(*, t_steps: int = 8, n_envs: int = 2, latent_k: int = 4) -> TensorDictRolloutBuffer:
    device = torch.device("cpu")
    buf = TensorDictRolloutBuffer(t_steps, n_envs, device=device)
    buf.register_field("z", dtype=torch.long)
    buf.register_field("z_logits", (latent_k,))
    buf.register_field("z_log_probs")
    buf.register_field("router_decision_valid", dtype=torch.bool, deferred=True)
    buf.register_field("advantages")
    buf.register_field("returns")
    buf.register_field("rewards")
    buf.register_field("terminated", dtype=torch.bool)
    buf.register_field("truncated", dtype=torch.bool)
    buf.register_field("opponent_id", dtype=torch.long)
    buf.register_field("global_state", (GLOBAL_STATE_V6I7_DIM,))
    buf.register_field("router_advantages")
    buf.register_field("router_reward")
    buf.register_field("router_returns")

    for t in range(t_steps):
        gs = torch.randn(n_envs, GLOBAL_STATE_V6I7_DIM)
        z = torch.randint(0, latent_k, (n_envs,), dtype=torch.long)
        logits = torch.randn(n_envs, latent_k)
        rdv = torch.rand(n_envs) > 0.5
        buf.add(
            z=z,
            z_logits=logits,
            z_log_probs=torch.log_softmax(logits, dim=-1).gather(1, z.unsqueeze(1)).squeeze(1),
            advantages=torch.randn(n_envs),
            returns=torch.randn(n_envs),
            rewards=torch.randn(n_envs),
            terminated=torch.zeros(n_envs, dtype=torch.bool),
            truncated=torch.zeros(n_envs, dtype=torch.bool),
            opponent_id=torch.randint(0, 3, (n_envs,), dtype=torch.long),
            global_state=gs,
            router_advantages=torch.randn(n_envs),
            router_reward=torch.randn(n_envs),
            router_returns=torch.randn(n_envs),
        )
        buf.fields["router_decision_valid"][t] = rdv

    buf.pos = t_steps
    buf.full = True
    return buf


class TestRouterRolloutDump(unittest.TestCase):
    def test_package_and_integrity(self) -> None:
        cfg = PPOConfig()
        cfg.router_reward_enabled = True
        cfg.recurrent_selector_hidden_dim = 0
        cfg.latent_k = 4
        cfg.latent_resample_every_n = 32
        cfg.map_layout = "map_b"

        buf = _synthetic_buffer()
        tensors, meta = package_rollout_tensors(buf, cfg=cfg, trainer=SimpleNamespace())
        integrity = assert_rollout_integrity(cfg=cfg, tensors=tensors, latent_k=4)
        self.assertTrue(integrity["integrity_passed"])
        self.assertEqual(tensors["strategy_context"].shape[1], GLOBAL_STATE_V6I7_DIM)
        self.assertEqual(tensors["selected_z"].shape[0], tensors["strategy_context"].shape[0])
        self.assertEqual(meta["advantage_source_used"], "router")

    def test_save_and_audit_roundtrip(self) -> None:
        cfg = PPOConfig()
        cfg.router_reward_enabled = True
        cfg.recurrent_selector_hidden_dim = 0
        cfg.latent_k = 4
        cfg.latent_resample_every_n = 32
        cfg.map_layout = "map_b"
        cfg.latent_strategy_ppo_coef = 0.10
        cfg.clip_range = 0.2
        cfg.router_ent_coef = 0.005

        buf = _synthetic_buffer()
        tensors, meta = package_rollout_tensors(buf, cfg=cfg, trainer=SimpleNamespace())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "update_0001.pt"
            save_router_rollout_audit(
                out,
                tensors=tensors,
                metadata=meta,
                cfg=cfg,
                trainer=None,
            )
            payload = torch.load(out, map_location="cpu", weights_only=False)
            report = audit_router_rollout_dump(payload, out_dir=Path(tmp) / "audit")
            self.assertIn("summary", report)
            self.assertTrue((Path(tmp) / "audit" / "router_advantage_summary.json").is_file())
            self.assertTrue((Path(tmp) / "audit" / "router_advantage_by_z.csv").is_file())


if __name__ == "__main__":
    unittest.main()
