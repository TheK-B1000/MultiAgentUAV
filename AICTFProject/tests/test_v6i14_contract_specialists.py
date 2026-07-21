"""Pinning tests for V6I14 contract-specialist repertoire birth."""

from __future__ import annotations

import dataclasses
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.contract_specialists import contract_specialist_reward
from rl.custom_ppo.trainer_audit import log_plan_faithful_audit
from rl.presets import apply_preset
from rl.training.banner import print_training_banner


class V6i14PresetContractTests(unittest.TestCase):
    _ALIASES = [
        "v6i14",
        "v6i14_contract_specialists",
        "v6i14_contract_specialist_repertoire",
        "latent_v6i14_contract_specialists",
        "plan_faithful_latent_v6i14_contract_specialists",
    ]

    def test_aliases_resolve_equal(self) -> None:
        base = dataclasses.asdict(apply_preset(PPOConfig(), self._ALIASES[0]))
        for alias in self._ALIASES[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_diff_vs_v6i9_repertoire_is_contract_scaffold_only(self) -> None:
        parent = dataclasses.asdict(apply_preset(PPOConfig(), "v6i9_mapaware_repertoire_hardpool"))
        cfg = dataclasses.asdict(apply_preset(PPOConfig(), "v6i14"))
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "latent_contract_specialist_coef",
                "latent_contract_specialist_enabled",
                "run_tag",
            },
        )

    def test_runtime_contract(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i14")
        self.assertEqual(cfg.experiment_id, "v6i14")
        self.assertEqual(cfg.run_tag, "v6i14_contract_specialists_OP8_OP9_OP10")
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")
        self.assertTrue(cfg.latent_contract_specialist_enabled)
        self.assertAlmostEqual(float(cfg.latent_contract_specialist_coef), 0.25)
        self.assertAlmostEqual(float(cfg.latent_contract_specialist_clip), 1.0)

    def test_audit_banner_is_contract_aware(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i14")
        model = SimpleNamespace(
            uses_latent_strategy=True,
            latent_actor=SimpleNamespace(z_adapter=None),
            actor_cnn_feature_dim=128,
            _scalar_per_agent=20,
            z_embed_dim=16,
            z_onehot_dim=0,
            _decentralized_actor_in_dim=164,
            global_state_dim=34,
        )
        trainer = SimpleNamespace(cfg=cfg, model=model)
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_plan_faithful_audit(trainer)
        out = buf.getvalue()
        self.assertIn("Contract-specialist diagnostic scaffold: ON", out)
        self.assertIn("Do not label this run paper-faithful or Summer-faithful", out)
        self.assertIn("contract_specialist_rewards", out)
        self.assertNotIn("no handcrafted strategy labels", out)

    def test_training_profile_banner_is_contract_aware(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i14")
        cfg.enable_metrics_csv = False
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_training_banner(cfg, curriculum=None, max_agents=2, team_size="2v2")
        out = buf.getvalue()
        self.assertIn("Training profile: contract-specialist diagnostic scaffold", out)
        self.assertNotIn("Training profile: default latent (Summer implementation)", out)


class V6i14ContractRewardTests(unittest.TestCase):
    def _cfg(self, enabled: bool = True, coef: float = 0.25) -> SimpleNamespace:
        return SimpleNamespace(
            latent_contract_specialist_enabled=enabled,
            latent_contract_specialist_coef=coef,
            latent_contract_specialist_clip=1.0,
        )

    def _states(self) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.zeros((4, 34), dtype=torch.float32)
        nxt = torch.zeros((4, 34), dtype=torch.float32)
        nxt[:, 17] = 0.0

        # z0: opening pressure.
        nxt[0, 20] = 1.0
        nxt[0, 28] = 1.0

        # z1: defensive recovery under home pressure.
        prev[1, 19] = 0.9
        nxt[1, 10] = 1.0
        nxt[1, 19] = 0.2
        nxt[1, 21] = 1.0

        # z2: friendly-carrier support.
        nxt[2, 11] = 1.0
        nxt[2, 24] = 0.8
        nxt[2, 25] = 1.0

        # z3: carrier conversion progress.
        prev[3, 23] = 0.8
        nxt[3, 11] = 1.0
        nxt[3, 23] = 0.2
        return prev, nxt

    def test_disabled_contract_returns_zero(self) -> None:
        prev, nxt = self._states()
        z = torch.tensor([0, 1, 2, 3])
        out = contract_specialist_reward(prev, nxt, z, self._cfg(enabled=False))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_contract_map_rewards_selected_latent_role(self) -> None:
        prev, nxt = self._states()
        z = torch.tensor([0, 1, 2, 3])
        out = contract_specialist_reward(prev, nxt, z, self._cfg())
        expected = torch.tensor(
            [
                0.25,       # z0: 1.0 raw opening pressure
                0.2125,     # z1: 0.85 raw defensive recovery
                0.2275,     # z2: 0.91 raw carrier support
                0.1650,     # z3: 0.66 raw conversion progress
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-5), out)


if __name__ == "__main__":
    unittest.main()
