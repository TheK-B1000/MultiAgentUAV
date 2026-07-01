"""Tests for OP8/OP9/OP10 hard scripted opponent pool.

Tests:
1.  OP8 params: attacker=1, defender=1
2.  OP8 params: role_switch_prob very low (<0.05)
3.  OP8 params: coordinated_attack rate > 0.7 (across 1000-sample draw)
4.  OP9 params: attacker_style=0
5.  OP9 params: defender_style=1
6.  OP9 params: coordinated_attack rate > 0.6
7.  OP10 params: attacker_style=1
8.  OP10 params: defender_style=0
9.  OP10 params: role_switch_prob low (<0.08)
10. Registration: OP8/9/10 accepted by sample_batched_opponent_params
11. Registration: aliases OP8_INTERCEPTOR/OP9_FORTRESS/OP10_ESCORT also accepted
12. Preset registry: hardpool preset names exist in the preset registry
13. Preset content: hardpool presets carry OP8/OP9/OP10 in opponent_pool
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from opponent_params import sample_batched_opponent_params


def _sample(key: str, n: int = 256) -> dict:
    return sample_batched_opponent_params(
        kind="SCRIPTED",
        key=key,
        phase=key,
        n_agents=2,
        batch_size=n,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Class 1: OP8 parameter checks
# ---------------------------------------------------------------------------

class TestOP8Params(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.p = _sample("OP8", n=1024)

    def test_attacker_style_is_medium(self) -> None:
        """OP8 is striker-heavy: attacker_style must be 1 for all samples."""
        self.assertTrue(
            (self.p["attacker_style"] == 1).all().item(),
            "OP8 attacker_style should be 1 (medium attacker)",
        )

    def test_defender_style_is_medium(self) -> None:
        """OP8 has a medium defender to intercept flag returns."""
        self.assertTrue(
            (self.p["defender_style"] == 1).all().item(),
            "OP8 defender_style should be 1 (medium defender)",
        )

    def test_role_switch_prob_low(self) -> None:
        """OP8 commits to pursuer/blocker roles: role_switch_prob < 0.05."""
        mean_rsp = self.p["role_switch_prob"].mean().item()
        self.assertLess(mean_rsp, 0.05, f"OP8 mean role_switch_prob={mean_rsp:.4f} should be < 0.05")


# ---------------------------------------------------------------------------
# Class 2: OP9 parameter checks
# ---------------------------------------------------------------------------

class TestOP9Params(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.p = _sample("OP9", n=1024)

    def test_attacker_style_is_easy(self) -> None:
        """OP9 fortress mode: attacker_style=0 (no offensive pressure)."""
        self.assertTrue(
            (self.p["attacker_style"] == 0).all().item(),
            "OP9 attacker_style should be 0",
        )

    def test_defender_style_is_medium(self) -> None:
        """OP9 uses medium defender for tight flag guardianship."""
        self.assertTrue(
            (self.p["defender_style"] == 1).all().item(),
            "OP9 defender_style should be 1",
        )

    def test_high_coordination_rate(self) -> None:
        """OP9 counterattack relies on coordination: >60% of envs coordinated."""
        coord_rate = self.p["coordinated_attack"].float().mean().item()
        self.assertGreater(coord_rate, 0.60, f"OP9 coord_rate={coord_rate:.3f} should be > 0.60")


# ---------------------------------------------------------------------------
# Class 3: OP10 parameter checks
# ---------------------------------------------------------------------------

class TestOP10Params(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2)
        self.p = _sample("OP10", n=1024)

    def test_attacker_style_is_medium(self) -> None:
        """OP10 escort carrier is striker-led: attacker_style=1."""
        self.assertTrue(
            (self.p["attacker_style"] == 1).all().item(),
            "OP10 attacker_style should be 1",
        )

    def test_defender_style_is_easy(self) -> None:
        """OP10 prioritises escort over defence: defender_style=0."""
        self.assertTrue(
            (self.p["defender_style"] == 0).all().item(),
            "OP10 defender_style should be 0",
        )

    def test_role_switch_prob_low(self) -> None:
        """OP10 escort stays committed: role_switch_prob < 0.08."""
        mean_rsp = self.p["role_switch_prob"].mean().item()
        self.assertLess(mean_rsp, 0.08, f"OP10 mean role_switch_prob={mean_rsp:.4f} should be < 0.08")


# ---------------------------------------------------------------------------
# Class 4: Registration
# ---------------------------------------------------------------------------

class TestOpponentRegistration(unittest.TestCase):
    def test_canonical_keys_accepted(self) -> None:
        """OP8/OP9/OP10 are accepted by sample_batched_opponent_params without error."""
        for key in ("OP8", "OP9", "OP10"):
            with self.subTest(key=key):
                p = _sample(key, n=4)
                self.assertIn("attacker_style", p)
                self.assertIn("defender_style", p)
                self.assertIn("role_switch_prob", p)
                self.assertIn("coordinated_attack", p)

    def test_alias_keys_accepted(self) -> None:
        """Alias keys OP8_INTERCEPTOR/OP9_FORTRESS/OP10_ESCORT also work."""
        aliases = ("OP8_INTERCEPTOR", "OP9_FORTRESS", "OP10_ESCORT")
        for key in aliases:
            with self.subTest(key=key):
                p = sample_batched_opponent_params(
                    kind="SCRIPTED", key=key, phase=key, n_agents=2, batch_size=4, device="cpu"
                )
                self.assertIn("attacker_style", p, f"alias {key} not handled")


# ---------------------------------------------------------------------------
# Class 5: Preset registration and content
# ---------------------------------------------------------------------------

class TestHardpoolPresets(unittest.TestCase):
    def _registry(self) -> dict:
        from rl.presets import PRESET_REGISTRY  # type: ignore[import]
        return PRESET_REGISTRY

    def test_hardpool_presets_in_registry(self) -> None:
        """Both hardpool preset aliases are resolvable from the registry."""
        try:
            registry = self._registry()
        except ImportError:
            self.skipTest("rl.presets.get_preset_registry not importable in this env")
        for name in (
            "v6i8_balanced_hardpool",
            "v6i8_adapter_balanced_hardpool",
            "v6i8_sparse_hardpool",
            "v6i8_adapter_sparse_hardpool",
        ):
            with self.subTest(name=name):
                self.assertIn(name, registry, f"Preset '{name}' not found in registry")

    def test_hardpool_opponent_pool_contents(self) -> None:
        """Hardpool presets expose OP8, OP9, OP10 in their opponent_pool field."""
        try:
            from rl.config_presets import (  # type: ignore[import]
                v6i8_adapter_balanced_hardpool_config,
                v6i8_adapter_sparse_hardpool_config,
            )
        except ImportError:
            self.skipTest("rl.config_presets not importable in this env")
        for fn in (v6i8_adapter_balanced_hardpool_config, v6i8_adapter_sparse_hardpool_config):
            cfg = fn()
            pool = tuple(str(k).upper() for k in cfg.opponent_pool)
            with self.subTest(fn=fn.__name__):
                self.assertIn("OP8", pool)
                self.assertIn("OP9", pool)
                self.assertIn("OP10", pool)
                self.assertNotIn("OP5", pool, "hardpool should not include OP5")
                self.assertNotIn("OP6", pool, "hardpool should not include OP6")
                self.assertNotIn("OP7", pool, "hardpool should not include OP7")


if __name__ == "__main__":
    unittest.main()
