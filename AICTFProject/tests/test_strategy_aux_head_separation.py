"""Step 5 — ``StrategyEncoder`` and the aux-return head are always distinct.

Pre-Step-5, ``SharedActorCentralizedCritic`` aliased a single
:class:`StrategyEncoder` instance to either ``strategy_encoder`` (aux head off)
or ``strategy_aux_return_head`` (aux head on). The same code path therefore
silently meant different things at runtime — that's the ambiguity the audit
flagged as "sometimes q_phi is acting as the aux-return head, sometimes not."

After Step 5:

* ``strategy_encoder`` is **always** present when latent strategy is enabled
  (it implements the q_phi(z|s) policy).
* ``strategy_aux_return_head`` is an **optional, separate** :class:`StrategyEncoder`
  instance instantiated only when the aux-return head is enabled.
* Legacy checkpoints saved when the aux head was the same module as q_phi
  load by mirroring the on-disk ``strategy_aux_return_head.*`` weights into
  the new ``strategy_encoder.*`` slot (so the trained q_phi behavior is
  preserved). The new aux head receives the same weights as a starting
  point when it is still enabled in the new run.
"""

from __future__ import annotations

import unittest
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo.policy import (
    SharedActorCentralizedCritic,
    _migrate_legacy_aliased_strategy_modules,
)
from rl.latent_marl import StrategyEncoder


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


def _build(use_aux_head: bool, *, seed: int = 1234) -> SharedActorCentralizedCritic:
    torch.manual_seed(seed)
    return SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
        use_strategy_aux_return_head=use_aux_head,
    )


class AlwaysDistinctModulesTests(unittest.TestCase):
    """``strategy_encoder`` and ``strategy_aux_return_head`` must not share weights."""

    def test_strategy_encoder_is_always_present_when_latent_on(self) -> None:
        for use_aux in (False, True):
            with self.subTest(use_aux=use_aux):
                model = _build(use_aux)
                self.assertIsNotNone(model.strategy_encoder)
                self.assertIsInstance(model.strategy_encoder, StrategyEncoder)

    def test_aux_head_off_means_aux_head_is_none(self) -> None:
        model = _build(use_aux_head=False)
        self.assertIsNone(model.strategy_aux_return_head)

    def test_aux_head_on_creates_separate_strategy_encoder_instance(self) -> None:
        model = _build(use_aux_head=True)
        self.assertIsNotNone(model.strategy_aux_return_head)
        self.assertIsNot(
            model.strategy_aux_return_head,
            model.strategy_encoder,
            "aux-return head must be a *separate* StrategyEncoder, not an alias",
        )
        # Concretely: a parameter object of one is not the same Python object
        # as the corresponding parameter of the other (no weight sharing).
        enc_w = model.strategy_encoder.net[0].weight
        aux_w = model.strategy_aux_return_head.net[0].weight
        self.assertIsNot(enc_w, aux_w)

    def test_aux_head_module_is_listed_in_named_modules(self) -> None:
        model = _build(use_aux_head=True)
        names = {n for n, _ in model.named_modules()}
        self.assertIn("strategy_encoder", names)
        self.assertIn("strategy_aux_return_head", names)

    def test_strategy_logits_reads_strategy_encoder_only(self) -> None:
        """With aux head on, z-logits must come from ``strategy_encoder``, not the aux head."""
        model = _build(use_aux_head=True).eval()
        gs = torch.randn(3, model.q_phi_input_dim)
        with torch.no_grad():
            logits = model.strategy_logits(gs)
            ref = model.strategy_encoder(gs.float())
        self.assertTrue(torch.equal(logits, ref))
        # Make sure we did NOT silently fall back to the aux head; flip the
        # aux-head weights and prove the logits are unaffected.
        with torch.no_grad():
            for p in model.strategy_aux_return_head.parameters():
                p.copy_(p * 0.0 + 7.0)
            logits_after = model.strategy_logits(gs)
        self.assertTrue(torch.equal(logits, logits_after))


def _state_dict_filter(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in sd.items() if k.startswith(prefix)}


class LegacyAliasedCheckpointMigrationTests(unittest.TestCase):
    """A pre-Step-5 ckpt where the aux head served as q_phi must still load."""

    def _legacy_aux_on_state_dict(self) -> Dict[str, torch.Tensor]:
        """Simulate a pre-Step-5 ckpt saved with ``use_strategy_aux_return_head=True``.

        In that layout ``self.strategy_encoder`` was ``None`` and the lone
        :class:`StrategyEncoder` lived under ``strategy_aux_return_head.*``.
        """
        ref = _build(use_aux_head=True, seed=2024)
        new_sd = ref.state_dict()
        legacy_sd: Dict[str, torch.Tensor] = {}
        for k, v in new_sd.items():
            if k.startswith("strategy_encoder."):
                # Pre-Step-5 had no separate ``strategy_encoder`` slot when
                # aux head was on, so we drop those keys to emulate the older
                # checkpoint shape.
                continue
            legacy_sd[k] = v
        # ``strategy_aux_return_head.*`` is what served as q_phi back then;
        # those are the keys the old trainer actually wrote to disk.
        self.assertTrue(any(k.startswith("strategy_aux_return_head.") for k in legacy_sd))
        self.assertFalse(any(k.startswith("strategy_encoder.") for k in legacy_sd))
        return legacy_sd

    def test_legacy_aux_on_ckpt_loads_into_new_aux_on_model(self) -> None:
        legacy_sd = self._legacy_aux_on_state_dict()
        torch.manual_seed(99)
        target = _build(use_aux_head=True)
        target.eval()
        target.load_state_dict(legacy_sd, strict=True)

        # After migration both heads must be initialized from the legacy
        # aliased q_phi/aux module.
        target_sd = target.state_dict()
        for k in target_sd:
            if k.startswith("strategy_encoder."):
                legacy_aux_key = "strategy_aux_return_head" + k[len("strategy_encoder"):]
                self.assertIn(legacy_aux_key, legacy_sd)
                self.assertTrue(
                    torch.equal(target_sd[k], legacy_sd[legacy_aux_key]),
                    f"strategy_encoder.* must be mirrored from the legacy aux head; mismatch at {k!r}",
                )
            elif k.startswith("strategy_aux_return_head."):
                self.assertIn(k, legacy_sd)
                self.assertTrue(torch.equal(target_sd[k], legacy_sd[k]))

    def test_legacy_aux_on_ckpt_loads_into_new_aux_off_model(self) -> None:
        legacy_sd = self._legacy_aux_on_state_dict()
        torch.manual_seed(7)
        target = _build(use_aux_head=False)
        target.eval()
        target.load_state_dict(legacy_sd, strict=True)
        target_sd = target.state_dict()
        # ``strategy_encoder.*`` weights must come from the legacy aux head.
        for k in target_sd:
            if k.startswith("strategy_encoder."):
                legacy_aux_key = "strategy_aux_return_head" + k[len("strategy_encoder"):]
                self.assertTrue(torch.equal(target_sd[k], legacy_sd[legacy_aux_key]))
        # And the loaded model has no aux head, so q_phi(z|s) is fully driven
        # by strategy_encoder.
        self.assertIsNone(target.strategy_aux_return_head)

    def test_modern_aux_off_ckpt_passes_through(self) -> None:
        """Step-5 aux-off ckpts already use ``strategy_encoder.*``; migration is no-op."""
        ref = _build(use_aux_head=False, seed=2024)
        sd = ref.state_dict()
        self.assertTrue(any(k.startswith("strategy_encoder.") for k in sd))
        self.assertFalse(any(k.startswith("strategy_aux_return_head.") for k in sd))
        target = _build(use_aux_head=False, seed=99)
        target.load_state_dict(sd, strict=True)
        # Bitwise param parity.
        for k, v in sd.items():
            self.assertTrue(torch.equal(target.state_dict()[k], v), f"mismatch at {k}")


class MigrationHelperUnitTests(unittest.TestCase):
    """Targeted tests for the pure migration helper itself."""

    def test_helper_is_idempotent_on_modern_sd(self) -> None:
        ref = _build(use_aux_head=True, seed=7).state_dict()
        once = _migrate_legacy_aliased_strategy_modules(
            ref, has_strategy_encoder=True, has_strategy_aux_return_head=True
        )
        twice = _migrate_legacy_aliased_strategy_modules(
            once, has_strategy_encoder=True, has_strategy_aux_return_head=True
        )
        self.assertEqual(set(once.keys()), set(twice.keys()))
        for k in once:
            self.assertTrue(torch.equal(once[k], twice[k]))

    def test_helper_mirrors_legacy_aux_into_encoder(self) -> None:
        legacy = {
            "strategy_aux_return_head.net.0.weight": torch.full((4, 8), 1.5),
            "strategy_aux_return_head.net.0.bias": torch.full((4,), 0.25),
        }
        out = _migrate_legacy_aliased_strategy_modules(
            legacy, has_strategy_encoder=True, has_strategy_aux_return_head=True
        )
        self.assertIn("strategy_encoder.net.0.weight", out)
        self.assertIn("strategy_encoder.net.0.bias", out)
        self.assertTrue(torch.equal(out["strategy_encoder.net.0.weight"], legacy["strategy_aux_return_head.net.0.weight"]))
        # Aux head keys must still be present because the model keeps the head.
        self.assertIn("strategy_aux_return_head.net.0.weight", out)

    def test_helper_drops_legacy_aux_when_model_lacks_head(self) -> None:
        legacy = {
            "strategy_aux_return_head.net.0.weight": torch.full((4, 8), 1.5),
            "strategy_aux_return_head.net.0.bias": torch.full((4,), 0.25),
        }
        out = _migrate_legacy_aliased_strategy_modules(
            legacy, has_strategy_encoder=True, has_strategy_aux_return_head=False
        )
        self.assertIn("strategy_encoder.net.0.weight", out)
        self.assertNotIn("strategy_aux_return_head.net.0.weight", out)

    def test_helper_passes_through_when_canonical_keys_present(self) -> None:
        sd = {
            "strategy_encoder.net.0.weight": torch.full((4, 8), 2.0),
            "strategy_aux_return_head.net.0.weight": torch.full((4, 8), 3.0),
        }
        out = _migrate_legacy_aliased_strategy_modules(
            sd, has_strategy_encoder=True, has_strategy_aux_return_head=True
        )
        self.assertTrue(torch.equal(out["strategy_encoder.net.0.weight"], sd["strategy_encoder.net.0.weight"]))
        self.assertTrue(torch.equal(out["strategy_aux_return_head.net.0.weight"], sd["strategy_aux_return_head.net.0.weight"]))

    def test_helper_supports_prefixed_keys(self) -> None:
        legacy = {
            "wrapper.strategy_aux_return_head.net.0.weight": torch.full((4, 8), 1.5),
            "wrapper.strategy_aux_return_head.net.0.bias": torch.full((4,), 0.25),
        }
        out = _migrate_legacy_aliased_strategy_modules(
            legacy,
            prefix="wrapper.",
            has_strategy_encoder=True,
            has_strategy_aux_return_head=True,
        )
        self.assertIn("wrapper.strategy_encoder.net.0.weight", out)
        self.assertIn("wrapper.strategy_aux_return_head.net.0.weight", out)


if __name__ == "__main__":
    unittest.main()
