"""Step 5 — config-key canonicalization for the latent aux-return head.

Legacy code paths used ``latent_strategy_q_head`` / ``latent_strategy_q_coef``;
the canonical names are ``latent_strategy_aux_return_head`` /
``latent_strategy_aux_return_coef`` (q_phi(z|s) is *not* an action-value Q-fn,
it is an auxiliary per-z return regression head).

This module pins:

* The canonicalization helper folds legacy → canonical exactly once and is
  idempotent (calling twice is a no-op).
* ``PPOConfig`` only exposes canonical attribute names; the ``run_config.json``
  snapshot consequently never contains legacy keys.
* ``read_custom_ppo_metadata`` returns canonical-only cfg even for checkpoints
  whose payload still carries the legacy names.
* ``_model_kwargs_from_cfg`` derives ``use_strategy_aux_return_head`` from
  whichever name the cfg uses, but reads exactly one canonical key internally.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, fields

import torch

from rl.custom_ppo.inference import (
    canonicalize_latent_strategy_cfg,
    _model_kwargs_from_cfg,
    read_custom_ppo_metadata,
)
from rl.train_ppo import PPOConfig, write_run_config_json
from rl.ruleset_identity import RunIdentity


def _identity_for_cfg(cfg: PPOConfig) -> RunIdentity:
    """Minimal formal identity for snapshot tests that do not build a live env."""
    return RunIdentity(
        run_id=str(cfg.run_tag),
        canonical_map="map_a",
        resolved_map="map_a_open",
        ruleset_id="RULESET_V2_AQUATICUS_10S",
        ruleset_fingerprint="0" * 64,
        ruleset={},
        formal_result_eligible=True,
        identity_override_used=False,
    )


class LatentStrategyCanonicalizationHelperTests(unittest.TestCase):
    def test_legacy_keys_fold_into_canonical_keys(self) -> None:
        cfg = {"latent_strategy_q_head": True, "latent_strategy_q_coef": 0.42}
        canon = canonicalize_latent_strategy_cfg(cfg)
        self.assertEqual(canon["latent_strategy_aux_return_head"], True)
        self.assertEqual(canon["latent_strategy_aux_return_coef"], 0.42)
        self.assertNotIn("latent_strategy_q_head", canon)
        self.assertNotIn("latent_strategy_q_coef", canon)

    def test_canonical_keys_win_over_legacy_when_both_present(self) -> None:
        cfg = {
            "latent_strategy_q_head": False,
            "latent_strategy_aux_return_head": True,
            "latent_strategy_q_coef": 0.1,
            "latent_strategy_aux_return_coef": 0.9,
        }
        canon = canonicalize_latent_strategy_cfg(cfg)
        self.assertTrue(canon["latent_strategy_aux_return_head"])
        self.assertAlmostEqual(canon["latent_strategy_aux_return_coef"], 0.9)
        self.assertNotIn("latent_strategy_q_head", canon)
        self.assertNotIn("latent_strategy_q_coef", canon)

    def test_canonicalize_is_idempotent(self) -> None:
        legacy = {"latent_strategy_q_head": True, "latent_strategy_q_coef": 0.25}
        once = canonicalize_latent_strategy_cfg(legacy)
        twice = canonicalize_latent_strategy_cfg(once)
        self.assertEqual(once, twice)

    def test_canonicalize_does_not_mutate_input(self) -> None:
        cfg = {"latent_strategy_q_head": True, "latent_strategy_q_coef": 0.25}
        snapshot = dict(cfg)
        _ = canonicalize_latent_strategy_cfg(cfg)
        self.assertEqual(cfg, snapshot)

    def test_canonicalize_passes_through_unrelated_keys(self) -> None:
        cfg = {"use_latent_strategy": True, "latent_k": 4, "other": "x"}
        canon = canonicalize_latent_strategy_cfg(cfg)
        self.assertEqual(canon, cfg)


class PPOConfigCanonicalSnapshotTests(unittest.TestCase):
    """The dataclass + write_run_config_json must only emit canonical names."""

    def test_ppoconfig_has_only_canonical_field_names(self) -> None:
        names = {f.name for f in fields(PPOConfig)}
        self.assertIn("latent_strategy_aux_return_head", names)
        self.assertIn("latent_strategy_aux_return_coef", names)
        self.assertNotIn("latent_strategy_q_head", names)
        self.assertNotIn("latent_strategy_q_coef", names)

    def test_run_config_snapshot_uses_canonical_keys_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PPOConfig()
            cfg.run_tag = "unittest_canonicalize_snapshot"
            cfg.checkpoint_dir = tmp
            cfg.use_latent_strategy = True
            cfg.latent_strategy_aux_return_head = True
            cfg.latent_strategy_aux_return_coef = 0.7
            path = write_run_config_json(
                cfg,
                argv=["train_ppo.py", "--preset", "x"],
                run_identity=_identity_for_cfg(cfg),
            )
            with open(path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            resolved = snapshot["resolved_ppo_config"]
            self.assertIn("latent_strategy_aux_return_head", resolved)
            self.assertIn("latent_strategy_aux_return_coef", resolved)
            self.assertNotIn("latent_strategy_q_head", resolved)
            self.assertNotIn("latent_strategy_q_coef", resolved)
            # Sanity: the canonical values reach the snapshot.
            self.assertTrue(resolved["latent_strategy_aux_return_head"])
            self.assertAlmostEqual(float(resolved["latent_strategy_aux_return_coef"]), 0.7)


class ModelKwargsFromCfgTests(unittest.TestCase):
    """``_model_kwargs_from_cfg`` must accept legacy cfgs and emit canonical kwargs."""

    def test_legacy_q_head_enables_aux_return_head_kwarg(self) -> None:
        cfg = {
            "use_latent_strategy": True,
            "latent_k": 4,
            "latent_strategy_q_head": True,
            "latent_strategy_q_coef": 0.5,
        }
        kwargs = _model_kwargs_from_cfg(cfg)
        self.assertTrue(kwargs["use_strategy_aux_return_head"])

    def test_canonical_aux_return_head_passes_through(self) -> None:
        cfg = {
            "use_latent_strategy": True,
            "latent_k": 4,
            "latent_strategy_aux_return_head": True,
        }
        kwargs = _model_kwargs_from_cfg(cfg)
        self.assertTrue(kwargs["use_strategy_aux_return_head"])

    def test_aux_return_head_off_by_default(self) -> None:
        cfg = {"use_latent_strategy": True, "latent_k": 4}
        kwargs = _model_kwargs_from_cfg(cfg)
        self.assertFalse(kwargs["use_strategy_aux_return_head"])


class ReadCustomPpoMetadataCanonicalizesCfg(unittest.TestCase):
    """``read_custom_ppo_metadata`` must canonicalize legacy cfg in old checkpoints."""

    def test_legacy_cfg_in_checkpoint_is_canonicalized_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "legacy.zip")
            payload = {
                "format": "custom_ppo_v2",
                "model_state_dict": {},
                "global_state_dim": 19,
                "cfg": {
                    "use_latent_strategy": False,
                    "latent_strategy_q_head": True,
                    "latent_strategy_q_coef": 0.33,
                },
            }
            torch.save(payload, ckpt_path)
            meta = read_custom_ppo_metadata(ckpt_path)
            cfg = meta["cfg"]
            self.assertIn("latent_strategy_aux_return_head", cfg)
            self.assertIn("latent_strategy_aux_return_coef", cfg)
            self.assertNotIn("latent_strategy_q_head", cfg)
            self.assertNotIn("latent_strategy_q_coef", cfg)
            self.assertTrue(cfg["latent_strategy_aux_return_head"])
            self.assertAlmostEqual(float(cfg["latent_strategy_aux_return_coef"]), 0.33)


if __name__ == "__main__":
    unittest.main()
