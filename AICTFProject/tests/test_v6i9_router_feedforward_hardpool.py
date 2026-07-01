"""Pins v6i9 feedforward router preset vs recurrent sibling."""
from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.presets.families.plan_faithful.v6_router_adapters import (
    apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool,
    apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool,
)
from tests.test_preset_resolution import SNAPSHOT_PATH


def _load_snapshot() -> dict:
  with open(SNAPSHOT_PATH, encoding="utf-8") as f:
    return json.load(f)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V6I9RouterFeedforwardPresetTests(unittest.TestCase):
  def test_feedforward_recurrence_disabled(self) -> None:
    cfg = _resolved("v6i9_mapaware_router_feedforward_hardpool")
    self.assertEqual(int(cfg.recurrent_selector_hidden_dim), 0)
    self.assertEqual(int(cfg.recurrent_seq_len), 0)
    self.assertEqual(int(cfg.recurrent_burn_in), 0)
    self.assertEqual(int(cfg.router_chunks_per_batch), 0)

  def test_router_stage_and_freeze_flags(self) -> None:
    cfg = _resolved("v6i9_mapaware_router_feedforward_hardpool")
    self.assertEqual(str(cfg.v6i9_training_stage), "router")
    self.assertTrue(bool(cfg.router_freeze_actor))
    self.assertEqual(str(cfg.latent_assignment_mode), "router")
    self.assertTrue(bool(cfg.train_router_when_forced))
    self.assertTrue(bool(cfg.train_router_critic_when_forced))

  def test_minimal_diff_vs_recurrent_sibling(self) -> None:
    recurrent = dataclasses.asdict(
      apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool(PPOConfig())
    )
    feedforward = dataclasses.asdict(
      apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool(PPOConfig())
    )
    diffs = {
      key: (recurrent.get(key), feedforward.get(key))
      for key in set(recurrent) | set(feedforward)
      if recurrent.get(key) != feedforward.get(key)
    }
    self.assertEqual(
      set(diffs),
      {
        "recurrent_selector_hidden_dim",
        "recurrent_seq_len",
        "recurrent_burn_in",
        "router_chunks_per_batch",
        "router_reinitialize_on_load",
        "run_tag",
      },
      f"unexpected diff keys: {sorted(diffs)}",
    )
    self.assertGreater(int(recurrent["recurrent_selector_hidden_dim"]), 0)
    self.assertTrue(bool(feedforward["router_reinitialize_on_load"]))

  def test_registry_aliases_resolve(self) -> None:
    for alias in (
      "v6i9_mapaware_router_feedforward_hardpool",
      "v6i9_router_feedforward_hardpool",
      "plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool",
    ):
      cfg = _resolved(alias)
      self.assertIn("feedforward", cfg.run_tag)

  def test_snapshot_entry_present(self) -> None:
    snapshot = _load_snapshot()
    key = "v6i9_mapaware_router_feedforward_hardpool"
    self.assertIn(key, snapshot, f"missing {key} in {SNAPSHOT_PATH}")
    entry = snapshot[key]
    self.assertEqual(int(entry["recurrent_selector_hidden_dim"]), 0)
    self.assertEqual(str(entry["v6i9_training_stage"]), "router")
    self.assertTrue(bool(entry["router_freeze_actor"]))

  def test_freeze_name_classification_contract(self) -> None:
    from rl.custom_ppo.trainer_optimizers import (
      is_shared_frozen_actor_param,
      is_z_specific_actor_param,
    )

    self.assertTrue(is_shared_frozen_actor_param("latent_actor.actor_cnn.conv.0.weight"))
    self.assertTrue(is_shared_frozen_actor_param("latent_actor.body.0.weight"))
    self.assertTrue(is_z_specific_actor_param("latent_actor.latent_adapters.0.weight"))
    self.assertTrue(is_z_specific_actor_param("latent_actor.strategy_embedding.weight"))
    self.assertFalse(is_z_specific_actor_param("strategy_encoder.net.0.weight"))


if __name__ == "__main__":
  unittest.main()
