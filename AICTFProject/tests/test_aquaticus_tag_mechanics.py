from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig


class AquaticusTagMechanicTests(unittest.TestCase):
    """RULESET_V1 sustained-pressure tagging (two simultaneous taggers).

    The V1 knobs are set explicitly because RULESET_V2 is now the default;
    V2's per-request eligibility model is covered by
    ``tests/test_aquaticus_tag_rules.py``.
    """

    def _make_core(self):
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=1,
            map_layout="map_b_split_lane",
            max_decision_steps=64,
            aquaticus_profile=True,
            rules_profile="OURS",
            taggers_required=2,
            tag_nearest_only=False,
            tag_min_interval_seconds=0.0,
            tag_channel_seconds=1.0,
            device="cpu",
            seed=123,
        )
        env = GPUCTFVecEnv(cfg)
        self.addCleanup(env.close)
        env.reset()
        return env.core

    def test_two_blue_defenders_sustained_in_range_tag_red_at_threshold(self) -> None:
        core = self._make_core()
        core.blue_x[:] = torch.tensor([[4.0, 4.8]])
        core.blue_y[:] = torch.tensor([[10.0, 10.0]])
        core.red_x[:] = torch.tensor([[4.4]])
        core.red_y[:] = torch.tensor([[10.0]])
        core.blue_tagged[:] = False
        core.red_tagged[:] = False
        core.blue_alive[:] = True
        core.red_alive[:] = True

        required_steps = int(torch.ceil(torch.tensor(core.cfg.tag_channel_seconds / core.dt)).item())
        observed_tag_step = None
        for step in range(required_steps + 2):
            core._apply_aquaticus_tag_rules(
                torch.zeros((1, 2), dtype=torch.bool),
                torch.zeros((1, 1), dtype=torch.bool),
            )
            if bool(core.red_tagged[0, 0].item()):
                observed_tag_step = step + 1
                break

        self.assertEqual(observed_tag_step, required_steps)
        self.assertAlmostEqual(float(core.cfg.tag_channel_seconds), 1.0)
        self.assertEqual(required_steps, 3)

    def test_losing_one_defender_resets_red_tag_accumulator(self) -> None:
        core = self._make_core()
        core.blue_x[:] = torch.tensor([[4.0, 4.8]])
        core.blue_y[:] = torch.tensor([[10.0, 10.0]])
        core.red_x[:] = torch.tensor([[4.4]])
        core.red_y[:] = torch.tensor([[10.0]])
        core.blue_tagged[:] = False
        core.red_tagged[:] = False
        core.blue_alive[:] = True
        core.red_alive[:] = True

        core._apply_aquaticus_tag_rules(
            torch.zeros((1, 2), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.bool),
        )
        self.assertGreater(float(core.red_tag_pressure_time[0, 0].item()), 0.0)

        core.blue_x[0, 1] = 14.0
        core._apply_aquaticus_tag_rules(
            torch.zeros((1, 2), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.bool),
        )

        self.assertEqual(float(core.red_tag_pressure_time[0, 0].item()), 0.0)
        self.assertFalse(bool(core.red_tagged[0, 0].item()))


if __name__ == "__main__":
    unittest.main()
