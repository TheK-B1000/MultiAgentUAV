"""Tests for OP6-OP12 adaptive memory flags under strategic niches."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_adaptive import _BTAdaptiveMixin
from gpu_env._core._bt_red import ROLE_ATTACKER, ROLE_INTERCEPTOR
from gpu_env._core._bt_profiles import (
    LRO_AUDITED_OPPONENT_POOL,
    build_profile_tensors,
    profile_for_level,
)
from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from opponent_params import sample_batched_opponent_params


def _core(opponent: str = "OP12"):
    env = GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=2,
            map_layout="map_b_split_lane",
            max_decision_steps=64,
            aquaticus_profile=True,
            rules_profile="OURS",
            device="cpu",
            seed=123,
        )
    )
    env.reset()
    core = env.core
    core._opponent_key[0] = opponent
    core.blue_alive[0] = True
    core.red_alive[0] = True
    core.blue_tagged[0] = False
    core.red_tagged[0] = False
    return core, env


def _op12_profile(core) -> dict[str, torch.Tensor]:
    return build_profile_tensors(
        ["OP12_LATE_CONVERTER"],
        device=core.device,
        batch_size=1,
    )


def _advance_adaptive_memory(core, prof: dict[str, torch.Tensor], steps: int) -> None:
    for t in range(steps):
        core.sim_step_count[0] = t
        core._update_adaptive_memory(prof)


def _run_live_op12_style(style: str, steps: int = 20, seed: int = 551001):
    env = GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=2,
            map_layout="map_b_split_lane",
            max_decision_steps=64,
            aquaticus_profile=True,
            rules_profile="OURS",
            device="cpu",
            seed=seed,
        )
    )
    env.env_method("set_phase", "OP12_LATE_CONVERTER")
    env.env_method("set_next_opponent", "SCRIPTED", "OP12_LATE_CONVERTER")
    env.reset()
    core = env.core
    env.env_method("set_phase", "OP12_LATE_CONVERTER")
    env.env_method("set_next_opponent", "SCRIPTED", "OP12_LATE_CONVERTER")
    core.blue_scripted = True
    core.set_blue_style(style)
    action = env.action_space.sample()
    zero_action = torch.zeros_like(torch.as_tensor(action)).cpu().numpy()
    for _ in range(steps):
        env.step_async(zero_action)
        env.step_wait()
    return core, env


class BTAdaptiveProfileTests(unittest.TestCase):
    def test_only_exploiter_and_converter_enable_adaptive(self) -> None:
        for lvl in range(5, 11):
            self.assertFalse(profile_for_level(lvl).adaptive_enabled, f"level {lvl}")
        self.assertTrue(profile_for_level(11).adaptive_enabled)
        self.assertTrue(profile_for_level(12).adaptive_enabled)

    def test_profile_tensors_export_adaptive_flag_for_audited_pool(self) -> None:
        keys = list(LRO_AUDITED_OPPONENT_POOL)
        prof = build_profile_tensors(keys, device=torch.device("cpu"), batch_size=len(keys))
        # Indices: OP6..OP10 False; OP11/OP12 True
        self.assertFalse(bool(prof["adaptive_enabled"][:5].any().item()))
        self.assertTrue(bool(prof["adaptive_enabled"][5:].all().item()))

    def test_matched_2v2_speed_bands_across_niches(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        for key in LRO_AUDITED_OPPONENT_POOL:
            params = sample_batched_opponent_params(
                "SCRIPTED",
                key,
                n_agents=2,
                batch_size=32,
                device="cpu",
                generator=gen,
            )
            self.assertGreaterEqual(float(params["speed_mult"].min().item()), 0.90, key)
            self.assertLessEqual(float(params["speed_mult"].max().item()), 1.05, key)

    def test_adaptive_hardpool_keys_include_audited_tags(self) -> None:
        for tag in LRO_AUDITED_OPPONENT_POOL:
            self.assertIn(tag, _BTAdaptiveMixin._ADAPTIVE_HARDPOOL_KEYS)

    def test_speed_overdrive_mask_allows_hardpool_red_to_exceed_base_cap(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                max_blue_agents=2,
                max_red_agents=2,
                map_layout="map_b",
                max_decision_steps=16,
                aquaticus_profile=True,
                rules_profile="OURS",
                device="cpu",
                seed=123,
            )
        )
        try:
            x = torch.full((1, 2), 5.0)
            y = torch.full((1, 2), 5.0)
            heading = torch.zeros((1, 2))
            speed = torch.full((1, 2), float(env.cfg.max_speed_cps))
            alive = torch.ones((1, 2), dtype=torch.bool)
            tx = torch.full((1, 2), 20.0)
            ty = torch.full((1, 2), 5.0)
            cap = torch.full((1, 2), float(env.cfg.max_speed_cps) * 1.20)
            mask = torch.ones((1, 2), dtype=torch.bool)
            _, _, _, speed2, _, _ = env.core._integrate_side(
                x,
                y,
                heading,
                speed,
                alive,
                tx,
                ty,
                speed_cap=cap,
                speed_overdrive_mask=mask,
            )
            self.assertGreater(float(speed2.max().item()), float(env.cfg.max_speed_cps))
        finally:
            env.close()

    def test_op12_split_detector_requires_separated_opposite_lanes(self) -> None:
        core, env = _core("OP12")
        try:
            prof = _op12_profile(core)
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 3.0
            core.blue_x[0, 1] = midline + 4.0
            core.blue_y[0, 0] = center_y - float(core.rows) * 0.35
            core.blue_y[0, 1] = center_y + float(core.rows) * 0.35

            _advance_adaptive_memory(core, prof, 4)

            self.assertEqual(int(core.bt_adapt_split_pressure_ticks[0].item()), 4)
            self.assertEqual(int(core.bt_adapt_split_first_trigger_step[0].item()), 3)
            self.assertGreater(int(core.bt_adapt_split_active_steps[0].item()), 0)
        finally:
            env.close()

    def test_op12_split_detector_does_not_trigger_on_clustered_rush(self) -> None:
        core, env = _core("OP12")
        try:
            prof = _op12_profile(core)
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 4.0
            core.blue_x[0, 1] = midline + 5.0
            core.blue_y[0, 0] = center_y - 0.5
            core.blue_y[0, 1] = center_y + 0.5

            _advance_adaptive_memory(core, prof, 4)

            self.assertEqual(int(core.bt_adapt_split_pressure_ticks[0].item()), 0)
            self.assertEqual(int(core.bt_adapt_split_first_trigger_step[0].item()), -1)
            self.assertEqual(int(core.bt_adapt_split_active_steps[0].item()), 0)
        finally:
            env.close()

    def test_op12_split_detector_resets_between_episodes(self) -> None:
        core, env = _core("OP12")
        try:
            prof = _op12_profile(core)
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 3.0
            core.blue_x[0, 1] = midline + 4.0
            core.blue_y[0, 0] = center_y - float(core.rows) * 0.35
            core.blue_y[0, 1] = center_y + float(core.rows) * 0.35
            _advance_adaptive_memory(core, prof, 8)

            core._reset_adaptive_memory(torch.tensor([True], device=core.device))

            self.assertEqual(int(core.bt_adapt_split_pressure_ticks[0].item()), 0)
            self.assertEqual(int(core.bt_adapt_split_first_trigger_step[0].item()), -1)
            self.assertEqual(int(core.bt_adapt_split_active_steps[0].item()), 0)
            self.assertEqual(float(core.bt_adapt_split_max_lateral_sep[0].item()), 0.0)
            self.assertEqual(float(core.bt_adapt_split_max_teammate_dist[0].item()), 0.0)
        finally:
            env.close()

    def test_op12_opening_escort_detector_triggers_on_lead_support(self) -> None:
        core, env = _core("OP12")
        try:
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 4.0
            core.blue_x[0, 1] = midline + 2.0
            core.blue_y[0, 0] = center_y + 0.75
            core.blue_y[0, 1] = center_y - 0.75
            core.sim_step_count[0] = 0

            for t in range(8):
                core.sim_step_count[0] = t
                core.blue_x[0, 0] += 0.45
                core.blue_x[0, 1] += 0.45
                core._get_bt_targets()

            self.assertGreaterEqual(int(core.bt_adapt_opening_escort_ticks[0].item()), 3)
            self.assertGreaterEqual(float(core.bt_adapt_opening_escort_score[0].item()), 3.0)
            self.assertGreater(int(core.bt_adapt_opening_escort_active_steps[0].item()), 0)
        finally:
            env.close()

    def test_op12_opening_escort_detector_does_not_trigger_on_wider_rush(self) -> None:
        core, env = _core("OP12")
        try:
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 4.0
            core.blue_x[0, 1] = midline + 1.0
            core.blue_y[0, 0] = center_y + 2.2
            core.blue_y[0, 1] = center_y - 2.2
            core.sim_step_count[0] = 0

            for t in range(8):
                core.sim_step_count[0] = t
                core.blue_x[0, 0] += 0.75
                core.blue_x[0, 1] += 0.75
                core._get_bt_targets()

            self.assertEqual(int(core.bt_adapt_opening_escort_ticks[0].item()), 0)
            self.assertEqual(int(core.bt_adapt_opening_escort_first_trigger_step[0].item()), -1)
            self.assertEqual(int(core.bt_adapt_opening_escort_active_steps[0].item()), 0)
        finally:
            env.close()

    def test_op12_opening_escort_detector_resets_between_episodes(self) -> None:
        core, env = _core("OP12")
        try:
            prof = _op12_profile(core)
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 4.0
            core.blue_x[0, 1] = midline + 1.0
            core.blue_y[0, 0] = center_y + 0.75
            core.blue_y[0, 1] = center_y - 0.75
            _advance_adaptive_memory(core, prof, 3)

            core._reset_adaptive_memory(torch.tensor([True], device=core.device))

            self.assertEqual(int(core.bt_adapt_opening_escort_ticks[0].item()), 0)
            self.assertEqual(int(core.bt_adapt_opening_escort_first_trigger_step[0].item()), -1)
            self.assertEqual(int(core.bt_adapt_opening_escort_active_steps[0].item()), 0)
        finally:
            env.close()

    def test_live_op12_opening_escort_score_separates_escort_from_rush(self) -> None:
        core, env = _run_live_op12_style("BLUE_ESCORT", steps=20, seed=551002)
        try:
            escort_score = float(core.bt_adapt_opening_escort_score[0].item())
        finally:
            env.close()

        core, env = _run_live_op12_style("BLUE_RUSH", steps=20, seed=551002)
        try:
            rush_score = float(core.bt_adapt_opening_escort_score[0].item())
            self.assertEqual(int(core.bt_adapt_opening_escort_first_trigger_step[0].item()), -1)
        finally:
            env.close()
        self.assertGreater(escort_score, rush_score)

    def test_op12_role_path_updates_opening_escort_detector(self) -> None:
        core, env = _core("OP12")
        try:
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 4.0
            core.blue_x[0, 1] = midline + 2.0
            core.blue_y[0, 0] = center_y + 0.75
            core.blue_y[0, 1] = center_y - 0.75
            core.blue_carrying[0] = False
            core.blue_alive[0] = True
            core.blue_tagged[0] = False
            core.red_alive[0] = True
            core.red_tagged[0] = False

            for t in range(5):
                core.sim_step_count[0] = t
                core.blue_x[0, 0] += 0.45
                core.blue_x[0, 1] += 0.45
                core._get_bt_targets()

            self.assertGreater(float(core.bt_adapt_opening_escort_score[0].item()), 0.0)
        finally:
            env.close()

    def test_op12_opening_escort_suspicion_does_not_force_intercept(self) -> None:
        core, env = _core("OP12")
        try:
            midline = float(core.cols) * 0.5
            center_y = float(core.rows) * 0.5
            core.blue_x[0, 0] = midline + 4.0
            core.blue_x[0, 1] = midline + 2.0
            core.blue_y[0, 0] = center_y + 0.75
            core.blue_y[0, 1] = center_y - 0.75
            core.blue_carrying[0] = False
            core.blue_alive[0] = True
            core.blue_tagged[0] = False
            core.red_alive[0] = True
            core.red_tagged[0] = False

            for t in range(8):
                core.bt_role_lock_ticks[0] = 0
                core.sim_step_count[0] = t
                core.blue_x[0, 0] += 0.45
                core.blue_x[0, 1] += 0.45
                core._get_bt_targets()

            self.assertGreaterEqual(int(core.bt_adapt_opening_escort_ticks[0].item()), 2)
            self.assertEqual(int(core.bt_adapt_escort_confirm_first_step[0].item()), -1)
            self.assertNotIn(ROLE_INTERCEPTOR, core.bt_red_role[0].tolist())
        finally:
            env.close()

    def test_op12_post_pickup_escort_confirmation_triggers_on_carrier_support(self) -> None:
        core, env = _core("OP12")
        try:
            center_y = float(core.rows) * 0.5
            core.blue_alive[0] = True
            core.blue_tagged[0] = False
            core.red_alive[0] = True
            core.red_tagged[0] = False
            core.blue_flag_home[0, 0] = 1.0
            core.blue_flag_home[0, 1] = center_y
            core.blue_carrying[0] = False
            core.blue_x[0, 0] = 21.0
            core.blue_x[0, 1] = 23.5
            core.blue_y[0, 0] = center_y
            core.blue_y[0, 1] = center_y + 1.0
            core._get_bt_targets()

            core.blue_carrying[0, 0] = True
            for t in range(1, 9):
                core.bt_role_lock_ticks[0] = 0
                core.sim_step_count[0] = t
                core.blue_x[0, 0] -= 0.5
                core.blue_x[0, 1] -= 0.5
                core._get_bt_targets()

            self.assertGreaterEqual(int(core.bt_adapt_escort_confirm_ticks[0].item()), 5)
            self.assertGreaterEqual(int(core.bt_adapt_escort_confirm_first_step[0].item()), 0)
            self.assertEqual(int(core.bt_adapt_escort_confirm_carrier_id[0].item()), 0)
            self.assertEqual(int(core.bt_adapt_escort_confirm_protector_id[0].item()), 1)
        finally:
            env.close()

    def test_blue_rush_and_escort_post_pickup_targets_are_separable(self) -> None:
        core, env = _core("OP12")
        try:
            center_y = float(core.rows) * 0.5
            max_y = float(core.rows - 1)
            core.blue_alive[0] = True
            core.blue_tagged[0] = False
            core.blue_carrying[0] = False
            core.blue_carrying[0, 0] = True
            core.blue_x[0, 0] = 14.0
            core.blue_y[0, 0] = center_y + 2.0
            core.blue_x[0, 1] = 15.0
            core.blue_y[0, 1] = center_y + 2.5
            core.red_flag_pos[0, 0] = float(core.cols - 2)
            core.red_flag_pos[0, 1] = center_y

            core.set_blue_style("BLUE_RUSH")
            rush_x, rush_y = core._assign_blue_style_targets()
            core.set_blue_style("BLUE_ESCORT")
            escort_x, escort_y = core._assign_blue_style_targets()

            self.assertLess(float(rush_y[0, 1].item()), max_y * 0.25)
            self.assertAlmostEqual(float(escort_y[0, 1].item()), float(core.blue_y[0, 0].item()), places=4)
            self.assertAlmostEqual(float(escort_x[0, 1].item()), float(core.blue_x[0, 0].item()) + 1.5, places=4)
        finally:
            env.close()

    def test_op12_opening_delays_carrier_intercept_until_late_phase(self) -> None:
        core, env = _core("OP12")
        try:
            core.blue_carrying[0, 0] = True
            core.blue_x[0, 0] = 12.0
            core.blue_y[0, 0] = 10.0
            core.blue_flag_home[0, 0] = 1.0
            core.blue_flag_home[0, 1] = 10.0
            core.red_x[0, 0] = 16.0
            core.red_y[0, 0] = 10.0
            core.red_x[0, 1] = 18.0
            core.red_y[0, 1] = 12.0

            core.sim_step_count[0] = 5
            core._get_bt_targets()
            self.assertNotIn(ROLE_INTERCEPTOR, core.bt_red_role[0].tolist())
            self.assertIn(ROLE_ATTACKER, core.bt_red_role[0].tolist())

            core.bt_role_lock_ticks[0] = 0
            core.sim_step_count[0] = 25
            core._get_bt_targets()
            self.assertIn(ROLE_INTERCEPTOR, core.bt_red_role[0].tolist())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
