from __future__ import annotations

from collections import deque
import unittest

import numpy as np
import torch

from game_field_gpu import (
    GPUCTFSingleEnv,
    GPUCTFVecEnv,
    GPUFieldConfig,
    MAP_A_OPEN,
    MAP_B_SPLIT_LANE,
    MAP_B_SPLIT_LANE_V2,
    RewardConfig,
    VEC_OBS_DIM,
)
from game_manager import (
    SPARSE_FLAG_CAPTURE_POINTS,
    SPARSE_MINE_TAG_POINTS,
    SPARSE_OOB_POINTS,
    SPARSE_TAG_NO_FLAG_POINTS,
    SPARSE_TAG_WITH_FLAG_POINTS,
)
from macro_actions import MacroAction
from rl.global_state import GLOBAL_STATE_DIM, GLOBAL_STATE_FIELD_NAMES


class EnvironmentContractTests(unittest.TestCase):
    def test_single_env_observation_space_and_info_contract(self) -> None:
        env = GPUCTFSingleEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=7))
        try:
            obs, info = env.reset(seed=7)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(info, {})

            action = np.zeros(env.action_space.shape, dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertEqual(info["action_mask"].shape, (2 * (5 + 50),))
            self.assertEqual(info["action_mask"].dtype, np.float32)
            self.assertEqual(info["agent_alive"].shape, (2,))
            self.assertEqual(info["global_state"].shape, (GLOBAL_STATE_DIM,))
            self.assertEqual(info["global_state"].dtype, np.float32)
            self.assertEqual(info["map_set"], "train")
            for key in (
                "reward_terminal",
                "reward_offense",
                "reward_pbrs",
                "reward_team",
                "reward_sparse",
                "reward_sparse_points",
                "reward_failure",
                "reward_total",
            ):
                self.assertIn(key, info)
                self.assertIsInstance(info[key], float)
        finally:
            env.close()

    def test_global_state_is_documented_structured_vector(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=2, n_agents_per_team=2, device="cpu", seed=11))
        try:
            env.reset()
            state = env.state()
            self.assertEqual(state.shape, (2, GLOBAL_STATE_DIM))
            self.assertEqual(state.dtype, np.float32)
            self.assertTrue(np.isfinite(state).all())

            core = env.core
            core.blue_score[0] = 2
            core.red_score[0] = 1
            scored_state = env.state()[0]
            score_den = float(core.score_limit)
            self.assertAlmostEqual(
                float(scored_state[GLOBAL_STATE_FIELD_NAMES.index("blue_score_norm")]),
                2.0 / score_den,
            )
            self.assertAlmostEqual(
                float(scored_state[GLOBAL_STATE_FIELD_NAMES.index("red_score_norm")]),
                1.0 / score_den,
            )
            self.assertAlmostEqual(
                float(scored_state[GLOBAL_STATE_FIELD_NAMES.index("score_diff_norm")]),
                1.0 / score_den,
            )

            env.step_async(np.zeros((2, 4), dtype=np.int64))
            _, _, _, infos = env.step_wait()
            for info in infos:
                self.assertEqual(info["global_state"].shape, (GLOBAL_STATE_DIM,))
                self.assertEqual(info["global_state"].dtype, np.float32)
        finally:
            env.close()

    def test_global_state_strategy_features_are_observable_not_labels(self) -> None:
        strategy_features = {
            "flag_pressure_blue",
            "flag_pressure_red",
            "home_defense_blue",
            "home_defense_red",
            "carrier_dist_home",
            "carrier_enemy_nearest_dist",
            "carrier_teammate_support",
            "mean_blue_red_dist",
            "min_blue_red_dist",
            "blue_near_enemy_flag_count",
            "red_near_enemy_flag_count",
            "blue_near_home_flag_count",
            "red_near_home_flag_count",
            "team_pairwise_distance_mean",
            "team_pairwise_distance_std",
        }
        forbidden_labels = {"opponent_id", "phase_id", "attack_label", "defend_label", "role_label"}

        self.assertEqual(len(GLOBAL_STATE_FIELD_NAMES), GLOBAL_STATE_DIM)
        self.assertTrue(strategy_features.issubset(set(GLOBAL_STATE_FIELD_NAMES)))
        self.assertTrue(forbidden_labels.isdisjoint(set(GLOBAL_STATE_FIELD_NAMES)))

    def test_reset_seed_reproducibility(self) -> None:
        cfg1 = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=123, map_set="train")
        cfg2 = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=999, map_set="train")
        env1 = GPUCTFSingleEnv(cfg1)
        env2 = GPUCTFSingleEnv(cfg2)
        try:
            obs1, _ = env1.reset(seed=42)
            obs2, _ = env2.reset(seed=42)
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])
            np.testing.assert_array_equal(env1.state(), env2.state())
        finally:
            env1.close()
            env2.close()

    def test_map_sets_are_reproducible_and_disjoint(self) -> None:
        train_env = GPUCTFSingleEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=42, map_set="train"))
        eval_env = GPUCTFSingleEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=42, map_set="eval"))
        try:
            train_obs, _ = train_env.reset(seed=42)
            eval_obs, _ = eval_env.reset(seed=42)
            self.assertFalse(np.array_equal(train_obs["vec"], eval_obs["vec"]))
            self.assertFalse(np.array_equal(train_env.state(), eval_env.state()))
        finally:
            train_env.close()
            eval_env.close()

    def test_team_size_configurations_scale_spaces(self) -> None:
        for n_agents in (2, 4, 6):
            for map_set in ("train", "eval"):
                env = GPUCTFSingleEnv(
                    GPUFieldConfig(n_envs=1, n_agents_per_team=n_agents, device="cpu", seed=5, map_set=map_set)
                )
                try:
                    obs, _ = env.reset(seed=5)
                    self.assertEqual(obs["grid"].shape, (n_agents, 7, 20, 20))
                    self.assertEqual(obs["vec"].shape, (n_agents, VEC_OBS_DIM))
                    self.assertEqual(obs["agent_mask"].shape, (n_agents,))
                    self.assertEqual(obs["mask"].shape, (n_agents * (5 + 50),))
                    self.assertEqual(len(env.action_space.nvec), n_agents * 2)
                    self.assertEqual(env.state().shape, (GLOBAL_STATE_DIM,))
                finally:
                    env.close()

    def test_default_layout_preserves_map_a_open_arena_contract(self) -> None:
        env = GPUCTFSingleEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=51))
        try:
            obs, _ = env.reset(seed=51)
            self.assertEqual(env.vec.core.map_layout, MAP_A_OPEN)
            self.assertEqual(env.vec.core.cfg.num_cnn_channels, 7)
            self.assertEqual(obs["grid"].shape, (2, 7, 20, 20))
            self.assertFalse(bool(env.vec.core.obstacle_active.any().item()))
            self.assertFalse(bool(env.vec.core.cfg.obstacle_obs_channel))
        finally:
            env.close()

    def test_map_b_split_lane_obstacles_and_observation_channel(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=1,
            n_agents_per_team=2,
            device="cpu",
            seed=52,
            map_layout=MAP_B_SPLIT_LANE,
            map_b_vertical_mirror_prob=0.0,
        )
        env = GPUCTFSingleEnv(cfg)
        try:
            obs, _ = env.reset(seed=52)
            core = env.vec.core
            self.assertEqual(core.map_layout, MAP_B_SPLIT_LANE)
            self.assertEqual(core.cfg.num_cnn_channels, 8)
            self.assertEqual(obs["grid"].shape, (2, 8, 20, 20))
            self.assertTrue(bool(core.obstacle_active[0, 0].item()))
            wall_channel = obs["grid"][:, 7, :, :]
            self.assertGreater(float(wall_channel.sum()), 0.0)
            flag_x = torch.stack([core.blue_flag_home[:, 0], core.red_flag_home[:, 0]], dim=1)
            flag_y = torch.stack([core.blue_flag_home[:, 1], core.red_flag_home[:, 1]], dim=1)
            self.assertFalse(bool(core._points_in_obstacles(flag_x, flag_y).any().item()))
            self.assertFalse(bool(core._points_in_obstacles(core.blue_x, core.blue_y).any().item()))
            self.assertFalse(bool(core._points_in_obstacles(core.red_x, core.red_y).any().item()))
        finally:
            env.close()

    def test_map_b_vertical_mirror_swaps_wall_band(self) -> None:
        base = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                device="cpu",
                seed=53,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        mirrored = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                device="cpu",
                seed=53,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=1.0,
            )
        )
        try:
            base.core.reset_all()
            mirrored.core.reset_all()
            base_rect = base.core.obstacle_rects[0, 0].detach().cpu().numpy()
            mirror_rect = mirrored.core.obstacle_rects[0, 0].detach().cpu().numpy()
            self.assertFalse(bool(base.core.map_vertical_mirror[0].item()))
            self.assertTrue(bool(mirrored.core.map_vertical_mirror[0].item()))
            self.assertAlmostEqual(float(base_rect[0]), float(mirror_rect[0]), places=5)
            self.assertAlmostEqual(float(base_rect[2]), float(mirror_rect[2]), places=5)
            self.assertGreater(float(mirror_rect[1]), float(base_rect[1]))
            self.assertGreater(float(mirror_rect[3]), float(base_rect[3]))
        finally:
            base.close()
            mirrored.close()

    def test_map_b_obstacle_collision_reverts_wall_entry(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                device="cpu",
                seed=54,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        try:
            core = env.core
            rect = core.obstacle_rects[0, 0]
            y_mid = ((rect[1] + rect[3]) * 0.5).item()
            prev_x = torch.tensor([[float(rect[0].item()) - 0.5, 2.0]], device=core.device)
            prev_y = torch.tensor([[y_mid, 2.0]], device=core.device)
            next_x = torch.tensor([[float(rect[0].item()) + 0.2, 2.0]], device=core.device)
            next_y = prev_y.clone()
            speed = torch.ones_like(prev_x)
            alive = torch.ones_like(prev_x, dtype=torch.bool)
            out_x, out_y, out_speed, hit = core._revert_obstacle_hits(prev_x, prev_y, next_x, next_y, speed, alive)
            self.assertTrue(bool(hit[0, 0].item()))
            self.assertAlmostEqual(float(out_x[0, 0].item()), float(prev_x[0, 0].item()), places=5)
            self.assertAlmostEqual(float(out_y[0, 0].item()), float(prev_y[0, 0].item()), places=5)
            self.assertAlmostEqual(float(out_speed[0, 0].item()), 0.0, places=5)
        finally:
            env.close()

    def test_map_b_route_targets_redirect_across_wall(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                device="cpu",
                seed=55,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        try:
            core = env.core
            rect = core.obstacle_rects[0, 0]
            own_x = torch.tensor([[float(rect[0].item()) - 2.0, float(rect[0].item()) - 2.0]], device=core.device)
            own_y = torch.tensor([[(float(rect[1].item()) + float(rect[3].item())) * 0.5, 2.0]], device=core.device)
            target_x = torch.tensor([[float(rect[2].item()) + 6.0, float(rect[2].item()) + 6.0]], device=core.device)
            target_y = torch.tensor([[(float(rect[1].item()) + float(rect[3].item())) * 0.5, 2.0]], device=core.device)
            routed_x, routed_y = core._route_targets_around_obstacles(own_x, own_y, target_x, target_y)
            self.assertNotAlmostEqual(float(routed_y[0, 0].item()), float(target_y[0, 0].item()), places=5)
            self.assertTrue(
                float(routed_y[0, 0].item()) < float(rect[1].item())
                or float(routed_y[0, 0].item()) > float(rect[3].item())
            )
            self.assertAlmostEqual(float(routed_x[0, 1].item()), float(target_x[0, 1].item()), places=5)
            self.assertAlmostEqual(float(routed_y[0, 1].item()), float(target_y[0, 1].item()), places=5)
        finally:
            env.close()

    def test_map_b_grid_reachability_has_upper_and_lower_routes(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                device="cpu",
                seed=56,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        try:
            core = env.core
            blocked = core._obstacle_grid_mask(side="blue")[0].detach().cpu().numpy() > 0.5
            rect = core.obstacle_rects[0, 0].detach().cpu().numpy()
            mid_col = blocked.shape[1] // 2
            open_rows = np.where(~blocked[:, mid_col])[0]
            self.assertTrue(np.any(open_rows < int(np.floor(rect[1]))))
            self.assertTrue(np.any(open_rows > int(np.ceil(rect[3]))))

            start = tuple(np.rint(core.blue_flag_home[0].detach().cpu().numpy()).astype(int)[::-1])
            goal = tuple(np.rint(core.red_flag_home[0].detach().cpu().numpy()).astype(int)[::-1])
            q: deque[tuple[int, int]] = deque([start])
            seen = {start}
            reachable = False
            while q:
                r, c = q.popleft()
                if (r, c) == goal:
                    reachable = True
                    break
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nc < 0 or nr >= blocked.shape[0] or nc >= blocked.shape[1]:
                        continue
                    if blocked[nr, nc] or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            self.assertTrue(reachable)
        finally:
            env.close()

    def test_map_b_render_and_terminal_payload_include_wall_route_telemetry(self) -> None:
        env = GPUCTFSingleEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=1,
                device="cpu",
                seed=57,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        try:
            obs, _ = env.reset(seed=57)
            frame = env.render(mode="rgb_array")
            self.assertGreater(int(((frame == np.array([70, 76, 86], dtype=np.uint8)).all(axis=2)).sum()), 0)
            action = np.zeros(env.action_space.shape, dtype=np.int64)
            _, _, _, truncated, info = env.step(action)
            self.assertTrue(truncated)
            self.assertEqual(info["map_layout"], MAP_B_SPLIT_LANE)
            self.assertIn("obstacle_collision_events_per_episode", info)
            self.assertIn("blue_route_upper_crossings", info)
            self.assertIn("blue_route_lower_crossings", info)
            self.assertIn("episode_result", info)
            self.assertEqual(info["episode_result"]["map_layout"], MAP_B_SPLIT_LANE)
            self.assertIn("obstacle_collision_events_per_episode", info["episode_result"])
            self.assertEqual(obs["grid"].shape[1], 8)
        finally:
            env.close()

    def test_map_b_v2_uses_lower_friction_wall_and_context_route_telemetry(self) -> None:
        base = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                device="cpu",
                seed=58,
                map_layout=MAP_B_SPLIT_LANE,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        v2 = GPUCTFSingleEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=1,
                device="cpu",
                seed=58,
                map_layout=MAP_B_SPLIT_LANE_V2,
                map_b_vertical_mirror_prob=0.0,
            )
        )
        try:
            base.core.reset_all()
            obs, _ = v2.reset(seed=58)
            core = v2.vec.core
            self.assertEqual(core.map_layout, MAP_B_SPLIT_LANE_V2)
            self.assertEqual(obs["grid"].shape, (2, 8, 20, 20))
            base_rect = base.core.obstacle_rects[0, 0].detach().cpu().numpy()
            v2_rect = core.obstacle_rects[0, 0].detach().cpu().numpy()
            base_area = float((base_rect[2] - base_rect[0]) * (base_rect[3] - base_rect[1]))
            v2_area = float((v2_rect[2] - v2_rect[0]) * (v2_rect[3] - v2_rect[1]))
            self.assertLess(v2_area, base_area)

            action = np.zeros(v2.action_space.shape, dtype=np.int64)
            _, _, _, truncated, info = v2.step(action)
            self.assertTrue(truncated)
            self.assertEqual(info["map_layout"], MAP_B_SPLIT_LANE_V2)
            for key in (
                "blue_attack_upper_crossings",
                "blue_return_lower_crossings",
                "blue_intercept_upper_crossings",
                "red_attack_lower_crossings",
                "red_return_upper_crossings",
                "red_intercept_lower_crossings",
            ):
                self.assertIn(key, info)
                self.assertIn(key, info["episode_result"])
        finally:
            base.close()
            v2.close()

    def test_time_limit_uses_truncated_not_terminated(self) -> None:
        env = GPUCTFSingleEnv(
            GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=1, device="cpu", seed=3)
        )
        try:
            env.reset(seed=3)
            action = np.zeros(env.action_space.shape, dtype=np.int64)
            _, _, terminated, truncated, info = env.step(action)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["terminated"])
            self.assertTrue(info["truncated"])
        finally:
            env.close()

    def test_terminal_episode_result_reports_real_trajectory_metrics(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=1, device="cpu", seed=31)
        )
        try:
            env.reset()
            action = np.zeros((1, 4), dtype=np.int64)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            self.assertTrue(bool(done[0]))
            episode_result = infos[0]["episode_result"]
            self.assertGreater(float(episode_result["zone_coverage"]), 0.0)
            self.assertIsNotNone(episode_result["mean_inter_robot_dist"])
            self.assertIn("collision_events_per_episode", episode_result)
            self.assertIn("near_misses_per_episode", episode_result)
            for key in (
                "reward_terminal",
                "reward_offense",
                "reward_pbrs",
                "reward_team",
                "reward_sparse",
                "reward_sparse_points",
                "reward_failure",
                "reward_total",
            ):
                self.assertIn(key, episode_result)
                self.assertIsInstance(episode_result[key], float)
        finally:
            env.close()

    def test_render_rgb_array_contract(self) -> None:
        env = GPUCTFSingleEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=9))
        try:
            env.reset(seed=9)
            frame = env.render(mode="rgb_array")
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
        finally:
            env.close()

    def test_positions_remain_in_bounds_after_actions(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=10))
        try:
            env.reset()
            action = np.array([[int(MacroAction.GO_TO), 49, int(MacroAction.GO_TO), 49]], dtype=np.int64)
            for _ in range(16):
                env.step_async(action)
                env.step_wait()
            core = env.core
            self.assertTrue(bool(torch.all(core.blue_x >= 0.0)))
            self.assertTrue(bool(torch.all(core.blue_y >= 0.0)))
            self.assertTrue(bool(torch.all(core.red_x >= 0.0)))
            self.assertTrue(bool(torch.all(core.red_y >= 0.0)))
            self.assertTrue(bool(torch.all(core.blue_x <= float(core.cols - 1))))
            self.assertTrue(bool(torch.all(core.blue_y <= float(core.rows - 1))))
            self.assertTrue(bool(torch.all(core.red_x <= float(core.cols - 1))))
            self.assertTrue(bool(torch.all(core.red_y <= float(core.rows - 1))))
        finally:
            env.close()

    def test_collision_guard_is_deterministic_for_close_agents(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=21)
        env1 = GPUCTFVecEnv(cfg)
        env2 = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=21))
        try:
            for env in (env1, env2):
                core = env.core
                core.reset_all()
                core.blue_x[0] = torch.tensor([4.00, 4.05], device=core.device)
                core.blue_y[0] = torch.tensor([8.00, 8.00], device=core.device)
                core.red_x[0] = torch.tensor([14.00, 14.05], device=core.device)
                core.red_y[0] = torch.tensor([8.00, 8.00], device=core.device)
                core._apply_avoid_collision_guard(
                    core.blue_x.clone(),
                    core.blue_y.clone(),
                    core.red_x.clone(),
                    core.red_y.clone(),
                )

            np.testing.assert_allclose(env1.core.blue_x.cpu().numpy(), env2.core.blue_x.cpu().numpy())
            np.testing.assert_allclose(env1.core.blue_y.cpu().numpy(), env2.core.blue_y.cpu().numpy())
            np.testing.assert_allclose(env1.core.red_x.cpu().numpy(), env2.core.red_x.cpu().numpy())
            np.testing.assert_allclose(env1.core.red_y.cpu().numpy(), env2.core.red_y.cpu().numpy())
            self.assertTrue(bool(torch.isfinite(env1.core.blue_x).all()))
            self.assertTrue(bool(torch.isfinite(env1.core.red_x).all()))
        finally:
            env1.close()
            env2.close()

    def test_tagging_rules_fire_once_per_event(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=22, tag_channel_seconds=0.0)
        )
        try:
            core = env.core
            core.reset_all()
            core.blue_x[0] = torch.tensor([5.0, 5.5], device=core.device)
            core.blue_y[0] = torch.tensor([7.0, 7.0], device=core.device)
            core.red_x[0] = torch.tensor([5.2, 18.0], device=core.device)
            core.red_y[0] = torch.tensor([7.0, 18.0], device=core.device)
            oob_blue = torch.zeros_like(core.blue_alive)
            oob_red = torch.zeros_like(core.red_alive)

            blue_tag_noflag, blue_tag_withflag, red_tag_total = core._apply_aquaticus_tag_rules(oob_blue, oob_red)
            self.assertEqual(float(blue_tag_noflag[0].item()), 1.0)
            self.assertEqual(float(blue_tag_withflag[0].item()), 0.0)
            self.assertEqual(float(red_tag_total[0].item()), 0.0)
            self.assertTrue(bool(core.red_tagged[0, 0].item()))

            blue_tag_noflag_2, blue_tag_withflag_2, red_tag_total_2 = core._apply_aquaticus_tag_rules(oob_blue, oob_red)
            self.assertEqual(float(blue_tag_noflag_2[0].item()), 0.0)
            self.assertEqual(float(blue_tag_withflag_2[0].item()), 0.0)
            self.assertEqual(float(red_tag_total_2[0].item()), 0.0)
        finally:
            env.close()

    def test_sparse_reward_points_account_nonzero_event_families(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=23))
        try:
            core = env.core
            b = torch.tensor([True], device=core.device)
            f = torch.tensor([False], device=core.device)
            blue_oob = torch.tensor([[True, False]], device=core.device)
            points = core._sparse_reward_points(
                blue_cap_env=b,
                red_cap_env=f,
                blue_tag_noflag=torch.tensor([1.0], device=core.device),
                blue_tag_withflag=torch.tensor([1.0], device=core.device),
                red_tag_total=torch.tensor([1.0], device=core.device),
                blue_oob=blue_oob,
                blue_mine_tags=torch.tensor([1.0], device=core.device),
                red_mine_tags=torch.tensor([0.0], device=core.device),
            )
            expected = (
                SPARSE_FLAG_CAPTURE_POINTS
                + SPARSE_TAG_NO_FLAG_POINTS
                + SPARSE_TAG_WITH_FLAG_POINTS
                - SPARSE_TAG_NO_FLAG_POINTS
                + SPARSE_MINE_TAG_POINTS
                + SPARSE_OOB_POINTS
            )
            self.assertAlmostEqual(float(points[0].item()), float(expected))
        finally:
            env.close()

    def test_reward_total_uses_dense_and_sparse_weights(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=1,
            n_agents_per_team=2,
            device="cpu",
            seed=24,
            dense_weight=0.25,
            sparse_weight=0.5,
            reward_scale=2.0,
            reward_clip=1.0,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            rterm = torch.tensor([1.0], device=core.device)
            roff = torch.tensor([0.5], device=core.device)
            rpbrs = torch.tensor([2.0], device=core.device)
            rteam = torch.tensor([1.0], device=core.device)
            sparse_points = torch.tensor([100.0], device=core.device)
            rfail = torch.tensor([-0.25], device=core.device)
            reward = core._reward_total(
                rterm,
                roff,
                rpbrs,
                rteam,
                sparse_points,
                rfail,
                torch.tensor([False], device=core.device),
            )
            raw = 1.0 + 0.5 - 0.25 + 0.25 * (2.0 + 1.0) + 0.5 * (100.0 / 100.0)
            expected = torch.tanh(torch.tensor(raw / 2.0)).item()
            self.assertAlmostEqual(float(reward[0].item()), float(expected), places=6)
        finally:
            env.close()

    def test_reward_config_overrides_legacy_flat_fields(self) -> None:
        profile = RewardConfig(
            flag_pickup_reward=0.25,
            dense_weight=0.4,
            sparse_weight=0.7,
            action_failed_punishment=-0.05,
        )
        cfg = GPUFieldConfig(
            n_envs=1,
            n_agents_per_team=2,
            flag_pickup_reward=99.0,
            dense_weight=99.0,
            reward_config=profile,
        )
        self.assertEqual(cfg.flag_pickup_reward, 0.25)
        self.assertEqual(cfg.dense_weight, 0.4)
        self.assertEqual(cfg.sparse_weight, 0.7)
        self.assertEqual(cfg.action_failed_punishment, -0.05)
        self.assertEqual(cfg.reward_config, profile)

    def test_reward_profile_defaults_reduce_dense_shaping_drift(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu")
        self.assertEqual(cfg.dense_weight, 0.25)
        self.assertEqual(cfg.pbrs_defense_coef, 0.5)
        self.assertEqual(cfg.reward_config.dense_weight, 0.25)
        self.assertEqual(cfg.reward_config.pbrs_defense_coef, 0.5)

    def test_pbrs_closeness_potential_increases_when_agents_approach_flag(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=25))
        try:
            core = env.core
            core.reset_all()
            core.red_flag_pos[0] = torch.tensor([18.0, 10.0], device=core.device)
            core.blue_carrying.zero_()
            core.red_carrying.zero_()

            far_x = torch.tensor([[2.0, 2.0]], device=core.device)
            near_x = torch.tensor([[10.0, 10.0]], device=core.device)
            y = torch.tensor([[10.0, 10.0]], device=core.device)
            far_attack, _, _ = core._compute_potentials(far_x, y, core.blue_carrying, core.red_carrying)
            near_attack, _, _ = core._compute_potentials(near_x, y, core.blue_carrying, core.red_carrying)

            self.assertGreater(float(near_attack[0].item()), float(far_attack[0].item()))
            core.blue_x = near_x.clone()
            core.blue_y = y.clone()
            shaped = core._pbrs_reward(far_x, y, core.blue_carrying)
            self.assertGreater(float(shaped[0].item()), 0.0)
        finally:
            env.close()

    def test_pbrs_does_not_punish_flag_pickup_phase_switch(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=26))
        try:
            core = env.core
            core.reset_all()
            core.red_flag_pos[0] = torch.tensor([18.0, 10.0], device=core.device)
            core.blue_flag_home[0] = torch.tensor([2.0, 10.0], device=core.device)
            prev_x = core.blue_x.clone()
            prev_y = core.blue_y.clone()
            prev_carrying = core.blue_carrying.clone()
            prev_x[0, 0] = 18.0
            prev_y[0, 0] = 10.0
            prev_carrying[0, 0] = False
            core.blue_x[0, 0] = 18.0
            core.blue_y[0, 0] = 10.0
            core.blue_carrying[0, 0] = True

            shaped = core._pbrs_reward(prev_x, prev_y, prev_carrying)

            self.assertGreater(float(shaped[0].item()), -0.05)
        finally:
            env.close()

    def test_carried_flag_position_update_is_per_env_not_global(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=2, n_agents_per_team=2, device="cpu", seed=261))
        try:
            core = env.core
            core.reset_all()
            core.blue_carrying.zero_()
            core.red_carrying.zero_()
            core.red_flag_pos[0] = torch.tensor([9.0, 9.0], device=core.device)
            core.red_flag_pos[1] = torch.tensor([18.0, 18.0], device=core.device)
            core.blue_flag_pos[0] = torch.tensor([1.0, 18.0], device=core.device)
            core.blue_flag_pos[1] = torch.tensor([2.0, 18.0], device=core.device)
            red_flag_env1_before = core.red_flag_pos[1].clone()
            blue_flag_env0_before = core.blue_flag_pos[0].clone()

            core.blue_x[0, 1] = 7.0
            core.blue_y[0, 1] = 8.0
            core.blue_carrying[0, 1] = True
            core.red_x[1, 0] = 12.0
            core.red_y[1, 0] = 13.0
            core.red_carrying[1, 0] = True

            core._apply_flag_rules()

            self.assertTrue(torch.allclose(core.red_flag_pos[0], torch.tensor([7.0, 8.0], device=core.device)))
            self.assertTrue(torch.allclose(core.blue_flag_pos[1], torch.tensor([12.0, 13.0], device=core.device)))
            self.assertTrue(torch.allclose(core.red_flag_pos[1], red_flag_env1_before))
            self.assertTrue(torch.allclose(core.blue_flag_pos[0], blue_flag_env0_before))
        finally:
            env.close()

    def test_dead_agents_cannot_grab_flags(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=262))
        try:
            core = env.core
            core.reset_all()
            core.blue_alive[0, 0] = False
            core.blue_tagged[0, 0] = False
            core.blue_x[0, 0] = core.red_flag_pos[0, 0]
            core.blue_y[0, 0] = core.red_flag_pos[0, 1]

            blue_grab_env, _, _, _ = core._apply_flag_rules()

            self.assertFalse(bool(blue_grab_env[0].item()))
            self.assertFalse(bool(core.blue_carrying[0].any().item()))
        finally:
            env.close()

    def test_sparse_reward_event_resets_stalemate_counter(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=263, stalemate_max_steps=2)
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            core.reset_all()
            core.stalemate_steps[0] = 1
            core._last_dense_progress[0] = 0.0
            flags = {
                "blue_grab_env": torch.tensor([False], device=core.device),
                "red_grab_env": torch.tensor([False], device=core.device),
                "blue_cap_env": torch.tensor([False], device=core.device),
                "red_cap_env": torch.tensor([False], device=core.device),
            }
            combat = {
                "blue_tag_noflag": torch.tensor([0.0], device=core.device),
                "blue_tag_withflag": torch.tensor([0.0], device=core.device),
                "red_tag_total": torch.tensor([0.0], device=core.device),
            }
            rewards = {
                "sparse_points": torch.tensor([100.0], device=core.device),
                "roff": torch.tensor([0.0], device=core.device),
                "rpbrs": torch.tensor([0.0], device=core.device),
                "rteam": torch.tensor([0.0], device=core.device),
                "rfail": torch.tensor([0.0], device=core.device),
            }

            terminal = core._advance_episode_end(flags, combat, rewards)

            self.assertFalse(bool(terminal["truncated"][0].item()))
            self.assertEqual(int(core.stalemate_steps[0].item()), 0)
        finally:
            env.close()

    def test_stalemate_progress_tracks_return_progress(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=27))
        try:
            core = env.core
            core.reset_all()
            core.blue_flag_home[0] = torch.tensor([2.0, 10.0], device=core.device)
            core.blue_carrying[0, 0] = True
            prev_x = core.blue_x.clone()
            prev_y = core.blue_y.clone()
            prev_carrying = core.blue_carrying.clone()
            prev_x[0, 0] = 18.0
            prev_y[0, 0] = 10.0
            core.blue_x[0, 0] = 14.0
            core.blue_y[0, 0] = 10.0

            shaped = core._pbrs_reward(prev_x, prev_y, prev_carrying)

            self.assertGreater(float(shaped[0].item()), 0.0)
            self.assertGreater(float(core._last_dense_progress[0].item()), 0.0)
        finally:
            env.close()

    def test_escort_reward_requires_noncarrier_teammate(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=1,
            n_agents_per_team=2,
            device="cpu",
            seed=28,
            idle_penalty_coef=0.0,
            spin_penalty_coef=0.0,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            core.reset_all()
            yaw = torch.zeros_like(core.blue_x)
            prev_x = core.blue_x.clone()
            prev_y = core.blue_y.clone()
            core.blue_carrying[0] = torch.tensor([True, False], device=core.device)
            core.blue_x[0] = torch.tensor([10.0, 2.0], device=core.device)
            core.blue_y[0] = torch.tensor([10.0, 2.0], device=core.device)

            no_escort = core._team_coordination_reward(prev_x, prev_y, yaw)
            core.blue_x[0, 1] = 12.0
            core.blue_y[0, 1] = 10.0
            escorted = core._team_coordination_reward(prev_x, prev_y, yaw)

            self.assertAlmostEqual(float(no_escort[0].item()), 0.0, places=6)
            self.assertGreater(float(escorted[0].item()), 0.0)
        finally:
            env.close()

    def test_respawn_timing_and_location_are_seed_deterministic(self) -> None:
        cfg1 = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=31, rules_profile="LEGACY")
        cfg2 = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=31, rules_profile="LEGACY")
        env1 = GPUCTFVecEnv(cfg1)
        env2 = GPUCTFVecEnv(cfg2)
        try:
            for env in (env1, env2):
                core = env.core
                core.reset_all()
                kill_blue = torch.zeros_like(core.blue_alive)
                kill_red = torch.zeros_like(core.red_alive)
                kill_blue[0, 0] = True
                core._kill_agents(kill_blue, kill_red)
                self.assertFalse(bool(core.blue_alive[0, 0].item()))
                for _ in range(5):
                    core._respawn_timers()

            self.assertTrue(bool(env1.core.blue_alive[0, 0].item()))
            self.assertTrue(bool(env2.core.blue_alive[0, 0].item()))
            np.testing.assert_allclose(env1.core.blue_x.cpu().numpy(), env2.core.blue_x.cpu().numpy())
            np.testing.assert_allclose(env1.core.blue_y.cpu().numpy(), env2.core.blue_y.cpu().numpy())
            self.assertLessEqual(float(env1.core.blue_x[0, 0].item()), float(env1.core.cols // 3))
        finally:
            env1.close()
            env2.close()

    def test_reset_clears_mutable_episode_state(self) -> None:
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=41))
        try:
            core = env.core
            core.reset_all()
            core.blue_score[0] = 2
            core.red_score[0] = 1
            core.blue_carrying[0, 0] = True
            core.red_carrying[0, 1] = True
            core.blue_tagged[0, 0] = True
            core.red_tagged[0, 1] = True
            core.blue_mine_active[0, 0] = True
            core.red_mine_active[0, 0] = True
            core.pickup_active[0] = False
            core.reset_all()

            self.assertEqual(int(core.blue_score[0].item()), 0)
            self.assertEqual(int(core.red_score[0].item()), 0)
            self.assertFalse(bool(core.blue_carrying.any().item()))
            self.assertFalse(bool(core.red_carrying.any().item()))
            self.assertFalse(bool(core.blue_tagged.any().item()))
            self.assertFalse(bool(core.red_tagged.any().item()))
            self.assertFalse(bool(core.blue_mine_active.any().item()))
            self.assertFalse(bool(core.red_mine_active.any().item()))
            self.assertTrue(bool(core.pickup_active.all().item()))
        finally:
            env.close()

    def test_flag_pickup_and_carrier_death_reset_flag_once(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=12, score_grace_steps=0)
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            core.reset_all()

            core.blue_x[0, 0] = core.red_flag_pos[0, 0]
            core.blue_y[0, 0] = core.red_flag_pos[0, 1]
            core.blue_x[0, 1] = 0.0
            core.blue_y[0, 1] = 0.0
            blue_grab_env, _, _, _ = core._apply_flag_rules()
            blue_grab_env_2, _, _, _ = core._apply_flag_rules()
            self.assertTrue(bool(blue_grab_env[0].item()))
            self.assertFalse(bool(blue_grab_env_2[0].item()))
            self.assertEqual(int(core.blue_carrying[0].sum().item()), 1)

            kill_blue = torch.zeros_like(core.blue_alive)
            kill_red = torch.zeros_like(core.red_alive)
            kill_blue[0, 0] = True
            core._kill_agents(kill_blue, kill_red)
            self.assertFalse(bool(core.blue_carrying[0, 0].item()))
            np.testing.assert_allclose(
                core.red_flag_pos[0].detach().cpu().numpy(),
                core.red_flag_home[0].detach().cpu().numpy(),
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
