from __future__ import annotations

import unittest
import numpy as np
import torch

from game_field_gpu import GPUCTFSingleEnv, GPUCTFVecEnv, GPUFieldConfig
from macro_actions import MacroAction


def blue_has_enemy_flag(env: GPUCTFSingleEnv) -> bool:
    return bool(env.vec.core.blue_carrying[0].any().item())


def get_blue_score(env: GPUCTFSingleEnv) -> int:
    return int(env.vec.core.blue_score[0].item())


def move_blue_agent_to_enemy_flag(env: GPUCTFSingleEnv):
    core = env.vec.core
    # Clear red defenders out of tag range first. This helper exists to exercise
    # the PICKUP/CAPTURE code path, not the tag rule. Under the Aquaticus-faithful
    # ruleset a single eligible defender tags on its own, and at this seed a red
    # agent sits 2.10 cells from the red flag -- inside tag_range_cells=2.5 -- so
    # the teleported carrier would be tagged and drop the flag before pickup could
    # be observed. (Under the superseded two-tagger rule that defender was
    # harmless, which is why this fixture used to pass.)
    core.red_x[0, :] = float(core.cols - 1)
    core.red_y[0, :] = 0.0
    # Teleport blue agent 0 directly onto the red flag position
    core.blue_x[0, 0] = core.red_flag_pos[0, 0]
    core.blue_y[0, 0] = core.red_flag_pos[0, 1]
    core.blue_commit_ticks_left[0, 0] = 0
    # Step the environment to trigger pickup
    action = np.zeros(env.action_space.shape, dtype=np.int64)
    env.step(action)


def move_blue_agent_to_blue_capture_zone(env: GPUCTFSingleEnv):
    core = env.vec.core
    # Teleport the carrier to the blue flag home (scoring zone)
    core.blue_x[0, 0] = core.blue_flag_home[0, 0]
    core.blue_y[0, 0] = core.blue_flag_home[0, 1]
    core.blue_commit_ticks_left[0, 0] = 0
    action = np.zeros(env.action_space.shape, dtype=np.int64)
    # Step twice because capture_confirm_frames = 2
    for _ in range(2):
        core.blue_x[0, 0] = core.blue_flag_home[0, 0]
        core.blue_y[0, 0] = core.blue_flag_home[0, 1]
        env.step(action)


class TestEnvironmentFeasibility(unittest.TestCase):
    """Gate 1: Mechanical Feasibility Checks."""

    def test_spawn_locations_and_flags_in_bounds(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, device="cpu", seed=42)
        env = GPUCTFSingleEnv(cfg)
        try:
            env.reset(seed=42)
            core = env.vec.core
            
            # Check spawns in bounds
            self.assertTrue(bool(torch.all(core.blue_x >= 0.0)))
            self.assertTrue(bool(torch.all(core.blue_x <= float(core.cols - 1))))
            self.assertTrue(bool(torch.all(core.blue_y >= 0.0)))
            self.assertTrue(bool(torch.all(core.blue_y <= float(core.rows - 1))))
            
            self.assertTrue(bool(torch.all(core.red_x >= 0.0)))
            self.assertTrue(bool(torch.all(core.red_x <= float(core.cols - 1))))
            self.assertTrue(bool(torch.all(core.red_y >= 0.0)))
            self.assertTrue(bool(torch.all(core.red_y <= float(core.rows - 1))))
            
            # Check flags in bounds
            self.assertTrue(bool(torch.all(core.blue_flag_home >= 0.0)))
            self.assertTrue(bool(torch.all(core.blue_flag_home[:, 0] <= float(core.cols - 1))))
            self.assertTrue(bool(torch.all(core.blue_flag_home[:, 1] <= float(core.rows - 1))))
            
            self.assertTrue(bool(torch.all(core.red_flag_home >= 0.0)))
            self.assertTrue(bool(torch.all(core.red_flag_home[:, 0] <= float(core.cols - 1))))
            self.assertTrue(bool(torch.all(core.red_flag_home[:, 1] <= float(core.rows - 1))))
            
            # No immediate captures
            self.assertFalse(bool(core.blue_carrying.any().item()))
            self.assertFalse(bool(core.red_carrying.any().item()))
            self.assertEqual(int(core.blue_score[0].item()), 0)
            self.assertEqual(int(core.red_score[0].item()), 0)
        finally:
            env.close()

    def test_minimum_distances(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, device="cpu", seed=42)
        env = GPUCTFSingleEnv(cfg)
        try:
            env.reset(seed=42)
            core = env.vec.core
            
            # Flags should be separated
            flag_dist = torch.dist(core.blue_flag_home[0], core.red_flag_home[0]).item()
            self.assertGreaterEqual(flag_dist, 10.0)
            
            # Spawns should be separated
            blue_mean_x = core.blue_x[0].mean().item()
            red_mean_x = core.red_x[0].mean().item()
            self.assertGreaterEqual(abs(blue_mean_x - red_mean_x), 6.0)
        finally:
            env.close()

    def test_flag_capture_path_deterministic(self) -> None:
        cfg = GPUFieldConfig(n_envs=1, device="cpu", seed=10, score_grace_steps=0)
        env = GPUCTFSingleEnv(cfg)
        try:
            env.reset(seed=10)
            
            # 1. Verify pickup path
            move_blue_agent_to_enemy_flag(env)
            self.assertTrue(blue_has_enemy_flag(env))
            
            # 2. Verify score path
            move_blue_agent_to_blue_capture_zone(env)
            self.assertGreaterEqual(get_blue_score(env), 1)
        finally:
            env.close()


class TestTacticalCoverageSmoke(unittest.TestCase):
    """Gate 2: Tactical coverage smoke test."""

    def test_tactical_coverage_metrics(self) -> None:
        # Run a small vectorized rollout using competent scripted behavior (n_envs=4, seed=123)
        cfg = GPUFieldConfig(n_envs=4, device="cpu", seed=123, score_grace_steps=0)
        env = GPUCTFVecEnv(cfg)
        
        # Enable competent scripted behaviors for both sides
        env.core.blue_scripted = True
        # opponent key OP3 is already SCRIPTED
        env.core.set_next_opponent(kind="SCRIPTED", key="OP3")
        
        try:
            env.reset()
            core = env.core
            
            flag_pickup_count = 0
            carry_home_steps = 0
            capture_count = 0
            own_flag_stolen_count = 0
            flag_return_count = 0
            escort_opportunity_count = 0
            defensive_emergency_count = 0
            
            # Step the environment for a short duration (200 steps)
            dummy_actions = np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.int64)
            for _ in range(200):
                prev_blue_carrying = core.blue_carrying.clone()
                prev_red_carrying = core.red_carrying.clone()
                prev_blue_score = core.blue_score.clone()
                prev_red_score = core.red_score.clone()
                
                env.step_async(dummy_actions)
                _, _, _, _ = env.step_wait()
                
                # Check pickups
                new_blue_pickup = core.blue_carrying & (~prev_blue_carrying)
                new_red_pickup = core.red_carrying & (~prev_red_carrying)
                flag_pickup_count += int(new_blue_pickup.sum().item() + new_red_pickup.sum().item())
                own_flag_stolen_count += int(new_red_pickup.sum().item())
                
                # Check carry steps
                carry_home_steps += int(core.blue_carrying.sum().item() + core.red_carrying.sum().item())
                
                # Check captures
                new_blue_cap = (core.blue_score > prev_blue_score)
                new_red_cap = (core.red_score > prev_red_score)
                capture_count += int(new_blue_cap.sum().item() + new_red_cap.sum().item())
                
                # Check flag returns (carrier lost flag but score didn't increase)
                blue_lost_no_score = prev_blue_carrying & (~core.blue_carrying) & (~new_blue_cap[:, None])
                red_lost_no_score = prev_red_carrying & (~core.red_carrying) & (~new_red_cap[:, None])
                flag_return_count += int(blue_lost_no_score.sum().item() + red_lost_no_score.sum().item())
                
                # Check emergency steps
                defensive_emergency_count += int(core.red_carrying.any(dim=1).sum().item() + core.blue_carrying.any(dim=1).sum().item())
                
                # Check escort opportunities (carrier exists, other alive teammates are within 6 cells)
                for b in range(core.B):
                    if core.blue_carrying[b].any():
                        c_idx = torch.argmax(core.blue_carrying[b].to(torch.int64)).item()
                        cx, cy = core.blue_x[b, c_idx], core.blue_y[b, c_idx]
                        for a in range(core.Nb):
                            if a != c_idx and core.blue_alive[b, a]:
                                dist = torch.hypot(core.blue_x[b, a] - cx, core.blue_y[b, a] - cy).item()
                                if dist <= 6.0:
                                    escort_opportunity_count += 1
                    if core.red_carrying[b].any():
                        c_idx = torch.argmax(core.red_carrying[b].to(torch.int64)).item()
                        cx, cy = core.red_x[b, c_idx], core.red_y[b, c_idx]
                        for a in range(core.Nr):
                            if a != c_idx and core.red_alive[b, a]:
                                dist = torch.hypot(core.red_x[b, a] - cx, core.red_y[b, a] - cy).item()
                                if dist <= 6.0:
                                    escort_opportunity_count += 1

            # Print stats for diagnostics
            print(f"\n[Smoke Test Tactical Telemetry]")
            print(f"  flag_pickup_count:         {flag_pickup_count}")
            print(f"  carry_home_steps:          {carry_home_steps}")
            print(f"  capture_count:             {capture_count}")
            print(f"  own_flag_stolen_count:     {own_flag_stolen_count}")
            print(f"  flag_return_count:         {flag_return_count}")
            print(f"  escort_opportunity_count:  {escort_opportunity_count}")
            print(f"  defensive_emergency_count: {defensive_emergency_count}")

            # Assertions to ensure environment is tactically active
            self.assertGreater(flag_pickup_count, 0, "No flags were picked up during rollout.")
            self.assertGreater(carry_home_steps, 0, "No steps were spent carrying flags.")
            self.assertGreater(defensive_emergency_count, 0, "No defensive emergencies occurred.")
            
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
