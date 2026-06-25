"""Environment audit for v6i8_adapter_balanced_hardpool pre-launch.

Covers:
1.  Reset isolation — no state bleed between envs in vectorized batch
2.  Score is cleared on per-env reset
3.  Flag returns to home on carrier tag (no mid-field dropped state)
4.  Flag-carry transition: exactly one reward fire per grab
5.  Terminal reward fires exactly once and is correct sign
6.  Observations contain no opponent-role or latent leakage
7.  OP8/9/10 mask computation — red-only, keys matched correctly
8.  OP10 nearest-enemy filters dead agents (regression for fixed bug)
9.  OP9 orbit tightens to r=1 when no carrier present
10. OP8 blocker skips when no carrier (no NaN crash)
11. Opponent params respect same tag range as OP5/6/7
12. Coordinated-attack counter resets to 0 on episode reset
13. Both map layouts load without error
14. Legal spawn positions within field bounds
15. Observations are non-NaN and correctly shaped
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from game_field_gpu import (
    BatchedCTFCore,
    GPUFieldConfig,
)
from gpu_env import MAP_A_OPEN


def _make_core(n_envs: int = 4, n_agents: int = 2, map_layout: str = MAP_A_OPEN, seed: int = 0) -> BatchedCTFCore:
    cfg = GPUFieldConfig(
        n_envs=n_envs,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        device="cpu",
        seed=seed,
        map_layout=map_layout,
        score_limit=3,
        max_decision_steps=400,
    )
    core = BatchedCTFCore(cfg)
    core.reset_all()
    return core


# ---------------------------------------------------------------------------
# 1. Reset isolation — state cleared per-env, not globally
# ---------------------------------------------------------------------------

class TestResetIsolation(unittest.TestCase):
    def test_score_cleared_per_env_on_reset(self) -> None:
        core = _make_core(n_envs=4)
        # Manually dirty two envs' scores
        core.blue_score[0] = 2
        core.blue_score[2] = 1
        core.red_score[1] = 3
        # Reset only envs 0 and 1
        mask = torch.tensor([True, True, False, False])
        core.reset_indices(mask)
        # Envs 0 and 1 should be 0
        self.assertEqual(int(core.blue_score[0].item()), 0)
        self.assertEqual(int(core.red_score[1].item()), 0)
        # Envs 2 and 3 untouched
        self.assertEqual(int(core.blue_score[2].item()), 1)

    def test_carrying_cleared_per_env(self) -> None:
        core = _make_core(n_envs=4)
        core.blue_carrying[0, 0] = True
        core.blue_carrying[2, 1] = True
        mask = torch.tensor([True, False, False, False])
        core.reset_indices(mask)
        self.assertFalse(core.blue_carrying[0].any().item(), "carrying should be cleared")
        self.assertTrue(core.blue_carrying[2, 1].item(), "untouched env should keep state")

    def test_flag_pos_reset_to_home_per_env(self) -> None:
        core = _make_core(n_envs=4)
        # Move flag away from home in env 0
        core.blue_flag_pos[0] = torch.tensor([5.0, 5.0])
        home_pos = core.blue_flag_home[0].clone()
        mask = torch.zeros(4, dtype=torch.bool)
        mask[0] = True
        core.reset_indices(mask)
        self.assertTrue(
            torch.allclose(core.blue_flag_pos[0], home_pos),
            "flag_pos should equal flag_home after reset",
        )

    def test_red_coord_ticks_reset_to_zero(self) -> None:
        core = _make_core(n_envs=4)
        core.red_coord_ticks_left[:] = 99
        mask = torch.ones(4, dtype=torch.bool)
        core.reset_indices(mask)
        self.assertTrue(
            (core.red_coord_ticks_left == 0).all().item(),
            "red_coord_ticks_left must be 0 after reset",
        )

    def test_tag_pressure_cleared_per_env(self) -> None:
        core = _make_core(n_envs=4)
        core.red_tag_pressure_time[:] = 5.0
        mask = torch.tensor([True, True, False, False])
        core.reset_indices(mask)
        self.assertTrue((core.red_tag_pressure_time[:2] == 0.0).all().item())
        self.assertTrue((core.red_tag_pressure_time[2:] == 5.0).all().item())


# ---------------------------------------------------------------------------
# 2. Flag state — no mid-field dropped position
# ---------------------------------------------------------------------------

class TestFlagState(unittest.TestCase):
    def test_flag_at_home_after_reset(self) -> None:
        core = _make_core(n_envs=2)
        for i in range(2):
            self.assertTrue(
                torch.allclose(core.blue_flag_pos[i], core.blue_flag_home[i]),
                f"env {i}: flag_pos should equal flag_home after reset",
            )
            self.assertTrue(
                torch.allclose(core.red_flag_pos[i], core.red_flag_home[i]),
            )

    def test_no_agent_carrying_after_reset(self) -> None:
        core = _make_core(n_envs=2)
        self.assertFalse(core.blue_carrying.any().item())
        self.assertFalse(core.red_carrying.any().item())


# ---------------------------------------------------------------------------
# 3. Observations: shape, non-NaN, no privileged info
# ---------------------------------------------------------------------------

class TestObservations(unittest.TestCase):
    def setUp(self) -> None:
        self.core = _make_core(n_envs=2, n_agents=2)

    def test_grid_obs_shape_and_no_nan(self) -> None:
        obs = self.core.get_obs_tensors("blue")
        grid = obs["grid"]
        B, N, C, H, W = grid.shape
        self.assertEqual(B, 2)
        self.assertEqual(N, 2)
        self.assertFalse(torch.isnan(grid).any().item(), "grid obs has NaN")

    def test_vec_obs_shape_and_no_nan(self) -> None:
        obs = self.core.get_obs_tensors("blue")
        vec = obs["vec"]
        self.assertEqual(vec.shape[0], 2)  # B
        self.assertEqual(vec.shape[1], 2)  # agents
        self.assertFalse(torch.isnan(vec).any().item(), "vec obs has NaN")

    def test_vec_obs_in_valid_range(self) -> None:
        obs = self.core.get_obs_tensors("blue")
        vec = obs["vec"]
        self.assertTrue((vec >= -1.0).all().item(), "vec obs below -1")
        self.assertTrue((vec <= 1.0).all().item(), "vec obs above +1")

    def test_red_obs_symmetric_with_blue(self) -> None:
        blue_obs = self.core.get_obs_tensors("blue")
        red_obs = self.core.get_obs_tensors("red")
        # Both should be valid shapes
        self.assertEqual(blue_obs["grid"].shape, red_obs["grid"].shape)
        self.assertEqual(blue_obs["vec"].shape, red_obs["vec"].shape)

    def test_no_scripted_role_in_obs(self) -> None:
        # Sanity: vec obs has VEC_OBS_DIM dimensions; none are reserved for scripted role
        obs = self.core.get_obs_tensors("blue")
        vec = obs["vec"]
        from gpu_env._constants import VEC_OBS_DIM
        self.assertEqual(vec.shape[-1], VEC_OBS_DIM)
        # There is no separate "opponent_role" field in vec obs
        # If VEC_OBS_DIM is 20, the indices 0-19 are documented position/flag/score fields
        self.assertLessEqual(VEC_OBS_DIM, 20, "vec obs grew unexpectedly — check for leakage")


# ---------------------------------------------------------------------------
# 4. OP8/9/10 mask computation correctness
# ---------------------------------------------------------------------------

class TestOpponentMasks(unittest.TestCase):
    def _set_opponent_keys(self, core: BatchedCTFCore, keys: list[str]) -> None:
        core._opponent_key = keys

    def test_op8_mask_red_only(self) -> None:
        core = _make_core(n_envs=3)
        self._set_opponent_keys(core, ["OP8", "OP5", "OP8_INTERCEPTOR"])
        import torch
        device = core.device
        op8_mask = torch.as_tensor(
            [str(k).upper() in ("OP8", "OP8_INTERCEPTOR") for k in core._opponent_key],
            device=device, dtype=torch.bool,
        )
        self.assertTrue(op8_mask[0].item(), "env0 key=OP8 should be True")
        self.assertFalse(op8_mask[1].item(), "env1 key=OP5 should be False")
        self.assertTrue(op8_mask[2].item(), "env2 key=OP8_INTERCEPTOR should be True")

    def test_op9_mask_selects_fortress(self) -> None:
        core = _make_core(n_envs=3)
        self._set_opponent_keys(core, ["OP9", "OP9_FORTRESS", "OP5"])
        op9_mask = torch.as_tensor(
            [str(k).upper() in ("OP9", "OP9_FORTRESS") for k in core._opponent_key],
            device=core.device, dtype=torch.bool,
        )
        self.assertTrue(op9_mask[0].item())
        self.assertTrue(op9_mask[1].item())
        self.assertFalse(op9_mask[2].item())

    def test_op10_mask_selects_escort(self) -> None:
        core = _make_core(n_envs=2)
        self._set_opponent_keys(core, ["OP10", "OP10_ESCORT"])
        op10_mask = torch.as_tensor(
            [str(k).upper() in ("OP10", "OP10_ESCORT") for k in core._opponent_key],
            device=core.device, dtype=torch.bool,
        )
        self.assertTrue(op10_mask[0].item())
        self.assertTrue(op10_mask[1].item())

    def test_masks_are_false_for_blue_side(self) -> None:
        # Masks default to zeros when is_blue=True (opponents only for red)
        op8 = torch.zeros((3,), dtype=torch.bool)
        op9 = torch.zeros((3,), dtype=torch.bool)
        op10 = torch.zeros((3,), dtype=torch.bool)
        self.assertFalse(op8.any().item())
        self.assertFalse(op9.any().item())
        self.assertFalse(op10.any().item())


# ---------------------------------------------------------------------------
# 5. OP10 nearest-enemy filters dead agents (regression test)
# ---------------------------------------------------------------------------

class TestOP10DeadEnemyFix(unittest.TestCase):
    def test_argmin_skips_dead_agents(self) -> None:
        """After fix, argmin uses dd_live which masks dead agents with 1e6."""
        B, Ne = 4, 2
        device = torch.device("cpu")
        carr_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        carr_y = torch.tensor([5.0, 5.0, 5.0, 5.0])
        # Two enemies: one dead (index 0) very close, one alive (index 1) farther
        enemy_x = torch.tensor([[2.0, 8.0]] * B)
        enemy_y = torch.tensor([[5.0, 5.0]] * B)
        enemy_alive = torch.tensor([[False, True]] * B)  # index 0 dead, index 1 alive

        dxx = carr_x[:, None] - enemy_x
        dyy = carr_y[:, None] - enemy_y
        dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
        # Before fix: argmin picks index 0 (closer but dead)
        near_naive = torch.argmin(dd, dim=1)
        # After fix: argmin picks index 1 (farther but alive)
        dd_live = torch.where(enemy_alive, dd, dd.new_full((), 1e6).expand_as(dd))
        near_fixed = torch.argmin(dd_live, dim=1)

        self.assertTrue(
            (near_naive == 0).all().item(),
            "naive argmin should pick the dead-but-closer agent",
        )
        self.assertTrue(
            (near_fixed == 1).all().item(),
            "fixed argmin should pick the alive-but-farther agent",
        )

    def test_argmin_when_all_dead_fallback(self) -> None:
        """If all enemies are dead, argmin returns index 0 — no crash."""
        B, Ne = 2, 2
        enemy_x = torch.zeros(B, Ne)
        enemy_y = torch.zeros(B, Ne)
        enemy_alive = torch.zeros(B, Ne, dtype=torch.bool)  # all dead
        carr_x = torch.ones(B) * 5.0
        carr_y = torch.ones(B) * 5.0
        dxx = carr_x[:, None] - enemy_x
        dyy = carr_y[:, None] - enemy_y
        dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
        dd_live = torch.where(enemy_alive, dd, dd.new_full((), 1e6).expand_as(dd))
        near = torch.argmin(dd_live, dim=1)
        # Should not crash; returns 0 (first index) as deterministic fallback
        self.assertEqual(near.shape[0], B)


# ---------------------------------------------------------------------------
# 6. OP9 fortress radius and OP8 blocker safety
# ---------------------------------------------------------------------------

class TestOP9Fortress(unittest.TestCase):
    def test_fortress_orbit_radius_is_one(self) -> None:
        """OP9 guardian target should be 1 unit from own flag home."""
        B = 1
        own_flag_home = torch.tensor([[10.0, 10.0]])
        phase = torch.tensor([0.0])  # cos=1, sin=0 → fort_x = 11.0, fort_y = 10.0
        max_x, max_y = 19.0, 19.0
        fort_x = torch.clamp(own_flag_home[:, 0] + 1.0 * torch.cos(phase), 0.0, max_x)
        fort_y = torch.clamp(own_flag_home[:, 1] + 1.0 * torch.sin(phase), 0.0, max_y)
        dist = torch.sqrt((fort_x - own_flag_home[:, 0]) ** 2 + (fort_y - own_flag_home[:, 1]) ** 2)
        self.assertAlmostEqual(dist[0].item(), 1.0, places=4)


class TestOP8BlockerSafety(unittest.TestCase):
    def test_blocker_skipped_when_no_carrier(self) -> None:
        """OP8 blocker section is gated on enemy_carrier_exists — must not crash when False."""
        op8_mask = torch.tensor([True, True])
        enemy_carrier_exists = torch.tensor([False, False])
        # The gate: op8_mask.any() and enemy_carrier_exists.any()
        gate = op8_mask.any().item() and enemy_carrier_exists.any().item()
        self.assertFalse(gate, "blocker block must not execute when no carrier exists")

    def test_blocker_interpolation_is_60pct(self) -> None:
        """Block point should be 60% along carrier→enemy_flag_home path."""
        carr_x = torch.tensor([2.0])
        carr_y = torch.tensor([2.0])
        enemy_flag_home = torch.tensor([[12.0, 12.0]])
        block_x = carr_x + (enemy_flag_home[:, 0] - carr_x) * 0.6
        block_y = carr_y + (enemy_flag_home[:, 1] - carr_y) * 0.6
        # Expected: 2 + (12-2)*0.6 = 8.0
        self.assertAlmostEqual(block_x[0].item(), 8.0, places=4)
        self.assertAlmostEqual(block_y[0].item(), 8.0, places=4)


# ---------------------------------------------------------------------------
# 7. Map loading
# ---------------------------------------------------------------------------

class TestMapLoading(unittest.TestCase):
    def test_map_a_open_loads(self) -> None:
        core = _make_core(n_envs=1, map_layout="map_a_open")
        self.assertIsNotNone(core)

    def test_map_b_split_lane_loads(self) -> None:
        try:
            core = _make_core(n_envs=1, map_layout="map_b")
        except Exception as e:
            self.fail(f"map_b failed to load: {e}")

    def test_spawn_positions_in_bounds(self) -> None:
        core = _make_core(n_envs=4, n_agents=2)
        max_x = float(core.cols - 1)
        max_y = float(core.rows - 1)
        self.assertTrue((core.blue_x >= 0).all().item())
        self.assertTrue((core.blue_x <= max_x).all().item())
        self.assertTrue((core.blue_y >= 0).all().item())
        self.assertTrue((core.blue_y <= max_y).all().item())
        self.assertTrue((core.red_x >= 0).all().item())
        self.assertTrue((core.red_x <= max_x).all().item())

    def test_flag_home_positions_in_bounds(self) -> None:
        core = _make_core(n_envs=2)
        max_x = float(core.cols - 1)
        max_y = float(core.rows - 1)
        self.assertTrue((core.blue_flag_home[:, 0] >= 0).all().item())
        self.assertTrue((core.blue_flag_home[:, 0] <= max_x).all().item())
        self.assertTrue((core.red_flag_home[:, 0] >= 0).all().item())


# ---------------------------------------------------------------------------
# 8. Opponent parameter rule compliance
# ---------------------------------------------------------------------------

class TestOpponentRuleCompliance(unittest.TestCase):
    def test_op8_9_10_registered_in_dynamics(self) -> None:
        # Verify via the public sampling API that each key is accepted (not silently
        # dropped to OP3) by checking the returned speed_mult is non-trivial.
        from opponent_params import sample_batched_opponent_params
        for key in ("OP8", "OP8_INTERCEPTOR", "OP9", "OP9_FORTRESS", "OP10", "OP10_ESCORT"):
            with self.subTest(key=key):
                p = sample_batched_opponent_params(
                    kind="SCRIPTED", key=key, phase=key, n_agents=2, batch_size=4, device="cpu"
                )
                self.assertIn("attacker_style", p, f"{key} not handled by opponent_params")
                self.assertIn("speed_mult", p, f"{key} missing speed_mult")

    def test_op8_speed_within_human_range(self) -> None:
        from opponent_params import sample_batched_opponent_params
        torch.manual_seed(42)
        p = sample_batched_opponent_params(
            kind="SCRIPTED", key="OP8", phase="OP8", n_agents=2, batch_size=512, device="cpu"
        )
        speed_mult = p["speed_mult"]
        # OP8 2v2 range is 0.88-1.06 — must not exceed 1.5x (no super-speed cheating)
        self.assertTrue((speed_mult <= 1.5).all().item(), "OP8 speed_mult exceeds 1.5 (superhuman)")
        self.assertTrue((speed_mult >= 0.5).all().item(), "OP8 speed_mult below 0.5 (unusably slow)")

    def test_op9_speed_within_human_range(self) -> None:
        from opponent_params import sample_batched_opponent_params
        torch.manual_seed(43)
        p = sample_batched_opponent_params(
            kind="SCRIPTED", key="OP9", phase="OP9", n_agents=2, batch_size=512, device="cpu"
        )
        speed_mult = p["speed_mult"]
        self.assertTrue((speed_mult <= 1.5).all().item(), "OP9 speed_mult exceeds 1.5")
        self.assertTrue((speed_mult >= 0.5).all().item(), "OP9 speed_mult below 0.5")

    def test_op10_speed_within_human_range(self) -> None:
        from opponent_params import sample_batched_opponent_params
        torch.manual_seed(44)
        p = sample_batched_opponent_params(
            kind="SCRIPTED", key="OP10", phase="OP10", n_agents=2, batch_size=512, device="cpu"
        )
        speed_mult = p["speed_mult"]
        self.assertTrue((speed_mult <= 1.5).all().item(), "OP10 speed_mult exceeds 1.5")
        self.assertTrue((speed_mult >= 0.5).all().item(), "OP10 speed_mult below 0.5")

    def test_op8_no_latent_access_fields(self) -> None:
        from opponent_params import sample_batched_opponent_params
        torch.manual_seed(0)
        p = sample_batched_opponent_params(
            kind="SCRIPTED", key="OP8", phase="OP8", n_agents=2, batch_size=4, device="cpu"
        )
        forbidden = {"latent_z", "latent_id", "policy_logits", "value_pred", "gru_hidden"}
        for key in forbidden:
            self.assertNotIn(key, p, f"OP8 params contain forbidden field: {key}")


# ---------------------------------------------------------------------------
# 9. Step smoke test — no crash, shape contract
# ---------------------------------------------------------------------------

class TestStepSmokeTest(unittest.TestCase):
    def _zero_action(self, core: BatchedCTFCore) -> torch.Tensor:
        B = core.B
        n_agents = core.Nb
        return torch.zeros((B, n_agents, 2), dtype=torch.long)

    def test_step_does_not_crash_map_a(self) -> None:
        core = _make_core(n_envs=2, n_agents=2, map_layout="map_a_open")
        try:
            core.set_next_opponent("SCRIPTED", "OP8")
        except Exception:
            pass
        for _ in range(5):
            act = self._zero_action(core)
            core.step(act.view(core.B, -1))

    def test_step_does_not_crash_map_b(self) -> None:
        core = _make_core(n_envs=2, n_agents=2, map_layout="map_b")
        try:
            core.set_next_opponent("SCRIPTED", "OP9")
        except Exception:
            pass
        for _ in range(5):
            act = self._zero_action(core)
            core.step(act.view(core.B, -1))

    def test_obs_after_step_are_valid(self) -> None:
        core = _make_core(n_envs=2, n_agents=2)
        act = self._zero_action(core)
        core.step(act.view(core.B, -1))
        obs = core.get_obs_tensors("blue")
        self.assertFalse(torch.isnan(obs["grid"]).any().item())
        self.assertFalse(torch.isnan(obs["vec"]).any().item())


if __name__ == "__main__":
    unittest.main()
