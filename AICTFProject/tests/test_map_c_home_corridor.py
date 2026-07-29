"""Pin Map C (home-corridor) as a single mandatory chokepoint.

Version 1 left open bypasses above and below the wall (~40%+ of field
height combined). Version 2 must keep the wall flush to the bottom edge
with exactly one navigable gap near the top.
"""
from __future__ import annotations

import unittest

from gpu_env._maps import (
    MAP_C_HOME_CORRIDOR,
    home_corridor_rect_norm,
    normalize_map_layout,
    norm_rect_to_cells,
)


class TestMapCHomeCorridorGeometry(unittest.TestCase):
    def test_alias_resolves(self) -> None:
        self.assertEqual(normalize_map_layout("map_c"), MAP_C_HOME_CORRIDOR)
        self.assertEqual(normalize_map_layout("home_corridor"), MAP_C_HOME_CORRIDOR)

    def test_wall_flush_to_bottom_single_top_gap(self) -> None:
        x0, y0, x1, y1 = home_corridor_rect_norm(mirror_y=False)
        self.assertAlmostEqual(y1, 1.0, places=6)
        self.assertLess(y0, 0.28)  # narrower than V1 top opening
        self.assertGreaterEqual(y0, 0.15)  # still passable under router clearance
        self.assertLess(x0, x1)
        # Near blue home (flag ~0.10), not midfield like Map B.
        self.assertLess(x1, 0.35)

    def test_cell_gaps_match_single_choke_contract(self) -> None:
        cols, rows = 20, 20
        max_y = float(rows - 1)
        x0, y0, x1, y1 = norm_rect_to_cells(
            home_corridor_rect_norm(mirror_y=False), cols=cols, rows=rows
        )
        top_gap = y0 - 0.0
        bot_gap = max_y - y1
        # Bottom sealed; top is the only passage and stays navigable.
        self.assertAlmostEqual(bot_gap, 0.0, places=5)
        self.assertGreaterEqual(top_gap, 3.0)
        self.assertLessEqual(top_gap, 5.0)
        # V1 dual-bypass regression: both ends were >5 cells open.
        self.assertLess(top_gap, 5.2)

    def test_mirror_still_single_gap(self) -> None:
        cols, rows = 20, 20
        max_y = float(rows - 1)
        _x0, y0, _x1, y1 = norm_rect_to_cells(
            home_corridor_rect_norm(mirror_y=True), cols=cols, rows=rows
        )
        top_gap = y0 - 0.0
        bot_gap = max_y - y1
        # Mirror flips the gap to the bottom; top becomes sealed.
        self.assertAlmostEqual(top_gap, 0.0, places=5)
        self.assertGreaterEqual(bot_gap, 3.0)
        self.assertLessEqual(bot_gap, 5.0)

    def test_router_disables_sealed_detour_end(self) -> None:
        """Flush edge must make the sealed detour end unusable."""
        import torch

        from gpu_env import GPUCTFVecEnv, GPUFieldConfig

        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=2,
            map_layout=MAP_C_HOME_CORRIDOR,
            map_b_vertical_mirror_prob=0.0,  # force non-mirrored (gap at top)
            aquaticus_profile=True,
            rules_profile="OURS",
            device="cpu",
            seed=7,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            env.reset()
            core = env.core
            rect = core.obstacle_rects[0, 0]
            y0 = float(rect[1].item())
            y1 = float(rect[3].item())
            max_y = float(max(0, core.rows - 1))
            clearance = 1.5
            top_y = max(0.0, min(max_y, y0 - clearance))
            bot_y = max(0.0, min(max_y, y1 + clearance))
            top_ok = top_y < (y0 - 1e-3)
            bot_ok = bot_y > (y1 + 1e-3)
            self.assertTrue(top_ok)
            self.assertFalse(bot_ok)
            # Midfield horizontal crossing must be forced through the top gap.
            own_x = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
            own_y = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
            tgt_x = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
            tgt_y = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
            aim_x, aim_y = core._route_targets_around_obstacles(
                own_x, own_y, tgt_x, tgt_y
            )
            # Aim should pull toward the open (top) corridor, not the sealed bottom.
            self.assertTrue(bool((aim_y < y0).all().item()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
