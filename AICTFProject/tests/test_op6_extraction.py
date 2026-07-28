"""OP6 post-pickup extraction support (carrier + screener)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_red import ROLE_ATTACKER, ROLE_ESCORT, _BTRedMixin
from tests.test_bt_op5_curriculum import _make_core, _run_bt
from tests.test_op6_failed_assault_recovery import _ensure_blue_anchor


class TestOP6ExtractionSupport(unittest.TestCase):
    def setUp(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = True

    def tearDown(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = True

    def test_extraction_assigns_screener_when_blue_abandons_home(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        # Both blues deep on red half — abandoned home.
        core.blue_x[0, 0] = midline + 3.0
        core.blue_x[0, 1] = midline + 4.0
        core.blue_alive[0] = True
        core.blue_carrying[0] = False
        core.red_carrying[0, 0] = True
        core.red_tagged[0] = False
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(roles[0], ROLE_ATTACKER)  # carrier
        self.assertEqual(roles[1], ROLE_ESCORT)  # screener
        self.assertGreater(int(core.bt_op6_extract_ticks[0].item()), 0)

    def test_no_extraction_screen_when_blue_anchor_present(self) -> None:
        core, _ = _make_core("OP6")
        _ensure_blue_anchor(core)
        core.red_carrying[0, 0] = True
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertNotEqual(roles[1], ROLE_ESCORT)
        self.assertEqual(int(core.bt_op6_extract_ticks[0].item()), 0)

    def test_carrier_direct_home_during_extraction(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        # Blues abandoned home but NOT on the carrier→home segment.
        core.blue_x[0, 0] = midline + 3.0
        core.blue_x[0, 1] = midline + 4.0
        core.blue_y[0, 0] = 1.0
        core.blue_y[0, 1] = 2.0
        core.blue_alive[0] = True
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = midline - 2.0
        hy = float(core.red_flag_home[0, 1].item())
        core.red_y[0, 0] = hy
        core.bt_role_lock_ticks[0] = 0
        _, tx, ty = _run_bt(core, "OP6")
        hx = float(core.red_flag_home[0, 0].item())
        self.assertAlmostEqual(float(tx[0]), hx, places=3)
        self.assertAlmostEqual(float(ty[0]), hy, places=3)

    def test_extraction_toggle_off(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = False
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 3.0
        core.blue_x[0, 1] = midline + 4.0
        core.blue_alive[0] = True
        core.red_carrying[0, 0] = True
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertNotEqual(roles[1], ROLE_ESCORT)

    def test_screen_break_peels_carrier_off_blocker_corridor(self) -> None:
        """Carrier takes safer corridor when a blue sits on the return line."""
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        max_y = float(max(0, core.rows - 1))
        # Blues abandoned home (deep).
        core.blue_x[0, 0] = midline + 2.0
        core.blue_x[0, 1] = midline + 3.0
        core.blue_y[0, 0] = float(core.red_flag_home[0, 1].item())
        core.blue_y[0, 1] = float(core.red_flag_home[0, 1].item()) + 1.0
        core.blue_alive[0] = True
        core.blue_carrying[0] = False
        # Red carrier between mid and home; blue 0 on the segment.
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = (midline + hx) * 0.5
        core.red_y[0, 0] = hy
        core.blue_x[0, 0] = (float(core.red_x[0, 0].item()) + hx) * 0.5
        core.blue_y[0, 0] = hy
        core.bt_role_lock_ticks[0] = 0
        roles, tx, ty = _run_bt(core, "OP6")
        self.assertEqual(roles[1], ROLE_ESCORT)
        # Carrier y should peel away from blocker y (=hy), not sit on hy mid-route.
        self.assertGreater(abs(float(ty[0]) - hy), 0.5)
        # Screener engages toward blocker, not deep dual-assault on blue flag.
        efx = float(core.blue_flag_pos[0, 0].item())
        self.assertGreater(abs(float(tx[1]) - efx), 1.0)
        import inspect
        from gpu_env._core import _bt_red

        src = inspect.getsource(_bt_red._BTRedMixin._bt_assign_roles)
        start = src.find("OP6 extraction support")
        end = src.find("Priority 1: flag retrieval")
        self.assertGreater(start, 0)
        chunk = src[start:end]
        for token in ("BLUE_TURTLE", "BLUE_RUSH", "BLUE_SPLIT", "style_id", "_STYLE_ID"):
            self.assertNotIn(token, chunk)


if __name__ == "__main__":
    unittest.main()
