"""OP7 separated-threat SPLIT lever — OP7-only, OP6/OP8 untouched."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_red import (  # noqa: E402
    ROLE_DEFENDER,
    ROLE_INTERCEPTOR,
    _BTRedMixin,
)
from tests.test_bt_op5_curriculum import _make_core, _run_bt  # noqa: E402


def _place_split_geometry(core) -> None:
    """Both blues offensive, opposite corridors, wide separation."""
    midline = float(core.cols) * 0.5
    center_y = float(core.rows) * 0.5
    core.blue_x[0, 0] = midline + 2.0
    core.blue_x[0, 1] = midline + 1.5
    core.blue_y[0, 0] = center_y + float(core.rows) * 0.35
    core.blue_y[0, 1] = center_y - float(core.rows) * 0.35
    core.blue_alive[0] = True
    core.blue_tagged[0] = False
    core.blue_carrying[0] = False
    # Reds near flag so defender assignment is available.
    core.red_x[0, 0] = midline + 4.0
    core.red_y[0, 0] = center_y + 1.0
    core.red_x[0, 1] = midline + 4.0
    core.red_y[0, 1] = center_y - 1.0
    core.bt_role_lock_ticks[0] = 0


def _place_rush_geometry(core) -> None:
    """Concentrated same-corridor push (compact latch, not SPLIT latch)."""
    midline = float(core.cols) * 0.5
    center_y = float(core.rows) * 0.5
    core.blue_x[0, 0] = midline + 2.0
    core.blue_x[0, 1] = midline + 1.0
    core.blue_y[0, 0] = center_y + 0.5
    core.blue_y[0, 1] = center_y - 0.4
    core.blue_alive[0] = True
    core.blue_tagged[0] = False
    core.blue_carrying[0] = False
    # One red deep near flag, one forward.
    core.red_x[0, 0] = float(core.red_flag_home[0, 0].item())
    core.red_y[0, 0] = float(core.red_flag_home[0, 1].item())
    core.red_x[0, 1] = midline + 3.0
    core.red_y[0, 1] = center_y
    core.bt_role_lock_ticks[0] = 0


class TestOP7SplitOvercommit(unittest.TestCase):
    def test_constants_positive(self) -> None:
        self.assertGreater(int(_BTRedMixin._OP7_SPLIT_PRESSURE_TICKS), 0)
        self.assertGreater(int(_BTRedMixin._OP7_SPLIT_RESPONSE_DURATION), 0)

    def test_split_geometry_arms_latch_and_collapses_corridor(self) -> None:
        core, _ = _make_core("OP7", step=0)
        _place_split_geometry(core)
        ticks = int(_BTRedMixin._OP7_SPLIT_PRESSURE_TICKS)
        for i in range(ticks):
            core.sim_step_count[0] = i
            core.step_count[0] = i
            roles, tx, ty = _run_bt(core, "OP7")
        self.assertGreaterEqual(int(core.bt_op7_split_first_trigger_step[0].item()), 0)
        self.assertGreaterEqual(int(core.bt_op7_split_primary_blue_idx[0].item()), 0)
        self.assertIn(ROLE_DEFENDER, roles)
        self.assertIn(ROLE_INTERCEPTOR, roles)
        # Both reds target the same corridor (primary y band).
        primary = int(core.bt_op7_split_primary_blue_idx[0].item())
        corridor_y = float(core.bt_op7_split_corridor_y[0].item())
        py = float(core.blue_y[0, primary].item())
        self.assertLess(abs(float(ty[0]) - py), 1.5)
        self.assertLess(abs(float(ty[1]) - corridor_y), 1.5)
        # Lateral targets should not diverge onto opposite blue.
        other = 1 - primary
        other_y = float(core.blue_y[0, other].item())
        self.assertGreater(abs(float(ty[0]) - other_y), 3.0)
        self.assertGreater(abs(float(ty[1]) - other_y), 3.0)
        # Compact must not steal the separated window.
        self.assertEqual(int(core.bt_op7_compact_first_trigger_step[0].item()), -1)

    def test_rush_geometry_does_not_arm_split_but_arms_compact(self) -> None:
        core, _ = _make_core("OP7", step=0)
        _place_rush_geometry(core)
        # Compact lever is HARD-STOPPED (disabled). Assert geometry still
        # fails the SPLIT latch and that compact stays off when disabled.
        was = bool(_BTRedMixin._OP7_COMPACT_LEVER_ENABLED)
        try:
            _BTRedMixin._OP7_COMPACT_LEVER_ENABLED = True
            ticks = max(
                int(_BTRedMixin._OP7_SPLIT_PRESSURE_TICKS),
                int(_BTRedMixin._OP7_COMPACT_PRESSURE_TICKS),
            ) + 2
            for i in range(ticks):
                core.sim_step_count[0] = i
                roles, tx, ty = _run_bt(core, "OP7")
            self.assertEqual(int(core.bt_op7_split_first_trigger_step[0].item()), -1)
            self.assertEqual(int(core.bt_op7_split_activations[0].item()), 0)
            self.assertGreaterEqual(int(core.bt_op7_compact_first_trigger_step[0].item()), 0)
            self.assertIn(ROLE_DEFENDER, roles)
            self.assertIn(ROLE_INTERCEPTOR, roles)
            self.assertGreater(
                abs(float(tx[0]) - float(tx[1])) + abs(float(ty[0]) - float(ty[1])),
                1.0,
            )
        finally:
            _BTRedMixin._OP7_COMPACT_LEVER_ENABLED = was

    def test_compact_lever_disabled_by_hard_stop(self) -> None:
        self.assertFalse(bool(_BTRedMixin._OP7_COMPACT_LEVER_ENABLED))
        core, _ = _make_core("OP7", step=0)
        _place_rush_geometry(core)
        ticks = int(_BTRedMixin._OP7_COMPACT_PRESSURE_TICKS) + 2
        for i in range(ticks):
            core.sim_step_count[0] = i
            _run_bt(core, "OP7")
        self.assertEqual(int(core.bt_op7_compact_first_trigger_step[0].item()), -1)
        self.assertEqual(int(core.bt_op7_compact_activations[0].item()), 0)

    def test_flag_pickup_clears_latch(self) -> None:
        core, _ = _make_core("OP7", step=0)
        _place_split_geometry(core)
        ticks = int(_BTRedMixin._OP7_SPLIT_PRESSURE_TICKS)
        for i in range(ticks):
            core.sim_step_count[0] = i
            _run_bt(core, "OP7")
        self.assertGreaterEqual(int(core.bt_op7_split_first_trigger_step[0].item()), 0)
        core.blue_carrying[0, 0] = True
        core.sim_step_count[0] = ticks
        _run_bt(core, "OP7")
        self.assertEqual(int(core.bt_op7_split_first_trigger_step[0].item()), -1)

    def test_op8_and_op6_paths_untouched(self) -> None:
        """Level gate: OP6/OP8 must not allocate OP7 latch response."""
        for key in ("OP6", "OP8"):
            core, _ = _make_core(key, step=0)
            _place_split_geometry(core)
            ticks = int(_BTRedMixin._OP7_SPLIT_PRESSURE_TICKS) + 1
            for i in range(ticks):
                core.sim_step_count[0] = i
                _run_bt(core, key)
            self.assertEqual(
                int(core.bt_op7_split_first_trigger_step[0].item()),
                -1,
                f"{key} should not arm OP7 latch",
            )
            self.assertEqual(
                int(core.bt_op7_compact_first_trigger_step[0].item()),
                -1,
                f"{key} should not arm OP7 compact",
            )


if __name__ == "__main__":
    unittest.main()
