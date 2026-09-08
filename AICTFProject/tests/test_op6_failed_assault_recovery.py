"""OP6 failed-assault recovery (defend-then-counter trap).

Legal triggers only: tag / flag-loss on blue's half. Never blue style ID.
OP9 path must remain untouched.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_red import ROLE_ATTACKER, ROLE_FLAG_RETR, _BTRedMixin
from tests.test_bt_op5_curriculum import _make_core, _run_bt


def _ensure_blue_anchor(core) -> None:
    """Place one non-carrier blue near its flag so recovery is allowed to arm."""
    midline = float(core.cols) * 0.5
    core.blue_alive[0] = True
    core.blue_carrying[0] = False
    hx = float(core.blue_flag_home[0, 0].item())
    hy = float(core.blue_flag_home[0, 1].item())
    core.blue_x[0, 0] = hx
    core.blue_y[0, 0] = hy
    # Keep second blue away so tests stay single-anchor.
    core.blue_x[0, 1] = midline + 2.0


class TestOP6FailedAssaultRecovery(unittest.TestCase):
    def test_recovery_duration_constant(self) -> None:
        self.assertGreaterEqual(int(_BTRedMixin._OP6_RECOVERY_DURATION), 20)

    def test_tag_on_blue_half_activates_recovery(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        # Dual-rush committed: both reds on blue half.
        core.red_x[0, 0] = midline - 2.0
        core.red_x[0, 1] = midline - 3.0
        core.red_y[0, 0] = 8.0
        core.red_y[0, 1] = 12.0
        _ensure_blue_anchor(core)
        core.red_carrying[0, 0] = False
        core.bt_op6_prev_red_carrying[0, 0] = False
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = False
        core.bt_role_lock_ticks[0] = 0
        _run_bt(core, "OP6")
        self.assertGreater(
            int(core.bt_op6_recovery_ticks[0, 0].item()),
            0,
            "committed dual-rush assault stop with blue anchor should arm recovery",
        )

    def test_no_recovery_when_blue_abandons_home(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        core.red_x[0, 0] = midline - 2.0
        core.red_x[0, 1] = midline - 3.0
        # Both blues on red's half — empty-home abandonment.
        core.blue_x[0, 0] = midline + 2.0
        core.blue_x[0, 1] = midline + 3.0
        core.blue_alive[0] = True
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = False
        core.bt_op6_prev_red_carrying[0, 0] = False
        _run_bt(core, "OP6")
        self.assertEqual(
            int(core.bt_op6_recovery_ticks[0, 0].item()),
            0,
            "empty-home blues must not receive OP6 recovery gift",
        )

    def test_solo_blue_half_tag_does_not_activate(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        core.red_x[0, 0] = midline - 2.0
        core.red_x[0, 1] = midline + 3.0  # partner still on red half
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = False
        core.bt_op6_prev_red_carrying[0, 0] = False
        core.red_carrying[0, 0] = False
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_recovery_ticks[0, 0].item()), 0)

    def test_tag_on_red_half_without_carry_does_not_activate(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        core.red_x[0, 0] = midline + 2.0
        core.red_carrying[0, 0] = False
        core.bt_op6_prev_red_carrying[0, 0] = False
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = False
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_recovery_ticks[0, 0].item()), 0)

    def test_carrier_stop_activates_recovery(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        _ensure_blue_anchor(core)
        core.red_x[0, 0] = midline - 2.0
        core.red_y[0, 0] = 8.0
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = False
        # Was carrying at previous step → carrier stop.
        core.bt_op6_prev_red_carrying[0, 0] = True
        core.red_carrying[0, 0] = False
        core.bt_role_lock_ticks[0] = 0
        _run_bt(core, "OP6")
        self.assertGreater(
            int(core.bt_op6_recovery_ticks[0, 0].item()),
            0,
            "carrier stop should arm OP6 recovery",
        )
        self.assertGreaterEqual(int(core.bt_op6_recovery_activations[0].item()), 1)
        self.assertGreaterEqual(int(core.bt_op6_failed_incursions[0].item()), 1)

    def test_no_renew_while_recovery_active(self) -> None:
        core, _ = _make_core("OP6")
        _ensure_blue_anchor(core)
        core.bt_op6_recovery_ticks[0, 0] = 10
        core.red_tagged[0, 0] = False  # countdown only while free to move
        core.bt_op6_prev_red_carrying[0, 0] = True
        core.red_carrying[0, 0] = False
        core.bt_op6_prev_red_tagged[0, 0] = False
        # Another fail while active must not renew.
        core.red_tagged[0, 0] = False
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_recovery_ticks[0, 0].item()), 9)
        self.assertEqual(int(core.bt_op6_recovery_activations[0].item()), 0)

    def test_recovery_ticks_pause_while_tagged(self) -> None:
        core, _ = _make_core("OP6")
        _ensure_blue_anchor(core)
        core.bt_op6_recovery_ticks[0, 0] = 10
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = True
        core.bt_op6_prev_red_carrying[0, 0] = False
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_recovery_ticks[0, 0].item()), 10)

    def test_flag_loss_activates_recovery(self) -> None:
        core, _ = _make_core("OP6")
        _ensure_blue_anchor(core)
        core.red_carrying[0, 1] = False
        core.bt_op6_prev_red_carrying[0, 1] = True
        core.bt_role_lock_ticks[0] = 0
        _run_bt(core, "OP6")
        self.assertGreater(int(core.bt_op6_recovery_ticks[0, 1].item()), 0)

    def test_recovery_keeps_attacker_and_blocks_flag_retr(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        _ensure_blue_anchor(core)
        # Agent 0 recovering; blue has stolen red flag → FLAG_RETR would fire
        # without recovery suppress.
        core.bt_op6_recovery_ticks[0, 0] = 20
        core.red_flag_pos[0, 0] = midline - 4.0
        core.red_flag_pos[0, 1] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(roles[0], ROLE_ATTACKER)
        self.assertNotIn(
            ROLE_FLAG_RETR,
            roles,
            "anchor-gated recovery must leave rear exposed (no FLAG_RETR)",
        )

    def test_recovery_route_is_midfield_not_enemy_flag(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        max_x = float(max(0, core.cols - 1))
        _ensure_blue_anchor(core)
        core.bt_op6_recovery_ticks[0, 0] = 20
        core.red_tagged[0, 0] = False
        core.red_x[0, 0] = midline - 3.0
        core.bt_role_lock_ticks[0] = 0
        roles, tx, _ = _run_bt(core, "OP6")
        self.assertEqual(roles[0], ROLE_ATTACKER)
        stage = midline + 0.20 * (max_x - midline)
        self.assertAlmostEqual(float(tx[0]), stage, places=3)
        # Must not chase the blue flag while recovering.
        efx = float(core.blue_flag_pos[0, 0].item())
        self.assertGreater(abs(float(tx[0]) - efx), 1.0)

    def test_op9_does_not_arm_recovery(self) -> None:
        core, _ = _make_core("OP9")
        midline = float(core.cols) * 0.5
        core.red_x[0, 0] = midline - 2.0
        core.red_tagged[0, 0] = True
        core.bt_op6_prev_red_tagged[0, 0] = False
        _run_bt(core, "OP9")
        self.assertEqual(int(core.bt_op6_recovery_ticks[0, 0].item()), 0)
        self.assertEqual(int(core.bt_op6_recovery_activations[0].item()), 0)

    def test_recovery_logic_has_no_blue_style_detector(self) -> None:
        import inspect
        from gpu_env._core import _bt_red

        src = inspect.getsource(_bt_red._BTRedMixin._bt_update_op6_recovery)
        forbidden = ("BLUE_TURTLE", "blue_style", "style_id", "_STYLE_ID")
        for token in forbidden:
            self.assertNotIn(token, src)


if __name__ == "__main__":
    unittest.main()
