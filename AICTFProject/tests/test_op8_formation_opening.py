"""OP8 formation-opening RUSH contract (Contract A) — OP8-only, OP9-safe."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_adaptive import _BTAdaptiveMixin
from gpu_env._core._bt_profiles import profile_for_level, role_gate_fingerprint
from gpu_env._core._bt_red import ROLE_ATTACKER, ROLE_DEFENDER, ROLE_INTERCEPTOR
from tests.test_bt_op5_curriculum import _make_core, _run_bt


class TestOP8FormationOpening(unittest.TestCase):
    def test_opening_steps_constant_is_positive(self) -> None:
        self.assertGreater(int(_BTAdaptiveMixin._OP8_FORMATION_OPENING_STEPS), 0)

    def test_opening_stages_not_enemy_flag(self) -> None:
        """Round-2: opening routes to midfield rally, not dual-rush to blue flag."""
        core, _ = _make_core("OP8", step=0)
        core.bt_role_lock_ticks[0] = 0
        roles, tx, _ty = _run_bt(core, "OP8")
        self.assertEqual(roles, [ROLE_ATTACKER, ROLE_ATTACKER], roles)
        midline = float(core.cols) * 0.5
        # Staging stays on the red side of midfield (red flag home is x > midline).
        self.assertGreater(float(tx[0]), midline - 0.5)
        self.assertGreater(float(tx[1]), midline - 0.5)
        # Not sitting on the blue flag.
        blue_flag_x = float(core.blue_flag_pos[0, 0].item())
        self.assertGreater(abs(float(tx[0]) - blue_flag_x), 2.0)

    def test_opening_forces_attackers_and_skips_defender(self) -> None:
        core, _ = _make_core("OP8", step=0)
        midline = float(core.cols) * 0.5
        # Intruder on red half would normally pull a DEFENDER after formation.
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP8")
        self.assertEqual(roles, [ROLE_ATTACKER, ROLE_ATTACKER], roles)
        self.assertNotIn(ROLE_DEFENDER, roles)

    def test_post_formation_can_defend_intruder(self) -> None:
        steps = int(_BTAdaptiveMixin._OP8_FORMATION_OPENING_STEPS)
        core, _ = _make_core("OP8", step=steps)
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP8")
        self.assertIn(ROLE_DEFENDER, roles, f"expected DEFENDER after formation, got {roles}")

    def test_blue_carry_ends_opening_and_allows_intercept(self) -> None:
        core, _ = _make_core("OP8", step=0)
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 10.0
        core.blue_y[0, 0] = 10.0
        core.red_x[0, 0] = 8.0
        core.red_y[0, 0] = 9.0
        core.red_x[0, 1] = 8.0
        core.red_y[0, 1] = 11.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP8")
        self.assertIn(ROLE_INTERCEPTOR, roles, f"expected INTERCEPTOR on blue carry, got {roles}")

    def test_op9_path_unchanged_with_intruder(self) -> None:
        """OP9 must not enter the OP8 opening gate (level-gated)."""
        core, _ = _make_core("OP9", step=0)
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP9")
        # OP9 retains defender response; it is not forced into dual ATTACKER opening.
        self.assertNotEqual(roles, [ROLE_ATTACKER, ROLE_ATTACKER], roles)
        self.assertIn(ROLE_DEFENDER, roles, f"OP9 should still defend, got {roles}")

    def test_op8_fingerprint_unchanged(self) -> None:
        # Opening is a temporal gate; structural niche fingerprint stays put.
        self.assertEqual(
            role_gate_fingerprint(8),
            (
                True,   # escort
                True,   # counter
                False,  # counter_always
                False,  # mines
                True,   # 2v1
                True,   # intercept
            ),
        )
        self.assertEqual(profile_for_level(8).name, "OP8_PROTECTED_CARRIER_ESCORT")


if __name__ == "__main__":
    unittest.main()
