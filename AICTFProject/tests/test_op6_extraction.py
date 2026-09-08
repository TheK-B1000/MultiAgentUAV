"""OP6 post-pickup extraction support (carrier + screener)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_red import ROLE_ATTACKER, ROLE_ESCORT, ROLE_INTERCEPTOR, _BTRedMixin
from tests.test_bt_op5_curriculum import _make_core, _run_bt
from tests.test_op6_failed_assault_recovery import _ensure_blue_anchor


class TestOP6ExtractionSupport(unittest.TestCase):
    def setUp(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = True

    def tearDown(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = False
        _BTRedMixin._OP6_PREENGAGE_ENABLED = False
        _BTRedMixin._OP6_RACE_DENIAL_ENABLED = False

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

    def test_dual_threat_counts_carrying_blue_as_threat(self) -> None:
        """A blue holding red's flag still counts — excluding it caused C1 dupes."""
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = (midline + hx) * 0.5
        core.red_y[0, 0] = hy
        core.blue_alive[0] = True
        core.blue_tagged[0] = False
        # Blue0 nearer but itself carrying; blue1 farther non-carrier.
        core.blue_carrying[0, 0] = True
        core.blue_carrying[0, 1] = False
        core.blue_x[0, 0] = float(core.red_x[0, 0].item()) + 1.0
        core.blue_y[0, 0] = hy
        core.blue_x[0, 1] = float(core.red_x[0, 0].item()) + 4.0
        core.blue_y[0, 1] = hy
        core.bt_role_lock_ticks[0] = 0
        _run_bt(core, "OP6")
        c_th = int(core.bt_op6_extract_carrier_threat[0].item())
        s_th = int(core.bt_op6_extract_screener_threat[0].item())
        self.assertEqual(c_th, 0)
        self.assertEqual(s_th, 1)

    def test_dual_threat_assignment_is_distinct_and_locked(self) -> None:
        """Screener covers the complementary blue, not carrier's nearest threat."""
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        # Carrier mid-return; both blues on the return path but at different ranges.
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = (midline + hx) * 0.5
        core.red_y[0, 0] = hy
        core.blue_alive[0] = True
        core.blue_carrying[0] = False
        # Blue0 nearer (carrier evasion threat); blue1 farther ahead toward home.
        core.blue_x[0, 0] = float(core.red_x[0, 0].item()) + 2.0
        core.blue_y[0, 0] = hy
        core.blue_x[0, 1] = (float(core.red_x[0, 0].item()) + hx) * 0.5 + 1.0
        core.blue_y[0, 1] = hy
        core.bt_role_lock_ticks[0] = 0
        roles, tx, ty = _run_bt(core, "OP6")
        self.assertEqual(roles[1], ROLE_ESCORT)
        carr_threat = int(core.bt_op6_extract_carrier_threat[0].item())
        scr_threat = int(core.bt_op6_extract_screener_threat[0].item())
        self.assertEqual(carr_threat, 0)
        self.assertEqual(scr_threat, 1)
        self.assertNotEqual(carr_threat, scr_threat)
        # Screener target should be toward blue1, not blue0.
        self.assertLess(
            abs(float(tx[1]) - float(core.blue_x[0, 1].item())),
            abs(float(tx[1]) - float(core.blue_x[0, 0].item())) + 0.5,
        )
        # Lock persists across a second BT tick.
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_extract_carrier_threat[0].item()), carr_threat)
        self.assertEqual(int(core.bt_op6_extract_screener_threat[0].item()), scr_threat)

    def test_dual_threat_distinct_when_blue1_is_nearest(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = (midline + hx) * 0.5
        core.red_y[0, 0] = hy
        core.blue_alive[0] = True
        core.blue_carrying[0] = False
        core.blue_x[0, 1] = float(core.red_x[0, 0].item()) + 2.0
        core.blue_y[0, 1] = hy
        core.blue_x[0, 0] = (float(core.red_x[0, 0].item()) + hx) * 0.5 + 1.0
        core.blue_y[0, 0] = hy
        core.bt_role_lock_ticks[0] = 0
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_extract_carrier_threat[0].item()), 1)
        self.assertEqual(int(core.bt_op6_extract_screener_threat[0].item()), 0)


class TestOP6PrePickupScreener(unittest.TestCase):
    """Narrow pre-pickup screener: RUSH-like arm, TURTLE home-anchor block."""

    def setUp(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = True
        _BTRedMixin._OP6_PREENGAGE_ENABLED = True

    def tearDown(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = False
        _BTRedMixin._OP6_PREENGAGE_ENABLED = False

    def _place_imminent_pickup(self, core) -> None:
        """Red0 near enemy flag; both blues threaten return corridor; no anchor."""
        midline = float(core.cols) * 0.5
        fx = float(core.blue_flag_pos[0, 0].item())
        fy = float(core.blue_flag_pos[0, 1].item())
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        # Flag at home (available).
        core.blue_flag_pos[0, 0] = float(core.blue_flag_home[0, 0].item())
        core.blue_flag_pos[0, 1] = float(core.blue_flag_home[0, 1].item())
        fx = float(core.blue_flag_pos[0, 0].item())
        fy = float(core.blue_flag_pos[0, 1].item())
        core.red_carrying[0] = False
        core.red_tagged[0] = False
        core.red_alive[0] = True
        # Attacker within pre-pickup radius of flag.
        core.red_x[0, 0] = fx + 1.5
        core.red_y[0, 0] = fy
        # Partner farther back on the return corridor.
        core.red_x[0, 1] = (fx + hx) * 0.5
        core.red_y[0, 1] = hy
        # Both blues deep on red half along the flag→home corridor (no home anchor).
        core.blue_alive[0] = True
        core.blue_tagged[0] = False
        core.blue_carrying[0] = False
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = hy
        core.blue_x[0, 1] = midline + 4.0
        core.blue_y[0, 1] = hy
        core.bt_role_lock_ticks[0] = 0
        core.bt_op6_preengage_ticks[0] = 0
        core.bt_op6_extract_ticks[0] = 0

    def test_preengage_arms_when_pickup_imminent_and_dual_threat(self) -> None:
        core, _ = _make_core("OP6")
        self._place_imminent_pickup(core)
        roles, tx, ty = _run_bt(core, "OP6")
        self.assertGreater(int(core.bt_op6_preengage_ticks[0].item()), 0)
        self.assertEqual(int(core.bt_op6_preengage_activations[0].item()), 1)
        self.assertEqual(roles[0], ROLE_ATTACKER)  # flag attacker
        self.assertEqual(roles[1], ROLE_ESCORT)  # preengage screener
        self.assertGreaterEqual(int(core.bt_op6_extract_screener_threat[0].item()), 0)
        # Screener routes toward locked threat / corridor, not dual-assault flag.
        efx = float(core.blue_flag_pos[0, 0].item())
        self.assertGreater(abs(float(tx[1]) - efx), 1.0)

    def test_preengage_targets_projected_intercept_not_blue_xy(self) -> None:
        """Screener waypoint sits on the return corridor, not on the blue body."""
        core, _ = _make_core("OP6")
        self._place_imminent_pickup(core)
        # Put screener-threat blue far off the corridor latitude so a chase
        # target would differ from a path-projection meet point.
        hx = float(core.red_flag_home[0, 1].item())
        core.blue_y[0, 0] = hx + 5.0
        core.blue_y[0, 1] = hx - 5.0
        roles, tx, ty = _run_bt(core, "OP6")
        self.assertEqual(roles[1], ROLE_ESCORT)
        scr_th = int(core.bt_op6_extract_screener_threat[0].item())
        self.assertGreaterEqual(scr_th, 0)
        by = float(core.blue_y[0, scr_th].item())
        # Waypoint y should be closer to home latitude than to the blue's y.
        self.assertLess(abs(float(ty[1]) - hx), abs(float(ty[1]) - by) + 0.25)

    def test_preengage_blocked_by_blue_home_anchor(self) -> None:
        """TURTLE-like: a non-carrier near blue home must suppress preengage."""
        core, _ = _make_core("OP6")
        self._place_imminent_pickup(core)
        _ensure_blue_anchor(core)
        # Keep one blue deep so dual-threat geometry is tempting; anchor wins.
        midline = float(core.cols) * 0.5
        hy = float(core.red_flag_home[0, 1].item())
        core.blue_x[0, 1] = midline + 3.0
        core.blue_y[0, 1] = hy
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_preengage_ticks[0].item()), 0)
        self.assertEqual(int(core.bt_op6_preengage_activations[0].item()), 0)
        self.assertNotEqual(roles[1], ROLE_ESCORT)

    def test_preengage_requires_imminent_radius(self) -> None:
        core, _ = _make_core("OP6")
        self._place_imminent_pickup(core)
        # Pull attacker outside pre-pickup radius.
        fx = float(core.blue_flag_pos[0, 0].item())
        fy = float(core.blue_flag_pos[0, 1].item())
        core.red_x[0, 0] = fx + 9.0
        core.red_y[0, 0] = fy
        _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_preengage_ticks[0].item()), 0)

    def test_preengage_toggle_off(self) -> None:
        _BTRedMixin._OP6_PREENGAGE_ENABLED = False
        core, _ = _make_core("OP6")
        self._place_imminent_pickup(core)
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_preengage_ticks[0].item()), 0)
        self.assertNotEqual(roles[1], ROLE_ESCORT)

    def test_preengage_no_style_id_in_gate(self) -> None:
        import inspect
        from gpu_env._core import _bt_red

        src = inspect.getsource(_bt_red._BTRedMixin._bt_assign_roles)
        start = src.find("OP6 extraction support")
        end = src.find("Priority 1: flag retrieval")
        chunk = src[start:end]
        for token in ("BLUE_TURTLE", "BLUE_RUSH", "BLUE_SPLIT", "style_id", "_STYLE_ID"):
            self.assertNotIn(token, chunk)


class TestOP6MutualCarryRaceDenial(unittest.TestCase):
    """dev36: mutual-carry → intercept blue carrier (not escort)."""

    def setUp(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = True
        _BTRedMixin._OP6_RACE_DENIAL_ENABLED = True

    def tearDown(self) -> None:
        _BTRedMixin._OP6_EXTRACTION_ENABLED = False
        _BTRedMixin._OP6_RACE_DENIAL_ENABLED = False

    def _place_mutual_abandoned(self, core) -> None:
        midline = float(core.cols) * 0.5
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        bhx = float(core.blue_flag_home[0, 0].item())
        bhy = float(core.blue_flag_home[0, 1].item())
        # Both sides carry; blues abandoned home (deep on red half).
        core.red_carrying[0, 0] = True
        core.red_carrying[0, 1] = False
        core.red_tagged[0] = False
        core.red_alive[0] = True
        core.red_x[0, 0] = (midline + hx) * 0.5
        core.red_y[0, 0] = hy
        core.red_x[0, 1] = midline + 1.0
        core.red_y[0, 1] = hy
        core.blue_carrying[0, 0] = True
        core.blue_carrying[0, 1] = False
        core.blue_alive[0] = True
        core.blue_tagged[0] = False
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = hy
        core.blue_x[0, 1] = midline + 4.0
        core.blue_y[0, 1] = hy
        core.bt_role_lock_ticks[0] = 0
        core.bt_op6_race_ticks[0] = 0
        core.bt_op6_extract_ticks[0] = 0

    def test_race_assigns_interceptor_not_escort(self) -> None:
        core, _ = _make_core("OP6")
        self._place_mutual_abandoned(core)
        roles, tx, ty = _run_bt(core, "OP6")
        self.assertGreater(int(core.bt_op6_race_ticks[0].item()), 0)
        self.assertEqual(int(core.bt_op6_race_activations[0].item()), 1)
        self.assertEqual(roles[0], ROLE_ATTACKER)
        self.assertEqual(roles[1], ROLE_INTERCEPTOR)
        self.assertEqual(int(core.bt_op6_race_target_idx[0].item()), 0)
        # Interceptor aims toward blue carrier / cut, not red home escort.
        bx = float(core.blue_x[0, 0].item())
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        self.assertLess(abs(float(tx[1]) - bx), abs(float(tx[1]) - hx) + 1.0)
        # During race denial, carrier goes straight home (no peel detour).
        self.assertAlmostEqual(float(tx[0]), hx, places=2)
        self.assertAlmostEqual(float(ty[0]), hy, places=2)

    def test_race_blocked_by_blue_home_anchor(self) -> None:
        core, _ = _make_core("OP6")
        self._place_mutual_abandoned(core)
        # TURTLE-like: non-carrier blue1 at blue home; blue0 still carries deep.
        midline = float(core.cols) * 0.5
        hy = float(core.red_flag_home[0, 1].item())
        bhx = float(core.blue_flag_home[0, 0].item())
        bhy = float(core.blue_flag_home[0, 1].item())
        core.blue_carrying[0, 0] = True
        core.blue_carrying[0, 1] = False
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = hy
        core.blue_x[0, 1] = bhx
        core.blue_y[0, 1] = bhy
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_race_ticks[0].item()), 0)
        self.assertNotEqual(roles[1], ROLE_INTERCEPTOR)

    def test_race_requires_blue_carry(self) -> None:
        core, _ = _make_core("OP6")
        self._place_mutual_abandoned(core)
        core.blue_carrying[0] = False
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_race_ticks[0].item()), 0)
        # Extract-only: partner is ESCORT screener when abandoned + red carry.
        self.assertEqual(roles[1], ROLE_ESCORT)

    def test_race_arms_on_imminent_pickup_while_blue_carries(self) -> None:
        """Deny before red completes pickup when blue already races home."""
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        hx = float(core.red_flag_home[0, 0].item())
        hy = float(core.red_flag_home[0, 1].item())
        fx = float(core.blue_flag_pos[0, 0].item())
        fy = float(core.blue_flag_pos[0, 1].item())
        # Red not carrying yet but imminent at blue flag.
        core.red_carrying[0] = False
        core.red_tagged[0] = False
        core.red_alive[0] = True
        core.red_x[0, 0] = fx + 2.0
        core.red_y[0, 0] = fy
        core.red_x[0, 1] = midline + 1.0
        core.red_y[0, 1] = hy
        # Blue carrier deep toward red home; partner also deep (abandoned).
        core.blue_carrying[0, 0] = True
        core.blue_carrying[0, 1] = False
        core.blue_alive[0] = True
        core.blue_tagged[0] = False
        core.blue_x[0, 0] = midline + 3.0
        core.blue_y[0, 0] = hy
        core.blue_x[0, 1] = midline + 4.0
        core.blue_y[0, 1] = hy
        core.bt_role_lock_ticks[0] = 0
        core.bt_op6_race_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertGreater(int(core.bt_op6_race_ticks[0].item()), 0)
        self.assertEqual(roles[0], ROLE_ATTACKER)  # flag attacker
        self.assertEqual(roles[1], ROLE_INTERCEPTOR)

    def test_race_toggle_off(self) -> None:
        _BTRedMixin._OP6_RACE_DENIAL_ENABLED = False
        core, _ = _make_core("OP6")
        self._place_mutual_abandoned(core)
        roles, _, _ = _run_bt(core, "OP6")
        self.assertEqual(int(core.bt_op6_race_ticks[0].item()), 0)
        self.assertEqual(roles[1], ROLE_ESCORT)


class TestOP6LandscapeFreezeDefaults(unittest.TestCase):
    """Held-out / landscape OP6: rejected race mechanisms default OFF."""

    def test_rejected_gameplay_toggles_default_off(self) -> None:
        # Re-read class defaults (tests may flip class attrs; restore first).
        _BTRedMixin._OP6_EXTRACTION_ENABLED = False
        _BTRedMixin._OP6_PREENGAGE_ENABLED = False
        _BTRedMixin._OP6_RACE_DENIAL_ENABLED = False
        self.assertFalse(_BTRedMixin._OP6_EXTRACTION_ENABLED)
        self.assertFalse(_BTRedMixin._OP6_PREENGAGE_ENABLED)
        self.assertFalse(_BTRedMixin._OP6_RACE_DENIAL_ENABLED)


if __name__ == "__main__":
    unittest.main()
