"""Deterministic scenario tests for OP5..OP12 behavior-tree opponents.

Tests cover the seven required tactical scenarios:
  1. Teammate carrying enemy flag while being pursued → ESCORT role fires.
  2. Enemy carrying own flag, intercept feasible → INTERCEPTOR role fires.
  3. Enemy carrying own flag, intercept infeasible → COUNTER role fires (OP12)
     or INTERCEPTOR is NOT assigned (OP11 non-trailing).
  4. Direct return route blocked by enemy → carrier evasion deviates from straight line.
  5. Two red agents able to coordinate against one blue enemy → 2V1_WING role fires.
  6. Defender decides: guard / intercept / counter depending on situation.
  7. BT fallback when no preferred action valid → returns to ATTACKER default.

Additional tests:
  - BT telemetry counters increment correctly.
  - OP11 and OP12 params accepted by sample_batched_opponent_params.
  - Opponent styles create observably different team behaviour (ESCORT vs COUNTER).
  - TacticalContext BT telemetry is populated.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_red import (
    ROLE_ATTACKER,
    ROLE_COUNTER,
    ROLE_DEFENDER,
    ROLE_ESCORT,
    ROLE_FLAG_RETR,
    ROLE_INTERCEPTOR,
    ROLE_2V1_WING,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture helper
# ──────────────────────────────────────────────────────────────────────────────

def _make_core(opponent: str, *, seed: int = 0, max_steps: int = 400,
               step: int = 0, red_score: int = 0, blue_score: int = 0):
    """Return a 2v2 core with n_envs=1 configured for the given BT opponent."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_layout="map_b", max_decision_steps=max_steps,
        aquaticus_profile=True, rules_profile="OURS",
        device="cpu", seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    core = env.core

    core.blue_score[0]     = blue_score
    core.red_score[0]      = red_score
    core.step_count[0]     = step
    core.sim_step_count[0] = step

    core._opponent_key[0]  = opponent
    # Ensure no role flip and no coordinated attack so tests are deterministic.
    core.red_script_role_flip[0]   = False
    core.red_coordinated_attack[0] = False
    core.red_alive[0]  = True
    core.blue_alive[0] = True
    # Disable deception / role-switching from scripted param noise.
    core.red_deception_prob[0]  = 0.0
    core.red_role_switch_prob[0] = 0.0

    return core, env


def _run_bt(core, opponent: str = "OP11") -> tuple:
    """Run one BT step and return (roles, target_x, target_y)."""
    core._opponent_key[0] = opponent
    tx, ty = core._get_bt_targets()
    roles = core.bt_red_role[0].tolist()
    return roles, tx[0].tolist(), ty[0].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 1: Own carrier pursued → ESCORT fires
# ──────────────────────────────────────────────────────────────────────────────

class TestEscortWhenCarrierPursued(unittest.TestCase):
    """When red has the enemy flag and a blue pursuer is close, one red agent
    should be assigned ESCORT to interpose between carrier and threat."""

    def test_escort_role_fires_with_carrier(self) -> None:
        core, _ = _make_core("OP11")
        # Red agent 0 carries the enemy flag.
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        # Blue pursuer is close to the carrier.
        core.blue_x[0, 0] = 13.0
        core.blue_y[0, 0] = 10.0
        core.blue_x[0, 1] = 5.0
        core.blue_y[0, 1] = 5.0

        roles, tx, ty = _run_bt(core, "OP11")
        # At least one non-carrier agent should be assigned ESCORT.
        self.assertIn(ROLE_ESCORT, roles,
                      f"Expected ESCORT when carrier exists, got roles={roles}")

    def test_escort_telemetry_increments(self) -> None:
        core, _ = _make_core("OP11")
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core.blue_x[0, 0] = 13.0
        core.blue_y[0, 0] = 10.0
        prev_escorts = int(core.bt_tel_escort_attempts[0].item())
        _run_bt(core, "OP11")
        new_escorts = int(core.bt_tel_escort_attempts[0].item())
        self.assertGreater(new_escorts, prev_escorts,
                           "escort_attempts counter should increment when ESCORT fires")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 2: Enemy carrier exists, intercept feasible → INTERCEPTOR fires
# ──────────────────────────────────────────────────────────────────────────────

class TestInterceptorFeasible(unittest.TestCase):
    """When blue has the red flag and at least one red agent can reach the
    midpoint of the carrier's path before the carrier, INTERCEPTOR fires."""

    def test_interceptor_role_fires_when_feasible(self) -> None:
        core, _ = _make_core("OP11")
        # Blue agent 0 carries the red flag and is mid-field.
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 10.0
        core.blue_y[0, 0] = 10.0
        core.blue_flag_home[0, 0] = 0.0
        core.blue_flag_home[0, 1] = 10.0
        # Red agents are close — intercept is clearly feasible.
        core.red_x[0, 0] = 8.0
        core.red_y[0, 0] = 9.0
        core.red_x[0, 1] = 8.0
        core.red_y[0, 1] = 11.0

        roles, tx, ty = _run_bt(core, "OP11")
        self.assertIn(ROLE_INTERCEPTOR, roles,
                      f"Expected INTERCEPTOR when intercept is feasible, got roles={roles}")

    def test_intercept_telemetry_increments(self) -> None:
        core, _ = _make_core("OP11")
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 10.0
        core.blue_y[0, 0] = 10.0
        core.red_x[0, 0] = 8.0
        core.red_y[0, 0] = 9.0
        core.red_x[0, 1] = 8.0
        core.red_y[0, 1] = 11.0
        prev = int(core.bt_tel_intercept_attempts[0].item())
        _run_bt(core, "OP11")
        new = int(core.bt_tel_intercept_attempts[0].item())
        self.assertGreater(new, prev,
                           "intercept_attempts counter should increment when INTERCEPTOR fires")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 3: Enemy carrier, intercept infeasible → COUNTER fires (OP12)
# ──────────────────────────────────────────────────────────────────────────────

class TestCounterCaptureWhenInterceptInfeasible(unittest.TestCase):
    """When the blue carrier is too far ahead to catch (infeasible intercept),
    OP12 should assign COUNTER (go for enemy flag) rather than chasing."""

    def _setup_infeasible(self, opponent: str, trailing: bool = False):
        core, _ = _make_core(opponent,
                              red_score=0 if trailing else 1,
                              blue_score=1 if trailing else 0)
        # Blue carrier is almost home — red cannot intercept.
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 1.0    # very close to blue home (left side)
        core.blue_y[0, 0] = 10.0
        core.blue_flag_home[0, 0] = 0.0
        core.blue_flag_home[0, 1] = 10.0
        # Red agents are far away on their side.
        core.red_x[0, 0] = 18.0
        core.red_y[0, 0] = 5.0
        core.red_x[0, 1] = 18.0
        core.red_y[0, 1] = 15.0
        return core

    def test_op12_counter_fires_when_infeasible(self) -> None:
        core = self._setup_infeasible("OP12")
        roles, _, _ = _run_bt(core, "OP12")
        self.assertIn(ROLE_COUNTER, roles,
                      f"OP12 should assign COUNTER when intercept infeasible, got roles={roles}")

    def test_op12_counter_telemetry_increments(self) -> None:
        core = self._setup_infeasible("OP12")
        prev = int(core.bt_tel_counter_captures[0].item())
        _run_bt(core, "OP12")
        new = int(core.bt_tel_counter_captures[0].item())
        self.assertGreater(new, prev,
                           "counter_captures counter should increment when COUNTER fires")

    def test_op11_trailing_counter_fires_when_infeasible(self) -> None:
        """OP11 also counter-captures when trailing and intercept is infeasible."""
        core = self._setup_infeasible("OP11", trailing=True)
        roles, _, _ = _run_bt(core, "OP11")
        self.assertIn(ROLE_COUNTER, roles,
                      f"OP11 trailing should COUNTER when infeasible, got roles={roles}")

    def test_op11_leading_no_counter_when_infeasible(self) -> None:
        """OP11 leading should NOT counter-capture when infeasible; some other role fires."""
        core = self._setup_infeasible("OP11", trailing=False)
        roles, _, _ = _run_bt(core, "OP11")
        # When leading and intercept is infeasible, OP11 does not flip to counter.
        # Agents may become ATTACKER or DEFENDER but not COUNTER.
        self.assertNotIn(ROLE_COUNTER, roles,
                         f"OP11 leading should NOT counter when infeasible, got roles={roles}")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 4: Direct return route blocked → carrier evasion deviates
# ──────────────────────────────────────────────────────────────────────────────

class TestCarrierEvasionDeviates(unittest.TestCase):
    """A red carrier with a blue agent directly on the straight-home path should
    receive a target that deviates from the straight line to own flag home."""

    def test_carrier_target_deviates_when_enemy_blocks(self) -> None:
        core, _ = _make_core("OP11")
        # Red agent 0 carries flag.
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 12.0
        core.red_y[0, 0] = 10.0
        # Red flag home is at the right side (~column 19).
        home_x = float(core.red_flag_home[0, 0].item())
        home_y = float(core.red_flag_home[0, 1].item())
        # Blue agent sits directly on the straight-line path.
        mid_x = (12.0 + home_x) * 0.5
        core.blue_x[0, 0] = mid_x
        core.blue_y[0, 0] = 10.0   # same y as carrier and home
        core.blue_x[0, 1] = 3.0
        core.blue_y[0, 1] = 3.0    # far from path

        roles, tx, ty = _run_bt(core, "OP11")
        # Carrier target (agent 0) should differ from straight home.
        straight_dist = abs(ty[0] - home_y)
        self.assertGreater(straight_dist, 0.5,
                           f"Carrier target y should deviate from home_y={home_y:.1f} "
                           f"when enemy blocks; got ty={ty[0]:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 5: Two red agents near same blue enemy → 2V1_WING role fires
# ──────────────────────────────────────────────────────────────────────────────

class TestTwoV1Wing(unittest.TestCase):
    """When two red agents are both within close range of the same blue agent,
    one should be promoted to 2V1_WING for a two-pronged attack orbit."""

    def test_2v1_wing_fires_when_two_red_close(self) -> None:
        core, _ = _make_core("OP11")
        # Both red agents close to blue agent 0.
        core.red_x[0, 0] = 7.0
        core.red_y[0, 0] = 10.0
        core.red_x[0, 1] = 8.0
        core.red_y[0, 1] = 9.0
        core.blue_x[0, 0] = 7.5
        core.blue_y[0, 0] = 9.5
        core.blue_x[0, 1] = 1.0   # far away
        core.blue_y[0, 1] = 1.0
        # No carrier so ESCORT/INTERCEPTOR don't steal the roles first.
        core.red_carrying[0]  = False
        core.blue_carrying[0] = False

        roles, _, _ = _run_bt(core, "OP11")
        self.assertIn(ROLE_2V1_WING, roles,
                      f"Expected ROLE_2V1_WING when two reds near one blue, got roles={roles}")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 6: Defender decides guard vs intercept vs counter
# ──────────────────────────────────────────────────────────────────────────────

class TestDefenderDecisionTree(unittest.TestCase):
    """The defender's target shifts based on game state:
    - No intruder / no carrier: moves to interception zone (ahead of own flag).
    - Intruder on own half: chases intruder.
    - Enemy carrier: chases enemy carrier.
    """

    def _defender_target(self, *, intruder_on_own: bool, enemy_carry: bool, trailing: bool = False):
        core, _ = _make_core(
            "OP11",
            red_score=0 if trailing else 1,
            blue_score=1 if trailing else 0,
        )
        core.red_carrying[0]  = False
        core.blue_carrying[0] = False
        core.red_x[0, 0] = 18.0   # red agent 0 = will be chosen as defender (closest to home)
        core.red_y[0, 0] = 10.0
        core.red_x[0, 1] = 18.0
        core.red_y[0, 1] = 10.0
        core.blue_x[0, 0] = 15.0   # blue agent on own half
        core.blue_y[0, 0] = 10.0
        core.blue_x[0, 1] = 5.0
        core.blue_y[0, 1] = 5.0
        core.blue_alive[0, 0] = intruder_on_own
        if enemy_carry:
            core.blue_carrying[0, 0] = True
        # Force blue agent 0 onto red half.
        if intruder_on_own:
            core.blue_x[0, 0] = 15.0   # > midline (10.0) → on red half
        else:
            core.blue_x[0, 0] = 3.0    # on blue half
        roles, tx, ty = _run_bt(core, "OP11")
        return roles, tx, ty, core

    def test_defender_chases_intruder(self) -> None:
        roles, tx, ty, core = self._defender_target(intruder_on_own=True, enemy_carry=False)
        self.assertIn(ROLE_DEFENDER, roles,
                      f"Expected DEFENDER with intruder on own half, got roles={roles}")
        # DEFENDER assigned agent should target closer to the intruder side.
        def_idx = roles.index(ROLE_DEFENDER)
        intruder_x = 15.0
        self.assertGreater(tx[def_idx], float(core.red_flag_home[0, 0].item()) * 0.5,
                           "Defender target should be closer to intruder than to own base")

    def test_defender_chases_carrier(self) -> None:
        roles, tx, ty, _ = self._defender_target(intruder_on_own=True, enemy_carry=True)
        self.assertIn(ROLE_DEFENDER, roles,
                      f"Expected DEFENDER when enemy carrier exists, got roles={roles}")
        # The target x should be near the carrier position (15.0).
        def_idx = roles.index(ROLE_DEFENDER)
        self.assertAlmostEqual(tx[def_idx], 15.0, delta=1.5,
                               msg=f"Defender should target carrier at x=15, got {tx[def_idx]:.2f}")

    def test_defender_zones_when_clear(self) -> None:
        roles, tx, ty, core = self._defender_target(intruder_on_own=False, enemy_carry=False)
        # With no intruder and no carrier, no one necessarily gets DEFENDER;
        # all may fall to ATTACKER.  If DEFENDER is assigned, verify it's in the zone.
        if ROLE_DEFENDER in roles:
            def_idx = roles.index(ROLE_DEFENDER)
            # Target should be between own flag home and midline (zone patrol).
            home_x = float(core.red_flag_home[0, 0].item())
            midline = float(core.cols) * 0.5
            self.assertGreater(tx[def_idx], midline * 0.5,
                               "Defender zone patrol target should be ahead of home base")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 7: BT fallback when no preferred action valid → ATTACKER default
# ──────────────────────────────────────────────────────────────────────────────

class TestBTFallbackToAttacker(unittest.TestCase):
    """When no special condition is true (no carrier, no intruder, no enemy carrier,
    own flag at home), all agents should default to ATTACKER role."""

    def test_all_attackers_in_neutral_state(self) -> None:
        core, _ = _make_core("OP11")
        # Clean neutral state: flags at home, no carriers, no threats.
        core.red_carrying[0]  = False
        core.blue_carrying[0] = False
        core.blue_alive[0]    = False   # no enemies visible
        # Reset BT role lock so reassignment is allowed.
        core.bt_role_lock_ticks[0] = 0

        roles, _, _ = _run_bt(core, "OP11")
        for j, r in enumerate(roles):
            self.assertEqual(r, ROLE_ATTACKER,
                             f"Agent {j} should default to ATTACKER in neutral state, got role={r}")


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry: objective change counter
# ──────────────────────────────────────────────────────────────────────────────

class TestObjectiveChangeTelemetry(unittest.TestCase):
    """Objective change counter should increment when a role changes between steps."""

    def test_objective_changes_count(self) -> None:
        core, _ = _make_core("OP11")
        # Force agents into ESCORT initially by having a carrier.
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core.bt_role_lock_ticks[0] = 0
        _run_bt(core, "OP11")  # first step: roles set

        # Now remove carrier — roles should shift away from ESCORT.
        core.red_carrying[0] = False
        core.bt_role_lock_ticks[0] = 0   # force change allowed
        prev = int(core.bt_tel_objective_changes[0].item())
        _run_bt(core, "OP11")
        new = int(core.bt_tel_objective_changes[0].item())
        self.assertGreater(new, prev,
                           "objective_changes should increment when role changes")


# ──────────────────────────────────────────────────────────────────────────────
# Hysteresis: role lock prevents thrashing
# ──────────────────────────────────────────────────────────────────────────────

class TestRoleHysteresis(unittest.TestCase):
    """When role_lock_ticks > 0, role should not change even if conditions change."""

    def test_locked_role_persists(self) -> None:
        core, _ = _make_core("OP11")
        # Pre-assign ESCORT and lock.
        core.bt_red_role[0, 1] = ROLE_ESCORT
        core.bt_role_lock_ticks[0, 1] = 20   # locked for 20 ticks
        core.red_carrying[0]  = False   # carrier gone
        core.blue_carrying[0] = False

        _run_bt(core, "OP11")
        # Agent 1 should still be ESCORT due to lock.
        self.assertEqual(int(core.bt_red_role[0, 1].item()), ROLE_ESCORT,
                         "Locked role should persist when lock ticks remain")


# ──────────────────────────────────────────────────────────────────────────────
# Flag retrieval role
# ──────────────────────────────────────────────────────────────────────────────

class TestFlagRetrieval(unittest.TestCase):
    """When own flag is not at home, FLAG_RETR role should be assigned."""

    def test_flag_retriever_fires(self) -> None:
        core, _ = _make_core("OP11")
        # Move red flag away from home.
        core.red_flag_pos[0, 0] = 10.0   # at midfield
        core.red_flag_pos[0, 1] = 10.0
        core.bt_role_lock_ticks[0] = 0

        roles, _, _ = _run_bt(core, "OP11")
        self.assertIn(ROLE_FLAG_RETR, roles,
                      f"Expected FLAG_RETR when own flag not at home, got roles={roles}")


# ──────────────────────────────────────────────────────────────────────────────
# OP11 vs OP12 observable behavioural difference
# ──────────────────────────────────────────────────────────────────────────────

class TestOP11vsOP12BehaviorDifference(unittest.TestCase):
    """OP12 should assign COUNTER more often than OP11 when intercept is infeasible
    and red is leading (OP11 does not counter when leading; OP12 always does)."""

    def _infeasible_carrier_state(self, opponent: str, red_score: int = 1):
        core, _ = _make_core(opponent, red_score=red_score, blue_score=0)
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 1.0     # almost home
        core.blue_y[0, 0] = 10.0
        core.blue_flag_home[0, 0] = 0.0
        core.blue_flag_home[0, 1] = 10.0
        core.red_x[0, 0] = 18.0
        core.red_y[0, 0] = 5.0
        core.red_x[0, 1] = 18.0
        core.red_y[0, 1] = 15.0
        core.bt_role_lock_ticks[0] = 0
        return core

    def test_op12_counters_when_leading(self) -> None:
        core = self._infeasible_carrier_state("OP12", red_score=1)
        roles, _, _ = _run_bt(core, "OP12")
        self.assertIn(ROLE_COUNTER, roles,
                      "OP12 should COUNTER even when leading, got roles=" + str(roles))

    def test_op11_does_not_counter_when_leading(self) -> None:
        core = self._infeasible_carrier_state("OP11", red_score=1)
        roles, _, _ = _run_bt(core, "OP11")
        self.assertNotIn(ROLE_COUNTER, roles,
                         "OP11 should NOT COUNTER when leading, got roles=" + str(roles))


# ──────────────────────────────────────────────────────────────────────────────
# Parameter tests: OP11 / OP12 accepted by sample_batched_opponent_params
# ──────────────────────────────────────────────────────────────────────────────

class TestOP11OP12Params(unittest.TestCase):
    def _sample(self, key: str, n: int = 64):
        from opponent_params import sample_batched_opponent_params  # type: ignore[import]
        return sample_batched_opponent_params(
            kind="SCRIPTED", key=key, phase=key, n_agents=2,
            batch_size=n, device="cpu",
        )

    def test_op11_canonical_accepted(self) -> None:
        p = self._sample("OP11")
        self.assertIn("attacker_style", p)
        self.assertIn("coordinated_attack", p)

    def test_op11_alias_accepted(self) -> None:
        p = self._sample("OP11_BT_BALANCED")
        self.assertIn("role_switch_prob", p)

    def test_op12_canonical_accepted(self) -> None:
        p = self._sample("OP12")
        self.assertIn("defender_style", p)

    def test_op12_alias_accepted(self) -> None:
        p = self._sample("OP12_COUNTER")
        self.assertIn("speed_mult", p)

    def test_op11_attacker_style_medium(self) -> None:
        """OP11 is a balanced BT opponent with medium attacker style."""
        p = self._sample("OP11", n=256)
        self.assertTrue(
            (p["attacker_style"] == 1).all().item(),
            "OP11 attacker_style should be 1 (medium)",
        )

    def test_op12_attacker_style_medium(self) -> None:
        """OP12 should have medium attacker style (aggressive counter capture)."""
        p = self._sample("OP12", n=256)
        self.assertTrue(
            (p["attacker_style"] == 1).all().item(),
            "OP12 attacker_style should be 1 (medium)",
        )


# ──────────────────────────────────────────────────────────────────────────────
# TacticalContext BT telemetry integration
# ──────────────────────────────────────────────────────────────────────────────

class TestTacticalContextBTTelemetry(unittest.TestCase):
    """extract_tactical_context should include bt field with telemetry data."""

    def test_bt_telemetry_in_context(self) -> None:
        from gpu_env._core._tactical_context import extract_tactical_context  # type: ignore[import]
        core, _ = _make_core("OP11")
        core.red_carrying[0, 0] = True
        _run_bt(core, "OP11")
        ctx = extract_tactical_context(core, env_idx=0)
        self.assertIsNotNone(ctx.bt, "TacticalContext.bt should not be None for OP11")
        assert ctx.bt is not None  # for type narrowing
        self.assertGreaterEqual(ctx.bt.escort_attempts, 0)
        self.assertIsInstance(ctx.bt.red_roles, list)
        self.assertEqual(len(ctx.bt.red_roles), core.Nr)

    def test_bt_active_branches_recorded(self) -> None:
        from gpu_env._core._tactical_context import extract_tactical_context  # type: ignore[import]
        core, _ = _make_core("OP12")
        # Trigger COUNTER.
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 1.0
        core.blue_y[0, 0] = 10.0
        core.blue_flag_home[0, 0] = 0.0
        core.blue_flag_home[0, 1] = 10.0
        core.red_x[0, 0] = 18.0
        core.red_y[0, 0] = 5.0
        core.red_x[0, 1] = 18.0
        core.red_y[0, 1] = 15.0
        _run_bt(core, "OP12")
        ctx = extract_tactical_context(core, env_idx=0)
        assert ctx.bt is not None
        self.assertIsInstance(ctx.bt.active_branches, list)
        self.assertEqual(len(ctx.bt.active_branches), core.Nr)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: _assign_scripted_targets_by_role dispatches to BT for OP11
# ──────────────────────────────────────────────────────────────────────────────

class TestScriptedDispatchToBT(unittest.TestCase):
    """_assign_scripted_targets_by_role should route OP5..OP12 through BT."""

    def test_op5_dispatch_populates_debug_targets(self) -> None:
        core, _ = _make_core("OP5")
        core._assign_scripted_targets_by_role("red")
        self.assertTrue(hasattr(core, "_debug_red_target_x"))
        self.assertEqual(tuple(core._debug_red_target_x.shape), (1, 2))

    def test_dispatch_populates_debug_targets(self) -> None:
        core, _ = _make_core("OP11")
        core._assign_scripted_targets_by_role("red")
        self.assertTrue(hasattr(core, "_debug_red_target_x"),
                        "_debug_red_target_x must be set after scripted dispatch")
        self.assertEqual(tuple(core._debug_red_target_x.shape), (1, 2))

    def test_targets_within_field_bounds(self) -> None:
        core, _ = _make_core("OP12")
        core._assign_scripted_targets_by_role("red")
        tx = core._debug_red_target_x
        ty = core._debug_red_target_y
        max_x = float(core.cols - 1)
        max_y = float(core.rows - 1)
        self.assertTrue((tx >= 0.0).all() and (tx <= max_x).all(),
                        f"All target_x should be in [0, {max_x}], got {tx}")
        self.assertTrue((ty >= 0.0).all() and (ty <= max_y).all(),
                        f"All target_y should be in [0, {max_y}], got {ty}")


if __name__ == "__main__":
    unittest.main()
