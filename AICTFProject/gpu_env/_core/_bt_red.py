"""
Behavior-Tree scripted red opponent for BatchedCTFCore.

Implements OP5 through OP12 scripted opponents using a team-blackboard pattern
and per-opponent tactical profiles (``_bt_profiles.py``).  All logic is
vectorized over the batch dimension (B) so it runs inside the normal GPU step
loop with no per-env Python loops.

Architecture
------------
1. ``_build_team_blackboard``  — per-step snapshot of shared team knowledge:
   teammate positions, flag states, enemy positions, carrier indices, and
   pre-computed interception feasibility.

2. ``_bt_assign_roles``        — dynamic role selection (one role per agent)
   written as utility scores so agents avoid clustering on identical objectives.

3. ``_bt_route_target``        — target computation per role, incorporating
   route scoring (distance, threat repulsion, teammate support, alternate lanes).

4. ``_bt_update_telemetry``    — writes per-env decision counters/branch names
   into ``bt_tel_*`` scratch tensors for diagnostics.

5. ``_get_bt_targets``         — top-level entry point called by the dispatch
   in ``_scripted_red.py`` when ``opponent_key`` is OP5 or higher.

Profiles
--------
Per-opponent tactical tuning lives in ``_bt_profiles.py`` (levels 5..12).
The BT engine is shared; profiles gate roles and route parameters.

Roles
-----
ROLE_ATTACKER   — go for enemy flag; use tangent evasion near defenders.
ROLE_DEFENDER   — guard own flag zone; intercept nearest intruder or carrier.
ROLE_ESCORT     — interpose between own carrier and nearest threat.
ROLE_INTERCEPTOR— cut off enemy carrier's path home (blocking fraction logic).
ROLE_FLAG_RETR  — retrieve own flag (when own flag is not at home).
ROLE_COUNTER    — ignore carrier chase; go capture enemy flag instead (counter-capture).
ROLE_2V1_WING   — orbit at 45° off a teammate who is tagging an enemy (2v1 flank).

Hysteresis
----------
``bt_role_lock_ticks`` tracks remaining commitment ticks per agent.
A role change is only allowed when ``bt_role_lock_ticks <= 0`` AND the
current role is no longer valid (e.g., own carrier no longer exists when
assigned ESCORT).

Fair play
---------
Opponents read only information available through the normal environment
observation tensors (positions, carrying flags, alive flags, flag positions,
scores, step count).  No hidden omniscient state beyond what ``_StateMixin``
exposes to both sides.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from macro_actions import MacroAction

from ._bt_profiles import BT_OPPONENT_KEYS, build_profile_tensors, is_bt_opponent
from ._bt_adaptive import _BTAdaptiveMixin


# ──────────────────────────────────────────────────────────────────────────────
# Role integer constants
# ──────────────────────────────────────────────────────────────────────────────
ROLE_ATTACKER   = 0
ROLE_DEFENDER   = 1
ROLE_ESCORT     = 2
ROLE_INTERCEPTOR = 3
ROLE_FLAG_RETR  = 4
ROLE_COUNTER    = 5
ROLE_2V1_WING   = 6
N_ROLES         = 7


class _BTRedMixin(_BTAdaptiveMixin):
    """
    Behavior-tree NPC brain for OP5..OP12 scripted opponents.

    Intended to be mixed into BatchedCTFCore alongside _ScriptedRedMixin.
    Relies on state tensors allocated by _ScratchStateMixin and augmented by
    ``_alloc_bt_telemetry``.
    """

    # OP6 failed-assault recovery window (sim steps after a legal failure).
    # Long enough for TURTLE's counter agent to reach red's flag; short enough
    # that empty-home styles still lose the dual-assault race before recovery
    # becomes the dominant tempo.
    _OP6_RECOVERY_DURATION = 36
    # OP6 post-pickup extraction: lock carrier+screener while returning.
    _OP6_EXTRACT_LOCK = 48
    # Gameplay defaults OFF — rejected extraction/race branch (dev29–dev36).
    # Landscape held-out freezes dual-assault + recovery only; keep buffers
    # and toggles for instrumentation / OFF-ON ablations.
    _OP6_EXTRACTION_ENABLED = False
    # Pre-pickup screener (dev35): rejected as race mechanism; instrumentation kept.
    _OP6_PREENGAGE_ENABLED = False
    # grab_r ≈ 1.2; start screener while attacker is closing, not at the grab
    # (dev35: lead_time was 0–1 at radius 2.5 — too late for C3).
    _OP6_PREPICKUP_RADIUS = 6.5
    _OP6_PREENGAGE_LOCK = 24
    # Mutual-carry race denial (dev36): rejected — interrupts RUSH but does not
    # flip first-score; not required for TURTLE payoff-tooth acceptance.
    _OP6_RACE_DENIAL_ENABLED = False
    _OP6_RACE_LOCK = 48
    # OP7 SPLIT host: separated-threat overcommit (first breached corridor).
    # Legal geometry only — never blue style ID. Thresholds mirror adaptive
    # split-pressure math so concentrated RUSH/ESCORT do not arm the latch.
    _OP7_SPLIT_LEVER_ENABLED = True
    _OP7_SPLIT_PRESSURE_TICKS = 2  # round-2: was 4; SPLIT armed only 4/8
    _OP7_SPLIT_RESPONSE_DURATION = 56  # round-2: was 40; longer overcommit window
    _OP7_SPLIT_LATERAL_FRAC = 0.45  # round-2: was 0.55
    _OP7_SPLIT_TEAMMATE_DIST = 10.0  # round-2: was 12.0
    _OP7_SPLIT_OFFENSIVE_BUFFER = 5.0  # round-2: was 3.0
    _OP7_SPLIT_COMMIT_GRACE = 6  # steps of lost dual-offense before clear
    # OP7 final lever: compact same-corridor containment (do not retune SPLIT latch).
    # Compact pressure → one deep defender + return-route interceptor.
    # HARD STOP (2026-07-28): micro-gates FAIL — compact fired on SPLIT/TURTLE
    # 8/8 and raised RUSH/ESCORT blue margins. Lever disabled; OP7 closed as
    # SPLIT host. SPLIT latch above stays enabled for near-miss/robustness.
    _OP7_COMPACT_LEVER_ENABLED = False
    _OP7_COMPACT_PRESSURE_TICKS = 3
    _OP7_COMPACT_RESPONSE_DURATION = 56
    _OP7_COMPACT_TEAMMATE_DIST = 4.0
    _OP7_COMPACT_LATERAL_MAX = 2.5
    _OP7_COMPACT_OFFENSIVE_BUFFER = 5.0
    _OP7_COMPACT_COMMIT_GRACE = 6

    def _bt_opponent_mask(self) -> torch.Tensor:
        """True for env rows whose opponent uses the BT tactical brain."""
        return torch.as_tensor(
            [is_bt_opponent(str(k)) for k in self._opponent_key],
            device=self.device,
            dtype=torch.bool,
        )

    # ──────────────────────────────────────────────────────────────────────
    # State allocation (called from _ScratchStateMixin._alloc_runtime_buffers)
    # ──────────────────────────────────────────────────────────────────────
    def _alloc_bt_telemetry(self, B: int, N: int, dev: torch.device) -> None:
        """Allocate per-env BT decision telemetry buffers.

        N is max(Nb, Nr) so the same allocation works for both sides.
        Called once during BatchedCTFCore.__init__ via _alloc_state.
        """
        f32 = torch.float32
        i32 = torch.int32
        # Active role per red agent: integer in [0, N_ROLES).
        self.bt_red_role       = torch.full((B, N), ROLE_ATTACKER, dtype=i32, device=dev)
        # Remaining ticks before role is allowed to change (hysteresis).
        self.bt_role_lock_ticks = torch.zeros((B, N), dtype=i32, device=dev)
        # Cumulative event counters (lifetime within episode, reset each episode).
        self.bt_tel_escort_attempts    = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_tel_intercept_attempts = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_tel_counter_captures   = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_tel_objective_changes  = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_tel_successful_tags    = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_tel_stuck_steps        = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_tel_mine_attempts        = torch.zeros((B,), dtype=i32, device=dev)
        # Per-agent deliberate mine placement (BT route layer).
        self.bt_mine_target_x = torch.zeros((B, N), dtype=f32, device=dev)
        self.bt_mine_target_y = torch.zeros((B, N), dtype=f32, device=dev)
        self.bt_want_mine = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.bt_mine_lock_ticks = torch.zeros((B, N), dtype=i32, device=dev)
        # Per-agent last position (for stuck detection, reset each episode).
        self.bt_last_x = torch.zeros((B, N), dtype=f32, device=dev)
        self.bt_last_y = torch.zeros((B, N), dtype=f32, device=dev)
        # Branch label cache: integer code for which BT branch fired last step.
        # Branch codes match role constants above.
        self.bt_active_branch = torch.full((B, N), ROLE_ATTACKER, dtype=i32, device=dev)
        # Searcher-only overlay. None keeps canonical OP6-OP12 profiles.
        self._bt_profile_override = None
        self._sds_opening_hold_steps = 0
        # OP6 failed-assault recovery (Contract B defend-then-counter trap).
        # Per-agent countdown after a legal failed incursion; OP6-only.
        self.bt_op6_recovery_ticks = torch.zeros((B, N), dtype=i32, device=dev)
        self.bt_op6_prev_red_tagged = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.bt_op6_prev_red_carrying = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.bt_op6_failed_incursions = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_recovery_activations = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_recovery_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        # OP6 extraction support countdown (env-level while returning).
        self.bt_op6_extract_ticks = torch.zeros((B,), dtype=i32, device=dev)
        # Locked dual-threat assignment for OP6 extract (carrier vs screener).
        # -1 = unset. Set once when extract arms; cleared when extract ends.
        self.bt_op6_extract_carrier_threat = torch.full(
            (B,), -1, dtype=i32, device=dev
        )
        self.bt_op6_extract_screener_threat = torch.full(
            (B,), -1, dtype=i32, device=dev
        )
        # Pre-pickup screener reserve (sticky until pickup / expire / cancel).
        self.bt_op6_preengage_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_preengage_activations = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_preengage_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        # Mutual-carry race denial (env-level while both sides carry).
        self.bt_op6_race_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_race_activations = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_race_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op6_race_target_idx = torch.full((B,), -1, dtype=i32, device=dev)
        # OP7 separated-threat latch (Contract: first-lane overcommit).
        self.bt_op7_split_pressure_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_split_first_trigger_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_op7_split_response_expiry_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_op7_split_primary_blue_idx = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_op7_split_corridor_y = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_op7_split_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_split_response_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_split_activations = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_split_max_lateral_sep = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_op7_split_max_teammate_dist = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_op7_split_lost_commit_ticks = torch.zeros((B,), dtype=i32, device=dev)
        # OP7 compact same-corridor containment (final RUSH/ESCORT lever).
        self.bt_op7_compact_pressure_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_compact_first_trigger_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_op7_compact_response_expiry_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_op7_compact_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_compact_response_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_compact_activations = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_compact_lost_commit_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_op7_compact_min_teammate_dist = torch.full((B,), 1.0e6, dtype=f32, device=dev)
        self.bt_op7_compact_max_lateral_sep = torch.zeros((B,), dtype=f32, device=dev)
        self._alloc_adaptive_memory(B, dev)

    def _reset_bt_telemetry(self, env_mask: torch.Tensor) -> None:
        """Zero all BT telemetry for environments in env_mask (called on episode reset)."""
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        N = self.bt_red_role.shape[1]
        self.bt_red_role[idx]          = ROLE_ATTACKER
        self.bt_role_lock_ticks[idx]   = 0
        self.bt_tel_escort_attempts[idx]    = 0
        self.bt_tel_intercept_attempts[idx] = 0
        self.bt_tel_counter_captures[idx]   = 0
        self.bt_tel_objective_changes[idx]  = 0
        self.bt_tel_successful_tags[idx]    = 0
        self.bt_tel_stuck_steps[idx]        = 0
        self.bt_tel_mine_attempts[idx]        = 0
        self.bt_mine_target_x[idx]            = 0.0
        self.bt_mine_target_y[idx]            = 0.0
        self.bt_want_mine[idx]                = False
        self.bt_mine_lock_ticks[idx]          = 0
        self.bt_last_x[idx]            = 0.0
        self.bt_last_y[idx]            = 0.0
        self.bt_active_branch[idx]     = ROLE_ATTACKER
        self.bt_op6_recovery_ticks[idx] = 0
        self.bt_op6_prev_red_tagged[idx] = False
        self.bt_op6_prev_red_carrying[idx] = False
        self.bt_op6_failed_incursions[idx] = 0
        self.bt_op6_recovery_activations[idx] = 0
        self.bt_op6_recovery_active_steps[idx] = 0
        self.bt_op6_extract_ticks[idx] = 0
        self.bt_op6_extract_carrier_threat[idx] = -1
        self.bt_op6_extract_screener_threat[idx] = -1
        self.bt_op6_preengage_ticks[idx] = 0
        self.bt_op6_preengage_activations[idx] = 0
        self.bt_op6_preengage_active_steps[idx] = 0
        self.bt_op6_race_ticks[idx] = 0
        self.bt_op6_race_activations[idx] = 0
        self.bt_op6_race_active_steps[idx] = 0
        self.bt_op6_race_target_idx[idx] = -1
        self.bt_op7_split_pressure_ticks[idx] = 0
        self.bt_op7_split_first_trigger_step[idx] = -1
        self.bt_op7_split_response_expiry_step[idx] = -1
        self.bt_op7_split_primary_blue_idx[idx] = -1
        self.bt_op7_split_corridor_y[idx] = 0.0
        self.bt_op7_split_active_steps[idx] = 0
        self.bt_op7_split_response_active_steps[idx] = 0
        self.bt_op7_split_activations[idx] = 0
        self.bt_op7_split_max_lateral_sep[idx] = 0.0
        self.bt_op7_split_max_teammate_dist[idx] = 0.0
        self.bt_op7_split_lost_commit_ticks[idx] = 0
        self.bt_op7_compact_pressure_ticks[idx] = 0
        self.bt_op7_compact_first_trigger_step[idx] = -1
        self.bt_op7_compact_response_expiry_step[idx] = -1
        self.bt_op7_compact_active_steps[idx] = 0
        self.bt_op7_compact_response_active_steps[idx] = 0
        self.bt_op7_compact_activations[idx] = 0
        self.bt_op7_compact_lost_commit_ticks[idx] = 0
        self.bt_op7_compact_min_teammate_dist[idx] = 1.0e6
        self.bt_op7_compact_max_lateral_sep[idx] = 0.0
        self._reset_adaptive_memory(env_mask)

    # ──────────────────────────────────────────────────────────────────────
    # Team blackboard: shared per-step situational snapshot
    # ──────────────────────────────────────────────────────────────────────
    def _build_team_blackboard(self, prof: Dict[str, torch.Tensor]) -> dict:
        """
        Construct a scalar/tensor snapshot of team-level situation for red side.

        Returns a dict of tensors (all shape [B] unless noted) that the role
        selector and target router consume.  This is the single read of shared
        state; individual BT nodes do not re-read environment tensors.
        """
        B, device = self.B, self.device
        idx_env = torch.arange(B, device=device)
        midline = float(self.cols) * 0.5
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))

        # ── Flag states ──────────────────────────────────────────────────
        red_flag_home     = self.red_flag_home        # [B, 2]
        blue_flag_home    = self.blue_flag_home       # [B, 2]
        red_flag_pos      = self.red_flag_pos         # [B, 2]
        blue_flag_pos     = self.blue_flag_pos        # [B, 2]

        own_flag_at_home  = (
            torch.abs(red_flag_pos[:, 0] - red_flag_home[:, 0]) < 1.5
        ) & (
            torch.abs(red_flag_pos[:, 1] - red_flag_home[:, 1]) < 1.5
        )

        # ── Carrier indices (-1 sentinel = no carrier) ───────────────────
        red_carry_any  = (self.red_carrying & self.red_alive & (~self.red_tagged)).any(dim=1)
        blue_carry_any = (self.blue_carrying & self.blue_alive & (~self.blue_tagged)).any(dim=1)
        red_carrier_idx  = torch.where(
            red_carry_any,
            torch.argmax((self.red_carrying & (~self.red_tagged)).to(torch.int64), dim=1),
            torch.full((B,), -1, dtype=torch.int64, device=device),
        )
        blue_carrier_idx = torch.where(
            blue_carry_any,
            torch.argmax((self.blue_carrying & (~self.blue_tagged)).to(torch.int64), dim=1),
            torch.full((B,), -1, dtype=torch.int64, device=device),
        )

        # ── Enemy carrier position (safe: fallback to flag pos) ──────────
        blue_ci_clamped = blue_carrier_idx.clamp(min=0)
        ec_x = torch.where(blue_carry_any, self.blue_x[idx_env, blue_ci_clamped], blue_flag_pos[:, 0])
        ec_y = torch.where(blue_carry_any, self.blue_y[idx_env, blue_ci_clamped], blue_flag_pos[:, 1])

        # ── Interception feasibility ─────────────────────────────────────
        # Red agent distances to the enemy carrier's current position.
        # Shape [B, Nr]
        dxx = self.red_x - ec_x[:, None]
        dyy = self.red_y - ec_y[:, None]
        red_dist_to_ec = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)

        # Distance enemy carrier still must travel to reach home.
        ec_to_home_x = blue_flag_home[:, 0] - ec_x
        ec_to_home_y = blue_flag_home[:, 1] - ec_y
        ec_to_home_dist = torch.sqrt(ec_to_home_x ** 2 + ec_to_home_y ** 2 + 1e-8)

        # An intercept is feasible if any red agent can reach the midpoint of
        # the carrier's path home before the carrier does (using speed ratio 1:1
        # as a conservative estimate; carrier may evade).
        midpoint_x = ec_x + ec_to_home_x * 0.5
        midpoint_y = ec_y + ec_to_home_y * 0.5
        red_dist_to_midpoint = torch.sqrt(
            (self.red_x - midpoint_x[:, None]) ** 2
            + (self.red_y - midpoint_y[:, None]) ** 2 + 1e-8
        )
        ratio = prof["intercept_feasibility_ratio"][:, None]
        intercept_feasible_per_agent = (
            blue_carry_any[:, None]
            & self.red_alive
            & (~self.red_tagged)
            & (red_dist_to_midpoint < (ec_to_home_dist[:, None] * ratio))
        )
        # Team-level: at least one agent can intercept.
        intercept_feasible = intercept_feasible_per_agent.any(dim=1)

        # ── Enemy intruders on own half ──────────────────────────────────
        enemy_on_own = self.blue_alive & (self.blue_x > midline)
        any_intruder = enemy_on_own.any(dim=1)
        intruder_count = enemy_on_own.sum(dim=1)

        # ── Time / score context ─────────────────────────────────────────
        time_frac = (self.step_count.float() / max(1, self.max_steps)).clamp(0.0, 1.0)
        late_game = time_frac > 0.75
        trailing  = self.red_score < self.blue_score
        leading   = self.red_score > self.blue_score

        # ── Alive red agent count ────────────────────────────────────────
        alive_count = self.red_alive.sum(dim=1)

        return dict(
            idx_env=idx_env,
            midline=midline,
            max_x=max_x,
            max_y=max_y,
            red_flag_home=red_flag_home,
            blue_flag_home=blue_flag_home,
            red_flag_pos=red_flag_pos,
            blue_flag_pos=blue_flag_pos,
            own_flag_at_home=own_flag_at_home,
            red_carry_any=red_carry_any,
            blue_carry_any=blue_carry_any,
            red_carrier_idx=red_carrier_idx,
            blue_carrier_idx=blue_carrier_idx,
            ec_x=ec_x,
            ec_y=ec_y,
            red_dist_to_ec=red_dist_to_ec,
            ec_to_home_dist=ec_to_home_dist,
            intercept_feasible=intercept_feasible,
            intercept_feasible_per_agent=intercept_feasible_per_agent,
            enemy_on_own=enemy_on_own,
            any_intruder=any_intruder,
            intruder_count=intruder_count,
            time_frac=time_frac,
            late_game=late_game,
            trailing=trailing,
            leading=leading,
            alive_count=alive_count,
            profile=prof,
            is_op12=prof["is_op12"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # OP6 failed-assault recovery (defend-then-counter trap)
    # ──────────────────────────────────────────────────────────────────────
    def _bt_update_op6_recovery(self, bb: dict, prof: Dict[str, torch.Tensor]) -> None:
        """Advance OP6 per-agent recovery windows from legal failure events.

        Triggers (no blue-style ID):
          * newly tagged while on blue's half (failed incursion / assault stop)
          * lost the flag / carrier stop (was carrying, now not)

        Carrier-only triggers (dev22) never armed vs TURTLE, which tags before
        pickup. Broad tags without renewal (dev21 renew) left OP6 soft for every
        style. This version keeps the assault-stop trigger but does not renew
        an active window.

        While recovery ticks remain, the agent keeps ATTACKER identity but is
        routed through a midfield redeploy waypoint so OP6 cannot instantly
        re-assault or peel to defense — leaving home temporarily exposed.
        """
        op6 = prof["bt_level"] == 6
        if not bool(op6.any().item()):
            # Still tick down any leftover state if opponent keys changed mid-run.
            self.bt_op6_recovery_ticks = torch.clamp(self.bt_op6_recovery_ticks - 1, min=0)
            return

        midline = float(bb["midline"])
        on_blue_half = self.red_x < midline
        newly_tagged = (~self.bt_op6_prev_red_tagged) & self.red_tagged
        was_carrying = self.bt_op6_prev_red_carrying
        lost_flag = was_carrying & (~self.red_carrying)
        carrier_stopped = newly_tagged & was_carrying
        # Assault-stop tags only count when the dual rush is committed
        # (both alive reds on blue's half).
        both_committed = (
            (self.red_alive & on_blue_half).sum(dim=1) >= 2
        )
        assault_stop = newly_tagged & on_blue_half & both_committed[:, None]
        # Home-exposure recovery only when blue still has a non-carrier
        # defensive anchor near its own flag. TURTLE keeps one; RUSH/SPLIT
        # that abandon home (or only return as carriers) do not receive the
        # redeploy gift. Legal geometry only — not style ID.
        home_x = self.blue_flag_home[:, 0:1]
        near_home = (self.blue_x - home_x).abs() <= 6.0
        blue_has_anchor = (
            self.blue_alive
            & (~self.blue_carrying)
            & (self.blue_x < midline)
            & near_home
        ).any(dim=1)
        fail = (
            (assault_stop | lost_flag | carrier_stopped)
            & op6[:, None]
            & self.red_alive
            & blue_has_anchor[:, None]
        )

        duration = int(self._OP6_RECOVERY_DURATION)
        ticks = self.bt_op6_recovery_ticks
        # Edge-trigger only when entering; no renew while window active.
        # Do not burn the window while tagged/frozen — the redeploy delay must
        # apply after the agent is free to move again (otherwise TURTLE's
        # counter window expires during the tag freeze).
        entering = fail & (ticks <= 0)
        untagged = ~self.red_tagged
        ticks = torch.where(
            entering,
            torch.full_like(ticks, duration),
            torch.where(
                untagged,
                torch.clamp(ticks - 1, min=0),
                ticks,
            ),
        )
        # If blue abandons its anchor mid-window, cancel recovery so dual
        # assault resumes against empty-home styles.
        ticks = torch.where(blue_has_anchor[:, None], ticks, torch.zeros_like(ticks))
        # Non-OP6 envs never carry recovery state.
        ticks = torch.where(op6[:, None], ticks, torch.zeros_like(ticks))

        active = ticks > 0
        self.bt_op6_failed_incursions = (
            self.bt_op6_failed_incursions + fail.any(dim=1).to(torch.int32)
        )
        self.bt_op6_recovery_activations = (
            self.bt_op6_recovery_activations + entering.any(dim=1).to(torch.int32)
        )
        self.bt_op6_recovery_active_steps = (
            self.bt_op6_recovery_active_steps + active.any(dim=1).to(torch.int32)
        )
        self.bt_op6_recovery_ticks = ticks.detach()
        self.bt_op6_prev_red_tagged = self.red_tagged.detach().clone()
        self.bt_op6_prev_red_carrying = self.red_carrying.detach().clone()

    # ──────────────────────────────────────────────────────────────────────
    # OP7 separated-threat latch (first breached corridor overcommit)
    # ──────────────────────────────────────────────────────────────────────
    def _bt_op7_split_response_active(
        self, prof: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """True while OP7's bounded first-lane overcommit window is live."""
        B, device = self.B, self.device
        if not bool(getattr(type(self), "_OP7_SPLIT_LEVER_ENABLED", True)):
            return torch.zeros((B,), dtype=torch.bool, device=device)
        step = self.sim_step_count.to(torch.int32)
        return (
            (prof["bt_level"] == 7)
            & (self.bt_op7_split_first_trigger_step >= 0)
            & (self.bt_op7_split_response_expiry_step >= step)
            & (self.bt_op7_split_primary_blue_idx >= 0)
        )

    def _bt_update_op7_split_latch(self, bb: dict, prof: Dict[str, torch.Tensor]) -> None:
        """Arm/clear OP7's legal separated-threat latch (no blue-style ID).

        Geometry (mirrors adaptive split-pressure thresholds):
          both blues offensively committed past midfield buffer
          + opposite lateral corridors
          + wide teammate separation
          + consecutive persistence ticks

        On arm: lock primary blue = deepest intruder on red half (first breach)
        and freeze that agent's corridor y. Response window is bounded; exit on
        timeout, any blue flag pickup, primary tag/death, or loss of dual
        offensive commitment.
        """
        enabled = bool(getattr(type(self), "_OP7_SPLIT_LEVER_ENABLED", True))
        op7 = (prof["bt_level"] == 7) & self._bt_opponent_mask()
        if (not enabled) or (not bool(op7.any().item())):
            return

        B, device = self.B, self.device
        midline = bb["midline"]
        center_y = float(self.rows) * 0.5
        step = self.sim_step_count.to(torch.int32)
        blue_alive = self.blue_alive & (~self.blue_tagged)
        offensive_buffer = float(self._OP7_SPLIT_OFFENSIVE_BUFFER)

        if self.Nb >= 2:
            blue0_off = blue_alive[:, 0] & (self.blue_x[:, 0] > midline - offensive_buffer)
            blue1_off = blue_alive[:, 1] & (self.blue_x[:, 1] > midline - offensive_buffer)
            lateral_sep = torch.abs(self.blue_y[:, 0] - self.blue_y[:, 1])
            dx01 = self.blue_x[:, 0] - self.blue_x[:, 1]
            dy01 = self.blue_y[:, 0] - self.blue_y[:, 1]
            teammate_dist = torch.sqrt(dx01 ** 2 + dy01 ** 2 + 1e-8)
            opposite_lanes = (
                ((self.blue_y[:, 0] > center_y) & (self.blue_y[:, 1] <= center_y))
                | ((self.blue_y[:, 1] > center_y) & (self.blue_y[:, 0] <= center_y))
            )
            both_offensive = blue0_off & blue1_off
            split_pressure_now = (
                both_offensive
                & opposite_lanes
                & (lateral_sep >= float(self.rows) * float(self._OP7_SPLIT_LATERAL_FRAC))
                & (teammate_dist >= float(self._OP7_SPLIT_TEAMMATE_DIST))
            )
            self.bt_op7_split_max_lateral_sep = torch.where(
                op7,
                torch.maximum(self.bt_op7_split_max_lateral_sep, lateral_sep),
                self.bt_op7_split_max_lateral_sep,
            )
            self.bt_op7_split_max_teammate_dist = torch.where(
                op7,
                torch.maximum(self.bt_op7_split_max_teammate_dist, teammate_dist),
                self.bt_op7_split_max_teammate_dist,
            )
        else:
            both_offensive = torch.zeros((B,), dtype=torch.bool, device=device)
            split_pressure_now = torch.zeros((B,), dtype=torch.bool, device=device)

        ticks = torch.where(
            op7 & split_pressure_now,
            self.bt_op7_split_pressure_ticks + 1,
            torch.zeros_like(self.bt_op7_split_pressure_ticks),
        )
        self.bt_op7_split_pressure_ticks = torch.where(
            op7, ticks, self.bt_op7_split_pressure_ticks
        )
        pressure_active = op7 & (
            self.bt_op7_split_pressure_ticks >= int(self._OP7_SPLIT_PRESSURE_TICKS)
        )
        self.bt_op7_split_active_steps = (
            self.bt_op7_split_active_steps + pressure_active.to(torch.int32)
        )

        new_trigger = pressure_active & (self.bt_op7_split_first_trigger_step < 0)
        if bool(new_trigger.any().item()):
            # First breached lane = deepest blue currently on red half.
            # Fallback: deepest alive blue if neither has crossed yet.
            on_own = bb["enemy_on_own"] & blue_alive
            depth = torch.where(
                on_own,
                self.blue_x,
                torch.full_like(self.blue_x, -1.0e9),
            )
            any_on_own = on_own.any(dim=1)
            fallback_depth = torch.where(
                blue_alive,
                self.blue_x,
                torch.full_like(self.blue_x, -1.0e9),
            )
            depth = torch.where(any_on_own[:, None], depth, fallback_depth)
            primary = torch.argmax(depth, dim=1).to(torch.int32)
            idx_env = torch.arange(B, device=device)
            corridor_y = self.blue_y[idx_env, primary.clamp(min=0).to(torch.int64)]
            self.bt_op7_split_primary_blue_idx = torch.where(
                new_trigger, primary, self.bt_op7_split_primary_blue_idx
            )
            self.bt_op7_split_corridor_y = torch.where(
                new_trigger, corridor_y, self.bt_op7_split_corridor_y
            )
            self.bt_op7_split_first_trigger_step = torch.where(
                new_trigger, step, self.bt_op7_split_first_trigger_step
            )
            self.bt_op7_split_activations = (
                self.bt_op7_split_activations + new_trigger.to(torch.int32)
            )

        # Bounded window: refresh expiry while pressure persists.
        duration = int(self._OP7_SPLIT_RESPONSE_DURATION)
        self.bt_op7_split_response_expiry_step = torch.where(
            pressure_active,
            step + duration,
            self.bt_op7_split_response_expiry_step,
        )

        # Exit conditions while a latch exists.
        latched = op7 & (self.bt_op7_split_first_trigger_step >= 0)
        primary_i = self.bt_op7_split_primary_blue_idx.clamp(min=0).to(torch.int64)
        idx_env = torch.arange(B, device=device)
        primary_alive = self.blue_alive[idx_env, primary_i] & (~self.blue_tagged[idx_env, primary_i])
        timed_out = self.bt_op7_split_response_expiry_step < step
        flag_pickup = bb["blue_carry_any"]
        # Graceful commitment exit: require several consecutive steps without
        # dual offensive commitment (single-step flicker used to kill the
        # window before the opposite lane could be exploited).
        lost_now = latched & (~both_offensive)
        lost_ticks = torch.where(
            lost_now,
            self.bt_op7_split_lost_commit_ticks + 1,
            torch.zeros_like(self.bt_op7_split_lost_commit_ticks),
        )
        self.bt_op7_split_lost_commit_ticks = torch.where(
            op7, lost_ticks, self.bt_op7_split_lost_commit_ticks
        )
        lost_commitment = self.bt_op7_split_lost_commit_ticks >= int(
            self._OP7_SPLIT_COMMIT_GRACE
        )
        should_clear = latched & (
            timed_out | flag_pickup | (~primary_alive) | lost_commitment
        )
        if bool(should_clear.any().item()):
            self.bt_op7_split_pressure_ticks = torch.where(
                should_clear, torch.zeros_like(self.bt_op7_split_pressure_ticks),
                self.bt_op7_split_pressure_ticks,
            )
            self.bt_op7_split_first_trigger_step = torch.where(
                should_clear,
                torch.full_like(self.bt_op7_split_first_trigger_step, -1),
                self.bt_op7_split_first_trigger_step,
            )
            self.bt_op7_split_response_expiry_step = torch.where(
                should_clear,
                torch.full_like(self.bt_op7_split_response_expiry_step, -1),
                self.bt_op7_split_response_expiry_step,
            )
            self.bt_op7_split_primary_blue_idx = torch.where(
                should_clear,
                torch.full_like(self.bt_op7_split_primary_blue_idx, -1),
                self.bt_op7_split_primary_blue_idx,
            )
            self.bt_op7_split_lost_commit_ticks = torch.where(
                should_clear,
                torch.zeros_like(self.bt_op7_split_lost_commit_ticks),
                self.bt_op7_split_lost_commit_ticks,
            )

        response_active = self._bt_op7_split_response_active(prof)
        self.bt_op7_split_response_active_steps = (
            self.bt_op7_split_response_active_steps + response_active.to(torch.int32)
        )

    def _bt_apply_op7_split_role_overrides(
        self,
        bb: dict,
        roles: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Lock nearer red as DEFENDER and partner as INTERCEPTOR on first lane."""
        active = self._bt_op7_split_response_active(prof)
        if not bool(active.any().item()):
            return roles

        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = torch.arange(B, device=device)
        primary = self.bt_op7_split_primary_blue_idx.clamp(min=0).to(torch.int64)
        px = self.blue_x[idx_env, primary]
        py = self.blue_y[idx_env, primary]
        dx = self.red_x - px[:, None]
        dy = self.red_y - py[:, None]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        alive = self.red_alive & (~self.red_tagged)
        dist_m = torch.where(alive, dist, dist.new_full((), 1e9).expand_as(dist))
        nearer = torch.zeros((B, Nr), dtype=torch.bool, device=device)
        nearer_idx = torch.argmin(dist_m, dim=1)
        nearer[idx_env, nearer_idx] = True
        slots = active[:, None] & alive
        roles = torch.where(slots & nearer, torch.full_like(roles, ROLE_DEFENDER), roles)
        roles = torch.where(slots & (~nearer), torch.full_like(roles, ROLE_INTERCEPTOR), roles)
        # Bound cross-lane reassignment for the response window.
        remaining = (
            self.bt_op7_split_response_expiry_step - self.sim_step_count.to(torch.int32)
        ).clamp(min=0)
        lock = self.bt_role_lock_ticks.clone()
        lock = torch.where(
            slots,
            torch.maximum(lock, remaining[:, None].expand(B, Nr)),
            lock,
        )
        self.bt_role_lock_ticks = lock.detach()
        self.bt_red_role = torch.where(slots, roles, self.bt_red_role)
        return roles

    def _bt_apply_op7_split_route_overrides(
        self,
        bb: dict,
        roles: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collapse both reds onto the first breached corridor."""
        active = self._bt_op7_split_response_active(prof)
        if not bool(active.any().item()):
            return tx, ty

        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = torch.arange(B, device=device)
        primary = self.bt_op7_split_primary_blue_idx.clamp(min=0).to(torch.int64)
        px = self.blue_x[idx_env, primary]
        py = self.blue_y[idx_env, primary]
        corridor_y = self.bt_op7_split_corridor_y
        dx = self.red_x - px[:, None]
        dy = self.red_y - py[:, None]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        alive = self.red_alive & (~self.red_tagged)
        dist_m = torch.where(alive, dist, dist.new_full((), 1e9).expand_as(dist))
        nearer = torch.zeros((B, Nr), dtype=torch.bool, device=device)
        nearer[idx_env, torch.argmin(dist_m, dim=1)] = True
        slots = active[:, None] & alive
        # Primary lock: chase the first-breached blue live.
        tx = torch.where(slots & nearer, px[:, None].expand(B, Nr), tx)
        ty = torch.where(slots & nearer, py[:, None].expand(B, Nr), ty)
        # Partner reinforces the same corridor (frozen y), not the opposite lane.
        sec_tx = px[:, None].expand(B, Nr)
        sec_ty = corridor_y[:, None].expand(B, Nr)
        tx = torch.where(slots & (~nearer), sec_tx, tx)
        ty = torch.where(slots & (~nearer), sec_ty, ty)
        return tx, ty

    # ──────────────────────────────────────────────────────────────────────
    # OP7 compact containment (final lever: same-corridor dual pressure)
    # ──────────────────────────────────────────────────────────────────────
    def _bt_op7_compact_response_active(
        self, prof: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """True while OP7's compact containment window is live (and SPLIT is not)."""
        B, device = self.B, self.device
        if not bool(getattr(type(self), "_OP7_COMPACT_LEVER_ENABLED", True)):
            return torch.zeros((B,), dtype=torch.bool, device=device)
        step = self.sim_step_count.to(torch.int32)
        split_busy = self._bt_op7_split_response_active(prof)
        return (
            (prof["bt_level"] == 7)
            & (~split_busy)
            & (self.bt_op7_compact_first_trigger_step >= 0)
            & (self.bt_op7_compact_response_expiry_step >= step)
        )

    def _bt_update_op7_compact_latch(self, bb: dict, prof: Dict[str, torch.Tensor]) -> None:
        """Arm/clear OP7 compact same-corridor containment (no blue-style ID).

        Geometry:
          both blues offensively committed
          + teammate distance below compact threshold
          + same corridor (small lateral separation)
          + persistence ticks

        Mutually exclusive with the SPLIT separated-threat latch: while SPLIT
        response is active, compact does not arm or apply. SPLIT latch
        thresholds are intentionally not retuned here.
        """
        enabled = bool(getattr(type(self), "_OP7_COMPACT_LEVER_ENABLED", True))
        op7 = (prof["bt_level"] == 7) & self._bt_opponent_mask()
        if (not enabled) or (not bool(op7.any().item())):
            return

        B, device = self.B, self.device
        midline = bb["midline"]
        center_y = float(self.rows) * 0.5
        step = self.sim_step_count.to(torch.int32)
        blue_alive = self.blue_alive & (~self.blue_tagged)
        offensive_buffer = float(self._OP7_COMPACT_OFFENSIVE_BUFFER)
        split_busy = self._bt_op7_split_response_active(prof)

        if self.Nb >= 2:
            blue0_off = blue_alive[:, 0] & (self.blue_x[:, 0] > midline - offensive_buffer)
            blue1_off = blue_alive[:, 1] & (self.blue_x[:, 1] > midline - offensive_buffer)
            lateral_sep = torch.abs(self.blue_y[:, 0] - self.blue_y[:, 1])
            dx01 = self.blue_x[:, 0] - self.blue_x[:, 1]
            dy01 = self.blue_y[:, 0] - self.blue_y[:, 1]
            teammate_dist = torch.sqrt(dx01 ** 2 + dy01 ** 2 + 1e-8)
            same_side = (
                ((self.blue_y[:, 0] > center_y) & (self.blue_y[:, 1] > center_y))
                | ((self.blue_y[:, 0] <= center_y) & (self.blue_y[:, 1] <= center_y))
            )
            same_corridor = same_side | (
                lateral_sep <= float(self._OP7_COMPACT_LATERAL_MAX)
            )
            both_offensive = blue0_off & blue1_off
            compact_now = (
                both_offensive
                & same_corridor
                & (teammate_dist <= float(self._OP7_COMPACT_TEAMMATE_DIST))
                & (lateral_sep <= float(self._OP7_COMPACT_LATERAL_MAX))
                & (~split_busy)
            )
            self.bt_op7_compact_min_teammate_dist = torch.where(
                op7 & both_offensive,
                torch.minimum(self.bt_op7_compact_min_teammate_dist, teammate_dist),
                self.bt_op7_compact_min_teammate_dist,
            )
            self.bt_op7_compact_max_lateral_sep = torch.where(
                op7 & both_offensive,
                torch.maximum(self.bt_op7_compact_max_lateral_sep, lateral_sep),
                self.bt_op7_compact_max_lateral_sep,
            )
        else:
            both_offensive = torch.zeros((B,), dtype=torch.bool, device=device)
            compact_now = torch.zeros((B,), dtype=torch.bool, device=device)

        ticks = torch.where(
            op7 & compact_now,
            self.bt_op7_compact_pressure_ticks + 1,
            torch.zeros_like(self.bt_op7_compact_pressure_ticks),
        )
        self.bt_op7_compact_pressure_ticks = torch.where(
            op7, ticks, self.bt_op7_compact_pressure_ticks
        )
        pressure_active = op7 & (
            self.bt_op7_compact_pressure_ticks >= int(self._OP7_COMPACT_PRESSURE_TICKS)
        )
        self.bt_op7_compact_active_steps = (
            self.bt_op7_compact_active_steps + pressure_active.to(torch.int32)
        )

        new_trigger = pressure_active & (self.bt_op7_compact_first_trigger_step < 0)
        self.bt_op7_compact_first_trigger_step = torch.where(
            new_trigger, step, self.bt_op7_compact_first_trigger_step
        )
        self.bt_op7_compact_activations = (
            self.bt_op7_compact_activations + new_trigger.to(torch.int32)
        )

        duration = int(self._OP7_COMPACT_RESPONSE_DURATION)
        # Refresh while compact persists, or while a blue carrier is returning
        # (the interceptor's job peaks after pickup).
        refresh = pressure_active | (
            (self.bt_op7_compact_first_trigger_step >= 0) & bb["blue_carry_any"] & (~split_busy)
        )
        self.bt_op7_compact_response_expiry_step = torch.where(
            refresh,
            step + duration,
            self.bt_op7_compact_response_expiry_step,
        )

        latched = op7 & (self.bt_op7_compact_first_trigger_step >= 0)
        timed_out = self.bt_op7_compact_response_expiry_step < step
        # Clear if SPLIT latch takes over (separated pressure wins).
        split_takeover = split_busy
        lost_now = latched & (~both_offensive) & (~bb["blue_carry_any"])
        lost_ticks = torch.where(
            lost_now,
            self.bt_op7_compact_lost_commit_ticks + 1,
            torch.zeros_like(self.bt_op7_compact_lost_commit_ticks),
        )
        self.bt_op7_compact_lost_commit_ticks = torch.where(
            op7, lost_ticks, self.bt_op7_compact_lost_commit_ticks
        )
        lost_commitment = self.bt_op7_compact_lost_commit_ticks >= int(
            self._OP7_COMPACT_COMMIT_GRACE
        )
        should_clear = latched & (timed_out | split_takeover | lost_commitment)
        if bool(should_clear.any().item()):
            self.bt_op7_compact_pressure_ticks = torch.where(
                should_clear,
                torch.zeros_like(self.bt_op7_compact_pressure_ticks),
                self.bt_op7_compact_pressure_ticks,
            )
            self.bt_op7_compact_first_trigger_step = torch.where(
                should_clear,
                torch.full_like(self.bt_op7_compact_first_trigger_step, -1),
                self.bt_op7_compact_first_trigger_step,
            )
            self.bt_op7_compact_response_expiry_step = torch.where(
                should_clear,
                torch.full_like(self.bt_op7_compact_response_expiry_step, -1),
                self.bt_op7_compact_response_expiry_step,
            )
            self.bt_op7_compact_lost_commit_ticks = torch.where(
                should_clear,
                torch.zeros_like(self.bt_op7_compact_lost_commit_ticks),
                self.bt_op7_compact_lost_commit_ticks,
            )

        response_active = self._bt_op7_compact_response_active(prof)
        self.bt_op7_compact_response_active_steps = (
            self.bt_op7_compact_response_active_steps + response_active.to(torch.int32)
        )

    def _bt_apply_op7_compact_role_overrides(
        self,
        bb: dict,
        roles: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One deep DEFENDER near own flag; partner INTERCEPTOR on return cut."""
        active = self._bt_op7_compact_response_active(prof)
        if not bool(active.any().item()):
            return roles

        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = torch.arange(B, device=device)
        # Deep defender = red closer to own flag home (fortress seat).
        rfh_x = bb["red_flag_home"][:, 0]
        rfh_y = bb["red_flag_home"][:, 1]
        dx = self.red_x - rfh_x[:, None]
        dy = self.red_y - rfh_y[:, None]
        dist_home = torch.sqrt(dx * dx + dy * dy + 1e-8)
        alive = self.red_alive & (~self.red_tagged)
        dist_m = torch.where(alive, dist_home, dist_home.new_full((), 1e9).expand_as(dist_home))
        deep = torch.zeros((B, Nr), dtype=torch.bool, device=device)
        deep[idx_env, torch.argmin(dist_m, dim=1)] = True
        slots = active[:, None] & alive
        roles = torch.where(slots & deep, torch.full_like(roles, ROLE_DEFENDER), roles)
        roles = torch.where(slots & (~deep), torch.full_like(roles, ROLE_INTERCEPTOR), roles)
        remaining = (
            self.bt_op7_compact_response_expiry_step - self.sim_step_count.to(torch.int32)
        ).clamp(min=0)
        lock = self.bt_role_lock_ticks.clone()
        lock = torch.where(
            slots,
            torch.maximum(lock, remaining[:, None].expand(B, Nr)),
            lock,
        )
        self.bt_role_lock_ticks = lock.detach()
        self.bt_red_role = torch.where(slots, roles, self.bt_red_role)
        return roles

    def _bt_apply_op7_compact_route_overrides(
        self,
        bb: dict,
        roles: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deep defender holds/chases; partner cuts carrier return route."""
        active = self._bt_op7_compact_response_active(prof)
        if not bool(active.any().item()):
            return tx, ty

        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = torch.arange(B, device=device)
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))
        rfh_x = bb["red_flag_home"][:, 0]
        rfh_y = bb["red_flag_home"][:, 1]
        dx = self.red_x - rfh_x[:, None]
        dy = self.red_y - rfh_y[:, None]
        dist_home = torch.sqrt(dx * dx + dy * dy + 1e-8)
        alive = self.red_alive & (~self.red_tagged)
        dist_m = torch.where(alive, dist_home, dist_home.new_full((), 1e9).expand_as(dist_home))
        deep = torch.zeros((B, Nr), dtype=torch.bool, device=device)
        deep[idx_env, torch.argmin(dist_m, dim=1)] = True
        slots = active[:, None] & alive

        # Deep defender: chase carrier if present, else deepest intruder on own half.
        blue_alive = self.blue_alive & (~self.blue_tagged)
        on_own = bb["enemy_on_own"] & blue_alive
        depth = torch.where(on_own, self.blue_x, torch.full_like(self.blue_x, -1.0e9))
        any_on_own = on_own.any(dim=1)
        fallback = torch.where(blue_alive, self.blue_x, torch.full_like(self.blue_x, -1.0e9))
        depth = torch.where(any_on_own[:, None], depth, fallback)
        primary = torch.argmax(depth, dim=1)
        def_tx = torch.where(
            bb["blue_carry_any"],
            bb["ec_x"],
            self.blue_x[idx_env, primary],
        )
        def_ty = torch.where(
            bb["blue_carry_any"],
            bb["ec_y"],
            self.blue_y[idx_env, primary],
        )
        tx = torch.where(slots & deep, def_tx[:, None].expand(B, Nr), tx)
        ty = torch.where(slots & deep, def_ty[:, None].expand(B, Nr), ty)

        # Interceptor: cut return path to blue home (do not pile onto first intruder).
        bhome_x = bb["blue_flag_home"][:, 0]
        bhome_y = bb["blue_flag_home"][:, 1]
        lead = torch.argmax(
            torch.where(blue_alive, self.blue_x, torch.full_like(self.blue_x, -1.0e9)),
            dim=1,
        )
        lead_x = self.blue_x[idx_env, lead]
        lead_y = self.blue_y[idx_env, lead]
        # When carrying: intercept between carrier and blue home.
        # Pre-pickup: pre-position on the lead agent's return corridor.
        cut_x = torch.where(
            bb["blue_carry_any"],
            0.55 * bb["ec_x"] + 0.45 * bhome_x,
            0.55 * lead_x + 0.45 * bhome_x,
        )
        cut_y = torch.where(
            bb["blue_carry_any"],
            0.55 * bb["ec_y"] + 0.45 * bhome_y,
            0.55 * lead_y + 0.45 * bhome_y,
        )
        cut_x = torch.clamp(cut_x, 0.0, max_x)
        cut_y = torch.clamp(cut_y, 0.0, max_y)
        tx = torch.where(slots & (~deep), cut_x[:, None].expand(B, Nr), tx)
        ty = torch.where(slots & (~deep), cut_y[:, None].expand(B, Nr), ty)
        return tx, ty

    # ──────────────────────────────────────────────────────────────────────
    # Role assignment: utility-based, with hysteresis
    # ──────────────────────────────────────────────────────────────────────
    def _bt_assign_roles(self, bb: dict) -> torch.Tensor:
        """
        Assign one role per red agent using per-agent utility scores.

        Returns roles tensor [B, Nr] of dtype int32 with role constants.

        Role priority (highest wins per slot, with teammate deduplication):
        1.  FLAG_RETR   — own flag stolen; assign cheapest agent.
        2.  ESCORT      — own carrier exists; assign nearest non-carrier.
        3.  INTERCEPTOR — enemy carrier feasible intercept; assign nearest feasible.
        4.  COUNTER     — OP12 or (enemy carrier, intercept NOT feasible, trailing).
        5.  DEFENDER    — intruder on own half, no carrier emergencies, assign one.
        6.  2V1_WING    — 2+ alive agents near same enemy, assign wing slot.
        7.  ATTACKER    — default; go for enemy flag.
        """
        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = bb["idx_env"]
        prof = bb["profile"]
        roles = self.bt_red_role.clone()                          # [B, Nr]
        lock  = self.bt_role_lock_ticks.clone()                  # [B, Nr]

        # Decrement lock counters (clamped to 0).
        lock = (lock - 1).clamp(min=0)

        # Boolean mask: which agents are eligible to change role this tick.
        can_change = (lock <= 0) & self.red_alive & (~self.red_tagged)  # [B, Nr]
        # Searcher-only delayed-intent hold. Default 0 is a no-op, so
        # canonical OP6-OP12 role assignment is unchanged.
        hold_n = int(getattr(self, "_sds_opening_hold_steps", 0) or 0)
        if hold_n > 0:
            sds_hold = self.sim_step_count.to(torch.int32) < hold_n
            if sds_hold.ndim == 0:
                sds_hold = sds_hold.repeat(B)
            hold_slots = sds_hold[:, None] & self.red_alive & (~self.red_tagged)
            roles = torch.where(
                hold_slots, torch.full_like(roles, ROLE_DEFENDER), roles)
            can_change = can_change & (~hold_slots)
        # Opening windows are opponent-specific (level-gated). OP9 never enters.
        # Legal signals only: sim_step_count, blue_carry_any, adapt_split_pressure.
        op8_opening = (
            (prof["bt_level"] == 8)
            & (self.sim_step_count.to(torch.int32) < int(self._OP8_FORMATION_OPENING_STEPS))
            & (~bb["blue_carry_any"])
        )
        op12_opening = (
            (prof["bt_level"] == 12)
            & (self.sim_step_count.to(torch.int32) < 20)
            & (~bb.get("adapt_split_pressure", torch.zeros((B,), dtype=torch.bool, device=device)))
        )
        opening_active = op8_opening | op12_opening
        late_or_ready = ~opening_active
        # OP7 RUSH-host redesign (2026-07-28) was abandoned after three
        # attempts (opening-window role suppression / Priority-5 gate /
        # widen-zone reasoning). OP7 is now the map_a/V3 SPLIT-host
        # candidate via `_bt_update_op7_split_latch` (separated-threat
        # first-lane overcommit). Do not reintroduce OP7 into
        # opening_active — that path handed OP7 an early dual-rush.
        opening_slots = opening_active[:, None] & self.red_alive & (~self.red_tagged)
        roles = torch.where(
            opening_slots,
            torch.full_like(roles, ROLE_ATTACKER),
            roles,
        )
        lock = torch.where(opening_slots, torch.zeros_like(lock), lock)
        can_change = can_change | opening_slots

        # OP6 failed-assault recovery: keep ATTACKER identity; block peel to
        # DEFENDER / INTERCEPTOR / FLAG_RETR / etc. for the recovering agent.
        recovering = (self.bt_op6_recovery_ticks > 0) & self.red_alive
        roles = torch.where(
            recovering,
            torch.full_like(roles, ROLE_ATTACKER),
            roles,
        )
        lock = torch.where(
            recovering,
            torch.maximum(lock, self.bt_op6_recovery_ticks),
            lock,
        )
        can_change = can_change & (~recovering)
        # While recovering under a blue home-anchor, suppress team FLAG_RETR so
        # the dual-rusher partner does not instantly cover the rear that
        # TURTLE's counter needs. Empty-home blues cancel recovery (no suppress).
        any_op6_recovery = recovering.any(dim=1) & (prof["bt_level"] == 6)

        # ── OP6 extraction support (pre-pickup screener + post-pickup lock) ─
        # Dominant failed-return modes vs RUSH/ESCORT (dev28 trace): tagged
        # from behind + agents crossing. When blue has no near-home non-carrier
        # anchor, lock carrier as ATTACKER and partner as ESCORT screener.
        # Pre-pickup (dev35): when grab is imminent and both blues threaten the
        # return corridor, reserve the partner as screener *before* pickup so
        # C3 one-tick lag does not decide the RUSH race. Geometry only — do
        # not inspect blue style ID. Leave recovery thresholds untouched.
        op6 = prof["bt_level"] == 6
        extract_enabled = bool(getattr(type(self), "_OP6_EXTRACTION_ENABLED", True))
        preengage_enabled = bool(getattr(type(self), "_OP6_PREENGAGE_ENABLED", True))
        if extract_enabled and bool(op6.any().item()):
            mid = bb["midline"]
            bhome_x = self.blue_flag_home[:, 0:1]
            near_bhome = (self.blue_x - bhome_x).abs() <= 6.0
            blue_anchor = (
                self.blue_alive
                & (~self.blue_carrying)
                & (self.blue_x < mid)
                & near_bhome
            ).any(dim=1)
            abandoned = ~blue_anchor
            extract_ticks = self.bt_op6_extract_ticks
            pre_ticks = self.bt_op6_preengage_ticks

            flag_x = bb["blue_flag_pos"][:, 0]
            flag_y = bb["blue_flag_pos"][:, 1]
            enemy_flag_at_home = (
                (self.blue_flag_pos[:, 0] - self.blue_flag_home[:, 0]).abs() < 1.5
            ) & (
                (self.blue_flag_pos[:, 1] - self.blue_flag_home[:, 1]).abs() < 1.5
            )
            r_flag_dist = torch.sqrt(
                (self.red_x - flag_x[:, None]) ** 2
                + (self.red_y - flag_y[:, None]) ** 2
                + 1e-8
            )
            r_flag_dist = torch.where(
                self.red_alive & (~self.red_tagged),
                r_flag_dist,
                r_flag_dist.new_full((), 1e9).expand_as(r_flag_dist),
            )
            flag_attacker = torch.argmin(r_flag_dist, dim=1)
            imminent = r_flag_dist.min(dim=1).values <= float(self._OP6_PREPICKUP_RADIUS)

            # Both blues threaten the corridor from blue flag → red home.
            home_x = bb["red_flag_home"][:, 0]
            home_y = bb["red_flag_home"][:, 1]
            vx = home_x - flag_x
            vy = home_y - flag_y
            vv = vx * vx + vy * vy + 1e-8
            wx = self.blue_x - flag_x[:, None]
            wy = self.blue_y - flag_y[:, None]
            t_proj = (wx * vx[:, None] + wy * vy[:, None]) / vv[:, None]
            proj_x = flag_x[:, None] + t_proj * vx[:, None]
            proj_y = flag_y[:, None] + t_proj * vy[:, None]
            lat = torch.sqrt(
                (self.blue_x - proj_x) ** 2 + (self.blue_y - proj_y) ** 2 + 1e-8
            )
            on_seg = (t_proj > -0.05) & (t_proj < 1.05) & (lat < 6.0)
            ahead = (self.blue_x > mid) & ((self.blue_y - home_y[:, None]).abs() < 7.0)
            blue_threat = self.blue_alive & (~self.blue_tagged) & (on_seg | ahead)
            both_threaten = blue_threat.sum(dim=1) >= 2

            # --- Pre-pickup screener reserve (narrow gate; sticky countdown) ---
            newly_pre = torch.zeros((B,), dtype=torch.bool, device=device)
            pre_active = torch.zeros((B,), dtype=torch.bool, device=device)
            if preengage_enabled:
                want_pre = (
                    op6
                    & (~bb["red_carry_any"])
                    & abandoned
                    & enemy_flag_at_home
                    & imminent
                    & both_threaten
                    & (extract_ticks <= 0)
                )
                newly_pre = want_pre & (pre_ticks <= 0)
                pre_ticks = torch.where(
                    newly_pre,
                    torch.full_like(pre_ticks, int(self._OP6_PREENGAGE_LOCK)),
                    pre_ticks,
                )
                # Cancel: pickup (extract owns), home anchor returns, flag gone.
                cancel_pre = (
                    bb["red_carry_any"]
                    | (~abandoned)
                    | (~enemy_flag_at_home)
                    | (~op6)
                )
                pre_ticks = torch.where(cancel_pre, torch.zeros_like(pre_ticks), pre_ticks)
                pre_ticks = torch.where(
                    (pre_ticks > 0) & (~bb["red_carry_any"]),
                    torch.clamp(pre_ticks - 1, min=0),
                    pre_ticks,
                )
                self.bt_op6_preengage_activations = (
                    self.bt_op6_preengage_activations + newly_pre.to(torch.int32)
                )
                pre_active = (pre_ticks > 0) & op6 & (~bb["red_carry_any"])
                self.bt_op6_preengage_active_steps = (
                    self.bt_op6_preengage_active_steps + pre_active.to(torch.int32)
                )
                self.bt_op6_preengage_ticks = pre_ticks.detach()
                if bool(newly_pre.any().item()):
                    # Lock threats as if carrier is at the pickup point.
                    self._bt_op6_lock_extract_threats(
                        bb, newly_pre, carr_x=flag_x, carr_y=flag_y
                    )
            else:
                self.bt_op6_preengage_ticks = torch.where(
                    op6,
                    torch.zeros_like(self.bt_op6_preengage_ticks),
                    self.bt_op6_preengage_ticks,
                )

            # --- Post-pickup extract latch ---
            want_extract = op6 & bb["red_carry_any"] & abandoned
            newly_armed = want_extract & (extract_ticks <= 0)
            extract_ticks = torch.where(
                newly_armed,
                torch.full_like(extract_ticks, int(self._OP6_EXTRACT_LOCK)),
                extract_ticks,
            )
            extract_ticks = torch.where(
                op6 & bb["red_carry_any"] & (extract_ticks > 0),
                torch.clamp(extract_ticks - 1, min=0),
                torch.where(
                    op6 & bb["red_carry_any"],
                    extract_ticks,
                    torch.zeros_like(extract_ticks),
                ),
            )
            extract_ticks = torch.where(
                op6 & bb["red_carry_any"],
                extract_ticks,
                torch.zeros_like(extract_ticks),
            )
            self.bt_op6_extract_ticks = extract_ticks.detach()
            extract_active = (extract_ticks > 0) & op6
            # Preserve preengage threat locks; only fill missing slots on pickup.
            need_lock = newly_armed & (
                (self.bt_op6_extract_carrier_threat < 0)
                | (self.bt_op6_extract_screener_threat < 0)
            )
            if bool(need_lock.any().item()):
                self._bt_op6_lock_extract_threats(bb, need_lock)
            clear_threats = op6 & (~extract_active) & (~pre_active)
            if bool(clear_threats.any().item()):
                self.bt_op6_extract_carrier_threat = torch.where(
                    clear_threats,
                    torch.full_like(self.bt_op6_extract_carrier_threat, -1),
                    self.bt_op6_extract_carrier_threat,
                )
                self.bt_op6_extract_screener_threat = torch.where(
                    clear_threats,
                    torch.full_like(self.bt_op6_extract_screener_threat, -1),
                    self.bt_op6_extract_screener_threat,
                )

            rc_idx = bb["red_carrier_idx"].clamp(min=0)
            agent_ids = torch.arange(Nr, device=device)[None, :]
            is_carrier = agent_ids == rc_idx[:, None]
            is_flag_atk = agent_ids == flag_attacker[:, None]
            extract_slots = (
                extract_active[:, None] & self.red_alive & (~self.red_tagged)
            )
            pre_slots = pre_active[:, None] & self.red_alive & (~self.red_tagged)
            roles = torch.where(
                extract_slots & is_carrier,
                torch.full_like(roles, ROLE_ATTACKER),
                roles,
            )
            roles = torch.where(
                extract_slots & (~is_carrier),
                torch.full_like(roles, ROLE_ESCORT),
                roles,
            )
            roles = torch.where(
                pre_slots & is_flag_atk,
                torch.full_like(roles, ROLE_ATTACKER),
                roles,
            )
            roles = torch.where(
                pre_slots & (~is_flag_atk),
                torch.full_like(roles, ROLE_ESCORT),
                roles,
            )
            lock = torch.where(
                extract_slots,
                torch.maximum(lock, extract_ticks[:, None].expand(B, Nr)),
                lock,
            )
            lock = torch.where(
                pre_slots,
                torch.maximum(
                    lock, self.bt_op6_preengage_ticks[:, None].expand(B, Nr)
                ),
                lock,
            )
            can_change = can_change & (~extract_slots) & (~pre_slots)

            # --- Mutual-carry / imminent race denial (dev36) ---
            # When blue already carries and home is abandoned, send the
            # non-attacker to cut the blue carrier — including the window
            # *before* red's pickup completes (dev36b: mutual-only still lost
            # first-score because blue was already closer to scoring).
            # Carrier/flag-attacker keeps ATTACKER + home/flag route. Peel and
            # dual-threat locks stay frozen outside race-home exception.
            race_enabled = bool(
                getattr(type(self), "_OP6_RACE_DENIAL_ENABLED", True)
            )
            if race_enabled:
                race_ticks = self.bt_op6_race_ticks
                want_race = (
                    op6
                    & bb["blue_carry_any"]
                    & abandoned
                    & (bb["red_carry_any"] | imminent)
                )
                newly_race = want_race & (race_ticks <= 0)
                race_ticks = torch.where(
                    newly_race,
                    torch.full_like(race_ticks, int(self._OP6_RACE_LOCK)),
                    race_ticks,
                )
                cancel_race = (
                    (~bb["blue_carry_any"])
                    | (~abandoned)
                    | (~op6)
                )
                race_ticks = torch.where(
                    cancel_race, torch.zeros_like(race_ticks), race_ticks
                )
                race_ticks = torch.where(
                    want_race & (race_ticks > 0),
                    torch.clamp(race_ticks - 1, min=0),
                    race_ticks,
                )
                self.bt_op6_race_activations = (
                    self.bt_op6_race_activations + newly_race.to(torch.int32)
                )
                race_active = (race_ticks > 0) & want_race
                self.bt_op6_race_active_steps = (
                    self.bt_op6_race_active_steps + race_active.to(torch.int32)
                )
                self.bt_op6_race_ticks = race_ticks.detach()
                # Lock blue carrier as interceptor target at arm.
                bc_idx = bb["blue_carrier_idx"].to(torch.int32)
                self.bt_op6_race_target_idx = torch.where(
                    newly_race,
                    bc_idx,
                    torch.where(
                        race_active,
                        self.bt_op6_race_target_idx,
                        torch.full_like(self.bt_op6_race_target_idx, -1),
                    ),
                )
                # Refresh target if blue carrier index changes while armed.
                self.bt_op6_race_target_idx = torch.where(
                    race_active & bb["blue_carry_any"],
                    bc_idx,
                    self.bt_op6_race_target_idx,
                )
                # Keep the red carrier (or imminent flag attacker) on offense;
                # partner becomes INTERCEPTOR.
                keep_atk = torch.where(
                    bb["red_carry_any"],
                    rc_idx,
                    flag_attacker,
                )
                is_keep = agent_ids == keep_atk[:, None]
                race_slots = (
                    race_active[:, None] & self.red_alive & (~self.red_tagged)
                )
                roles = torch.where(
                    race_slots & is_keep,
                    torch.full_like(roles, ROLE_ATTACKER),
                    roles,
                )
                roles = torch.where(
                    race_slots & (~is_keep),
                    torch.full_like(roles, ROLE_INTERCEPTOR),
                    roles,
                )
                lock = torch.where(
                    race_slots,
                    torch.maximum(
                        lock, race_ticks[:, None].expand(B, Nr)
                    ),
                    lock,
                )
                can_change = can_change & (~race_slots)
            elif hasattr(self, "bt_op6_race_ticks"):
                self.bt_op6_race_ticks = torch.where(
                    op6,
                    torch.zeros_like(self.bt_op6_race_ticks),
                    self.bt_op6_race_ticks,
                )
                self.bt_op6_race_target_idx = torch.where(
                    op6,
                    torch.full_like(self.bt_op6_race_target_idx, -1),
                    self.bt_op6_race_target_idx,
                )
        elif hasattr(self, "bt_op6_extract_ticks"):
            self.bt_op6_extract_ticks = torch.where(
                op6, torch.zeros_like(self.bt_op6_extract_ticks), self.bt_op6_extract_ticks
            )
            if hasattr(self, "bt_op6_preengage_ticks"):
                self.bt_op6_preengage_ticks = torch.where(
                    op6,
                    torch.zeros_like(self.bt_op6_preengage_ticks),
                    self.bt_op6_preengage_ticks,
                )
            if hasattr(self, "bt_op6_race_ticks"):
                self.bt_op6_race_ticks = torch.where(
                    op6,
                    torch.zeros_like(self.bt_op6_race_ticks),
                    self.bt_op6_race_ticks,
                )
                self.bt_op6_race_target_idx = torch.where(
                    op6,
                    torch.full_like(self.bt_op6_race_target_idx, -1),
                    self.bt_op6_race_target_idx,
                )
            if hasattr(self, "bt_op6_extract_carrier_threat"):
                self.bt_op6_extract_carrier_threat = torch.where(
                    op6,
                    torch.full_like(self.bt_op6_extract_carrier_threat, -1),
                    self.bt_op6_extract_carrier_threat,
                )
                self.bt_op6_extract_screener_threat = torch.where(
                    op6,
                    torch.full_like(self.bt_op6_extract_screener_threat, -1),
                    self.bt_op6_extract_screener_threat,
                )

        # ── Priority 1: flag retrieval ───────────────────────────────────
        need_retr = (
            (~bb["own_flag_at_home"])
            & prof["enable_flag_retr"]
            & late_or_ready
            & (~any_op6_recovery)
        )
        if need_retr.any():
            # Pick the closest eligible agent per env.
            flag_dx = self.red_x - bb["red_flag_pos"][:, 0:1]
            flag_dy = self.red_y - bb["red_flag_pos"][:, 1:2]
            flag_dist = torch.sqrt(flag_dx ** 2 + flag_dy ** 2 + 1e-8)
            flag_dist_masked = torch.where(can_change & need_retr[:, None], flag_dist,
                                           flag_dist.new_full((), 1e9).expand_as(flag_dist))
            retr_agent = torch.argmin(flag_dist_masked, dim=1)  # [B]
            for j in range(Nr):
                is_j = (retr_agent == j)
                assign = need_retr & is_j & can_change[:, j]
                roles[:, j] = torch.where(assign, torch.full((B,), ROLE_FLAG_RETR, dtype=torch.int32, device=device), roles[:, j])
                lock[:, j]  = torch.where(assign, prof["lock_flag_retr"], lock[:, j])
                can_change[:, j] = can_change[:, j] & (~assign)

        # ── Priority 2: escort own carrier ───────────────────────────────
        # Opening windows (OP8 formation / OP12 stage-1) force ATTACKER above
        # and suppress escort so red does not instantly convert its own
        # opening pickup into a protected return. Gating uses legal state
        # only (level + time / blue_carry / adapt_split_pressure) — never
        # blue style ID. OP9 is excluded by level gates.
        have_carrier = bb["red_carry_any"] & prof["enable_escort"] & late_or_ready
        if have_carrier.any():
            rc_idx = bb["red_carrier_idx"].clamp(min=0)
            carr_x = self.red_x[idx_env, rc_idx]
            carr_y = self.red_y[idx_env, rc_idx]
            # Agent distances to own carrier position [B, Nr].
            cx_diff = self.red_x - carr_x[:, None]
            cy_diff = self.red_y - carr_y[:, None]
            cdist = torch.sqrt(cx_diff ** 2 + cy_diff ** 2 + 1e-8)
            # Exclude the carrier itself from escort assignment.
            is_carrier_slot = (torch.arange(Nr, device=device)[None, :] == rc_idx[:, None])
            cdist_masked = torch.where(
                can_change & (~is_carrier_slot) & have_carrier[:, None],
                cdist, cdist.new_full((), 1e9).expand_as(cdist)
            )
            escort_agent = torch.argmin(cdist_masked, dim=1)
            for j in range(Nr):
                is_j = (escort_agent == j)
                assign = have_carrier & is_j & can_change[:, j] & (~is_carrier_slot[:, j])
                roles[:, j] = torch.where(assign, torch.full((B,), ROLE_ESCORT, dtype=torch.int32, device=device), roles[:, j])
                lock[:, j]  = torch.where(assign, prof["lock_escort"], lock[:, j])
                can_change[:, j] = can_change[:, j] & (~assign)

        # ── Priority 3: intercept enemy carrier (feasible) ───────────────
        e_carry = bb["blue_carry_any"]
        feas    = bb["intercept_feasible"] & prof["enable_intercept"] & late_or_ready
        if (e_carry & feas).any():
            # Among feasible agents, pick the one closest to intercept midpoint.
            feas_per = bb["intercept_feasible_per_agent"]  # [B, Nr]
            ec_x_exp = bb["ec_x"][:, None]
            ec_y_exp = bb["ec_y"][:, None]
            home_x_exp = bb["blue_flag_home"][:, 0:1]
            home_y_exp = bb["blue_flag_home"][:, 1:2]
            mid_x = ec_x_exp + (home_x_exp - ec_x_exp) * 0.5
            mid_y = ec_y_exp + (home_y_exp - ec_y_exp) * 0.5
            mid_dist = torch.sqrt((self.red_x - mid_x) ** 2 + (self.red_y - mid_y) ** 2 + 1e-8)
            mid_dist_m = torch.where(
                can_change & feas_per & e_carry[:, None],
                mid_dist, mid_dist.new_full((), 1e9).expand_as(mid_dist)
            )
            intc_agent = torch.argmin(mid_dist_m, dim=1)
            for j in range(Nr):
                is_j = (intc_agent == j)
                assign = e_carry & feas & is_j & can_change[:, j]
                roles[:, j] = torch.where(assign, torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device), roles[:, j])
                lock[:, j]  = torch.where(assign, prof["lock_intercept"], lock[:, j])
                can_change[:, j] = can_change[:, j] & (~assign)

        # ── Priority 4: counter-capture ──────────────────────────────────
        counter_ok = (
            prof["enable_counter"]
            & e_carry
            & (~feas)
            & late_or_ready
            & (prof["counter_always"] | (prof["counter_when_trailing"] & bb["trailing"]))
        )
        if counter_ok.any():
            # Pick the agent closest to the enemy flag.
            efx = bb["blue_flag_pos"][:, 0:1]
            efy = bb["blue_flag_pos"][:, 1:2]
            ef_dist = torch.sqrt((self.red_x - efx) ** 2 + (self.red_y - efy) ** 2 + 1e-8)
            ef_dist_m = torch.where(
                can_change & counter_ok[:, None],
                ef_dist, ef_dist.new_full((), 1e9).expand_as(ef_dist)
            )
            ctr_agent = torch.argmin(ef_dist_m, dim=1)
            for j in range(Nr):
                is_j = (ctr_agent == j)
                assign = counter_ok & is_j & can_change[:, j]
                roles[:, j] = torch.where(assign, torch.full((B,), ROLE_COUNTER, dtype=torch.int32, device=device), roles[:, j])
                lock[:, j]  = torch.where(assign, prof["lock_counter"], lock[:, j])
                can_change[:, j] = can_change[:, j] & (~assign)

        # ── Priority 5: defender ─────────────────────────────────────────
        # Assign one agent to defend when intruders exist and enough agents alive.
        need_def = (
            prof["enable_defender"]
            & bb["any_intruder"]
            & late_or_ready
            & (bb["alive_count"] >= prof["min_alive_for_defender"])
        )
        if need_def.any():
            # Pick agent closest to own flag.
            rfh_x = bb["red_flag_home"][:, 0:1]
            rfh_y = bb["red_flag_home"][:, 1:2]
            home_dist = torch.sqrt((self.red_x - rfh_x) ** 2 + (self.red_y - rfh_y) ** 2 + 1e-8)
            home_dist_m = torch.where(
                can_change & need_def[:, None],
                home_dist, home_dist.new_full((), 1e9).expand_as(home_dist)
            )
            def_agent = torch.argmin(home_dist_m, dim=1)
            for j in range(Nr):
                is_j = (def_agent == j)
                assign = need_def & is_j & can_change[:, j]
                roles[:, j] = torch.where(assign, torch.full((B,), ROLE_DEFENDER, dtype=torch.int32, device=device), roles[:, j])
                lock[:, j]  = torch.where(assign, prof["lock_defender"], lock[:, j])
                can_change[:, j] = can_change[:, j] & (~assign)

        # ── Priority 6: 2v1 wing (if 2+ agents near same enemy) ─────────
        if prof["enable_2v1"].any() and Nr >= 2 and bb["alive_count"].max() >= 2:
            # Find the nearest alive enemy for each red agent.
            dxx = self.red_x[:, :, None] - self.blue_x[:, None, :]  # [B, Nr, Nb]
            dyy = self.red_y[:, :, None] - self.blue_y[:, None, :]
            ddist = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
            big = ddist.new_full((), 1e9)
            ddist_m = torch.where(self.blue_alive[:, None, :], ddist, big.expand_as(ddist))
            nearest_enemy = torch.argmin(ddist_m, dim=2)  # [B, Nr]
            # Check if at least 2 agents share the same nearest enemy and are close.
            close_thresh = 8.0
            min_dist_val = ddist_m.min(dim=2)[0]  # [B, Nr]
            for j in range(Nr):
                if Nr < 2:
                    break
                partner_shares = torch.zeros((B,), dtype=torch.bool, device=device)
                for k in range(Nr):
                    if k == j:
                        continue
                    partner_shares = partner_shares | (
                        (nearest_enemy[:, j] == nearest_enemy[:, k])
                        & self.red_alive[:, k]
                        & (min_dist_val[:, k] < close_thresh)
                    )
                close_to_enemy = (min_dist_val[:, j] < close_thresh)
                assign = (
                    prof["enable_2v1"]
                    & late_or_ready
                    & partner_shares & close_to_enemy & can_change[:, j]
                    & self.red_alive[:, j]
                    # Only promote to wing if the agent is already in attacker mode
                    # (avoid pulling an escort or defender into a 2v1 unnecessarily).
                    & (roles[:, j] == ROLE_ATTACKER)
                )
                roles[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_2V1_WING, dtype=torch.int32, device=device),
                    roles[:, j]
                )
                lock[:, j] = torch.where(assign, prof["lock_2v1"], lock[:, j])
                can_change[:, j] = can_change[:, j] & (~assign)

        # ── Priority 7: default attacker ─────────────────────────────────
        for j in range(Nr):
            reset_to_atk = can_change[:, j] & self.red_alive[:, j]
            roles[:, j] = torch.where(
                reset_to_atk,
                torch.full((B,), ROLE_ATTACKER, dtype=torch.int32, device=device),
                roles[:, j]
            )
            lock[:, j] = torch.where(
                reset_to_atk & (roles[:, j] != self.bt_red_role[:, j]),
                prof["lock_attacker"],
                lock[:, j]
            )

        # Persist updated roles and lock counters (BT opponents only).
        bt_active = self._bt_opponent_mask()[:, None]
        roles = torch.where(bt_active, roles, self.bt_red_role)
        lock = torch.where(bt_active, lock, self.bt_role_lock_ticks)
        self.bt_red_role       = roles
        self.bt_role_lock_ticks = lock
        return roles

    # ──────────────────────────────────────────────────────────────────────
    # Route scoring: choose target position for each role
    # ──────────────────────────────────────────────────────────────────────
    def _bt_route_target(self, bb: dict, roles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (target_x, target_y) for every red agent based on assigned role.

        Route selection incorporates:
        - Distance to objective (raw).
        - Threat repulsion: tangent detour when enemies are within threat_radius.
        - Alternate lane selection: upper/lower lane preference with hysteresis via
          ``red_script_lane_sign`` (already per-episode randomized).
        - Flag carrier status: evasion routing is delegated to ``_carrier_evasion_target``.
        - Flag carrier return: alternate lane when direct path is blocked.

        Returns target_x, target_y each [B, Nr].
        """
        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = bb["idx_env"]
        prof = bb["profile"]
        max_x, max_y = bb["max_x"], bb["max_y"]
        center_y = float(self.rows) * 0.5

        target_x = torch.zeros((B, Nr), dtype=torch.float32, device=device)
        target_y = torch.zeros((B, Nr), dtype=torch.float32, device=device)

        # Precompute lane y-values for offense routing.
        lane_mid = torch.full((B,), center_y, device=device)
        lane_amp = max_y * prof["lane_amplitude_frac"]
        lane_y_pref = torch.clamp(lane_mid + self.red_script_lane_sign * lane_amp, 0.0, max_y)
        # OP6 dual-assault (Contract B): the two attackers take opposite lanes.
        # Shared red_script_lane_sign alone would put both on the same corridor.
        op6_mask = prof["bt_level"] == 6
        lane_y_opp = torch.clamp(lane_mid - self.red_script_lane_sign * lane_amp, 0.0, max_y)

        # OP9-style late-game pressure: evasion only when trailing near end of match.
        late_press = bb["late_game"] & bb["trailing"] & prof["late_game_evasion_unlock"]
        base_threat = prof["threat_radius"]
        press_threat = torch.where(late_press, base_threat, torch.zeros_like(base_threat))
        threat_radius = torch.where(prof["late_game_evasion_unlock"], press_threat, base_threat)

        for j in range(Nr):
            rx = self.red_x[:, j]
            ry = self.red_y[:, j]
            role_j = roles[:, j]  # [B]
            # Agent 0 keeps script lane; agent 1 takes the opposite corridor under OP6.
            atk_lane_y = torch.where(
                op6_mask & (j == 1),
                lane_y_opp,
                lane_y_pref,
            )

            tx = torch.zeros((B,), device=device)
            ty = torch.zeros((B,), device=device)

            # ── ATTACKER ──────────────────────────────────────────────────
            # Lane approach + tangent evasion near defenders.
            efx = bb["blue_flag_pos"][:, 0]
            efy = bb["blue_flag_pos"][:, 1]
            dist_to_flag = torch.sqrt((rx - efx) ** 2 + (ry - efy) ** 2 + 1e-8)
            # Use lane waypoint when far from flag; converge directly when close.
            # OP6: converge earlier so dual-assault reaches the flag on map_a.
            converge_dist = torch.where(op6_mask, 8.0, 4.0)
            atk_tx = torch.where(dist_to_flag > converge_dist, efx, efx)
            atk_ty = torch.where(dist_to_flag > converge_dist, atk_lane_y, efy)
            # Tangent evasion from nearest blue agent.
            atk_tx, atk_ty = self._bt_tangent_evade(
                rx, ry, atk_tx, atk_ty, threat_radius, max_x, max_y,
            )
            tx = torch.where(role_j == ROLE_ATTACKER, atk_tx, tx)
            ty = torch.where(role_j == ROLE_ATTACKER, atk_ty, ty)

            # OP6 failed-assault recovery route: midfield redeploy on red's half
            # (not own flag home). Burns assault tempo and leaves home exposed
            # while ATTACKER identity stays stable. Legal state only.
            recovering_j = (
                (self.bt_op6_recovery_ticks[:, j] > 0)
                & self.red_alive[:, j]
                & (~self.red_tagged[:, j])
                & op6_mask
            )
            mid = bb["midline"]
            # Red home is typically x > midline; stage just past midfield.
            stage_x = mid + 0.20 * (max_x - mid)
            stage_y = atk_lane_y
            tx = torch.where(recovering_j & (role_j == ROLE_ATTACKER), stage_x, tx)
            ty = torch.where(recovering_j & (role_j == ROLE_ATTACKER), stage_y, ty)

            # ── DEFENDER ─────────────────────────────────────────────────
            # If enemy carrier exists: chase carrier directly.
            # Otherwise: patrol an interception zone ahead of own flag.
            rfh_x = bb["red_flag_home"][:, 0]
            rfh_y = bb["red_flag_home"][:, 1]
            # Intercept zone ahead of own flag (profile-tuned depth).
            zone_frac = prof["defender_zone_frac"]
            zone_x = rfh_x + (bb["midline"] - rfh_x) * zone_frac
            zone_y = rfh_y
            orbit_r = prof["defender_orbit_radius"]
            phase = self.sim_step_count.to(torch.float32) * 0.12
            def_tx = torch.where(
                orbit_r > 0.0,
                torch.clamp(rfh_x + orbit_r * torch.cos(phase), 0.0, max_x),
                zone_x,
            )
            def_ty = torch.where(
                orbit_r > 0.0,
                torch.clamp(rfh_y + orbit_r * torch.sin(phase), 0.0, max_y),
                zone_y,
            )
            # Chase enemy carrier when active.
            def_tx = torch.where(bb["blue_carry_any"], bb["ec_x"], def_tx)
            def_ty = torch.where(bb["blue_carry_any"], bb["ec_y"], def_ty)
            # Chase nearest intruder when no carrier.
            intruder_x, intruder_y = self._bt_nearest_enemy_on_own(rx, ry, bb)
            def_tx = torch.where(
                (~bb["blue_carry_any"]) & bb["any_intruder"],
                intruder_x, def_tx
            )
            def_ty = torch.where(
                (~bb["blue_carry_any"]) & bb["any_intruder"],
                intruder_y, def_ty
            )
            tx = torch.where(role_j == ROLE_DEFENDER, def_tx, tx)
            ty = torch.where(role_j == ROLE_DEFENDER, def_ty, ty)

            # ── ESCORT ───────────────────────────────────────────────────
            # Interpose between own carrier and nearest enemy threat.
            rc_idx = bb["red_carrier_idx"].clamp(min=0)
            carr_x = self.red_x[idx_env, rc_idx]
            carr_y = self.red_y[idx_env, rc_idx]
            cdx = carr_x[:, None] - self.blue_x
            cdy = carr_y[:, None] - self.blue_y
            cdist_all = torch.sqrt(cdx ** 2 + cdy ** 2 + 1e-8)
            cdist_live = torch.where(self.blue_alive, cdist_all, cdist_all.new_full((), 1e9).expand_as(cdist_all))
            near_threat_idx = torch.argmin(cdist_live, dim=1)
            threat_x = self.blue_x[idx_env, near_threat_idx]
            threat_y = self.blue_y[idx_env, near_threat_idx]
            interpose_tx = torch.clamp((carr_x + threat_x) * 0.5, 0.0, max_x)
            interpose_ty = torch.clamp((carr_y + threat_y) * 0.5, 0.0, max_y)
            home_dx = bb["red_flag_home"][:, 0] - carr_x
            home_dy = bb["red_flag_home"][:, 1] - carr_y
            home_n = torch.sqrt(home_dx ** 2 + home_dy ** 2 + 1e-8)
            home_ux = home_dx / home_n
            home_uy = home_dy / home_n
            perp_x = -home_uy
            perp_y = home_ux
            parallel_tx = torch.clamp(carr_x + perp_x * 4.0, 0.0, max_x)
            parallel_ty = torch.clamp(carr_y + perp_y * 4.0, 0.0, max_y)
            no_threat = cdist_live.min(dim=1)[0] > base_threat
            escort_tx = torch.where(prof["escort_interpose"], interpose_tx, parallel_tx)
            escort_ty = torch.where(prof["escort_interpose"], interpose_ty, parallel_ty)
            escort_tx = torch.where(
                prof["escort_perpendicular_fallback"] & no_threat,
                parallel_tx,
                escort_tx,
            )
            escort_ty = torch.where(
                prof["escort_perpendicular_fallback"] & no_threat,
                parallel_ty,
                escort_ty,
            )
            tx = torch.where((role_j == ROLE_ESCORT) & bb["red_carry_any"], escort_tx, tx)
            ty = torch.where((role_j == ROLE_ESCORT) & bb["red_carry_any"], escort_ty, ty)
            # OP6 extraction screener: engage the *locked* complementary threat
            # (not generic nearest). Carrier already owns evasion-nearest.
            # Pre-pickup: same geometry with the enemy flag as virtual carrier.
            op6_extract = (
                (prof["bt_level"] == 6)
                & (self.bt_op6_extract_ticks > 0)
                & bb["red_carry_any"]
            )
            op6_pre = (
                (prof["bt_level"] == 6)
                & (self.bt_op6_preengage_ticks > 0)
                & (~bb["red_carry_any"])
            )
            op6_screen = op6_extract | op6_pre
            if bool(op6_screen.any().item()):
                scr_idx = self.bt_op6_extract_screener_threat.clamp(min=0)
                has_scr = self.bt_op6_extract_screener_threat >= 0
                threat_x = torch.where(
                    has_scr,
                    self.blue_x[idx_env, scr_idx],
                    threat_x,
                )
                threat_y = torch.where(
                    has_scr,
                    self.blue_y[idx_env, scr_idx],
                    threat_y,
                )
                # Virtual carrier = real carrier, else enemy flag (preengage).
                vc_x = torch.where(bb["red_carry_any"], carr_x, bb["blue_flag_pos"][:, 0])
                vc_y = torch.where(bb["red_carry_any"], carr_y, bb["blue_flag_pos"][:, 1])
                # Post-pickup: keep existing interpose blend (frozen engage geometry).
                screen_tx = 0.55 * vc_x + 0.45 * threat_x
                screen_ty = 0.55 * vc_y + 0.45 * threat_y
                home_dx = bb["red_flag_home"][:, 0] - vc_x
                home_dy = bb["red_flag_home"][:, 1] - vc_y
                home_n = torch.sqrt(home_dx ** 2 + home_dy ** 2 + 1e-8)
                behind_x = vc_x - 2.0 * home_dx / home_n
                behind_y = vc_y - 2.0 * home_dy / home_n
                screen_tx = 0.6 * screen_tx + 0.4 * behind_x
                screen_ty = 0.6 * screen_ty + 0.4 * behind_y
                # Pre-pickup only: meet the locked threat on the return corridor
                # (projected intercept), do not chase its current position.
                if bool(op6_pre.any().item()):
                    ix, iy = self._bt_op6_projected_intercept_xy(
                        bb, vc_x, vc_y, scr_idx, max_x, max_y
                    )
                    screen_tx = torch.where(op6_pre, ix, screen_tx)
                    screen_ty = torch.where(op6_pre, iy, screen_ty)
                screen_tx = torch.clamp(screen_tx, 0.0, max_x)
                screen_ty = torch.clamp(screen_ty, 0.0, max_y)
                use_screen = op6_screen & (role_j == ROLE_ESCORT)
                tx = torch.where(use_screen, screen_tx, tx)
                ty = torch.where(use_screen, screen_ty, ty)
            # If escort but no carrier exists, fall back to attacker target
            # unless OP6 preengage already owns the screener route.
            no_carr_escort = (role_j == ROLE_ESCORT) & (~bb["red_carry_any"])
            if hasattr(self, "bt_op6_preengage_ticks"):
                no_carr_escort = no_carr_escort & ~(
                    (prof["bt_level"] == 6) & (self.bt_op6_preengage_ticks > 0)
                )
            tx = torch.where(no_carr_escort, atk_tx, tx)
            ty = torch.where(no_carr_escort, atk_ty, ty)

            # ── INTERCEPTOR ──────────────────────────────────────────────
            block_frac = (
                prof["intercept_block_base"]
                + prof["intercept_block_trailing_bonus"] * bb["trailing"].float()
            )
            bx = bb["ec_x"] + (bb["blue_flag_home"][:, 0] - bb["ec_x"]) * block_frac
            by_ = bb["ec_y"] + (bb["blue_flag_home"][:, 1] - bb["ec_y"]) * block_frac
            intc_tx = torch.clamp(bx, 0.0, max_x)
            intc_ty = torch.clamp(by_, 0.0, max_y)
            tx = torch.where((role_j == ROLE_INTERCEPTOR) & bb["blue_carry_any"], intc_tx, tx)
            ty = torch.where((role_j == ROLE_INTERCEPTOR) & bb["blue_carry_any"], intc_ty, ty)
            tx = torch.where((role_j == ROLE_INTERCEPTOR) & (~bb["blue_carry_any"]), def_tx, tx)
            ty = torch.where((role_j == ROLE_INTERCEPTOR) & (~bb["blue_carry_any"]), def_ty, ty)
            # OP6 mutual-carry denial: chase/cut the blue carrier for a tag or
            # forced detour (closer than the generic block fraction).
            if hasattr(self, "bt_op6_race_ticks"):
                op6_race = (
                    (prof["bt_level"] == 6)
                    & (self.bt_op6_race_ticks > 0)
                    & bb["blue_carry_any"]
                )
                if bool(op6_race.any().item()):
                    tgt = self.bt_op6_race_target_idx.clamp(min=0)
                    has_tgt = self.bt_op6_race_target_idx >= 0
                    chase_x = torch.where(
                        has_tgt, self.blue_x[idx_env, tgt], bb["ec_x"]
                    )
                    chase_y = torch.where(
                        has_tgt, self.blue_y[idx_env, tgt], bb["ec_y"]
                    )
                    # Cut tightly on the blue carrier (tag / forced drop).
                    # Larger lead fractions delayed contact without delaying score.
                    cut_x = chase_x + (bb["blue_flag_home"][:, 0] - chase_x) * 0.08
                    cut_y = chase_y + (bb["blue_flag_home"][:, 1] - chase_y) * 0.08
                    cut_x = torch.clamp(cut_x, 0.0, max_x)
                    cut_y = torch.clamp(cut_y, 0.0, max_y)
                    use_race = op6_race & (role_j == ROLE_INTERCEPTOR)
                    tx = torch.where(use_race, cut_x, tx)
                    ty = torch.where(use_race, cut_y, ty)

            # ── FLAG_RETRIEVER ───────────────────────────────────────────
            retr_tx = bb["red_flag_pos"][:, 0]
            retr_ty = bb["red_flag_pos"][:, 1]
            tx = torch.where(role_j == ROLE_FLAG_RETR, retr_tx, tx)
            ty = torch.where(role_j == ROLE_FLAG_RETR, retr_ty, ty)

            # ── COUNTER-CAPTURE ──────────────────────────────────────────
            # Deliberately ignore the enemy carrier; go capture enemy flag instead.
            # Use alternate lane (opposite to script_lane_sign) to avoid colliding
            # with defenders on the direct route.
            alt_lane_y = torch.clamp(lane_mid - self.red_script_lane_sign * lane_amp, 0.0, max_y)
            ctr_tx = efx
            ctr_ty = torch.where(dist_to_flag > 4.0, alt_lane_y, efy)
            ctr_tx, ctr_ty = self._bt_tangent_evade(rx, ry, ctr_tx, ctr_ty, threat_radius, max_x, max_y)
            tx = torch.where(role_j == ROLE_COUNTER, ctr_tx, tx)
            ty = torch.where(role_j == ROLE_COUNTER, ctr_ty, ty)

            # ── 2V1 WING ─────────────────────────────────────────────────
            # Orbit 45° off the nearest enemy at a tight radius to create
            # a two-pronged pressure angle that's harder to evade.
            dxx = rx[:, None] - self.blue_x
            dyy = ry[:, None] - self.blue_y
            ddist_j = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
            ddist_jm = torch.where(self.blue_alive, ddist_j, ddist_j.new_full((), 1e9).expand_as(ddist_j))
            near_e_idx = torch.argmin(ddist_jm, dim=1)
            nex = self.blue_x[idx_env, near_e_idx]
            ney = self.blue_y[idx_env, near_e_idx]
            angle_offset = torch.full((B,), 0.785, device=device)  # ~45 degrees
            phase_j = self.sim_step_count.to(torch.float32) * 0.15
            orbit_r = torch.full((B,), 3.0, device=device)
            wing_tx = torch.clamp(nex + orbit_r * torch.cos(phase_j + angle_offset), 0.0, max_x)
            wing_ty = torch.clamp(ney + orbit_r * torch.sin(phase_j + angle_offset), 0.0, max_y)
            tx = torch.where(role_j == ROLE_2V1_WING, wing_tx, tx)
            ty = torch.where(role_j == ROLE_2V1_WING, wing_ty, ty)

            target_x[:, j] = tx
            target_y[:, j] = ty

        # OP8 Contract A round-2: during formation opening, stage at midfield
        # carrier/protector rally points instead of dual-rushing the blue flag.
        # Home defense stays incomplete (no DEFENDER/INTERCEPTOR), but reds are
        # not both deep in blue territory — that mutual race caused 0-0 timeouts
        # in round-1 micro-gates. OP8-only; OP9 never enters.
        op8_opening = (
            (prof["bt_level"] == 8)
            & (self.sim_step_count.to(torch.int32) < int(self._OP8_FORMATION_OPENING_STEPS))
            & (~bb["blue_carry_any"])
        )
        if op8_opening.any():
            rfh_x = bb["red_flag_home"][:, 0]
            midline = bb["midline"]
            # Stage just on the red side of midfield, split upper/lower lanes.
            stage_x = rfh_x + (midline - rfh_x) * 0.72
            stage_y0 = torch.full((B,), float(self.rows) * 0.28, device=device)
            stage_y1 = torch.full((B,), float(self.rows) * 0.72, device=device)
            opening_slots = op8_opening[:, None] & self.red_alive & (~self.red_tagged)
            target_x = torch.where(opening_slots, stage_x[:, None].expand(B, Nr), target_x)
            for j in range(Nr):
                stage_y = stage_y0 if j % 2 == 0 else stage_y1
                target_y[:, j] = torch.where(opening_slots[:, j], stage_y, target_y[:, j])
            target_x = torch.clamp(target_x, 0.0, max_x)
            target_y = torch.clamp(target_y, 0.0, max_y)

        # ── Carrier evasion override (any role carrying a flag) ──────────
        # Carriers always use multi-threat tangent routing toward home;
        # this overrides whatever role-target was computed above.
        if self.red_carrying.any():
            evade_tx, evade_ty = self._carrier_evasion_target(
                self.red_x, self.red_y,
                bb["red_flag_home"][:, 0], bb["red_flag_home"][:, 1],
                self.blue_x, self.blue_y, self.blue_alive,
                self.red_carrying,
                side="red",
            )
            target_x = torch.where(self.red_carrying, evade_tx, target_x)
            target_y = torch.where(self.red_carrying, evade_ty, target_y)

        # OP6 extraction: shortest stable home route (no evasion churn).
        # Trace dominant failure included carrier_route_detour; direct home
        # while extract ticks are active. OP6-only; recovery untouched.
        if bool(getattr(type(self), "_OP6_EXTRACTION_ENABLED", True)):
            op6_direct = (
                (prof["bt_level"] == 6)[:, None]
                & (self.bt_op6_extract_ticks > 0)[:, None]
                & self.red_carrying
            )
            home_tx = bb["red_flag_home"][:, 0:1].expand(B, Nr)
            home_ty = bb["red_flag_home"][:, 1:2].expand(B, Nr)
            target_x = torch.where(op6_direct, home_tx, target_x)
            target_y = torch.where(op6_direct, home_ty, target_y)
            # Screen-break: when a blue non-carrier sits on the return corridor
            # and blue has no near-home defender, peel to the safer of two
            # corridors and send the partner to engage/draw the blocker.
            # Freezes latch/recovery; geometry only (no blue style ID).
            target_x, target_y = self._bt_op6_screen_break_routes(
                bb, prof, roles, target_x, target_y, max_x, max_y
            )

        return target_x, target_y

    def _bt_op6_projected_intercept_xy(
        self,
        bb: dict,
        vc_x: torch.Tensor,
        vc_y: torch.Tensor,
        threat_idx: torch.Tensor,
        max_x: float,
        max_y: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Meet-the-blocker waypoint on the return corridor (preengage only).

        Projects the locked blue threat onto the virtual-carrier→red-home path,
        then leads slightly toward home so the screener arrives ahead of a chase.
        Uses closing velocity when adaptive prev-blue buffers are valid.
        """
        idx_env = bb["idx_env"]
        ti = threat_idx.clamp(min=0).to(torch.int64)
        bx = self.blue_x[idx_env, ti]
        by = self.blue_y[idx_env, ti]
        home_x = bb["red_flag_home"][:, 0]
        home_y = bb["red_flag_home"][:, 1]
        vx = home_x - vc_x
        vy = home_y - vc_y
        vv = vx * vx + vy * vy + 1e-8
        wx = bx - vc_x
        wy = by - vc_y
        t = (wx * vx + wy * vy) / vv
        # Lead along the corridor so we meet rather than trail the blocker.
        lead = torch.full_like(t, 0.12)
        if hasattr(self, "bt_adapt_prev_blue_valid"):
            prev_ok = self.bt_adapt_prev_blue_valid
            bvx = bx - self.bt_adapt_prev_blue_x[idx_env, ti]
            bvy = by - self.bt_adapt_prev_blue_y[idx_env, ti]
            # Component of blue motion along the return path (toward home > 0).
            along = (bvx * vx + bvy * vy) / torch.sqrt(vv)
            # Extra lead when the threat is already charging the corridor.
            lead_dyn = lead + 0.08 * torch.clamp(along, 0.0, 2.0)
            lead = torch.where(prev_ok, lead_dyn, lead)
        t_meet = torch.clamp(t + lead, 0.08, 0.92)
        ix = vc_x + t_meet * vx
        iy = vc_y + t_meet * vy
        return torch.clamp(ix, 0.0, max_x), torch.clamp(iy, 0.0, max_y)

    def _bt_op6_projected_threat_danger(
        self,
        bb: dict,
        carr_x: torch.Tensor,
        carr_y: torch.Tensor,
    ) -> torch.Tensor:
        """Projected interception danger per blue vs the returning red carrier.

        Higher = more dangerous. Uses legal geometry only (no style IDs):
        path projection, ahead/behind, distance to carrier / red home, closing.
        """
        home_x = bb["red_flag_home"][:, 0]
        home_y = bb["red_flag_home"][:, 1]
        mid = bb["midline"]
        vx = home_x - carr_x
        vy = home_y - carr_y
        vv = vx * vx + vy * vy + 1e-8
        wx = self.blue_x - carr_x[:, None]
        wy = self.blue_y - carr_y[:, None]
        t = (wx * vx[:, None] + wy * vy[:, None]) / vv[:, None]
        proj_x = carr_x[:, None] + t * vx[:, None]
        proj_y = carr_y[:, None] + t * vy[:, None]
        lat = torch.sqrt(
            (self.blue_x - proj_x) ** 2 + (self.blue_y - proj_y) ** 2 + 1e-8
        )
        dist_carr = torch.sqrt(wx * wx + wy * wy + 1e-8)
        dist_home = torch.sqrt(
            (self.blue_x - home_x[:, None]) ** 2
            + (self.blue_y - home_y[:, None]) ** 2
            + 1e-8
        )
        # Ahead on return corridor (t in path segment, small lateral).
        on_corridor = (t > 0.0) & (t < 1.05) & (lat < 6.0)
        ahead_score = torch.clamp(t, 0.0, 1.0) * on_corridor.float()
        ahead_score = ahead_score + 0.5 * (
            (self.blue_x > mid) & (self.blue_x > carr_x[:, None])
        ).float()

        closing = torch.zeros_like(dist_carr)
        if hasattr(self, "bt_adapt_prev_blue_valid"):
            prev_ok = self.bt_adapt_prev_blue_valid[:, None]
            bvx = self.blue_x - self.bt_adapt_prev_blue_x
            bvy = self.blue_y - self.bt_adapt_prev_blue_y
            # Positive = blue stepping toward carrier.
            closing = torch.where(
                prev_ok,
                (bvx * (-wx) + bvy * (-wy)) / dist_carr,
                closing,
            )
        approaching = closing > 0.05
        eta = torch.where(
            approaching,
            dist_carr / closing.clamp(min=0.05),
            dist_carr / 0.15 + 20.0,
        )
        # Path-arrival proxy: lateral gap / modest approach rate.
        eta_path = lat / 0.20 + torch.clamp(1.0 - t, min=0.0) * 8.0

        danger = (
            3.0 / (eta + 1.0)
            + 2.0 / (eta_path + 1.0)
            + 2.0 * ahead_score
            + 1.5 / (dist_carr + 1.0)
            + 1.0 / (dist_home + 1.0)
            + 0.5 * torch.clamp(closing, 0.0, 2.0)
        )
        live = self.blue_alive
        if hasattr(self, "blue_tagged"):
            live = live & (~self.blue_tagged)
        return torch.where(live, danger, danger.new_full((), -1e9).expand_as(danger))

    def _bt_op6_lock_extract_threats(
        self,
        bb: dict,
        newly_armed: torch.Tensor,
        carr_x: torch.Tensor | None = None,
        carr_y: torch.Tensor | None = None,
    ) -> None:
        """Lock complementary carrier/screener blue threats at extract arm.

        Carrier threat = evasion-nearest alive blue (what carrier routing already
        responds to). Screener = the other alive blue when ≥2 are up (hard
        complementary 2v2 contract — no duplicated targeting). Blues that are
        themselves carrying still count as threats (excluding them caused C1
        duplicates when one blue held red's flag at arm time).

        Optional ``carr_x`` / ``carr_y`` override the reference point (used by
        pre-pickup arming where the virtual carrier is the enemy flag).
        """
        idx_env = bb["idx_env"]
        if carr_x is None or carr_y is None:
            rc = bb["red_carrier_idx"].clamp(min=0)
            carr_x = self.red_x[idx_env, rc]
            carr_y = self.red_y[idx_env, rc]
        # Tagged-out agents are not threats; carrying blues still are.
        live = self.blue_alive & (~self.blue_tagged)
        dist = torch.sqrt(
            (self.blue_x - carr_x[:, None]) ** 2
            + (self.blue_y - carr_y[:, None]) ** 2
            + 1e-8
        )
        dist_m = torch.where(live, dist, dist.new_full((), 1e9).expand_as(dist))
        carrier_threat = torch.argmin(dist_m, dim=1).to(torch.int32)
        any_live = live.any(dim=1)
        carrier_threat = torch.where(
            any_live,
            carrier_threat,
            torch.full_like(carrier_threat, -1),
        )

        danger = self._bt_op6_projected_threat_danger(bb, carr_x, carr_y)
        # Prefer danger among live agents; carrying blues keep their danger.
        danger = torch.where(live, danger, danger.new_full((), -1e9).expand_as(danger))
        nb = danger.shape[1]
        threat_ids = torch.arange(nb, device=danger.device)[None, :]
        danger_scr = torch.where(
            threat_ids == carrier_threat[:, None],
            danger.new_full((), -1e9).expand_as(danger),
            danger,
        )
        screener_threat = torch.argmax(danger_scr, dim=1).to(torch.int32)
        scr_ok = (danger_scr.max(dim=1).values > -1e8) & any_live
        n_live = live.to(torch.int32).sum(dim=1)
        screener_threat = torch.where(
            scr_ok,
            screener_threat,
            torch.where(any_live, carrier_threat, torch.full_like(screener_threat, -1)),
        )
        # Hard complementary when ≥2 live blues (2v2: no duplicated targeting).
        if nb >= 2:
            alt = torch.where(
                carrier_threat == 0,
                torch.ones_like(carrier_threat),
                torch.zeros_like(carrier_threat),
            )
            alt_live = live.gather(1, alt.clamp(min=0)[:, None].to(torch.int64)).squeeze(1)
            force_alt = (n_live >= 2) & (carrier_threat >= 0) & alt_live
            screener_threat = torch.where(force_alt, alt.to(torch.int32), screener_threat)

        self.bt_op6_extract_carrier_threat = torch.where(
            newly_armed, carrier_threat, self.bt_op6_extract_carrier_threat
        )
        self.bt_op6_extract_screener_threat = torch.where(
            newly_armed, screener_threat, self.bt_op6_extract_screener_threat
        )

    def _bt_op6_screen_break_routes(
        self,
        bb: dict,
        prof: Dict[str, torch.Tensor],
        roles: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        max_x: float,
        max_y: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OP6-only corridor peel + complementary-threat screener engage.

        Carrier peel geometry is unchanged (away from on-path / ahead blocker).
        Screener engages the *locked* complementary threat, not the peel blue,
        so dual RUSH interceptors are not both covered by one assignment.
        """
        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = bb["idx_env"]
        op6 = prof["bt_level"] == 6
        extract = (self.bt_op6_extract_ticks > 0) & op6 & bb["red_carry_any"]
        if not bool(extract.any().item()):
            return target_x, target_y

        mid = bb["midline"]
        bhome_x = self.blue_flag_home[:, 0:1]
        near_bhome = (self.blue_x - bhome_x).abs() <= 6.0
        blue_home_def = (
            self.blue_alive
            & (~self.blue_carrying)
            & (self.blue_x < mid)
            & near_bhome
        ).any(dim=1)
        no_home_def = ~blue_home_def

        rc = bb["red_carrier_idx"].clamp(min=0)
        carr_x = self.red_x[idx_env, rc]
        carr_y = self.red_y[idx_env, rc]
        home_x = bb["red_flag_home"][:, 0]
        home_y = bb["red_flag_home"][:, 1]
        vx = home_x - carr_x
        vy = home_y - carr_y
        vv = vx * vx + vy * vy + 1e-8
        dist_carr_home = torch.sqrt(vv)

        # (1) On-segment blocker (lateral band around return line).
        wx = self.blue_x - carr_x[:, None]
        wy = self.blue_y - carr_y[:, None]
        t = (wx * vx[:, None] + wy * vy[:, None]) / vv[:, None]
        proj_x = carr_x[:, None] + t * vx[:, None]
        proj_y = carr_y[:, None] + t * vy[:, None]
        lat = torch.sqrt((self.blue_x - proj_x) ** 2 + (self.blue_y - proj_y) ** 2 + 1e-8)
        on_seg = (t > 0.05) & (t < 0.98) & (lat < 5.0)

        # (2) Ahead-near-home blocker.
        dist_b_home = torch.sqrt(
            (self.blue_x - home_x[:, None]) ** 2 + (self.blue_y - home_y[:, None]) ** 2 + 1e-8
        )
        ahead = (
            (self.blue_x > mid)
            & (self.blue_x > carr_x[:, None])
            & (dist_b_home < (dist_carr_home[:, None] + 1.0))
            & ((self.blue_y - home_y[:, None]).abs() < 6.0)
        )

        eligible = (
            self.blue_alive
            & (~self.blue_carrying)
            & (on_seg | ahead)
        )
        # Carrier peel still uses nearest eligible (route unchanged), except
        # during mutual-carry race denial: then go straight home (scoring-race
        # tempo). Peel geometry itself is frozen for non-race extract returns.
        d_carr = torch.sqrt(
            (self.blue_x - carr_x[:, None]) ** 2 + (self.blue_y - carr_y[:, None]) ** 2 + 1e-8
        )
        d_m = torch.where(eligible, d_carr, d_carr.new_full((), 1e9).expand_as(d_carr))
        peel_idx = torch.argmin(d_m, dim=1)
        race_active = torch.zeros((B,), dtype=torch.bool, device=device)
        if hasattr(self, "bt_op6_race_ticks"):
            race_active = (self.bt_op6_race_ticks > 0) & op6 & bb["red_carry_any"]
        has_peel = (
            eligible.any(dim=1) & extract & no_home_def & (~race_active)
        )
        peel_x = self.blue_x[idx_env, peel_idx]
        peel_y = self.blue_y[idx_env, peel_idx]

        amp = float(max(3.0, min(7.0, max_y * 0.30)))
        y_hi = torch.clamp(home_y + amp, 0.0, max_y)
        y_lo = torch.clamp(home_y - amp, 0.0, max_y)
        use_hi = (y_hi - peel_y).abs() >= (y_lo - peel_y).abs()
        safe_y = torch.where(use_hi, y_hi, y_lo)
        c_tx = home_x
        c_ty = torch.where(dist_carr_home > 2.5, safe_y, home_y)

        # Screener engages locked complementary threat (not peel nearest).
        scr_idx = self.bt_op6_extract_screener_threat.clamp(min=0)
        has_scr = (self.bt_op6_extract_screener_threat >= 0) & extract & no_home_def
        # Fall back to non-peel eligible / any live if lock missing this frame.
        if bool((extract & (~has_scr)).any().item()):
            danger = self._bt_op6_projected_threat_danger(bb, carr_x, carr_y)
            danger = torch.where(
                torch.arange(danger.shape[1], device=device)[None, :] == peel_idx[:, None],
                danger.new_full((), -1e9).expand_as(danger),
                danger,
            )
            fb = torch.argmax(danger, dim=1)
            scr_idx = torch.where(has_scr, scr_idx, fb)
            has_scr = has_scr | (extract & no_home_def & (danger.max(dim=1).values > -1e8))

        bx = self.blue_x[idx_env, scr_idx]
        by_ = self.blue_y[idx_env, scr_idx]
        engage_x = torch.clamp(bx, 0.0, max_x)
        engage_y = torch.clamp(by_, 0.0, max_y)

        agent_ids = torch.arange(Nr, device=device)[None, :]
        is_carrier = agent_ids == rc[:, None]
        peel_slots = has_peel[:, None] & self.red_alive & (~self.red_tagged)
        scr_slots = has_scr[:, None] & self.red_alive & (~self.red_tagged)
        target_x = torch.where(
            peel_slots & is_carrier & self.red_carrying,
            c_tx[:, None].expand(B, Nr),
            target_x,
        )
        target_y = torch.where(
            peel_slots & is_carrier & self.red_carrying,
            c_ty[:, None].expand(B, Nr),
            target_y,
        )
        target_x = torch.where(
            scr_slots & (~is_carrier) & (roles == ROLE_ESCORT),
            engage_x[:, None].expand(B, Nr),
            target_x,
        )
        target_y = torch.where(
            scr_slots & (~is_carrier) & (roles == ROLE_ESCORT),
            engage_y[:, None].expand(B, Nr),
            target_y,
        )
        return target_x, target_y

    # ──────────────────────────────────────────────────────────────────────
    # Deliberate mine placement (BT decides whether; routes decide where)
    # ──────────────────────────────────────────────────────────────────────
    def _bt_mine_site_clear(
        self,
        mine_x: torch.Tensor,
        mine_y: torch.Tensor,
        min_spacing: torch.Tensor,
    ) -> torch.Tensor:
        """True when proposed site is not within ``min_spacing`` of an active mine."""
        amx = self.red_mine_x
        amy = self.red_mine_y
        active = self.red_mine_active
        dx = mine_x[:, :, None] - amx[:, None, :]
        dy = mine_y[:, :, None] - amy[:, None, :]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        too_close = (dist < min_spacing[:, None, None]) & active[:, None, :]
        return ~too_close.any(dim=2)

    def _bt_plan_mines(
        self,
        bb: dict,
        roles: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return proposed mine sites and per-agent placement intent."""
        B, Nr = self.B, self.Nr
        device = self.device
        enable = prof["enable_mines"][:, None]
        if not bool(enable.any().item()):
            zeros_f = torch.zeros((B, Nr), dtype=torch.float32, device=device)
            zeros_b = torch.zeros((B, Nr), dtype=torch.bool, device=device)
            return zeros_f, zeros_f, zeros_b

        max_x, max_y = bb["max_x"], bb["max_y"]
        has_free = (~self.red_mine_active).any(dim=1, keepdim=True)
        has_charge = self.red_mine_charges > 0
        not_carrier = ~self.red_carrying
        alive = self.red_alive & (~self.red_tagged)

        rfh_x = bb["red_flag_home"][:, 0]
        rfh_y = bb["red_flag_home"][:, 1]
        midline_x = float(bb["midline"])
        lane_frac = prof["mine_defender_lane_frac"][:, None]
        def_mine_x = torch.clamp(
            rfh_x[:, None] + (midline_x - rfh_x[:, None]) * lane_frac,
            0.0,
            max_x,
        )
        def_mine_y = rfh_y[:, None].expand(B, Nr)

        block_frac = (
            prof["intercept_block_base"][:, None]
            + prof["intercept_block_trailing_bonus"][:, None] * bb["trailing"].float()[:, None]
        )
        int_mine_x = torch.clamp(
            bb["ec_x"][:, None]
            + (bb["blue_flag_home"][:, 0:1] - bb["ec_x"][:, None]) * block_frac,
            0.0,
            max_x,
        )
        int_mine_y = torch.clamp(
            bb["ec_y"][:, None]
            + (bb["blue_flag_home"][:, 1:2] - bb["ec_y"][:, None]) * block_frac,
            0.0,
            max_y,
        )

        is_def = roles == ROLE_DEFENDER
        is_int = roles == ROLE_INTERCEPTOR
        mine_x = torch.zeros((B, Nr), dtype=torch.float32, device=device)
        mine_y = torch.zeros((B, Nr), dtype=torch.float32, device=device)
        mine_x = torch.where(is_def, def_mine_x, mine_x)
        mine_y = torch.where(is_def, def_mine_y, mine_y)
        mine_x = torch.where(is_int, int_mine_x, mine_x)
        mine_y = torch.where(is_int, int_mine_y, mine_y)

        def_want = is_def & (bb["any_intruder"][:, None] | bb["blue_carry_any"][:, None])
        dist_ec = torch.sqrt(
            (self.red_x - bb["ec_x"][:, None]) ** 2
            + (self.red_y - bb["ec_y"][:, None]) ** 2
            + 1e-8
        )
        int_want = (
            is_int
            & bb["blue_carry_any"][:, None]
            & bb["intercept_feasible_per_agent"]
            & (dist_ec > 4.0)
        )
        role_want = def_want | int_want

        forbidden = (
            (roles == ROLE_ESCORT)
            | (roles == ROLE_FLAG_RETR)
            | (roles == ROLE_COUNTER)
            | (roles == ROLE_ATTACKER)
            | (roles == ROLE_2V1_WING)
        )

        want = (
            enable
            & has_free
            & has_charge
            & not_carrier
            & alive
            & role_want
            & (~forbidden)
        )
        want = want & self._bt_mine_site_clear(mine_x, mine_y, prof["mine_min_spacing"])

        cooldown = prof["mine_cooldown_steps"][:, None].clamp(min=1)
        lead = prof["mine_approach_lead_steps"][:, None]
        step_mod = torch.remainder(
            self.sim_step_count[:, None].expand(B, Nr),
            cooldown,
        )
        want = want & (step_mod < lead)

        prev_want = self.bt_want_mine[:, :Nr]
        lock = self.bt_mine_lock_ticks[:, :Nr]
        lock = (lock - 1).clamp(min=0)
        persist = (
            prev_want
            & (lock > 0)
            & has_charge
            & not_carrier
            & alive
            & (~forbidden)
        )
        want = want | persist
        new_want = want & (~prev_want)
        lock = torch.where(new_want, prof["mine_lock_ticks"][:, None], lock)
        lock = torch.where(~want, torch.zeros_like(lock), lock)
        self.bt_mine_lock_ticks[:, :Nr] = lock

        return mine_x, mine_y, want

    def _bt_apply_mine_routes(
        self,
        tx: torch.Tensor,
        ty: torch.Tensor,
        mine_x: torch.Tensor,
        mine_y: torch.Tensor,
        want: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route agents toward mine sites before issuing PLACE_MINE at arrival."""
        place_r = prof["mine_place_radius"][:, None]
        dist = torch.sqrt((self.red_x - mine_x) ** 2 + (self.red_y - mine_y) ** 2 + 1e-8)
        route_mine = want & (dist > place_r)
        tx = torch.where(route_mine, mine_x, tx)
        ty = torch.where(route_mine, mine_y, ty)
        return tx, ty

    def _bt_resolved_profile_tensors(self) -> Dict[str, torch.Tensor]:
        """Canonical profiles, with a searcher overlay if present.

        ``_bt_profile_override`` accepts either form:

        * ``None``        -> no overlay (bit-identical to the pre-searcher path)
        * a ``BTProfile`` -> broadcast to all B envs (the original behaviour,
          used by the searcher and by every confirmation run to date)
        * a ``list``      -> one entry per env, each ``BTProfile`` or ``None``

        The per-env list form exists because a balanced A/B training mixture
        needs pole A's overlay on the OP6 envs ONLY. Broadcasting a single
        override across a mixed-opponent batch would apply OP6-derived overlay
        knobs to the OP7 envs, silently corrupting the B pole.
        ``build_profile_tensors`` already accepts a per-env list, so this is
        plumbing, not a behaviour change.
        """
        ov = getattr(self, "_bt_profile_override", None)
        if ov is None:
            overrides = None
        elif isinstance(ov, (list, tuple)):
            overrides = list(ov)
            if len(overrides) != int(self.B):
                raise ValueError(
                    f"_bt_profile_override list has {len(overrides)} entries, "
                    f"expected one per env (B={int(self.B)})")
        else:
            overrides = [ov] * int(self.B)
        return build_profile_tensors(
            self._opponent_key,
            device=self.device,
            batch_size=self.B,
            overrides=overrides,
        )

    def _bt_scripted_red_macros(self) -> Optional[torch.Tensor]:
        """Emit PLACE_MINE macros for BT mine opponents; None when inactive."""
        prof = self._bt_resolved_profile_tensors()
        bt_mine_envs = self._bt_opponent_mask() & prof["enable_mines"]
        if not bool(bt_mine_envs.any().item()):
            return None

        B, Nr = self.B, self.Nr
        device = self.device
        macro = torch.full(
            (B, Nr),
            int(MacroAction.GO_TO),
            dtype=torch.int64,
            device=device,
        )

        bt_active = self._bt_opponent_mask()
        step_50 = (self.sim_step_count % 50) == 0
        legacy_place = (~bt_active) & step_50 & (self.red_mine_charges[:, 0] > 0)
        macro[:, 0] = torch.where(
            legacy_place,
            torch.full((B,), int(MacroAction.PLACE_MINE), dtype=torch.int64, device=device),
            macro[:, 0],
        )

        enable_env = prof["enable_mines"][:, None]
        dist = torch.sqrt(
            (self.red_x - self.bt_mine_target_x[:, :Nr]) ** 2
            + (self.red_y - self.bt_mine_target_y[:, :Nr]) ** 2
            + 1e-8
        )
        at_site = (
            enable_env
            & bt_active[:, None]
            & self.bt_want_mine[:, :Nr]
            & (dist <= prof["mine_place_radius"][:, None])
            & (self.red_mine_charges > 0)
        )
        macro = torch.where(
            at_site,
            torch.full_like(macro, int(MacroAction.PLACE_MINE)),
            macro,
        )
        self.bt_tel_mine_attempts += (
            at_site.any(dim=1).to(torch.int32) & bt_mine_envs.to(torch.int32)
        )
        return macro

    # ──────────────────────────────────────────────────────────────────────
    # Helper: tangent evasion from nearest blue agent
    # ──────────────────────────────────────────────────────────────────────
    def _bt_tangent_evade(
        self,
        rx: torch.Tensor,
        ry: torch.Tensor,
        goal_x: torch.Tensor,
        goal_y: torch.Tensor,
        threat_radius,
        max_x: float,
        max_y: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a tangent-evasion waypoint for a single agent column.

        When the nearest blue agent is within threat_radius the waypoint curves
        perpendicular to the goal direction; further away, returns goal unchanged.
        """
        dxx = rx[:, None] - self.blue_x
        dyy = ry[:, None] - self.blue_y
        ddist = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
        ddist_m = torch.where(self.blue_alive, ddist, ddist.new_full((), 1e9).expand_as(ddist))
        min_dist = ddist_m.min(dim=1)[0].clamp(min=1e-6)
        if not isinstance(threat_radius, torch.Tensor):
            threat_radius = torch.full((rx.shape[0],), float(threat_radius), device=rx.device, dtype=rx.dtype)
        in_range = min_dist < threat_radius
        safe_radius = threat_radius.clamp(min=1e-3)

        goal_dx = goal_x - rx
        goal_dy = goal_y - ry
        goal_n  = torch.sqrt(goal_dx ** 2 + goal_dy ** 2 + 1e-8)
        goal_ux = goal_dx / goal_n
        goal_uy = goal_dy / goal_n
        # Tangent perpendicular to goal direction.
        tan_x = -goal_uy
        tan_y =  goal_ux
        repulsion = torch.pow(
            torch.clamp(threat_radius - min_dist, min=0.0) / safe_radius, 2.0
        ) * in_range.float()
        center_y = float(self.rows) * 0.5
        side_bias = torch.where(ry > center_y, 1.0, -1.0)

        evade_x = rx + goal_ux * 2.0 + tan_x * side_bias * repulsion * 6.0
        evade_y = ry + goal_uy * 2.0 + tan_y * side_bias * repulsion * 6.0
        evade_x = torch.clamp(evade_x, 0.0, max_x)
        evade_y = torch.clamp(evade_y, 0.0, max_y)
        out_x = torch.where(in_range, evade_x, goal_x)
        out_y = torch.where(in_range, evade_y, goal_y)
        return out_x, out_y

    # ──────────────────────────────────────────────────────────────────────
    # Helper: nearest intruding enemy on own half
    # ──────────────────────────────────────────────────────────────────────
    def _bt_nearest_enemy_on_own(
        self,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bb: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return position of the nearest enemy on own half for a given agent row."""
        dxx = rx[:, None] - self.blue_x
        dyy = ry[:, None] - self.blue_y
        ddist = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
        big = ddist.new_full((), 1e9)
        ddist_m = torch.where(bb["enemy_on_own"], ddist, big.expand_as(ddist))
        near_idx = torch.argmin(ddist_m, dim=1)
        idx_env  = bb["idx_env"]
        intruder_x = self.blue_x[idx_env, near_idx]
        intruder_y = self.blue_y[idx_env, near_idx]
        return intruder_x, intruder_y

    # ──────────────────────────────────────────────────────────────────────
    # Telemetry update
    # ──────────────────────────────────────────────────────────────────────
    def _bt_update_telemetry(self, bb: dict, roles: torch.Tensor, prev_roles: torch.Tensor) -> None:
        """
        Increment per-env counters for diagnostic observation of BT branches.

        Called after role assignment and before target computation so counters
        reflect what actually fired this step.
        """
        active = self._bt_opponent_mask()
        # Escort attempts: any agent assigned ESCORT this step.
        escort_active = (roles == ROLE_ESCORT).any(dim=1)
        active = self._bt_opponent_mask()
        self.bt_tel_escort_attempts += (escort_active & active).to(torch.int32)

        intc_active = (roles == ROLE_INTERCEPTOR).any(dim=1)
        self.bt_tel_intercept_attempts += (intc_active & active).to(torch.int32)

        ctr_active = (roles == ROLE_COUNTER).any(dim=1)
        self.bt_tel_counter_captures += (ctr_active & active).to(torch.int32)

        changed = (roles != prev_roles).any(dim=1)
        self.bt_tel_objective_changes += (changed & active).to(torch.int32)

        # Stuck detection: agent hasn't moved more than 0.2 cells in last step.
        moved = torch.sqrt(
            (self.red_x - self.bt_last_x[:, :self.Nr]) ** 2
            + (self.red_y - self.bt_last_y[:, :self.Nr]) ** 2 + 1e-8
        )
        stuck_agents = self.red_alive & (~self.red_tagged) & (moved < 0.2)
        stuck_envs = stuck_agents.any(dim=1)
        self.bt_tel_stuck_steps += (stuck_envs & active).to(torch.int32)

        # Update last-position buffer.
        N_buf = self.bt_last_x.shape[1]
        self.bt_last_x[:, :self.Nr] = self.red_x.detach()
        self.bt_last_y[:, :self.Nr] = self.red_y.detach()

        # Branch label cache.
        self.bt_active_branch[:, :self.Nr] = torch.where(
            active[:, None],
            roles.detach(),
            self.bt_active_branch[:, :self.Nr],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Top-level entry point called from _scripted_red.py dispatch
    # ──────────────────────────────────────────────────────────────────────
    def _get_bt_targets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full BT brain: blackboard → role assignment → route → telemetry.

        Opponent-specific tuning is resolved from ``self._opponent_key`` via
        ``build_profile_tensors``. A searcher overlay, if set, replaces knobs
        only; the dispatch level is unchanged.

        Returns
        -------
        target_x, target_y : [B, Nr] float tensors.
        """
        prof = self._bt_resolved_profile_tensors()
        self._update_adaptive_memory(prof)
        prev_roles = self.bt_red_role.clone()
        bb = self._build_team_blackboard(prof)
        bb = self._extend_blackboard_adaptive(bb, prof)
        self._bt_update_op6_recovery(bb, prof)
        self._bt_update_op7_split_latch(bb, prof)
        self._bt_update_op7_compact_latch(bb, prof)
        roles = self._bt_assign_roles(bb)
        roles = self._bt_apply_adaptive_role_overrides(bb, roles, prof)
        roles = self._bt_apply_op7_split_role_overrides(bb, roles, prof)
        roles = self._bt_apply_op7_compact_role_overrides(bb, roles, prof)
        self._bt_update_telemetry(bb, roles, prev_roles)
        tx, ty = self._bt_route_target(bb, roles)
        tx, ty = self._bt_apply_adaptive_route_overrides(bb, roles, tx, ty, prof)
        tx, ty = self._bt_apply_op7_split_route_overrides(bb, roles, tx, ty, prof)
        tx, ty = self._bt_apply_op7_compact_route_overrides(bb, roles, tx, ty, prof)
        mine_x, mine_y, want = self._bt_plan_mines(bb, roles, prof)
        tx, ty = self._bt_apply_mine_routes(tx, ty, mine_x, mine_y, want, prof)
        self.bt_mine_target_x[:, :self.Nr] = mine_x.detach()
        self.bt_mine_target_y[:, :self.Nr] = mine_y.detach()
        self.bt_want_mine[:, :self.Nr] = want.detach()
        self._debug_red_target_x = tx.detach()
        self._debug_red_target_y = ty.detach()
        return tx, ty
