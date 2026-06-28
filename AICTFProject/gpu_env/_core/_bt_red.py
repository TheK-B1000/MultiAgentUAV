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

from typing import Dict, Tuple

import torch

from ._bt_profiles import BT_OPPONENT_KEYS, build_profile_tensors, is_bt_opponent


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


class _BTRedMixin:
    """
    Behavior-tree NPC brain for OP5..OP12 scripted opponents.

    Intended to be mixed into BatchedCTFCore alongside _ScriptedRedMixin.
    Relies on state tensors allocated by _ScratchStateMixin and augmented by
    ``_alloc_bt_telemetry``.
    """

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
        # Per-agent last position (for stuck detection, reset each episode).
        self.bt_last_x = torch.zeros((B, N), dtype=f32, device=dev)
        self.bt_last_y = torch.zeros((B, N), dtype=f32, device=dev)
        # Branch label cache: integer code for which BT branch fired last step.
        # Branch codes match role constants above.
        self.bt_active_branch = torch.full((B, N), ROLE_ATTACKER, dtype=i32, device=dev)

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
        self.bt_last_x[idx]            = 0.0
        self.bt_last_y[idx]            = 0.0
        self.bt_active_branch[idx]     = ROLE_ATTACKER

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
            time_frac=time_frac,
            late_game=late_game,
            trailing=trailing,
            leading=leading,
            alive_count=alive_count,
            profile=prof,
            is_op12=prof["is_op12"],
        )

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

        # ── Priority 1: flag retrieval ───────────────────────────────────
        need_retr = (~bb["own_flag_at_home"]) & prof["enable_flag_retr"]
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
        have_carrier = bb["red_carry_any"] & prof["enable_escort"]
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
        feas    = bb["intercept_feasible"] & prof["enable_intercept"]
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

        # OP9-style late-game pressure: evasion only when trailing near end of match.
        late_press = bb["late_game"] & bb["trailing"] & prof["late_game_evasion_unlock"]
        base_threat = prof["threat_radius"]
        press_threat = torch.where(late_press, base_threat, torch.zeros_like(base_threat))
        threat_radius = torch.where(prof["late_game_evasion_unlock"], press_threat, base_threat)

        for j in range(Nr):
            rx = self.red_x[:, j]
            ry = self.red_y[:, j]
            role_j = roles[:, j]  # [B]

            tx = torch.zeros((B,), device=device)
            ty = torch.zeros((B,), device=device)

            # ── ATTACKER ──────────────────────────────────────────────────
            # Lane approach + tangent evasion near defenders.
            efx = bb["blue_flag_pos"][:, 0]
            efy = bb["blue_flag_pos"][:, 1]
            dist_to_flag = torch.sqrt((rx - efx) ** 2 + (ry - efy) ** 2 + 1e-8)
            # Use lane waypoint when far from flag; converge directly when close.
            atk_tx = torch.where(dist_to_flag > 4.0, efx, efx)
            atk_ty = torch.where(dist_to_flag > 4.0, lane_y_pref, efy)
            # Tangent evasion from nearest blue agent.
            atk_tx, atk_ty = self._bt_tangent_evade(
                rx, ry, atk_tx, atk_ty, threat_radius, max_x, max_y,
            )
            tx = torch.where(role_j == ROLE_ATTACKER, atk_tx, tx)
            ty = torch.where(role_j == ROLE_ATTACKER, atk_ty, ty)

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
            # If escort but no carrier exists, fall back to attacker target.
            tx = torch.where((role_j == ROLE_ESCORT) & (~bb["red_carry_any"]), atk_tx, tx)
            ty = torch.where((role_j == ROLE_ESCORT) & (~bb["red_carry_any"]), atk_ty, ty)

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

        return target_x, target_y

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
        ``build_profile_tensors``.

        Returns
        -------
        target_x, target_y : [B, Nr] float tensors.
        """
        prof = build_profile_tensors(self._opponent_key, device=self.device, batch_size=self.B)
        prev_roles = self.bt_red_role.clone()
        bb = self._build_team_blackboard(prof)
        roles = self._bt_assign_roles(bb)
        self._bt_update_telemetry(bb, roles, prev_roles)
        tx, ty = self._bt_route_target(bb, roles)
        self._debug_red_target_x = tx.detach()
        self._debug_red_target_y = ty.detach()
        return tx, ty
