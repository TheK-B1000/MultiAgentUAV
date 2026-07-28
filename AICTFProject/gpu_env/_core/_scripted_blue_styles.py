"""Scripted blue-team style probes for the pool-admissibility protocol.

Fixed, deterministic blue controllers used ONLY to test whether the red
opponent presets (OP6-OP12) reward or punish different blue playstyles at
all -- independent of whether the RL-trained LRO latent branches (z0..z3)
can learn/select such styles. This question came up because Level 1a/1b
(run_v6i26_usable_selector_eval.py / run_v6i26_level1b_short_history_selector.py)
found the router had almost no usable payoff complementarity to select
between, raising the question of whether OP6-OP12 create strategic niches
for ANY blue play at all.

Each style is a hand-coded target-selection policy (per-agent tx, ty),
consumed exactly the way _get_scripted_targets("blue") already is (via
BatchedCTFCore.blue_scripted=True) -- no other engine code needs to change
to use these; the caller just steps the env without ever calling a blue
policy's predict().

These are intentionally isolated from _scripted_red.py's shared "unified NPC
brain" (_assign_scripted_targets_by_role) rather than threading new branches
into that 400+ line, heavily-row-masked function: correctness here matters
for a numeric claim about the opponent pool, and the existing function's
surface area (OP5-OP12 masks, BT overrides, coordinated-attack state) is
large enough that isolation is the lower-risk choice.

Deliberately 2-agent role scripts (Agent 0 / Agent 1), matching this
project's standard n_blue=2 evaluation convention -- see module docstring
discussion in run_v6i26_usable_selector_eval.py and friends. Shared rules
common to all four styles (per the locked spec):
  - Deterministic tie-breaking by agent index (no RNG draws anywhere below).
  - Legal observed game state only (positions, alive/carrying/tagged flags,
    flag positions) -- no opponent-profile labels, no hidden information.
  - No stochastic role switching.
  - Carrying the enemy flag always takes priority (return-it evasion),
    applied uniformly across all four styles at the end of dispatch, via the
    same _carrier_evasion_target multi-threat router carriers already use
    elsewhere in this codebase.
  - Role assignment is a pure function of CURRENT alive/carrying state, not
    persisted mutable state -- so a respawned agent automatically resumes
    its style's assigned role with no separate reset needed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

BLUE_STYLE_NAMES = ("BLUE_RUSH", "BLUE_TURTLE", "BLUE_SPLIT", "BLUE_ESCORT")
_STYLE_ID = {name: i + 1 for i, name in enumerate(BLUE_STYLE_NAMES)}  # 0 = no style (legacy generic brain)


class _ScriptedBlueStylesMixin:
    def set_blue_style(self, style: Optional[str]) -> None:
        """Select a fixed blue probe style (or None to fall back to the
        legacy generic blue brain in _assign_scripted_targets_by_role).
        Does NOT itself enable scripted blue -- callers must also set
        self.blue_scripted = True, matching the existing convention where
        blue_scripted gates scripted-vs-RL target computation in _step.py."""
        if style is None:
            self._blue_style_id = 0
            self._blue_turtle_counter_ticks = None
            self._blue_turtle_prev_red_tagged = None
            self._blue_turtle_prev_red_carrying = None
            self._blue_split_escape_ticks = None
            self._blue_split_escape_lane_y = None
            self._blue_split_prev_carrying = None
            return
        style_u = str(style).upper()
        if style_u not in _STYLE_ID:
            raise ValueError(f"Unknown blue style {style!r}, expected one of {BLUE_STYLE_NAMES}")
        self._blue_style_id = _STYLE_ID[style_u]
        self._blue_turtle_counter_ticks = None
        self._blue_turtle_prev_red_tagged = None
        self._blue_turtle_prev_red_carrying = None
        self._blue_split_escape_ticks = None
        self._blue_split_escape_lane_y = None
        self._blue_split_prev_carrying = None

    def _blue_style_active(self) -> bool:
        return bool(getattr(self, "_blue_style_id", 0))

    @staticmethod
    def _dist(ax: torch.Tensor, ay: torch.Tensor, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + 1e-8)

    def _assign_blue_style_targets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        style_id = int(getattr(self, "_blue_style_id", 0))
        B, N = self.B, self.Nb
        idx_env = torch.arange(B, device=self.device)
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))
        midline = float(self.cols) * 0.5

        own_x, own_y = self.blue_x, self.blue_y
        own_carrying = self.blue_carrying
        own_alive = self.blue_alive
        own_flag_home = self.blue_flag_home
        enemy_x, enemy_y = self.red_x, self.red_y
        enemy_alive = self.red_alive
        enemy_tagged = self.red_tagged
        enemy_flag_pos = self.red_flag_pos

        if N < 2:
            # These are 2-agent role scripts; degenerate team sizes fall back
            # to a plain direct flag rush rather than guessing at a role split.
            target_x = enemy_flag_pos[:, 0:1].expand(B, N).clone()
            target_y = enemy_flag_pos[:, 1:2].expand(B, N).clone()
        elif style_id == _STYLE_ID["BLUE_RUSH"]:
            target_x, target_y = self._blue_rush_targets(own_x, own_y, enemy_x, enemy_y, enemy_alive, enemy_flag_pos, idx_env)
        elif style_id == _STYLE_ID["BLUE_TURTLE"]:
            target_x, target_y = self._blue_turtle_targets(own_x, own_y, own_flag_home, enemy_x, enemy_y, enemy_alive, enemy_tagged, enemy_flag_pos, midline, idx_env)
        elif style_id == _STYLE_ID["BLUE_SPLIT"]:
            target_x, target_y = self._blue_split_targets(own_x, own_y, enemy_flag_pos, max_y)
        elif style_id == _STYLE_ID["BLUE_ESCORT"]:
            target_x, target_y = self._blue_escort_targets(own_x, own_y, own_alive, enemy_x, enemy_y, enemy_alive, enemy_flag_pos, idx_env)
        else:
            target_x = enemy_flag_pos[:, 0:1].expand(B, N).clone()
            target_y = enemy_flag_pos[:, 1:2].expand(B, N).clone()

        # Shared rule: carrying the enemy flag -> returning it takes priority,
        # for every style, via the same multi-threat evasion router legacy
        # carriers already use.
        if own_carrying.any():
            if style_id == _STYLE_ID["BLUE_SPLIT"]:
                target_x, target_y = self._blue_split_post_pickup_targets(
                    target_x,
                    target_y,
                    own_x,
                    own_y,
                    own_alive,
                    own_carrying,
                    own_flag_home,
                    enemy_x,
                    enemy_y,
                    enemy_alive,
                    max_y,
                    midline,
                    idx_env,
                )
            else:
                evade_tx, evade_ty = self._carrier_evasion_target(
                    own_x, own_y, own_flag_home[:, 0], own_flag_home[:, 1],
                    enemy_x, enemy_y, enemy_alive, own_carrying, side="blue",
                )
                target_x = torch.where(own_carrying, evade_tx, target_x)
                target_y = torch.where(own_carrying, evade_ty, target_y)
            if style_id == _STYLE_ID["BLUE_RUSH"]:
                carrier_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
                other_idx = torch.where(
                    carrier_idx == 0,
                    torch.ones_like(carrier_idx),
                    torch.zeros_like(carrier_idx),
                )
                carrier_y = own_y[idx_env, carrier_idx]
                upper_lane_y = torch.full((B,), max_y * 0.90, dtype=own_y.dtype, device=own_y.device)
                lower_lane_y = torch.full((B,), max_y * 0.10, dtype=own_y.dtype, device=own_y.device)
                pressure_lane_y = torch.where(carrier_y >= max_y * 0.5, lower_lane_y, upper_lane_y)
                noncarrier_slot = torch.arange(N, device=self.device)[None, :] == other_idx[:, None]
                rush_pressure = own_carrying.any(dim=1)[:, None] & noncarrier_slot
                target_x = torch.where(rush_pressure, enemy_flag_pos[:, 0:1], target_x)
                target_y = torch.where(rush_pressure, pressure_lane_y[:, None], target_y)
            if style_id == _STYLE_ID["BLUE_ESCORT"]:
                carrier_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
                other_idx = torch.where(
                    carrier_idx == 0,
                    torch.ones_like(carrier_idx),
                    torch.zeros_like(carrier_idx),
                )
                carrier_x = own_x[idx_env, carrier_idx]
                carrier_y = own_y[idx_env, carrier_idx]
                other_x = own_x[idx_env, other_idx]
                other_y = own_y[idx_env, other_idx]
                pair_dist = torch.sqrt((carrier_x - other_x) ** 2 + (carrier_y - other_y) ** 2 + 1e-8)
                carrier_wait = (pair_dist > 2.0) & own_carrying.any(dim=1)
                carrier_slot = torch.arange(N, device=self.device)[None, :] == carrier_idx[:, None]
                # Protective follow: track the carrier's evasion waypoint so the
                # escort stays between carrier and threats on the return trip.
                # OP12 commit 861696f briefly replaced this with
                # ``carrier_x + 1.5`` to tighten detector geometry; that made
                # ESCORT collapse against OP11 (held-out second place +1.38
                # became development last place -1.125 with blue_score=0 on
                # all 8 matched seeds) because the escort stopped covering the
                # actual escape path. Compactness for detectors still comes
                # from pre-pickup shield_dist and the carrier_wait gate above.
                follow_x = evade_tx[idx_env, carrier_idx]
                follow_y = evade_ty[idx_env, carrier_idx]
                escort_follow = own_carrying.any(dim=1)[:, None] & (~own_carrying)
                target_x = torch.where(carrier_wait[:, None] & carrier_slot, other_x[:, None], target_x)
                target_y = torch.where(carrier_wait[:, None] & carrier_slot, other_y[:, None], target_y)
                target_x = torch.where(escort_follow, follow_x[:, None], target_x)
                target_y = torch.where(escort_follow, follow_y[:, None], target_y)
        elif style_id == _STYLE_ID["BLUE_SPLIT"]:
            self._blue_split_prev_carrying = own_carrying.detach().clone()
            ticks = getattr(self, "_blue_split_escape_ticks", None)
            if ticks is not None and ticks.shape[0] == B:
                self._blue_split_escape_ticks = torch.zeros_like(ticks).detach()

        target_x = torch.clamp(target_x, 0.0, max_x)
        target_y = torch.clamp(target_y, 0.0, max_y)
        self._debug_blue_target_x = target_x.detach()
        self._debug_blue_target_y = target_y.detach()
        return target_x, target_y

    # ------------------------------------------------------------------
    # BLUE_RUSH: both agents commit directly to the enemy flag. No lane
    # offsets, no defender pressure, no tangent evasion -- direct routing
    # except for the shared carrying-priority override above.
    # ------------------------------------------------------------------
    def _blue_rush_targets(self, own_x, own_y, enemy_x, enemy_y, enemy_alive, enemy_flag_pos, idx_env):
        efx, efy = enemy_flag_pos[:, 0], enemy_flag_pos[:, 1]
        max_y = float(max(0, self.rows - 1))
        t0x, t0y = efx, efy
        t1x = efx
        t1y = torch.clamp(efy + 4.0, 0.0, max_y)
        return torch.stack([t0x, t1x], dim=1), torch.stack([t0y, t1y], dim=1)

    # ------------------------------------------------------------------
    # BLUE_TURTLE: agent 0 anchors near home; agent 1 patrols the own half
    # and intercepts any red found there. Counterattack (agent 1 -> enemy
    # flag) unlocks only after the rush is either fully committed or contained
    # by tags/deaths; this keeps defense first, but lets successful defense
    # become payoff rather than permanent stalling. No escort logic is ever
    # introduced, so "non-carrier reverts to defense rather than escorting"
    # holds by construction once the enemy flag is obtained.
    # ------------------------------------------------------------------
    def _blue_turtle_targets(self, own_x, own_y, own_flag_home, enemy_x, enemy_y, enemy_alive, enemy_tagged, enemy_flag_pos, midline, idx_env):
        B = own_x.shape[0]
        hx, hy = own_flag_home[:, 0], own_flag_home[:, 1]
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))

        enemy_threat_alive = enemy_alive & (~enemy_tagged)
        enemy_on_own = enemy_threat_alive & (enemy_x < midline)
        enemy_inbound = enemy_threat_alive
        any_intruder = enemy_on_own.any(dim=1)
        any_inbound = enemy_inbound.any(dim=1)

        window_len = int(getattr(self, "blue_turtle_post_tag_counter_steps", 20))
        ticks = getattr(self, "_blue_turtle_counter_ticks", None)
        if ticks is None or ticks.shape[0] != B:
            ticks = torch.zeros((B,), dtype=torch.int32, device=own_x.device)
        prev_red_tagged = getattr(self, "_blue_turtle_prev_red_tagged", None)
        if prev_red_tagged is None or prev_red_tagged.shape != enemy_tagged.shape:
            prev_red_tagged = torch.zeros_like(enemy_tagged)
        prev_red_carrying = getattr(self, "_blue_turtle_prev_red_carrying", None)
        if prev_red_carrying is None or prev_red_carrying.shape != self.red_carrying.shape:
            prev_red_carrying = torch.zeros_like(self.red_carrying)
        newly_red_tagged = (~prev_red_tagged) & enemy_tagged
        red_carrier_stopped = (newly_red_tagged & prev_red_carrying).any(dim=1)
        dual_rush_stopped = (enemy_tagged & enemy_alive).sum(dim=1) >= 2
        counter_trigger = red_carrier_stopped | dual_rush_stopped
        ticks = torch.where(
            counter_trigger,
            torch.full_like(ticks, window_len),
            torch.clamp(ticks - 1, min=0),
        )

        red_progress = torch.where(enemy_inbound, midline - enemy_x, torch.full_like(enemy_x, -1e9))
        urgent_idx = torch.argmax(red_progress, dim=1)
        active_accum = self.red_tag_pressure_time[:, :2] > 0.0
        accum_choice = torch.argmax(self.red_tag_pressure_time[:, :2], dim=1)
        urgent_idx = torch.where(active_accum.any(dim=1), accum_choice, urgent_idx)
        other_idx = 1 - urgent_idx
        urgent_x = enemy_x[idx_env, urgent_idx]
        urgent_y = enemy_y[idx_env, urgent_idx]

        # Pinch the same target: defender 0 blocks slightly ahead on the route
        # to the blue flag, defender 1 closes from the rear/side. If an actual
        # tag accumulator is active, the same target is held until it completes
        # or resets, preventing last-step target churn.
        to_flag_x = hx - urgent_x
        to_flag_y = hy - urgent_y
        to_flag_n = torch.sqrt(to_flag_x ** 2 + to_flag_y ** 2 + 1e-8)
        ux = to_flag_x / to_flag_n
        uy = to_flag_y / to_flag_n
        perp_x = -uy
        perp_y = ux
        side = torch.where(urgent_y >= hy, 1.0, -1.0)
        t0x_patrol = torch.clamp(urgent_x + ux * 0.8 + perp_x * side * 0.2, 0.0, max_x)
        t0y_patrol = torch.clamp(urgent_y + uy * 0.8 + perp_y * side * 0.2, 0.0, max_y)
        t1x_patrol = torch.clamp(urgent_x - ux * 0.2 - perp_x * side * 0.2, 0.0, max_x)
        t1y_patrol = torch.clamp(urgent_y - uy * 0.2 - perp_y * side * 0.2, 0.0, max_y)

        active_accum_any = active_accum.any(dim=1)
        t0x_patrol = torch.where(active_accum_any, urgent_x, t0x_patrol)
        t0y_patrol = torch.where(active_accum_any, urgent_y, t0y_patrol)
        t1x_patrol = torch.where(active_accum_any, urgent_x, t1x_patrol)
        t1y_patrol = torch.where(active_accum_any, urgent_y, t1y_patrol)

        # Carrier priority: collapse both defenders on the flag carrier.
        red_carry_any = self.red_carrying.any(dim=1)
        red_carrier_idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
        carrier_x = enemy_x[idx_env, red_carrier_idx]
        carrier_y = enemy_y[idx_env, red_carrier_idx]
        t0x_patrol = torch.where(red_carry_any, carrier_x, t0x_patrol)
        t0y_patrol = torch.where(red_carry_any, carrier_y, t0y_patrol)
        t1x_patrol = torch.where(red_carry_any, carrier_x, t1x_patrol)
        t1y_patrol = torch.where(red_carry_any, carrier_y, t1y_patrol)

        fallback_x = torch.clamp(hx + (midline - hx) * 0.35, 0.0, max_x)
        fallback_y0 = torch.clamp(hy - 1.0, 0.0, max_y)
        fallback_y1 = torch.clamp(hy + 1.0, 0.0, max_y)
        t0x_patrol = torch.where(any_inbound, t0x_patrol, fallback_x)
        t0y_patrol = torch.where(any_inbound, t0y_patrol, fallback_y0)
        t1x_patrol = torch.where(any_inbound, t1x_patrol, fallback_x)
        t1y_patrol = torch.where(any_inbound, t1y_patrol, fallback_y1)

        # "Red significantly committed": every alive red agent is on blue's
        # side of the midline. Tagged/dead agents do not block the trigger.
        committed_or_neutralized = (~enemy_alive) | enemy_tagged | (enemy_x < midline)
        all_committed = committed_or_neutralized.all(dim=1) & enemy_alive.any(dim=1)
        rush_contained = (~any_intruder) & ((~enemy_alive) | enemy_tagged).any(dim=1)
        counter_window_active = (ticks > 0) & (~any_intruder) & (~red_carry_any) & (~active_accum_any)
        counter_unlocked = (rush_contained | counter_window_active) & (~active_accum_any)
        efx, efy = enemy_flag_pos[:, 0], enemy_flag_pos[:, 1]
        # Keep agent 0 on defense; only agent 1 converts. Dual-agent counter was
        # tried (dev17) and rejected: tags are temporary, so both blues leaving
        # home lets untagged reds score into an empty base.
        t0x = t0x_patrol
        t0y = t0y_patrol
        t1x = torch.where(counter_unlocked, efx, t1x_patrol)
        t1y = torch.where(counter_unlocked, efy, t1y_patrol)

        self._blue_turtle_counter_ticks = torch.where(any_intruder | red_carry_any, torch.zeros_like(ticks), ticks).detach()
        self._blue_turtle_prev_red_tagged = enemy_tagged.detach().clone()
        self._blue_turtle_prev_red_carrying = self.red_carrying.detach().clone()

        return torch.stack([t0x, t1x], dim=1), torch.stack([t0y, t1y], dim=1)

    # ------------------------------------------------------------------
    # BLUE_SPLIT: agent 0 = upper lane, agent 1 = lower lane, fixed by
    # index for determinism. Each holds its lane's y-band until close to
    # the enemy flag in x, then converges -- same "hold lane, converge near
    # the goal" pattern already used for red's default striker role, just
    # with two independently-lanced agents instead of one striker.
    # ------------------------------------------------------------------
    def _blue_split_targets(self, own_x, own_y, enemy_flag_pos, max_y):
        efx, efy = enemy_flag_pos[:, 0], enemy_flag_pos[:, 1]
        lane_y_upper = torch.full_like(efy, min(max_y, max_y * 0.90))
        lane_y_lower = torch.full_like(efy, max(0.0, max_y * 0.10))

        d0 = torch.abs(own_x[:, 0] - efx)
        d1 = torch.abs(own_x[:, 1] - efx)
        t0y = torch.where(d0 > 1.5, lane_y_upper, efy)
        t1y = torch.where(d1 > 1.5, lane_y_lower, efy)
        t0x = efx
        t1x = efx
        return torch.stack([t0x, t1x], dim=1), torch.stack([t0y, t1y], dim=1)

    def _blue_split_post_pickup_targets(
        self,
        target_x,
        target_y,
        own_x,
        own_y,
        own_alive,
        own_carrying,
        own_flag_home,
        enemy_x,
        enemy_y,
        enemy_alive,
        max_y,
        midline,
        idx_env,
    ):
        B, N = own_x.shape
        lane_y_upper = torch.full((B,), min(max_y, max_y * 0.90), dtype=own_y.dtype, device=own_y.device)
        lane_y_lower = torch.full((B,), max(0.0, max_y * 0.10), dtype=own_y.dtype, device=own_y.device)
        home_x = own_flag_home[:, 0]
        home_y = own_flag_home[:, 1]

        ticks = getattr(self, "_blue_split_escape_ticks", None)
        if ticks is None or ticks.shape[0] != B:
            ticks = torch.zeros((B,), dtype=torch.int32, device=own_x.device)
        lane_y = getattr(self, "_blue_split_escape_lane_y", None)
        if lane_y is None or lane_y.shape[0] != B:
            lane_y = torch.where(home_y >= max_y * 0.5, lane_y_lower, lane_y_upper)
        prev_carrying = getattr(self, "_blue_split_prev_carrying", None)
        if prev_carrying is None or prev_carrying.shape != own_carrying.shape:
            prev_carrying = torch.zeros_like(own_carrying)

        any_carrying = own_carrying.any(dim=1)
        carrier_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
        carrier_x = own_x[idx_env, carrier_idx]
        carrier_y = own_y[idx_env, carrier_idx]
        newly_picked = ((~prev_carrying) & own_carrying).any(dim=1)

        probe_x = torch.clamp((carrier_x + home_x) * 0.5, 0.0, float(max(0, self.cols - 1)))
        upper_len = torch.sqrt((carrier_x - home_x) ** 2 + (carrier_y - lane_y_upper) ** 2 + (lane_y_upper - home_y) ** 2 + 1e-8)
        lower_len = torch.sqrt((carrier_x - home_x) ** 2 + (carrier_y - lane_y_lower) ** 2 + (lane_y_lower - home_y) ** 2 + 1e-8)
        upper_clear = self._lane_clearance(probe_x, lane_y_upper, enemy_x, enemy_y, enemy_alive)
        lower_clear = self._lane_clearance(probe_x, lane_y_lower, enemy_x, enemy_y, enemy_alive)
        upper_score = upper_clear - 0.15 * upper_len
        lower_score = lower_clear - 0.15 * lower_len
        picked_lane_y = torch.where(upper_score >= lower_score, lane_y_upper, lane_y_lower)

        lock_len = int(getattr(self, "blue_split_escape_lock_steps", 5))
        ticks = torch.where(newly_picked, torch.full_like(ticks, lock_len), torch.clamp(ticks - 1, min=0))
        lane_y = torch.where(newly_picked, picked_lane_y, lane_y)

        route_probe_clear = self._lane_clearance(probe_x, lane_y, enemy_x, enemy_y, enemy_alive)
        blocked = route_probe_clear < float(getattr(self.cfg, "tag_range_cells", 1.5)) * 0.9
        lane_y = torch.where((ticks <= 0) | blocked, picked_lane_y, lane_y)
        ticks = torch.where(any_carrying, ticks, torch.zeros_like(ticks))

        return_lane_x = torch.where(carrier_x > midline, torch.full_like(carrier_x, midline - 1.0), home_x)
        carrier_target_x = torch.where(carrier_x > midline, return_lane_x, home_x)
        carrier_target_y = torch.where(carrier_x > midline, lane_y, home_y)

        teammate_idx = torch.where(carrier_idx == 0, torch.ones_like(carrier_idx), torch.zeros_like(carrier_idx))
        opposite_lane_y = torch.where(lane_y >= max_y * 0.5, lane_y_lower, lane_y_upper)
        teammate_target_x = torch.full_like(carrier_target_x, float(max(0, self.cols - 1)))
        teammate_target_y = opposite_lane_y

        carrier_slot = torch.arange(N, device=own_x.device)[None, :] == carrier_idx[:, None]
        teammate_slot = torch.arange(N, device=own_x.device)[None, :] == teammate_idx[:, None]
        target_x = torch.where(any_carrying[:, None] & carrier_slot, carrier_target_x[:, None], target_x)
        target_y = torch.where(any_carrying[:, None] & carrier_slot, carrier_target_y[:, None], target_y)
        target_x = torch.where(any_carrying[:, None] & teammate_slot & own_alive, teammate_target_x[:, None], target_x)
        target_y = torch.where(any_carrying[:, None] & teammate_slot & own_alive, teammate_target_y[:, None], target_y)

        self._blue_split_escape_ticks = ticks.detach()
        self._blue_split_escape_lane_y = lane_y.detach()
        self._blue_split_prev_carrying = own_carrying.detach().clone()
        return target_x, target_y

    def _lane_clearance(self, probe_x, probe_y, enemy_x, enemy_y, enemy_alive):
        dx = enemy_x - probe_x[:, None]
        dy = enemy_y - probe_y[:, None]
        d = torch.sqrt(dx * dx + dy * dy + 1e-8)
        big = torch.full_like(d, 1e6)
        return torch.where(enemy_alive, d, big).min(dim=1).values

    # ------------------------------------------------------------------
    # BLUE_ESCORT: agent 0 is the default/intended carrier; agent 1 shields
    # it with a perpendicular offset biased away from the nearest live
    # enemy (same geometry as the legacy carrier-shielding behavior
    # elsewhere in this codebase). Role reassignment on carrier failure is
    # recomputed fresh every step from CURRENT alive/carrying state (no
    # persisted "who is carrier" flag to go stale across a respawn).
    # ------------------------------------------------------------------
    def _blue_escort_targets(self, own_x, own_y, own_alive, enemy_x, enemy_y, enemy_alive, enemy_flag_pos, idx_env):
        B = own_x.shape[0]
        own_carrying = self.blue_carrying
        efx, efy = enemy_flag_pos[:, 0], enemy_flag_pos[:, 1]

        any_carrying = own_carrying.any(dim=1)
        carrier_idx_if_carrying = torch.argmax(own_carrying.to(torch.int64), dim=1)
        agent0_down = ~own_alive[:, 0]
        intended_carrier = torch.where(
            agent0_down,
            torch.ones(B, dtype=torch.int64, device=own_x.device),
            torch.zeros(B, dtype=torch.int64, device=own_x.device),
        )
        carrier_idx = torch.where(any_carrying, carrier_idx_if_carrying, intended_carrier)

        carrier_x = own_x[idx_env, carrier_idx]
        carrier_y = own_y[idx_env, carrier_idx]

        # Pre-pickup carrier target: the enemy flag directly. (Post-pickup,
        # the shared carrying-priority override in _assign_blue_style_targets
        # takes over with evasion-toward-home, so this branch only matters
        # before the flag is grabbed.)
        t_carrier_x, t_carrier_y = efx, efy

        # Escort: protective perpendicular offset from the carrier, biased
        # away from the nearest live enemy -- keeps the pair spatially
        # concentrated and interposes the escort between carrier and threat.
        shield_dist = 0.3  # tighter than the legacy generic-brain shield (4.0): "both agents remain
        # spatially concentrated" is an explicit ESCORT requirement, not just a side effect. Must
        # also decisively beat SPLIT's INCIDENTAL closeness near flag pickup (both SPLIT agents
        # converge toward the same (enemy_flag_x, enemy_flag_y) point when close to the flag).
        dxx = carrier_x[:, None] - enemy_x
        dyy = carrier_y[:, None] - enemy_y
        dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
        big = torch.full_like(dd, 1e9)
        dd_live = torch.where(enemy_alive, dd, big)
        near_enemy = torch.argmin(dd_live, dim=1)
        nex = enemy_x[idx_env, near_enemy]
        ney = enemy_y[idx_env, near_enemy]
        to_enemy_x = nex - carrier_x
        to_enemy_y = ney - carrier_y
        to_enemy_n = torch.sqrt(to_enemy_x ** 2 + to_enemy_y ** 2 + 1e-8)
        perp_x = -(to_enemy_y / to_enemy_n)
        perp_y = (to_enemy_x / to_enemy_n)
        t_escort_x = carrier_x + perp_x * shield_dist
        t_escort_y = carrier_y + perp_y * shield_dist

        is_agent0_carrier = carrier_idx == 0
        t0x = torch.where(is_agent0_carrier, t_carrier_x, t_escort_x)
        t0y = torch.where(is_agent0_carrier, t_carrier_y, t_escort_y)
        t1x = torch.where(is_agent0_carrier, t_escort_x, t_carrier_x)
        t1y = torch.where(is_agent0_carrier, t_escort_y, t_carrier_y)
        return torch.stack([t0x, t1x], dim=1), torch.stack([t0y, t1y], dim=1)
