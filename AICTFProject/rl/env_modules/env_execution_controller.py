"""
ExecutionController: Guided GoTo + safety for Sprint A.

Makes training easier without reducing sim realism by:
- Providing guided pathfinding with safety checks
- Deterministic conflict resolution (no random action fails)
- Fallback strategies when primary actions fail
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from macro_actions import MacroAction


class ExecutionController:
    """
    ExecutionController provides guided action execution with safety checks.
    
    Key features:
    - Guided GoTo: Helps agents reach targets more reliably
    - Safety checks: Avoids dangerous paths deterministically
    - Conflict resolution: Deterministic fallbacks when actions fail
    """
    
    def __init__(
        self,
        *,
        enable_guided_goto: bool = True,
        enable_safety_checks: bool = True,
        enable_deterministic_fallback: bool = True,
        seed: int = 0,
    ) -> None:
        self._enable_guided_goto = bool(enable_guided_goto)
        self._enable_safety_checks = bool(enable_safety_checks)
        self._enable_deterministic_fallback = bool(enable_deterministic_fallback)
        self._rng = np.random.RandomState(int(seed) + 2000)
        self._action_failure_history: Dict[str, int] = {}  # Track failures per agent
    
    def reset_episode(self) -> None:
        """Reset episode-level state."""
        self._action_failure_history.clear()
    
    def execute_guided_goto(
        self,
        agent: Any,
        target: Tuple[int, int],
        game_field: Any,
        *,
        avoid_enemies: bool = True,
        safety_radius: int = 1,
    ) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """
        Execute guided GoTo with safety checks.
        
        Returns:
            (success, path) - success=True if path found, path is list of cells or None
        """
        if not self._enable_guided_goto:
            return True, None  # Let default pathfinding handle it
        
        if agent is None:
            return False, None
        
        try:
            start = self._agent_cell_pos(agent, game_field)
            if start is None:
                return False, None
            
            tgt = self._clamp_cell(target[0], target[1], game_field)
            if start == tgt:
                return True, [tgt]  # Already at target
            
            # Use game_field's pathfinder
            pathfinder = getattr(game_field, "pathfinder", None)
            if pathfinder is None:
                return True, None  # Fallback to default
            
            # Safety: Check for enemies if avoid_enemies is True
            if self._enable_safety_checks and avoid_enemies:
                danger = self._compute_danger_map(agent, game_field, safety_radius)
                if danger and hasattr(pathfinder, "setDangerCosts"):
                    pathfinder.setDangerCosts(danger)
            
            # Find path
            path = pathfinder.astar(start, tgt)
            
            if path and len(path) > 0:
                return True, path
            else:
                # Deterministic fallback: try direct path or nearest safe cell
                if self._enable_deterministic_fallback:
                    fallback_path = self._deterministic_fallback_path(
                        agent, target, game_field, avoid_enemies, safety_radius
                    )
                    return fallback_path is not None, fallback_path
                return False, None
                
        except Exception:
            return False, None
    
    def resolve_action_conflict(
        self,
        agent: Any,
        intended_action: MacroAction,
        intended_target: Optional[Tuple[int, int]],
        game_field: Any,
    ) -> Tuple[MacroAction, Optional[Tuple[int, int]], bool]:
        """
        Deterministically resolve action conflicts.
        
        When an action fails (e.g., path blocked, target invalid), provides
        deterministic fallback instead of random failure.
        
        Returns:
            (resolved_action, resolved_target, was_fallback) - 
            resolved action/target and whether fallback was used
        """
        if not self._enable_deterministic_fallback:
            return intended_action, intended_target, False
        
        agent_key = self._get_agent_key(agent)
        
        # Check if action is valid
        is_valid, reason = self._validate_action(
            agent, intended_action, intended_target, game_field
        )
        
        if is_valid:
            return intended_action, intended_target, False
        
        # Deterministic fallback based on action type
        fallback_action, fallback_target = self._get_deterministic_fallback(
            agent, intended_action, intended_target, game_field, reason
        )
        
        # Track failure for debugging
        if agent_key:
            self._action_failure_history[agent_key] = (
                self._action_failure_history.get(agent_key, 0) + 1
            )
        
        return fallback_action, fallback_target, True
    
    def _validate_action(
        self,
        agent: Any,
        action: MacroAction,
        target: Optional[Tuple[int, int]],
        game_field: Any,
    ) -> Tuple[bool, str]:
        """Validate if action can be executed."""
        if agent is None:
            return False, "no_agent"
        
        if not getattr(agent, "isEnabled", lambda: True)():
            return False, "agent_disabled"
        
        if action == MacroAction.GO_TO:
            if target is None:
                return False, "no_target"
            # Check if target is reachable (basic check)
            start = self._agent_cell_pos(agent, game_field)
            if start is None:
                return False, "no_start_pos"
            # Pathfinding will handle detailed validation
        
        elif action == MacroAction.GET_FLAG:
            if getattr(agent, "isCarryingFlag", lambda: False)():
                return False, "already_carrying"
        
        elif action == MacroAction.GO_HOME:
            # Always valid if agent exists
            pass
        
        elif action == MacroAction.GRAB_MINE:
            # Check if mines available
            side = str(getattr(agent, "side", "blue")).lower()
            mine_pickups = getattr(game_field, "mine_pickups", [])
            my_pickups = [p for p in mine_pickups if getattr(p, "owner_side", None) == side]
            if not my_pickups:
                return False, "no_mines_available"
        
        elif action == MacroAction.PLACE_MINE:
            charges = int(getattr(agent, "mine_charges", 0))
            if charges <= 0:
                return False, "no_mine_charges"
        
        return True, "valid"
    
    def _get_deterministic_fallback(
        self,
        agent: Any,
        intended_action: MacroAction,
        intended_target: Optional[Tuple[int, int]],
        game_field: Any,
        failure_reason: str,
    ) -> Tuple[MacroAction, Optional[Tuple[int, int]]]:
        """Get deterministic fallback action based on failure reason."""
        side = str(getattr(agent, "side", "blue")).lower()
        
        # Fallback priority: GO_HOME > GO_TO (safe position) > GO_TO (current position)
        if intended_action == MacroAction.GET_FLAG and failure_reason == "already_carrying":
            # If already carrying, go home
            home = self._get_home_position(agent, game_field)
            return MacroAction.GO_HOME, home
        
        elif intended_action == MacroAction.GO_TO:
            # Try to find a safe nearby position
            safe_target = self._find_safe_fallback_target(agent, intended_target, game_field)
            if safe_target:
                return MacroAction.GO_TO, safe_target
            # Last resort: stay put (current position)
            current_pos = self._agent_cell_pos(agent, game_field)
            return MacroAction.GO_TO, current_pos if current_pos else (0, 0)
        
        elif intended_action == MacroAction.GRAB_MINE and failure_reason == "no_mines_available":
            # No mines available, go to flag or patrol
            flag_pos = self._get_enemy_flag_position(agent, game_field)
            if flag_pos:
                return MacroAction.GO_TO, flag_pos
            # Fallback to home
            home = self._get_home_position(agent, game_field)
            return MacroAction.GO_TO, home
        
        elif intended_action == MacroAction.PLACE_MINE and failure_reason == "no_mine_charges":
            # No charges, go grab mines or go to flag
            side = str(getattr(agent, "side", "blue")).lower()
            mine_pickups = getattr(game_field, "mine_pickups", [])
            my_pickups = [p for p in mine_pickups if getattr(p, "owner_side", None) == side]
            if my_pickups:
                nearest = min(my_pickups, key=lambda p: self._distance_to_agent(agent, p, game_field))
                return MacroAction.GRAB_MINE, None
            # Fallback to flag
            flag_pos = self._get_enemy_flag_position(agent, game_field)
            if flag_pos:
                return MacroAction.GO_TO, flag_pos
            home = self._get_home_position(agent, game_field)
            return MacroAction.GO_TO, home
        
        # Default fallback: GO_TO current position (stay put)
        current_pos = self._agent_cell_pos(agent, game_field)
        return MacroAction.GO_TO, current_pos if current_pos else (0, 0)
    
    def _deterministic_fallback_path(
        self,
        agent: Any,
        target: Tuple[int, int],
        game_field: Any,
        avoid_enemies: bool,
        safety_radius: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Find deterministic fallback path when primary pathfinding fails."""
        start = self._agent_cell_pos(agent, game_field)
        if start is None:
            return None
        
        tgt = self._clamp_cell(target[0], target[1], game_field)
        
        # Try direct path first (no obstacles)
        if self._is_direct_path_clear(start, tgt, game_field):
            return [start, tgt]
        
        # Try intermediate waypoint
        mid_x = (start[0] + tgt[0]) // 2
        mid_y = (start[1] + tgt[1]) // 2
        waypoint = self._clamp_cell(mid_x, mid_y, game_field)
        
        if waypoint != start and waypoint != tgt:
            if self._is_direct_path_clear(start, waypoint, game_field):
                return [start, waypoint, tgt]
        
        # Last resort: return path to nearest safe cell toward target
        safe_cell = self._find_nearest_safe_cell_toward_target(
            agent, target, game_field, safety_radius
        )
        if safe_cell and safe_cell != start:
            return [start, safe_cell]
        
        return None
    
    def _compute_danger_map(
        self,
        agent: Any,
        game_field: Any,
        radius: int,
    ) -> Dict[Tuple[int, int], float]:
        """Compute danger map for enemy avoidance."""
        danger: Dict[Tuple[int, int], float] = {}
        side = str(getattr(agent, "side", "blue")).lower()
        enemy_team = game_field.red_agents if side == "blue" else game_field.blue_agents
        
        base_penalty = 3.0
        max_penalty = 8.0
        rr = max(1, radius)
        
        for e in enemy_team:
            if e is None:
                continue
            try:
                if hasattr(e, "isEnabled") and (not e.isEnabled()):
                    continue
            except Exception:
                pass
            
            ex, ey = self._agent_cell_pos(e, game_field)
            if ex is None or ey is None:
                continue
            
            for dx in range(-rr, rr + 1):
                for dy in range(-rr, rr + 1):
                    cx, cy = ex + dx, ey + dy
                    if not (0 <= cx < getattr(game_field, "col_count", 20) and 
                            0 <= cy < getattr(game_field, "row_count", 20)):
                        continue
                    d = max(abs(dx), abs(dy))
                    pen = base_penalty * float(rr + 1 - d)
                    if pen <= 0.0:
                        continue
                    prev = float(danger.get((cx, cy), 0.0))
                    danger[(cx, cy)] = min(max_penalty, max(prev, float(pen)))
        
        return danger
    
    def _find_safe_fallback_target(
        self,
        agent: Any,
        original_target: Optional[Tuple[int, int]],
        game_field: Any,
    ) -> Optional[Tuple[int, int]]:
        """Find safe fallback target when original is blocked."""
        if original_target is None:
            return None
        
        start = self._agent_cell_pos(agent, game_field)
        if start is None:
            return None
        
        # Try cells near original target
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                candidate = (original_target[0] + dx, original_target[1] + dy)
                clamped = self._clamp_cell(candidate[0], candidate[1], game_field)
                if clamped != start and self._is_cell_safe(clamped, agent, game_field):
                    return clamped
        
        # Fallback to home
        return self._get_home_position(agent, game_field)
    
    def _find_nearest_safe_cell_toward_target(
        self,
        agent: Any,
        target: Tuple[int, int],
        game_field: Any,
        safety_radius: int,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest safe cell in direction of target."""
        start = self._agent_cell_pos(agent, game_field)
        if start is None:
            return None
        
        # Try cells in direction of target
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        
        # Normalize direction
        if dx != 0 or dy != 0:
            dist = max(abs(dx), abs(dy))
            step_x = dx // dist if dist > 0 else 0
            step_y = dy // dist if dist > 0 else 0
            
            # Try 1-3 steps toward target
            for steps in [1, 2, 3]:
                candidate = (start[0] + step_x * steps, start[1] + step_y * steps)
                clamped = self._clamp_cell(candidate[0], candidate[1], game_field)
                if clamped != start and self._is_cell_safe(clamped, agent, game_field):
                    return clamped
        
        return None
    
    def _is_direct_path_clear(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        game_field: Any,
    ) -> bool:
        """Check if direct path is clear (simple line-of-sight)."""
        # Simple check: if start and end are adjacent or same, path is clear
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        return dx <= 1 and dy <= 1
    
    def _is_cell_safe(
        self,
        cell: Tuple[int, int],
        agent: Any,
        game_field: Any,
    ) -> bool:
        """Check if cell is safe (not blocked by obstacles)."""
        # Basic check: cell is within bounds
        cols = getattr(game_field, "col_count", 20)
        rows = getattr(game_field, "row_count", 20)
        if not (0 <= cell[0] < cols and 0 <= cell[1] < rows):
            return False
        
        # Check grid for obstacles (if available)
        grid = getattr(game_field, "grid", None)
        if grid is not None:
            try:
                if grid[cell[1]][cell[0]] != 0:  # Assuming 0 is free
                    return False
            except Exception:
                pass
        
        return True
    
    def _agent_cell_pos(self, agent: Any, game_field: Any) -> Optional[Tuple[int, int]]:
        """Get agent's cell position."""
        if agent is None:
            return None
        try:
            if hasattr(game_field, "_agent_cell_pos"):
                return game_field._agent_cell_pos(agent)
            x = int(getattr(agent, "x", 0))
            y = int(getattr(agent, "y", 0))
            return (x, y)
        except Exception:
            return None
    
    def _clamp_cell(self, x: int, y: int, game_field: Any) -> Tuple[int, int]:
        """Clamp cell coordinates to valid bounds."""
        cols = getattr(game_field, "col_count", 20)
        rows = getattr(game_field, "row_count", 20)
        return (max(0, min(cols - 1, x)), max(0, min(rows - 1, y)))
    
    def _get_home_position(self, agent: Any, game_field: Any) -> Optional[Tuple[int, int]]:
        """Get agent's home/base position."""
        try:
            side = str(getattr(agent, "side", "blue")).lower()
            gm = getattr(game_field, "manager", None)
            if gm is not None:
                home = gm.get_team_zone_center(side)
                if home:
                    return (int(home[0]), int(home[1]))
        except Exception:
            pass
        return None
    
    def _get_enemy_flag_position(self, agent: Any, game_field: Any) -> Optional[Tuple[int, int]]:
        """Get enemy flag position."""
        try:
            side = str(getattr(agent, "side", "blue")).lower()
            gm = getattr(game_field, "manager", None)
            if gm is not None:
                if side == "blue":
                    flag_pos = getattr(gm, "red_flag_position", None)
                else:
                    flag_pos = getattr(gm, "blue_flag_position", None)
                if flag_pos:
                    return (int(flag_pos[0]), int(flag_pos[1]))
        except Exception:
            pass
        return None
    
    def _distance_to_agent(self, agent: Any, obj: Any, game_field: Any) -> float:
        """Compute distance from agent to object."""
        agent_pos = self._agent_cell_pos(agent, game_field)
        if agent_pos is None:
            return float('inf')
        
        try:
            obj_x = int(getattr(obj, "x", 0))
            obj_y = int(getattr(obj, "y", 0))
            dx = agent_pos[0] - obj_x
            dy = agent_pos[1] - obj_y
            return float(dx * dx + dy * dy)
        except Exception:
            return float('inf')
    
    def _get_agent_key(self, agent: Any) -> Optional[str]:
        """Get unique key for agent (for tracking)."""
        if agent is None:
            return None
        try:
            side = str(getattr(agent, "side", "blue"))
            agent_id = int(getattr(agent, "agent_id", 0))
            return f"{side}_{agent_id}"
        except Exception:
            return None
    
    def get_failure_stats(self) -> Dict[str, int]:
        """Get action failure statistics."""
        return dict(self._action_failure_history)


__all__ = ["ExecutionController"]
