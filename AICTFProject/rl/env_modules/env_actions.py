"""
Action sanitization, masks, and noise module for CTFGameFieldSB3Env.

Handles:
- Action execution noise (flip probability)
- Action sanitization (mask enforcement)
- Role-based macro masking
- Target mask validation and fallback
- Macro usage tracking
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import numpy as np

from macro_actions import MacroAction


class EnvActionManager:
    """Manages action sanitization, masking, and noise."""

    def __init__(
        self,
        *,
        enforce_masks: bool = True,
        action_flip_prob: float = 0.0,
        n_macros: int = 5,
        n_targets: int = 8,
        blue_role_macros: Optional[Tuple[List[int], List[int]]] = None,
        seed: int = 0,
    ) -> None:
        self._enforce_masks = bool(enforce_masks)
        self._action_flip_prob = float(max(0.0, min(1.0, action_flip_prob)))
        self._n_macros = int(n_macros)
        self._n_targets = int(n_targets)
        self._blue_role_macros = blue_role_macros
        self._action_rng = np.random.RandomState(int(seed) + 1000)
        self._episode_macro_counts: List[List[int]] = []
        self._episode_mine_counts: List[Dict[str, int]] = []

    def reset_episode(self, n_agents: int) -> None:
        """Reset episode-level action tracking."""
        self._episode_macro_counts = [[0 for _ in range(self._n_macros)] for _ in range(n_agents)]
        self._episode_mine_counts = [{"grab": 0, "place": 0} for _ in range(n_agents)]

    def apply_noise_and_sanitize(
        self,
        intended_actions: List[Tuple[int, int]],
        n_blue_agents: int,
        game_field: Any,
        role_macro_mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[List[Tuple[int, int]], int, int, int]:
        """
        Apply action noise and sanitize actions.
        
        Returns:
            (executed_actions, flip_count_step, macro_flip_count_step, target_flip_count_step)
        """
        executed: List[Tuple[int, int]] = []
        flip_count_step = 0
        macro_flip_count_step = 0
        target_flip_count_step = 0

        for i in range(min(n_blue_agents, len(intended_actions))):
            m, t = intended_actions[i]

            # Apply noise
            if self._action_flip_prob > 0.0:
                if self._action_rng.rand() < self._action_flip_prob:
                    m = int(self._action_rng.randint(0, self._n_macros))
                    flip_count_step += 1
                    macro_flip_count_step += 1
                if self._action_rng.rand() < self._action_flip_prob:
                    t = int(self._action_rng.randint(0, self._n_targets))
                    flip_count_step += 1
                    target_flip_count_step += 1

            # Sanitize
            m, t = self._sanitize_action_for_agent(
                i, m, t, game_field, role_macro_mask_fn
            )

            executed.append((m, t))
            self._record_macro_usage(i, m, game_field)

        # Target deconfliction: if two agents choose same target cell, perturb one to nearest free cell
        executed = self._deconflict_targets(executed, n_blue_agents, game_field)

        # Pad to n_slots
        n_slots = len(intended_actions)
        while len(executed) < n_slots:
            executed.append((0, 0))

        return executed, flip_count_step, macro_flip_count_step, target_flip_count_step

    def _sanitize_action_for_agent(
        self,
        blue_index: int,
        macro: int,
        tgt: int,
        game_field: Any,
        role_macro_mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[int, int]:
        """Sanitize action for agent (enforce masks, validate targets)."""
        macro = int(macro) % max(1, self._n_macros)
        tgt = int(tgt) % max(1, self._n_targets)

        if not self._enforce_masks:
            return macro, tgt

        blue_agents = getattr(game_field, "blue_agents", []) or []
        if blue_index >= len(blue_agents):
            return 0, 0

        agent = blue_agents[blue_index]
        if agent is None:
            return 0, 0

        mm = np.asarray(game_field.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        if role_macro_mask_fn is not None:
            mm = role_macro_mask_fn(blue_index, mm)
        if mm.shape != (self._n_macros,) or (not mm.any()):
            return macro, tgt

        if not bool(mm[macro]):
            macro = 0  # GO_TO fallback

        if self._macro_uses_target(macro, game_field):
            tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
            if tm.shape == (self._n_targets,) and tm.any():
                if not bool(tm[tgt]):
                    tgt = self._nearest_valid_target(agent, tm, game_field)

        return macro, tgt

    def _deconflict_targets(
        self,
        actions: List[Tuple[int, int]],
        n_blue_agents: int,
        game_field: Any,
    ) -> List[Tuple[int, int]]:
        """
        Target deconfliction: if two agents choose same target cell, perturb one to nearest free cell.
        Only applies to macros that should NOT allow duplicates: GO_TO, GET_FLAG, PLACE_MINE.
        Allows duplicates for swarm-allowed macros: GO_HOME (DEFEND_HOME), GRAB_MINE, etc.
        """
        if n_blue_agents < 2 or len(actions) < 2:
            return actions
        
        # Macros that should be deconflicted (no duplicates allowed)
        deconflict_macros = {MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.PLACE_MINE}
        
        # Get target cells for each agent (only for deconflict-eligible macros)
        target_cells: Dict[int, Tuple[int, int]] = {}  # agent_idx -> (x, y) cell
        macro_types: Dict[int, MacroAction] = {}  # agent_idx -> MacroAction
        
        for i in range(min(n_blue_agents, len(actions))):
            m, t = actions[i]
            
            # Get macro action enum
            macro_action = None
            if game_field is not None:
                try:
                    macro_action = game_field.macro_order[int(m)]
                except Exception:
                    try:
                        macro_action = MacroAction(int(m))
                    except Exception:
                        pass
            
            # Only deconflict if macro is in the deconflict set
            if macro_action is not None and macro_action in deconflict_macros:
                macro_types[i] = macro_action
                if self._macro_uses_target(m, game_field):
                    try:
                        cell = game_field.get_macro_target(int(t))
                        if cell is not None:
                            target_cells[i] = tuple(cell) if isinstance(cell, (tuple, list)) else (int(cell[0]), int(cell[1]))
                    except Exception:
                        pass
        
        # Find duplicates (only among deconflict-eligible macros)
        cell_to_agents: Dict[Tuple[int, int], List[int]] = {}
        for agent_idx, cell in target_cells.items():
            # Only consider duplicates if both agents have the same macro type
            macro = macro_types.get(agent_idx)
            if macro is None:
                continue
            
            # Group by (cell, macro) to avoid deconflicting different macro types at same cell
            key = (cell, macro)
            if key not in cell_to_agents:
                cell_to_agents[key] = []
            cell_to_agents[key].append(agent_idx)
        
        # For each duplicate, perturb one agent to nearest free cell
        new_actions = list(actions)
        for (cell, macro), agent_indices in cell_to_agents.items():
            if len(agent_indices) <= 1:
                continue
            
            # Keep first agent's target, perturb others
            for agent_idx in agent_indices[1:]:
                m, t = new_actions[agent_idx]
                if not self._macro_uses_target(m, game_field):
                    continue
                
                # Find nearest free target cell
                try:
                    free_tgt = self._nearest_free_target(agent_idx, cell, game_field, list(target_cells.values()))
                    if free_tgt is not None:
                        new_actions[agent_idx] = (m, free_tgt)
                        # Update target_cells for this agent to avoid double-deconfliction
                        try:
                            new_cell = game_field.get_macro_target(int(free_tgt))
                            if new_cell is not None:
                                target_cells[agent_idx] = tuple(new_cell) if isinstance(new_cell, (tuple, list)) else (int(new_cell[0]), int(new_cell[1]))
                        except Exception:
                            pass
                except Exception:
                    pass  # Keep original target if deconfliction fails
        
        return new_actions

    def _nearest_free_target(
        self,
        agent_idx: int,
        avoid_cell: Tuple[int, int],
        game_field: Any,
        used_cells: List[Tuple[int, int]],
    ) -> Optional[int]:
        """Find nearest target cell that's not in used_cells."""
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if agent_idx >= len(blue_agents) or blue_agents[agent_idx] is None:
            return None
        
        agent = blue_agents[agent_idx]
        try:
            tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
            if tm.shape != (self._n_targets,) or not tm.any():
                return None
            
            used_set = set(used_cells)
            best_tgt = None
            best_dist = float('inf')
            
            for tgt_idx in range(self._n_targets):
                if not bool(tm[tgt_idx]):
                    continue
                try:
                    tgt_cell = game_field.get_macro_target(int(tgt_idx))
                    if tgt_cell is None:
                        continue
                    tgt_tuple = tuple(tgt_cell) if isinstance(tgt_cell, (tuple, list)) else (int(tgt_cell[0]), int(tgt_cell[1]))
                    
                    # Skip if already used
                    if tgt_tuple in used_set:
                        continue
                    
                    # Compute distance from avoid_cell
                    dist = abs(tgt_tuple[0] - avoid_cell[0]) + abs(tgt_tuple[1] - avoid_cell[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_tgt = tgt_idx
                except Exception:
                    continue
            
            return best_tgt
        except Exception:
            return None

    def _apply_role_macro_mask(self, blue_index: int, mm: np.ndarray) -> np.ndarray:
        """Apply role-based macro mask if configured."""
        if self._blue_role_macros is None:
            return mm
        if not isinstance(self._blue_role_macros, (tuple, list)):
            return mm
        if blue_index >= len(self._blue_role_macros):
            return mm

        allowed = self._blue_role_macros[blue_index]
        if not allowed:
            return mm

        role_mask = np.zeros_like(mm, dtype=np.bool_)
        for idx in allowed:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < role_mask.size:
                role_mask[i] = True

        out = mm & role_mask
        return out if out.any() else mm

    def _macro_uses_target(self, macro_idx: int, game_field: Any) -> bool:
        """Check if macro action requires a target."""
        if game_field is None:
            return True
        try:
            action = game_field.macro_order[int(macro_idx)]
        except Exception:
            return True
        return action in (MacroAction.GO_TO, MacroAction.PLACE_MINE)

    def _nearest_valid_target(
        self, agent: Any, tgt_mask: np.ndarray, game_field: Any
    ) -> int:
        """Find nearest valid target when requested target is masked."""
        valid = np.flatnonzero(tgt_mask)
        if valid.size == 0:
            return 0

        try:
            ax = float(getattr(agent, "x", 0.0))
            ay = float(getattr(agent, "y", 0.0))
            best_idx = int(valid[0])
            best_d2 = None
            for i in valid:
                tx, ty = game_field.get_macro_target(int(i))
                d2 = (float(tx) - ax) ** 2 + (float(ty) - ay) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_idx = int(i)
            return best_idx
        except Exception:
            return int(valid[0])

    def _record_macro_usage(self, blue_index: int, macro_idx: int, game_field: Any) -> None:
        """Record macro usage for episode metrics."""
        if blue_index >= len(self._episode_macro_counts):
            return
        if not self._episode_macro_counts:
            return
        if 0 <= macro_idx < len(self._episode_macro_counts[blue_index]):
            self._episode_macro_counts[blue_index][macro_idx] += 1

        if not self._episode_mine_counts or blue_index >= len(self._episode_mine_counts):
            return

        action = None
        if game_field is not None:
            try:
                action = game_field.macro_order[int(macro_idx)]
            except Exception:
                action = None

        if action is None:
            try:
                action = MacroAction(int(macro_idx))
            except Exception:
                action = None

        if action == MacroAction.GRAB_MINE:
            self._episode_mine_counts[blue_index]["grab"] += 1
        elif action == MacroAction.PLACE_MINE:
            self._episode_mine_counts[blue_index]["place"] += 1

    def get_episode_macro_counts(self) -> List[List[int]]:
        return [list(row) for row in self._episode_macro_counts]

    def get_episode_mine_counts(self) -> List[Dict[str, int]]:
        return [dict(row) for row in self._episode_mine_counts]

    def get_macro_order_names(self, game_field: Any) -> List[str]:
        """Get macro order names for logging."""
        if game_field is None:
            return []
        names: List[str] = []
        try:
            for m in game_field.macro_order:
                n = getattr(m, "name", None)
                names.append(str(n) if n is not None else str(m))
        except Exception:
            pass
        return names


__all__ = ["EnvActionManager"]
