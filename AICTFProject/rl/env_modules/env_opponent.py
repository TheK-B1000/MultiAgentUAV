"""
Opponent switching module for CTFGameFieldSB3Env.

Handles opponent hot-swapping:
- Scripted opponents (OP1, OP2, OP3, etc.)
- Species opponents (BALANCED, RUSHER, CAMPER)
- Snapshot opponents (from saved checkpoints)
- Opponent context ID for observation embedding
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

try:
    from rl.episode_result import path_to_snapshot_key
except Exception:
    def path_to_snapshot_key(path: str) -> str:
        if not (path or str(path).strip()):
            return "SNAPSHOT:unknown"
        base = os.path.basename(str(path).strip())
        name, _ = os.path.splitext(base)
        return f"SNAPSHOT:{name}" if name else "SNAPSHOT:unknown"

from red_opponents import make_species_wrapper, make_snapshot_wrapper

# Try to import v2 wrapper for 4v4/8v8 support
try:
    from snapshot_wrapper_v2 import make_snapshot_wrapper_v2
    _HAS_V2_WRAPPER = True
except ImportError:
    _HAS_V2_WRAPPER = False
    make_snapshot_wrapper_v2 = None

# Step 4.1: max opponent context id (stable int in [0, MAX_OPPONENT_CONTEXT_IDS-1])
MAX_OPPONENT_CONTEXT_IDS = 256


class EnvOpponentManager:
    """Manages opponent switching and identity tracking."""

    def __init__(self, default_kind: str = "SCRIPTED", default_key: str = "OP1") -> None:
        self.default_opponent_kind = str(default_kind).upper()
        self.default_opponent_key = str(default_key).upper()
        self._next_opponent: Optional[Tuple[str, str]] = None
        self._opponent_kind = "scripted"  # "scripted" | "species" | "snapshot"
        self._opponent_snapshot_path: Optional[str] = None
        self._opponent_species_tag: str = "BALANCED"
        self._opponent_scripted_tag: str = "OP1"

    def set_next_opponent(self, kind: str, key: str) -> None:
        """Queue opponent for next reset."""
        k = str(kind).upper()
        v = str(key).upper() if k != "SNAPSHOT" else str(key)
        self._next_opponent = (k, v)

    def set_opponent_scripted(self, scripted_tag: str, game_field: Optional[Any] = None) -> None:
        """Set scripted opponent (OP1, OP2, OP3, etc.)."""
        tag = str(scripted_tag).upper()
        self._opponent_kind = "scripted"
        self._opponent_scripted_tag = tag
        self._opponent_species_tag = "BALANCED"
        self._opponent_snapshot_path = None
        if game_field is None:
            return
        game_field.set_red_opponent(tag)
        game_field.set_red_policy_wrapper(None)

    def set_opponent_species(self, species_tag: str, game_field: Optional[Any] = None) -> None:
        """Set species opponent (BALANCED, RUSHER, CAMPER)."""
        self._opponent_kind = "species"
        self._opponent_species_tag = str(species_tag).upper()
        self._opponent_snapshot_path = None
        self._opponent_scripted_tag = "OP3"
        if game_field is None:
            return
        wrapper = make_species_wrapper(self._opponent_species_tag)
        game_field.set_red_policy_wrapper(wrapper)

    def set_opponent_snapshot(self, snapshot_path: str, game_field: Optional[Any] = None) -> None:
        """Set snapshot opponent (from saved checkpoint)."""
        path = str(snapshot_path).strip()
        candidates = [path]
        if not path.lower().endswith(".zip"):
            candidates.append(path + ".zip")
        candidates.append(os.path.normpath(path))
        candidates.append(os.path.normpath(path + ("" if path.lower().endswith(".zip") else ".zip")))
        if path != path.lower():
            candidates.append(path.lower())
            candidates.append(os.path.normpath(path.lower()))
        found = None
        for p in candidates:
            if p and os.path.isfile(p):
                found = p
                break
        if found is None:
            try:
                print(f"[WARN] Snapshot not found: {path!r}; falling back to SCRIPTED:OP3")
            except Exception:
                pass
            self.set_opponent_scripted("OP3", game_field)
            return
        self._opponent_kind = "snapshot"
        self._opponent_snapshot_path = found
        self._opponent_species_tag = "BALANCED"
        self._opponent_scripted_tag = "OP3"
        if game_field is None:
            return
        
        # Use v2 wrapper for 4v4/8v8 (new obs space), v1 for 2v2 (legacy)
        # Detect agent count from game_field
        max_agents = int(getattr(game_field, "agents_per_team", 2) or 2)
        # Also check red_agents count as fallback (more reliable for 4v4/8v8)
        if hasattr(game_field, "red_agents"):
            red_agents = getattr(game_field, "red_agents", []) or []
            red_count = len([a for a in red_agents if a is not None])
            if red_count > 2:
                max_agents = max(max_agents, red_count)
        # Also check blue_agents count (for consistency)
        if hasattr(game_field, "blue_agents"):
            blue_agents = getattr(game_field, "blue_agents", []) or []
            blue_count = len([a for a in blue_agents if a is not None])
            if blue_count > 2:
                max_agents = max(max_agents, blue_count)
        
        try:
            if _HAS_V2_WRAPPER and max_agents > 2:
                try:
                    wrapper = make_snapshot_wrapper_v2(
                        self._opponent_snapshot_path,
                        max_agents=max_agents,
                        n_macros=int(getattr(game_field, "num_macro_actions", 5) or 5),
                        n_targets=int(getattr(game_field, "num_macro_targets", 8) or 8),
                    )
                except Exception as e:
                    print(f"[WARN] Failed to load snapshot with v2 wrapper: {e}; falling back to v1")
                    wrapper = make_snapshot_wrapper(self._opponent_snapshot_path)
            else:
                wrapper = make_snapshot_wrapper(self._opponent_snapshot_path)
            game_field.set_red_policy_wrapper(wrapper)
        except Exception as e:
            print(f"[WARN] Snapshot load failed (corrupt?): {self._opponent_snapshot_path!r}: {e}; using SCRIPTED:OP3")
            self._opponent_snapshot_path = None
            self._opponent_kind = "SCRIPTED"
            self._opponent_scripted_tag = "OP3"
            self.set_opponent_scripted("OP3", game_field)
            try:
                from opponent_params import sample_opponent_params
                rng = __import__("random").Random()
                n_agents = int(getattr(game_field, "agents_per_team", 2))
                params = sample_opponent_params(kind="SCRIPTED", key="OP3", phase="OP3", rng=rng, n_agents=n_agents)
                if hasattr(game_field, "set_opponent_params"):
                    game_field.set_opponent_params(params)
            except Exception:
                pass

    def _opponent_context_id(self) -> int:
        """Stable opponent embedding id in [0, MAX_OPPONENT_CONTEXT_IDS-1]."""
        kind = str(self._opponent_kind or "scripted").upper()
        if kind == "SCRIPTED":
            tag = str(self._opponent_scripted_tag or "OP3").upper()
            scripted_map = {"OP1": 0, "OP2": 1, "OP3": 2}
            return scripted_map.get(tag, 2)
        if kind == "SPECIES":
            tag = str(self._opponent_species_tag or "BALANCED").upper()
            species_map = {"BALANCED": 3, "RUSHER": 4, "CAMPER": 5}
            return species_map.get(tag, 3)
        if kind == "SNAPSHOT":
            key = path_to_snapshot_key(self._opponent_snapshot_path or "")
            h = hash(key) % (MAX_OPPONENT_CONTEXT_IDS - 6)
            return 6 + (h if h >= 0 else h + (MAX_OPPONENT_CONTEXT_IDS - 6))
        return 0

    def get_opponent_kind(self) -> str:
        return self._opponent_kind

    def get_opponent_scripted_tag(self) -> str:
        return self._opponent_scripted_tag

    def get_opponent_species_tag(self) -> str:
        return self._opponent_species_tag

    def get_opponent_snapshot_path(self) -> Optional[str]:
        return self._opponent_snapshot_path

    def apply_opponent_at_reset(self, game_field: Any, phase_name: str, seed: int) -> None:
        """Apply opponent setting at reset (handles next_opponent queue and OpponentParams)."""
        if self._next_opponent is not None:
            kind, key = self._next_opponent
            kind = str(kind).upper()
            key = str(key) if kind == "SNAPSHOT" else str(key).upper()
            self._next_opponent = None
        else:
            kind = self.default_opponent_kind
            key = self.default_opponent_key

        if kind == "SCRIPTED":
            self.set_opponent_scripted(key, game_field)
        elif kind == "SPECIES":
            self.set_opponent_species(key, game_field)
        elif kind == "SNAPSHOT":
            self.set_opponent_snapshot(key, game_field)

        # Apply OpponentParams (speed multipliers, etc.); 4v4 uses easier OP3 via n_agents
        try:
            from opponent_params import sample_opponent_params
            phase = str(phase_name).upper()
            rng = __import__("random").Random(int(seed) + 1)
            n_agents = int(getattr(game_field, "agents_per_team", 2))
            params = sample_opponent_params(kind=kind, key=key, phase=phase, rng=rng, n_agents=n_agents)
            if hasattr(game_field, "set_opponent_params"):
                game_field.set_opponent_params(params)
        except Exception:
            pass


__all__ = ["EnvOpponentManager", "MAX_OPPONENT_CONTEXT_IDS"]
