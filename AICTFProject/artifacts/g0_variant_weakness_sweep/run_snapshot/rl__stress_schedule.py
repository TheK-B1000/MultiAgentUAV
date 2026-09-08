"""Per-phase water-current / drift overrides used by ``BatchedCTFCore._apply_profile_runtime``.

Replaces the old ``rl.curriculum`` module after curriculum removal. Phases OP1–OP3 still
appear in scripted training; empty dicts mean \"use GPUFieldConfig defaults only\".
"""

from __future__ import annotations

from typing import Any, Dict

# Keys optional per phase: current_strength_cps, drift_sigma_cells (see game_field_gpu).
STRESS_BY_PHASE: Dict[str, Dict[str, Any]] = {
    "OP1": {},
    "OP2": {},
    "OP3": {},
    "SELF_PLAY": {},
    "FIXED": {},
}
