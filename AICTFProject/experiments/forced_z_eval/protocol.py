"""Canonical matched-seed forced-z evaluation protocol.

All forced-z runners and analysis passes must use these defaults unless an
experiment explicitly documents a deviation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Tuple

# Grid defaults (V6I8/V6I9 hard pool).
DEFAULT_OPPONENTS: Tuple[str, ...] = ("OP8", "OP9", "OP10")
DEFAULT_MAPS: Tuple[str, ...] = ("map_b", "map_b_split_lane_v2")
DEFAULT_LATENTS: Tuple[int, ...] = (0, 1, 2, 3)

# Matched-seed contract.
DEFAULT_BASE_SEED: int = 42
DEFAULT_EPISODES_PER_CELL: int = 100
DEFAULT_DETERMINISTIC_ACTIONS: bool = True  # greedy argmax; set False only with --stochastic
DEFAULT_MAX_DECISION_STEPS: int = 400

# Artifact names inside a run directory.
EPISODE_RESULTS_CSV = "episode_results.csv"
RUN_MANIFEST_JSON = "run_manifest.json"
STAGE_C_JSON = "stage_c_report.json"
COMPLEMENTARITY_JSON = "complementarity_report.json"
ORACLE_JSON = "oracle_report.json"
BEHAVIOR_JSON = "behavior_report.json"


@dataclass
class ForcedZProtocol:
    """Resolved protocol for one forced-z evaluation run."""

    checkpoint: str
    opponents: Tuple[str, ...] = DEFAULT_OPPONENTS
    maps: Tuple[str, ...] = DEFAULT_MAPS
    latents: Tuple[int, ...] = DEFAULT_LATENTS
    episodes_per_cell: int = DEFAULT_EPISODES_PER_CELL
    base_seed: int = DEFAULT_BASE_SEED
    deterministic_actions: bool = DEFAULT_DETERMINISTIC_ACTIONS
    max_decision_steps: int = DEFAULT_MAX_DECISION_STEPS
    env_reward_kwargs: dict[str, Any] = field(default_factory=dict)
    training_run_config: str | None = None
    device: str = "cuda"
    collect_behavior_mean: bool = True
    progress_every: int = 25

    def cell_seed(self, opponent_index: int, map_index: int) -> int:
        """Environment seed base for a (opponent, map) block (shared across z)."""
        return int(self.base_seed) + 1000 * int(opponent_index) + 100 * int(map_index)

    def episode_seed(self, cell_seed: int, episode_index: int) -> int:
        """Per-episode seed (matched across z for the same episode_index)."""
        return int(cell_seed) + int(episode_index)

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def audit_protocol_note(protocol: ForcedZProtocol | None = None) -> str:
    """Human-readable contract summary for run logs."""
    steps = DEFAULT_MAX_DECISION_STEPS
    surface_note = ""
    if protocol is not None:
        steps = int(protocol.max_decision_steps)
        if protocol.env_reward_kwargs:
            keys = sorted(protocol.env_reward_kwargs)
            surface_note = (
                " env_reward_overrides={"
                + ", ".join(f"{k}={protocol.env_reward_kwargs[k]}" for k in keys)
                + "},"
            )
        if protocol.training_run_config:
            surface_note += f" training_run_config={protocol.training_run_config!r},"
    return (
        "Forced-z protocol: "
        f"base_seed={DEFAULT_BASE_SEED if protocol is None else protocol.base_seed}, "
        f"deterministic_actions={DEFAULT_DETERMINISTIC_ACTIONS if protocol is None else protocol.deterministic_actions}, "
        "cell_seed=base+1000*opp_idx+100*map_idx, "
        "episode_seed=cell_seed+ep_idx, "
        f"max_decision_steps={steps},{surface_note} "
        "sampling=argmax unless --stochastic."
    )


__all__ = [
    "COMPLEMENTARITY_JSON",
    "BEHAVIOR_JSON",
    "DEFAULT_BASE_SEED",
    "DEFAULT_DETERMINISTIC_ACTIONS",
    "DEFAULT_EPISODES_PER_CELL",
    "DEFAULT_LATENTS",
    "DEFAULT_MAPS",
    "DEFAULT_MAX_DECISION_STEPS",
    "DEFAULT_OPPONENTS",
    "EPISODE_RESULTS_CSV",
    "ForcedZProtocol",
    "ORACLE_JSON",
    "RUN_MANIFEST_JSON",
    "STAGE_C_JSON",
    "audit_protocol_note",
]
