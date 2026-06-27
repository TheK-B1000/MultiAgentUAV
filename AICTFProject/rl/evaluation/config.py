"""Immutable configuration for the V6I9 map-awareness evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MapAwarenessEvaluationConfig:
    """All parameters that control one evaluation run.

    Constructed once from CLI arguments; passed down to every sub-component
    instead of forwarding ``argparse.Namespace``.
    """

    baseline_checkpoint: Path
    candidate_checkpoint: Path
    maps: tuple[str, ...]
    opponents: tuple[str, ...]
    episodes_per_cell: int
    seed_start: int
    device: str
    output_dir: Path
    max_decision_steps: int
    counterfactual_steps: int
    # gate thresholds
    obs_weight_threshold: float
    gradient_threshold: float
    counterfactual_kl_threshold: float
    counterfactual_action_threshold: float
    navigation_improvement_threshold: float
    route_difference_threshold: float
    minimum_win_rate: float
    competence_retention_tolerance: float
    saturation_win_rate: float

    # CNN channel counts for each policy role
    baseline_cnn_channels: int = 7
    candidate_cnn_channels: int = 8

    def validate(self) -> None:
        """Raise ValueError for invalid combinations."""
        if self.episodes_per_cell < 1:
            raise ValueError("episodes_per_cell must be >= 1.")
        if self.max_decision_steps < 1:
            raise ValueError("max_decision_steps must be >= 1.")
        if not self.maps:
            raise ValueError("At least one map is required.")
        if not self.opponents:
            raise ValueError("At least one opponent is required.")
        if not self.baseline_checkpoint.is_file():
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {self.baseline_checkpoint}"
            )
        if not self.candidate_checkpoint.is_file():
            raise FileNotFoundError(
                f"Candidate checkpoint not found: {self.candidate_checkpoint}"
            )

    @property
    def reference_map(self) -> str:
        """Last map in the list — used for probes and preflight."""
        return self.maps[-1]

    @property
    def reference_opponent(self) -> str:
        """First opponent in the list — used for probes."""
        return self.opponents[0]
