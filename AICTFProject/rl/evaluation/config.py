"""Immutable configuration for the V6I9 map-awareness evaluation."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from rl.evaluation.errors import EvaluationConfigError


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
    obs_weight_threshold: float
    gradient_threshold: float
    counterfactual_kl_threshold: float
    counterfactual_action_threshold: float
    navigation_improvement_threshold: float
    route_difference_threshold: float
    minimum_win_rate: float
    competence_retention_tolerance: float
    saturation_win_rate: float
    allow_saturated_pool: bool = False
    baseline_cnn_channels: int = 7
    candidate_cnn_channels: int = 8

    def validate(self) -> None:
        """Raise for invalid combinations before execution starts."""
        if self.episodes_per_cell < 1:
            raise EvaluationConfigError("episodes_per_cell must be >= 1.")
        if self.max_decision_steps < 1:
            raise EvaluationConfigError("max_decision_steps must be >= 1.")
        if not self.maps:
            raise EvaluationConfigError("At least one map is required.")
        if not self.opponents:
            raise EvaluationConfigError("At least one opponent is required.")
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
        """Last map in the list, used for probes and preflight."""
        return self.maps[-1]

    @property
    def reference_opponent(self) -> str:
        """First opponent in the list, used for probes."""
        return self.opponents[0]


def config_from_namespace(args: Namespace) -> MapAwarenessEvaluationConfig:
    """Build typed config from the legacy evaluator CLI namespace."""
    config = MapAwarenessEvaluationConfig(
        baseline_checkpoint=Path(args.baseline),
        candidate_checkpoint=Path(args.candidate),
        maps=tuple(args.maps),
        opponents=tuple(args.opponents),
        episodes_per_cell=int(args.episodes),
        seed_start=int(args.seed_start),
        device=str(args.device),
        output_dir=Path(args.output_dir),
        max_decision_steps=int(args.max_decision_steps),
        counterfactual_steps=int(args.counterfactual_steps),
        obs_weight_threshold=float(args.obs_weight_threshold),
        gradient_threshold=float(args.gradient_threshold),
        counterfactual_kl_threshold=float(args.counterfactual_kl_threshold),
        counterfactual_action_threshold=float(args.counterfactual_action_threshold),
        navigation_improvement_threshold=float(args.navigation_improvement_threshold),
        route_difference_threshold=float(args.route_difference_threshold),
        minimum_win_rate=float(args.minimum_win_rate),
        competence_retention_tolerance=float(args.competence_retention_tolerance),
        saturation_win_rate=float(args.saturation_win_rate),
        allow_saturated_pool=bool(getattr(args, "allow_saturated_pool", False)),
    )
    config.validate()
    return config


__all__ = ["MapAwarenessEvaluationConfig", "config_from_namespace"]
