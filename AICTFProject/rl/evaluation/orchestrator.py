"""Evaluation orchestration for the V6I9 map-awareness pipeline."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rl.evaluation.aggregation import aggregate_conditions
from rl.evaluation.artifact_writer import report_text, write_csv
from rl.evaluation.config import MapAwarenessEvaluationConfig
from rl.evaluation.gates import build_summary
from rl.evaluation.manifest import (
    EvaluationManifest,
    begin_manifest,
    complete_manifest,
    fail_manifest,
    interrupt_manifest,
    json_safe,
)
from rl.evaluation.matched_seed import matched_seed_evaluation
from rl.evaluation.policy_loader import LoadedEvaluationPolicy, load_evaluation_policy, read_checkpoint_dimensions


@dataclass(frozen=True)
class EvaluationRunResult:
    exit_code: int
    output_dir: Path
    summary: Mapping[str, Any]
    episodes: Sequence[Mapping[str, Any]]
    conditions: Sequence[Mapping[str, Any]]
    probes: Mapping[str, Any]
    manifest: EvaluationManifest


@dataclass(frozen=True)
class EvaluationRuntime:
    project_root: Path
    command: Sequence[str]
    validate_opponent_name: Callable[[str], str]
    preflight_opponents: Callable[..., None]
    preflight_distribution_contract: Callable[..., None]
    inspect_obstacle_weights: Callable[[Any], Any]
    gradient_probe: Callable[..., Any]
    obstacle_counterfactual: Callable[..., Any]
    run_episode: Callable[..., dict[str, Any]]
    write_json_text: Callable[[Path, Any], None]


def namespace_from_config(config: MapAwarenessEvaluationConfig) -> Namespace:
    return Namespace(
        baseline=str(config.baseline_checkpoint),
        candidate=str(config.candidate_checkpoint),
        maps=list(config.maps),
        opponents=list(config.opponents),
        episodes=config.episodes_per_cell,
        seed_start=config.seed_start,
        device=config.device,
        output_dir=str(config.output_dir),
        max_decision_steps=config.max_decision_steps,
        counterfactual_steps=config.counterfactual_steps,
        obs_weight_threshold=config.obs_weight_threshold,
        gradient_threshold=config.gradient_threshold,
        counterfactual_kl_threshold=config.counterfactual_kl_threshold,
        counterfactual_action_threshold=config.counterfactual_action_threshold,
        navigation_improvement_threshold=config.navigation_improvement_threshold,
        route_difference_threshold=config.route_difference_threshold,
        minimum_win_rate=config.minimum_win_rate,
        competence_retention_tolerance=config.competence_retention_tolerance,
        saturation_win_rate=config.saturation_win_rate,
        allow_saturated_pool=config.allow_saturated_pool,
    )


def _artifact_paths(output_dir: Path) -> list[Path]:
    names = (
        "obstacle_probe.json",
        "episode_results.csv",
        "condition_summary.csv",
        "per_episode.csv",
        "per_condition.csv",
        "final_report.json",
        "summary.json",
        "final_report.txt",
        "evaluation_manifest.json",
    )
    return [output_dir / name for name in names if (output_dir / name).exists()]


def _load_and_preflight_policies(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
) -> tuple[LoadedEvaluationPolicy, LoadedEvaluationPolicy]:
    print("Loading native 7-channel baseline checkpoint...")
    baseline = load_evaluation_policy(
        "baseline",
        str(config.baseline_checkpoint),
        device=config.device,
        cnn_channels=config.baseline_cnn_channels,
    )
    print("... baseline loaded.")

    print("Loading native 8-channel candidate checkpoint...")
    candidate = load_evaluation_policy(
        "candidate",
        str(config.candidate_checkpoint),
        device=config.device,
        cnn_channels=config.candidate_cnn_channels,
    )
    print("... candidate loaded.")

    print("Preflighting public distribution contract...")
    runtime.preflight_distribution_contract(baseline.policy, label="baseline")
    runtime.preflight_distribution_contract(candidate.policy, label="candidate")
    print("... distribution contract OK.")
    return baseline, candidate


def _run_probes(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
    *,
    baseline_policy: Any,
    candidate_policy: Any,
    n_agents: int,
) -> dict[str, Any]:
    print("Running obstacle probes...")
    probes = {
        "baseline_weights": runtime.inspect_obstacle_weights(baseline_policy),
        "candidate_weights": runtime.inspect_obstacle_weights(candidate_policy),
        "candidate_gradient": runtime.gradient_probe(
            candidate_policy,
            device=config.device,
            map_name=config.reference_map,
            opponent=config.reference_opponent,
            n_agents=n_agents,
        ),
        "candidate_counterfactual": runtime.obstacle_counterfactual(
            candidate_policy,
            device=config.device,
            map_name=config.reference_map,
            opponent=config.reference_opponent,
            n_agents=n_agents,
            steps=config.counterfactual_steps,
        ),
    }
    print("... obstacle probes complete.")
    return probes


def _write_probe_artifact(config: MapAwarenessEvaluationConfig, runtime: EvaluationRuntime, probes: Mapping[str, Any]) -> None:
    probe_json = {
        key: value.to_json_dict() if hasattr(value, "to_json_dict") else value
        for key, value in probes.items()
    }
    runtime.write_json_text(config.output_dir / "obstacle_probe.json", probe_json)


def _write_result_artifacts(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
    *,
    summary: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(config.output_dir / "episode_results.csv", episodes)
    write_csv(config.output_dir / "condition_summary.csv", conditions)
    write_csv(config.output_dir / "per_episode.csv", episodes)
    write_csv(config.output_dir / "per_condition.csv", conditions)

    runtime.write_json_text(config.output_dir / "final_report.json", summary)
    runtime.write_json_text(config.output_dir / "summary.json", summary)

    report = report_text(summary)
    print("\n" + report)
    (config.output_dir / "final_report.txt").write_text(report + "\n", encoding="utf-8")


def run_evaluation(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
) -> EvaluationRunResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    args = namespace_from_config(config)
    args.opponents = [runtime.validate_opponent_name(opponent) for opponent in args.opponents]

    baseline_metadata, baseline_agents, _, _ = read_checkpoint_dimensions(str(config.baseline_checkpoint))
    candidate_metadata, candidate_agents, _, _ = read_checkpoint_dimensions(str(config.candidate_checkpoint))
    if baseline_agents != candidate_agents:
        raise ValueError(
            f"Baseline uses {baseline_agents} agents per team, but candidate uses {candidate_agents}."
        )
    n_agents = candidate_agents

    manifest = begin_manifest(
        config,
        command=runtime.command,
        project_root=runtime.project_root,
        baseline_metadata=baseline_metadata,
        candidate_metadata=candidate_metadata,
        n_agents=n_agents,
    )

    try:
        runtime.preflight_opponents(
            opponents=args.opponents,
            n_agents=n_agents,
            map_name=config.reference_map,
            device=config.device,
            max_steps=config.max_decision_steps,
        )

        baseline, candidate = _load_and_preflight_policies(config, runtime)
        probes = _run_probes(
            config,
            runtime,
            baseline_policy=baseline.policy,
            candidate_policy=candidate.policy,
            n_agents=n_agents,
        )
        _write_probe_artifact(config, runtime, probes)

        print("Running matched-seed evaluation...")
        episodes = matched_seed_evaluation(
            args,
            baseline.policy,
            candidate.policy,
            n_agents,
            run_episode_fn=runtime.run_episode,
            validate_opponent_name=runtime.validate_opponent_name,
        )
        conditions = aggregate_conditions(episodes)
        print("... matched-seed evaluation complete.")

        summary = build_summary(args, probes, episodes, conditions)
        _write_result_artifacts(
            config,
            runtime,
            summary=summary,
            episodes=episodes,
            conditions=conditions,
        )
        complete_manifest(manifest)
        print(f"\nArtifacts written to: {config.output_dir.resolve()}")
        stage2_ready = bool(summary.get("stage2_eligible"))
        return EvaluationRunResult(
            exit_code=0 if stage2_ready else 1,
            output_dir=config.output_dir,
            summary=summary,
            episodes=episodes,
            conditions=conditions,
            probes=probes,
            manifest=manifest,
        )
    except KeyboardInterrupt:
        interrupt_manifest(manifest, artifact_paths=_artifact_paths(config.output_dir))
        raise
    except Exception as exc:
        fail_manifest(manifest, exc, artifact_paths=_artifact_paths(config.output_dir))
        raise


__all__ = [
    "EvaluationRunResult",
    "EvaluationRuntime",
    "namespace_from_config",
    "run_evaluation",
]
