"""Evaluation orchestration for the V6I9 map-awareness pipeline.

Formal identity border (mandatory when ``require_formal_identity`` is True):

    create live evaluation environment
            ↓
    resolve evaluation RunIdentity from that live environment
            ↓
    load checkpoint *metadata* / identity payload (no model weights yet)
            ↓
    verify checkpoint identity against evaluation identity
            ↓
    write stamped evaluation artifacts (manifest first)
            ↓
    only then load model weights / run policy inference
"""
from __future__ import annotations

import uuid
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from rl.evaluation.aggregation import aggregate_conditions
from rl.evaluation.artifact_writer import report_text, write_csv
from rl.evaluation.config import MapAwarenessEvaluationConfig
from rl.evaluation.env_factory import make_env
from rl.evaluation.gates import build_summary
from rl.evaluation.manifest import (
    EvaluationManifest,
    begin_manifest,
    complete_manifest,
    fail_manifest,
    interrupt_manifest,
)
from rl.evaluation.matched_seed import matched_seed_evaluation
from rl.evaluation.policy_loader import LoadedEvaluationPolicy, load_evaluation_policy, read_checkpoint_dimensions
from rl.ruleset_identity import (
    RunIdentity,
    RunIdentityError,
    assert_checkpoint_compatible_with_evaluation_identity,
    build_evaluation_run_identity,
    maps_compatible,
    stamp_csv_row,
    stamp_json_artifact,
    validate_bundle,
)
from rl.training.run_artifacts import write_result_summary_json


@dataclass(frozen=True)
class EvaluationRunResult:
    exit_code: int
    output_dir: Path
    summary: Mapping[str, Any]
    episodes: Sequence[Mapping[str, Any]]
    conditions: Sequence[Mapping[str, Any]]
    probes: Mapping[str, Any]
    manifest: EvaluationManifest
    evaluation_identity: Optional[RunIdentity] = None
    lineage: Mapping[str, Any] | None = None


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
    # Test instrumentation: fired at the first model-weight load / forward path.
    on_model_execution: Callable[[], None] | None = None


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
        "episode_rows.csv",
        "condition_summary.csv",
        "per_episode.csv",
        "per_condition.csv",
        "final_report.json",
        "summary.json",
        "result_summary.json",
        "final_report.txt",
        "evaluation_manifest.json",
    )
    return [output_dir / name for name in names if (output_dir / name).exists()]


def _notify_model_execution(runtime: EvaluationRuntime) -> None:
    cb = runtime.on_model_execution
    if cb is not None:
        cb()


def _resolve_eval_run_id(config: MapAwarenessEvaluationConfig) -> str:
    return str(config.evaluation_run_id or f"eval_{uuid.uuid4().hex[:12]}")


def _assert_requested_maps_compatible_with_identity(
    config: MapAwarenessEvaluationConfig,
    identity: RunIdentity,
) -> None:
    """CLI-requested maps must not disagree with the live-resolved identity."""
    for requested in config.maps:
        if not maps_compatible(
            identity.canonical_map,
            identity.resolved_map,
            requested,
            requested,
        ):
            raise RunIdentityError(
                f"Requested evaluation map {requested!r} is incompatible with "
                f"live evaluation identity "
                f"({identity.canonical_map!r}/{identity.resolved_map!r}); "
                "refusing inference."
            )


def gate_evaluation_identity(
    config: MapAwarenessEvaluationConfig,
    *,
    n_agents: int,
) -> tuple[Any, RunIdentity, dict]:
    """Create the live eval env, resolve identity, verify checkpoints.

    Returns ``(eval_env, evaluation_identity, lineage)``. Model weights are
    NOT loaded here — only checkpoint identity metadata is read.
    """
    eval_env = make_env(
        n_agents=n_agents,
        map_name=config.reference_map,
        device=config.device,
        seed=int(config.seed_start),
        max_steps=int(config.max_decision_steps),
        instrumented=False,
    )
    try:
        evaluation_run_id = _resolve_eval_run_id(config)
        evaluation_identity, lineage = build_evaluation_run_identity(
            eval_env,
            evaluation_run_id=evaluation_run_id,
            source_checkpoint=config.candidate_checkpoint,
            allow_override=bool(config.allow_identity_override),
        )
        assert_checkpoint_compatible_with_evaluation_identity(
            evaluation_identity,
            config.baseline_checkpoint,
            allow_override=bool(config.allow_identity_override),
            context=str(config.baseline_checkpoint),
        )
        _assert_requested_maps_compatible_with_identity(config, evaluation_identity)
        return eval_env, evaluation_identity, lineage
    except Exception:
        eval_env.close()
        raise


def _load_and_preflight_policies(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
) -> tuple[LoadedEvaluationPolicy, LoadedEvaluationPolicy]:
    _notify_model_execution(runtime)

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


def _write_probe_artifact(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
    probes: Mapping[str, Any],
) -> None:
    probe_json = {
        key: value.to_json_dict() if hasattr(value, "to_json_dict") else value
        for key, value in probes.items()
    }
    runtime.write_json_text(config.output_dir / "obstacle_probe.json", probe_json)


def _stamp_episode_rows(
    episodes: Sequence[Mapping[str, Any]],
    identity: RunIdentity,
) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for ep in episodes:
        row = dict(ep)
        stamp_csv_row(row, identity)
        stamped.append(row)
    return stamped


def _write_formal_evaluation_artifacts(
    config: MapAwarenessEvaluationConfig,
    *,
    identity: RunIdentity,
    lineage: Mapping[str, Any],
    summary: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
    runtime: EvaluationRuntime,
    manifest: EvaluationManifest | None = None,
) -> list[dict[str, Any]]:
    """Stamp episode_rows / result_summary; keep evaluation_manifest on ``manifest``."""
    stamped_rows = _stamp_episode_rows(episodes, identity)

    if manifest is not None:
        stamp_json_artifact(manifest.data, identity)
        manifest.data.update(dict(lineage))
        manifest.data["evaluation_run_id"] = identity.run_id
        manifest.data["source_training_run_id"] = lineage.get("source_training_run_id", "")
        manifest.data["source_checkpoint_id"] = lineage.get(
            "source_checkpoint_id", lineage.get("source_checkpoint_fingerprint", "")
        )
        manifest.data["source_checkpoint_ruleset_fingerprint"] = lineage.get(
            "source_checkpoint_ruleset_fingerprint", ""
        )
        manifest.data["scope"] = "standalone_evaluation"
        manifest.data["live_canonical_map"] = identity.canonical_map
        manifest.data["live_resolved_map"] = identity.resolved_map
        manifest.write()

    write_csv(config.output_dir / "episode_rows.csv", stamped_rows)
    write_csv(config.output_dir / "episode_results.csv", stamped_rows)
    write_csv(config.output_dir / "condition_summary.csv", conditions)
    write_csv(config.output_dir / "per_episode.csv", stamped_rows)
    write_csv(config.output_dir / "per_condition.csv", conditions)

    formal_summary = stamp_json_artifact(dict(summary), identity)
    formal_summary["n_episode_rows"] = len(stamped_rows)
    formal_summary["evaluation_run_id"] = identity.run_id
    formal_summary.update({k: lineage[k] for k in lineage})

    write_result_summary_json(
        str(config.output_dir / "result_summary.json"),
        formal_summary,
        run_identity=identity,
        source_rows=stamped_rows if stamped_rows else None,
    )

    runtime.write_json_text(config.output_dir / "final_report.json", formal_summary)
    runtime.write_json_text(config.output_dir / "summary.json", formal_summary)
    report = report_text(formal_summary)
    print("\n" + report)
    (config.output_dir / "final_report.txt").write_text(report + "\n", encoding="utf-8")

    if stamped_rows and manifest is not None:
        validate_bundle(
            {
                "evaluation_manifest.json": manifest.data,
                "result_summary.json": formal_summary,
            },
            {"episode_rows.csv": stamped_rows},
            require_formal=bool(identity.formal_result_eligible),
        )
    return stamped_rows


def _write_result_artifacts(
    config: MapAwarenessEvaluationConfig,
    runtime: EvaluationRuntime,
    *,
    summary: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> None:
    """Legacy unstamped writer kept for ``require_formal_identity=False`` tests."""
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

    # Dimension metadata only — does not execute the policy network.
    baseline_metadata, baseline_agents, _, _ = read_checkpoint_dimensions(
        str(config.baseline_checkpoint)
    )
    candidate_metadata, candidate_agents, _, _ = read_checkpoint_dimensions(
        str(config.candidate_checkpoint)
    )
    if baseline_agents != candidate_agents:
        raise ValueError(
            f"Baseline uses {baseline_agents} agents per team, but candidate uses {candidate_agents}."
        )
    n_agents = candidate_agents

    evaluation_identity: RunIdentity | None = None
    lineage: dict[str, Any] | None = None
    eval_env = None

    if bool(config.require_formal_identity):
        eval_env, evaluation_identity, lineage = gate_evaluation_identity(
            config, n_agents=n_agents
        )
        print(
            f"[EVAL] Identity: {evaluation_identity.ruleset_id} "
            f"map={evaluation_identity.canonical_map}/"
            f"{evaluation_identity.resolved_map} "
            f"fingerprint={evaluation_identity.ruleset_fingerprint[:12]} "
            f"formal_eligible={evaluation_identity.formal_result_eligible}"
        )

    manifest = begin_manifest(
        config,
        command=runtime.command,
        project_root=runtime.project_root,
        baseline_metadata=baseline_metadata,
        candidate_metadata=candidate_metadata,
        n_agents=n_agents,
    )
    if evaluation_identity is not None and lineage is not None:
        stamp_json_artifact(manifest.data, evaluation_identity)
        manifest.data.update(dict(lineage))
        manifest.data["evaluation_run_id"] = evaluation_identity.run_id
        manifest.write()

    try:
        runtime.preflight_opponents(
            opponents=args.opponents,
            n_agents=n_agents,
            map_name=config.reference_map,
            device=config.device,
            max_steps=config.max_decision_steps,
        )

        # Model weights load ONLY after the identity gate has passed.
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
        if evaluation_identity is not None and lineage is not None:
            _write_formal_evaluation_artifacts(
                config,
                identity=evaluation_identity,
                lineage=lineage,
                summary=summary,
                episodes=episodes,
                conditions=conditions,
                runtime=runtime,
                manifest=manifest,
            )
        else:
            _write_result_artifacts(
                config,
                runtime,
                summary=summary,
                episodes=episodes,
                conditions=conditions,
            )
        complete_manifest(manifest, artifact_paths=_artifact_paths(config.output_dir))
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
            evaluation_identity=evaluation_identity,
            lineage=lineage,
        )
    except KeyboardInterrupt:
        interrupt_manifest(manifest, artifact_paths=_artifact_paths(config.output_dir))
        raise
    except Exception as exc:
        fail_manifest(manifest, exc, artifact_paths=_artifact_paths(config.output_dir))
        raise
    finally:
        if eval_env is not None:
            eval_env.close()


__all__ = [
    "EvaluationRunResult",
    "EvaluationRuntime",
    "gate_evaluation_identity",
    "namespace_from_config",
    "run_evaluation",
]
