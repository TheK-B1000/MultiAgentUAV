"""Standalone evaluation identity border — real orchestrator entrypoint tests.

Proves the live evaluation environment, checkpoint, and evaluation artifacts
share one identity universe, and that mismatches fail BEFORE model execution.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from rl.custom_ppo.probe_result import (  # noqa: E402
    PROBE_SUCCESS,
    CounterfactualProbeResult,
    GradientProbeResult,
    WeightProbeResult,
)
from rl.evaluation.config import MapAwarenessEvaluationConfig  # noqa: E402
from rl.evaluation.orchestrator import (  # noqa: E402
    EvaluationRuntime,
    gate_evaluation_identity,
    run_evaluation,
)
from rl.evaluation.policy_loader import LoadedEvaluationPolicy  # noqa: E402
from rl.ruleset_identity import (  # noqa: E402
    ARTIFACT_IDENTITY_KEY,
    RULESET_FIELDS,
    RunIdentityError,
    build_formal_run_identity,
    ruleset_fingerprint_hash,
    validate_bundle,
)

V2 = dict(
    ruleset_id="RULESET_V2_AQUATICUS_10S",
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)


def _live_env(map_layout: str = "map_a", seed: int = 2_700_001, **rules):
    field_rules = {
        "taggers_required": 1,
        "tag_min_interval_seconds": 10.0,
        "tag_nearest_only": True,
        "tag_channel_seconds": 0.0,
        "suppression_attackers_required": 2,
    }
    field_rules.update({k: v for k, v in rules.items() if k != "ruleset_id"})
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_set="train",
        map_layout=map_layout,
        max_decision_steps=64,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=seed,
        obstacle_obs_channel=True,
        **field_rules,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    return env


def _write_stamped_checkpoint(
    path: Path,
    *,
    identity,
    ruleset: dict | None = None,
    omit_artifact_identity: bool = False,
    omit_ruleset: bool = False,
    map_override: dict | None = None,
):
    rs = dict(ruleset if ruleset is not None else identity.ruleset)
    if "ruleset_id" not in rs:
        rs["ruleset_id"] = identity.ruleset_id
    ai = identity.artifact_identity()
    if map_override:
        ai.update(map_override)
    if ruleset is not None:
        ai["ruleset_id"] = rs["ruleset_id"]
        ai["ruleset_fingerprint"] = ruleset_fingerprint_hash(rs)
    payload = {"model_state_dict": {}, "cfg": {}, "format": "custom_ppo"}
    if not omit_ruleset:
        payload["ruleset"] = rs
    if not omit_artifact_identity:
        payload[ARTIFACT_IDENTITY_KEY] = ai
    torch.save(payload, path)
    return path


def _config(tmp_path: Path, *, baseline: Path, candidate: Path, maps=("map_a",), **kwargs):
    defaults = dict(
        baseline_checkpoint=baseline,
        candidate_checkpoint=candidate,
        maps=tuple(maps),
        opponents=("OP3",),
        episodes_per_cell=1,
        seed_start=2_700_100,
        device="cpu",
        output_dir=tmp_path / "out",
        max_decision_steps=32,
        counterfactual_steps=1,
        obs_weight_threshold=1e-4,
        gradient_threshold=0.0,
        counterfactual_kl_threshold=1e-5,
        counterfactual_action_threshold=0.01,
        navigation_improvement_threshold=0.1,
        route_difference_threshold=0.1,
        minimum_win_rate=0.0,
        competence_retention_tolerance=1.0,
        saturation_win_rate=1.0,
        evaluation_run_id="eval_formal_border",
        require_formal_identity=True,
        allow_identity_override=False,
        baseline_cnn_channels=7,
        candidate_cnn_channels=7,
    )
    defaults.update(kwargs)
    return MapAwarenessEvaluationConfig(**defaults)


def _runtime(calls: list[str], *, on_model_execution=None):
    def write_json(path: Path, payload) -> None:
        calls.append(f"write:{path.name}")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return EvaluationRuntime(
        project_root=Path.cwd(),
        command=["eval_formal.py"],
        validate_opponent_name=lambda o: o.upper(),
        preflight_opponents=lambda **kwargs: calls.append("preflight_opponents"),
        preflight_distribution_contract=lambda policy, *, label: calls.append(
            f"preflight_distribution:{label}"
        ),
        inspect_obstacle_weights=lambda policy: calls.append("weights")
        or WeightProbeResult(
            status=PROBE_SUCCESS,
            has_obstacle_channel=True,
            cnn_channels=7,
            obstacle_weight_l2=1.0,
        ),
        gradient_probe=lambda *a, **k: calls.append("gradient")
        or GradientProbeResult(status=PROBE_SUCCESS, obstacle_gradient_l2=1.0),
        obstacle_counterfactual=lambda *a, **k: calls.append("counterfactual")
        or CounterfactualProbeResult(
            status=PROBE_SUCCESS,
            states_evaluated=1,
            mean_action_kl=1.0,
            mean_logit_l2=1.0,
            argmax_action_change_rate=1.0,
        ),
        run_episode=lambda **kwargs: calls.append(f"episode:{kwargs['policy_name']}")
        or {
            "policy": kwargs["policy_name"],
            "map": kwargs["map_name"],
            "requested_opponent": kwargs["opponent"],
            "resolved_opponent": kwargs["opponent"],
            "opponent": kwargs["opponent"],
            "seed": kwargs["seed"],
            "blue_score": 1.0,
            "red_score": 0.0,
            "win": 1,
            "loss": 0,
            "draw": 0,
            "score_margin": 1.0,
            "collision_metric_source": "environment_exact",
            "stuck_metric_source": "environment_exact",
            "route_metric_source": "environment_exact",
            "wall_collisions": 0.0,
            "blocked_movement_events": 0.0,
            "stuck_steps": 0.0,
            "upper_lane_use": 0.0,
            "lower_lane_use": 0.0,
            "episode_steps": 1,
        },
        write_json_text=write_json,
        on_model_execution=on_model_execution,
    )


# --- gate helpers (reject before inference) ---------------------------------


def test_matching_v2_checkpoint_and_map_accepted(tmp_path):
    env = _live_env()
    try:
        train_id = build_formal_run_identity(env, run_id="train_mapa_v2")
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        candidate = _write_stamped_checkpoint(tmp_path / "candidate.zip", identity=train_id)
        cfg = _config(tmp_path, baseline=baseline, candidate=candidate, maps=("map_a",))

        model_executed = {"hit": False}

        def _sentinel():
            model_executed["hit"] = True

        from unittest.mock import DEFAULT, patch

        calls: list[str] = []
        with patch.multiple(
            "rl.evaluation.orchestrator",
            read_checkpoint_dimensions=DEFAULT,
            load_evaluation_policy=DEFAULT,
        ) as mocks:
            mocks["read_checkpoint_dimensions"].side_effect = [
                ({}, 2, 5, 50),
                ({}, 2, 5, 50),
            ]
            mocks["load_evaluation_policy"].side_effect = [
                LoadedEvaluationPolicy("baseline", str(baseline), object(), {}, 2, 5, 50, 7),
                LoadedEvaluationPolicy("candidate", str(candidate), object(), {}, 2, 5, 50, 7),
            ]
            result = run_evaluation(cfg, _runtime(calls, on_model_execution=_sentinel))

        assert model_executed["hit"] is True
        assert result.evaluation_identity is not None
        assert result.evaluation_identity.run_id == "eval_formal_border"
        assert result.evaluation_identity.run_id != "train_mapa_v2"
        assert result.lineage["source_training_run_id"] == "train_mapa_v2"
        assert result.lineage["source_checkpoint_id"]
        assert result.lineage["source_checkpoint_ruleset_fingerprint"] == train_id.ruleset_fingerprint

        out = cfg.output_dir
        manifest = json.loads((out / "evaluation_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((out / "result_summary.json").read_text(encoding="utf-8"))
        with open(out / "episode_rows.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        ref = validate_bundle(
            {"evaluation_manifest.json": manifest, "result_summary.json": summary},
            {"episode_rows.csv": rows},
            require_formal=True,
        )
        assert ref["run_id"] == "eval_formal_border"
        assert ref["formal_result_eligible"] is True
        for row in rows:
            assert row["run_id"] == "eval_formal_border"
            assert row["run_id"] != "train_mapa_v2"
        assert manifest[ARTIFACT_IDENTITY_KEY]["formal_result_eligible"] is True
        assert manifest["source_training_run_id"] == "train_mapa_v2"
        assert manifest["source_checkpoint_id"]
        assert manifest["source_checkpoint_ruleset_fingerprint"] == train_id.ruleset_fingerprint
    finally:
        env.close()


def test_mismatched_map_rejected_before_model_execution(tmp_path):
    env = _live_env(map_layout="map_a")
    try:
        train_id = build_formal_run_identity(env, run_id="train_mapa")
        # Checkpoint stamped as map_b while live eval env is map_a.
        bad = _write_stamped_checkpoint(
            tmp_path / "candidate.zip",
            identity=train_id,
            map_override={"canonical_map": "map_b_split_lane", "resolved_map": "map_b_split_lane"},
        )
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        cfg = _config(tmp_path, baseline=baseline, candidate=bad, maps=("map_a",))

        model_executed = {"hit": False}
        with pytest.raises(RunIdentityError, match="map mismatch"):
            gate_evaluation_identity(cfg, n_agents=2)
        assert model_executed["hit"] is False

        from unittest.mock import DEFAULT, patch

        calls: list[str] = []
        with patch.multiple(
            "rl.evaluation.orchestrator",
            read_checkpoint_dimensions=DEFAULT,
            load_evaluation_policy=DEFAULT,
        ) as mocks:
            mocks["read_checkpoint_dimensions"].side_effect = [
                ({}, 2, 5, 50),
                ({}, 2, 5, 50),
            ]
            mocks["load_evaluation_policy"].side_effect = AssertionError("model load must not run")
            with pytest.raises(RunIdentityError, match="map mismatch"):
                run_evaluation(cfg, _runtime(calls, on_model_execution=lambda: model_executed.__setitem__("hit", True)))
        assert model_executed["hit"] is False
        assert not any(c.startswith("episode:") for c in calls)
    finally:
        env.close()


def test_matching_label_mismatched_fingerprint_rejected(tmp_path):
    env = _live_env()
    try:
        train_id = build_formal_run_identity(env, run_id="train_v2")
        sneaky = dict(V2)
        sneaky["tag_min_interval_seconds"] = 30.0  # same label, different game
        bad = _write_stamped_checkpoint(
            tmp_path / "candidate.zip", identity=train_id, ruleset=sneaky
        )
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        cfg = _config(tmp_path, baseline=baseline, candidate=bad)

        model_executed = {"hit": False}
        with pytest.raises(RunIdentityError):
            gate_evaluation_identity(cfg, n_agents=2)
        assert model_executed["hit"] is False
    finally:
        env.close()


def test_legacy_checkpoint_rejected(tmp_path):
    env = _live_env()
    try:
        train_id = build_formal_run_identity(env, run_id="train_v2")
        legacy = tmp_path / "legacy.zip"
        torch.save({"model_state_dict": {}, "cfg": {}}, legacy)
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        cfg = _config(tmp_path, baseline=baseline, candidate=legacy)
        with pytest.raises(RunIdentityError, match="legacy|missing|complete ruleset"):
            gate_evaluation_identity(cfg, n_agents=2)
    finally:
        env.close()


def test_missing_checkpoint_identity_rejected(tmp_path):
    env = _live_env()
    try:
        train_id = build_formal_run_identity(env, run_id="train_v2")
        missing_ai = _write_stamped_checkpoint(
            tmp_path / "candidate.zip",
            identity=train_id,
            omit_artifact_identity=True,
        )
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        cfg = _config(tmp_path, baseline=baseline, candidate=missing_ai)
        with pytest.raises(RunIdentityError, match="artifact_identity"):
            gate_evaluation_identity(cfg, n_agents=2)
    finally:
        env.close()


def test_diagnostic_override_runs_ineligible(tmp_path):
    env = _live_env()
    try:
        train_id = build_formal_run_identity(env, run_id="train_v2")
        sneaky = dict(V2)
        sneaky["tag_min_interval_seconds"] = 30.0
        bad = _write_stamped_checkpoint(
            tmp_path / "candidate.zip", identity=train_id, ruleset=sneaky
        )
        # Baseline also needs to pass under override — stamp it with same sneaky ruleset.
        baseline = _write_stamped_checkpoint(
            tmp_path / "baseline.zip", identity=train_id, ruleset=sneaky
        )
        cfg = _config(
            tmp_path,
            baseline=baseline,
            candidate=bad,
            allow_identity_override=True,
        )

        model_executed = {"hit": False}
        from unittest.mock import DEFAULT, patch

        calls: list[str] = []
        with patch.multiple(
            "rl.evaluation.orchestrator",
            read_checkpoint_dimensions=DEFAULT,
            load_evaluation_policy=DEFAULT,
        ) as mocks:
            mocks["read_checkpoint_dimensions"].side_effect = [
                ({}, 2, 5, 50),
                ({}, 2, 5, 50),
            ]
            mocks["load_evaluation_policy"].side_effect = [
                LoadedEvaluationPolicy("baseline", str(baseline), object(), {}, 2, 5, 50, 7),
                LoadedEvaluationPolicy("candidate", str(bad), object(), {}, 2, 5, 50, 7),
            ]
            with pytest.warns(RuntimeWarning, match="OVERRIDE"):
                result = run_evaluation(
                    cfg,
                    _runtime(calls, on_model_execution=lambda: model_executed.__setitem__("hit", True)),
                )
        assert model_executed["hit"] is True
        assert result.evaluation_identity is not None
        assert result.evaluation_identity.formal_result_eligible is False
        assert result.evaluation_identity.identity_override_used is True
        summary = json.loads(
            (cfg.output_dir / "result_summary.json").read_text(encoding="utf-8")
        )
        assert summary[ARTIFACT_IDENTITY_KEY]["formal_result_eligible"] is False
        with pytest.raises(RunIdentityError, match="formal_result_eligible"):
            validate_bundle(
                {"result_summary.json": summary},
                require_formal=True,
            )
    finally:
        env.close()


def test_cli_map_a_live_env_wins_alias(tmp_path):
    """CLI map_a with live resolving map_a_open is accepted against map_a ckpt."""
    env = _live_env(map_layout="map_a")
    try:
        train_id = build_formal_run_identity(env, run_id="train_mapa")
        assert train_id.canonical_map == "map_a"
        assert train_id.resolved_map == "map_a_open"
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        candidate = _write_stamped_checkpoint(tmp_path / "candidate.zip", identity=train_id)
        # Request map_a_open explicitly — alias-compatible with live map_a identity.
        cfg = _config(
            tmp_path,
            baseline=baseline,
            candidate=candidate,
            maps=("map_a_open",),
        )
        eval_env, identity, lineage = gate_evaluation_identity(cfg, n_agents=2)
        try:
            assert identity.formal_result_eligible is True
            assert lineage["source_training_run_id"] == "train_mapa"
        finally:
            eval_env.close()
    finally:
        env.close()


def test_requested_foreign_map_rejected_even_if_ckpt_matches_live(tmp_path):
    env = _live_env(map_layout="map_a")
    try:
        train_id = build_formal_run_identity(env, run_id="train_mapa")
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        candidate = _write_stamped_checkpoint(tmp_path / "candidate.zip", identity=train_id)
        # reference_map is the LAST entry — keep live env on map_a so the
        # checkpoint matches, then reject because maps also requests map_b.
        cfg = _config(
            tmp_path,
            baseline=baseline,
            candidate=candidate,
            maps=("map_b_split_lane", "map_a"),
        )
        with pytest.raises(RunIdentityError, match="incompatible with live evaluation identity"):
            gate_evaluation_identity(cfg, n_agents=2)
    finally:
        env.close()


def test_checkpoint_map_a_vs_live_other_map_rejected(tmp_path):
    """Checkpoint trained on map_a must not evaluate against a live foreign map."""
    env = _live_env(map_layout="map_a")
    try:
        train_id = build_formal_run_identity(env, run_id="train_mapa")
        baseline = _write_stamped_checkpoint(tmp_path / "baseline.zip", identity=train_id)
        candidate = _write_stamped_checkpoint(tmp_path / "candidate.zip", identity=train_id)
        cfg = _config(
            tmp_path,
            baseline=baseline,
            candidate=candidate,
            maps=("map_b_split_lane",),
        )
        with pytest.raises(RunIdentityError, match="map mismatch"):
            gate_evaluation_identity(cfg, n_agents=2)
    finally:
        env.close()
