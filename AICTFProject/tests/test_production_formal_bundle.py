"""Production entrypoint integration: one identity stamps the whole bundle.

Not helper-level unit tests — this drives ``orchestrate_training_run`` (the
real production path behind ``train_ppo``) in artifact-bundle-only mode and
asserts the on-disk travelers share one passport.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rl.config.ppo_config import PPOConfig  # noqa: E402
from rl.ruleset_identity import (  # noqa: E402
    ARTIFACT_IDENTITY_KEY,
    RunIdentityError,
    read_artifact_identity,
    validate_bundle,
)
from rl.training.orchestrator import orchestrate_training_run  # noqa: E402
from rl.training.run_artifacts import write_run_config_json  # noqa: E402
from rl.custom_ppo.checkpoints.loader import (  # noqa: E402
    read_checkpoint_payload,
    read_checkpoint_ruleset,
)


def _tiny_formal_cfg(tmp_path: Path) -> PPOConfig:
    cfg = PPOConfig()
    cfg.run_tag = "formal_bundle_wiring"
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.metrics_csv_path = str(tmp_path / "metrics.csv")
    cfg.episode_csv_path = str(tmp_path / "episode_rows.csv")
    cfg.seed = 2_600_001
    cfg.device = "cpu"
    cfg.n_envs = 1
    cfg.n_steps = 1
    cfg.batch_size = 1
    cfg.n_epochs = 1
    cfg.total_timesteps = 1
    cfg.max_decision_steps = 32
    cfg.max_blue_agents = 2
    cfg.map_layout = "map_a"
    cfg.mode = "FIXED_OPPONENT"
    cfg.fixed_opponent_tag = "OP3"
    cfg.use_latent_strategy = False
    cfg.use_stable_marl_ppo = False
    cfg.gpu_native_env = True
    cfg.enable_progress_bar = False
    cfg.verbose_training = False
    cfg.training_telemetry_mode = __import__(
        "rl.telemetry_mode", fromlist=["TrainingTelemetryMode"]
    ).TrainingTelemetryMode.OFF
    # Artifact-only production path: writers + checkpoint, no learn loop.
    cfg.formal_artifact_bundle_only = True  # type: ignore[attr-defined]
    return cfg


def test_production_entrypoint_writes_one_formal_bundle(tmp_path):
    cfg = _tiny_formal_cfg(tmp_path)
    orchestrate_training_run(cfg)

    base = tmp_path
    run_config_path = base / f"{cfg.run_tag}_run_config.json"
    # run_config may live next to metrics CSV
    if not run_config_path.exists():
        run_config_path = Path(cfg.metrics_csv_path).parent / f"{cfg.run_tag}_run_config.json"

    expected = {
        "run_config.json": run_config_path,
        "training_manifest.json": base / "training_manifest.json",
        "episode_rows.csv": Path(cfg.episode_csv_path),
        "evaluation_manifest.json": base / "evaluation_manifest.json",
        "result_summary.json": base / "result_summary.json",
    }
    ckpt = Path(cfg.checkpoint_dir) / f"final_{cfg.run_tag}.zip"

    for label, path in expected.items():
        assert path.is_file(), f"missing production artifact {label}: {path}"
    assert ckpt.is_file(), f"missing checkpoint: {ckpt}"

    json_artifacts = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in expected.items()
        if label.endswith(".json")
    }
    with open(expected["episode_rows.csv"], encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "episode CSV must contain stamped rows"

    ref = validate_bundle(json_artifacts, {"episode_rows.csv": rows}, require_formal=True)
    assert ref["formal_result_eligible"] is True or json_artifacts["run_config.json"][
        ARTIFACT_IDENTITY_KEY
    ]["formal_result_eligible"] is True
    assert ref["canonical_map"] == "map_a"
    assert ref["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"

    # Every CSV row matches the passport byte-for-byte.
    for i, row in enumerate(rows):
        for key in ("run_id", "canonical_map", "resolved_map", "ruleset_id",
                    "ruleset_fingerprint"):
            assert str(row[key]) == str(ref[key]), f"row {i} {key} mismatch"

    # Checkpoint ruleset + artifact_identity agree with the live-run passport.
    payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
    ckpt_ai = payload.get(ARTIFACT_IDENTITY_KEY)
    assert isinstance(ckpt_ai, dict)
    for key in ("canonical_map", "resolved_map", "ruleset_id", "ruleset_fingerprint"):
        assert str(ckpt_ai[key]) == str(ref[key])
    rs = read_checkpoint_ruleset(str(ckpt))
    assert rs["ruleset_id"] == ref["ruleset_id"]

    # Result summary validates against its source rows.
    summary_ai = read_artifact_identity(json_artifacts["result_summary.json"])
    assert summary_ai["ruleset_fingerprint"] == ref["ruleset_fingerprint"]
    assert json_artifacts["result_summary.json"]["n_episode_rows"] == len(rows)


def test_missing_identity_on_writer_fails_before_rollout(tmp_path):
    """Removing identity from a production writer must fail closed."""
    cfg = _tiny_formal_cfg(tmp_path)
    with pytest.raises(RunIdentityError, match="requires the run's resolved RunIdentity"):
        write_run_config_json(cfg, argv=["train_ppo.py"])
    assert not list(tmp_path.glob("*_run_config.json"))
    assert not (tmp_path / "training_manifest.json").exists()


def test_standalone_eval_identity_requires_checkpoint_match(tmp_path, live_env=None):
    """Standalone evaluation may have its own run_id but must match checkpoint ruleset."""
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.ruleset_identity import build_evaluation_run_identity, build_formal_run_identity

    v2 = dict(
        taggers_required=1,
        tag_min_interval_seconds=10.0,
        tag_nearest_only=True,
        tag_channel_seconds=0.0,
        suppression_attackers_required=2,
    )
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_set="train",
        map_layout="map_a",
        max_decision_steps=64,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=2_600_101,
        obstacle_obs_channel=True,
        **v2,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    try:
        train_id = build_formal_run_identity(env, run_id="train_run")
        eval_id, lineage = build_evaluation_run_identity(
            env,
            evaluation_run_id="eval_run",
            source_training_run_id=train_id.run_id,
            source_checkpoint_fingerprint="abc",
            source_checkpoint_ruleset_fingerprint=train_id.ruleset_fingerprint,
        )
        assert eval_id.run_id == "eval_run"
        assert eval_id.ruleset_fingerprint == train_id.ruleset_fingerprint
        assert eval_id.formal_result_eligible is True
        assert lineage["source_training_run_id"] == "train_run"

        with pytest.raises(RunIdentityError, match="does not match the source checkpoint"):
            build_evaluation_run_identity(
                env,
                evaluation_run_id="eval_bad",
                source_training_run_id=train_id.run_id,
                source_checkpoint_fingerprint="abc",
                source_checkpoint_ruleset_fingerprint="0" * 64,
            )

        overridden, _ = build_evaluation_run_identity(
            env,
            evaluation_run_id="eval_diag",
            source_training_run_id=train_id.run_id,
            source_checkpoint_fingerprint="abc",
            source_checkpoint_ruleset_fingerprint="0" * 64,
            allow_override=True,
        )
        assert overridden.formal_result_eligible is False
        assert overridden.identity_override_used is True
    finally:
        env.close()