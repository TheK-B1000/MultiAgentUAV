"""``tag_telemetry_enabled`` must travel from PPOConfig to the live env and artifacts.

The formal smoke had to inject this setting from a harness because it was not
plumbed through. These tests pin the production path so a formal run can never
again depend on harness-only wiring:

    PPOConfig -> env factory -> live GPUCTFVecEnv -> run_config / manifest

They assert against the LIVE environment object and the on-disk artifacts, not
against the config that was handed in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.config.ppo_config import PPOConfig  # noqa: E402
from rl.training.env_factory import build_training_env  # noqa: E402
from rl.training.orchestrator import orchestrate_training_run  # noqa: E402
from rl.training.resolved_config import resolve_training_config  # noqa: E402


def _base_cfg(tmp_path: Path) -> PPOConfig:
    cfg = PPOConfig()
    cfg.run_tag = "telemetry_wiring"
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.metrics_csv_path = str(tmp_path / "metrics.csv")
    cfg.episode_csv_path = str(tmp_path / "episode_rows.csv")
    cfg.seed = 2_800_001
    cfg.device = "cpu"
    cfg.n_envs = 2
    cfg.n_steps = 4
    cfg.batch_size = 4
    cfg.n_epochs = 1
    cfg.total_timesteps = 8
    cfg.max_decision_steps = 32
    cfg.max_blue_agents = 2
    cfg.map_layout = "map_a"
    cfg.mode = "FIXED_OPPONENT"
    cfg.fixed_opponent_tag = "OP6"
    cfg.use_latent_strategy = False
    cfg.use_stable_marl_ppo = False
    cfg.gpu_native_env = True
    cfg.enable_progress_bar = False
    cfg.verbose_training = False
    return cfg


def _g0_v2_shaped_cfg(tmp_path: Path) -> PPOConfig:
    """Minimal config shaped like the formal G0-v2 training path."""
    cfg = _base_cfg(tmp_path)
    cfg.run_tag = "g0_v2_telemetry_gate"
    cfg.mode = "OPPONENT_POOL"
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")
    cfg.formal_run = True
    return cfg


def _live_flag(env) -> bool:
    return bool(env.core.cfg.tag_telemetry_enabled)


@pytest.mark.parametrize("enabled", [True, False])
def test_config_flag_reaches_live_environment(tmp_path, enabled):
    cfg = _base_cfg(tmp_path)
    cfg.tag_telemetry_enabled = enabled
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        assert _live_flag(env) is enabled
    finally:
        env.close()


def test_enabled_env_actually_emits_identified_events(tmp_path):
    """Wiring is only real if the live env produces the events it promises."""
    cfg = _base_cfg(tmp_path)
    cfg.tag_telemetry_enabled = True
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        env.reset()
        core = env.core
        core.drain_tag_events()
        # GPUCTFVecEnv wants a flat (B * Nb * 2) int64 action vector.
        n_actions = int(core.B) * int(core.Nb) * 2
        actions = np.zeros(n_actions, dtype=np.int64)
        events = []
        for _ in range(cfg.max_decision_steps + 1):
            env.step(actions)
            events.extend(core.drain_tag_events())
        assert events, "telemetry enabled but no events were emitted"
        for e in events:
            assert "event_sequence" in e and "episode_id" in e and "env_index" in e
    finally:
        env.close()


def test_disabled_env_emits_nothing(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.tag_telemetry_enabled = False
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        env.reset()
        assert env.core.drain_tag_events() == []
    finally:
        env.close()


def test_formal_run_without_telemetry_is_refused_before_env_build(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.formal_run = True
    cfg.tag_telemetry_enabled = False
    with pytest.raises(ValueError, match="tag_telemetry_enabled"):
        orchestrate_training_run(cfg)
    # nothing may be written by a run that must not start
    assert not list(tmp_path.glob("*run_config.json"))
    assert not (tmp_path / "training_manifest.json").exists()


def test_formal_g0_v2_omitted_telemetry_is_refused_before_rollout(tmp_path):
    """Formal G0-v2 shape + default/omitted telemetry must fail closed."""
    cfg = _g0_v2_shaped_cfg(tmp_path)
    # Do not set tag_telemetry_enabled — the PPOConfig default is False.
    assert cfg.tag_telemetry_enabled is False
    with pytest.raises(ValueError, match="tag_telemetry_enabled"):
        orchestrate_training_run(cfg)
    assert not list(tmp_path.glob("*run_config.json"))
    assert not (tmp_path / "training_manifest.json").exists()
    assert not list((tmp_path / "ckpts").glob("*")) if (tmp_path / "ckpts").exists() else True


def test_production_chain_config_to_live_env_to_artifacts(tmp_path):
    """Decisive chain: PPOConfig(True) -> live env -> run_config -> manifest."""
    cfg = _base_cfg(tmp_path)
    cfg.tag_telemetry_enabled = True
    cfg.formal_run = True
    cfg.formal_artifact_bundle_only = True  # type: ignore[attr-defined]

    # Prove the factory path independently before the orchestrator writes.
    resolved = resolve_training_config(cfg)
    probe_env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        assert _live_flag(probe_env) is True
    finally:
        probe_env.close()

    orchestrate_training_run(cfg)

    run_config = tmp_path / f"{cfg.run_tag}_run_config.json"
    manifest = tmp_path / "training_manifest.json"
    assert run_config.is_file() and manifest.is_file()

    for path in (run_config, manifest):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["tag_telemetry_enabled"] is True, path.name
        assert payload["formal_run"] is True, path.name


def test_flag_is_recorded_in_run_config_and_training_manifest(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg.tag_telemetry_enabled = True
    cfg.formal_run = True
    cfg.formal_artifact_bundle_only = True  # type: ignore[attr-defined]
    orchestrate_training_run(cfg)

    run_config = tmp_path / f"{cfg.run_tag}_run_config.json"
    manifest = tmp_path / "training_manifest.json"
    assert run_config.is_file() and manifest.is_file()

    for path in (run_config, manifest):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["tag_telemetry_enabled"] is True, path.name
        assert payload["formal_run"] is True, path.name
