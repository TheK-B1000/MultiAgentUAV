"""Production artifact wiring: identity must reach the real writers.

The helper-level tests (test_artifact_bundle_identity.py) prove the passport
office works. These prove the actual travelers get stamped -- i.e. that the
PRODUCTION writers receive the run's single resolved RunIdentity rather than
rebuilding one from config defaults.

Two properties matter most:

  1. A production writer invoked WITHOUT identity fails, before the run can
     produce artifacts that cannot later be validated.
  2. Identity is resolved from the LIVE environment, after it is built, so the
     stamp describes the game that actually ran.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from rl.ruleset_identity import (  # noqa: E402
    ARTIFACT_IDENTITY_KEY,
    RunIdentityError,
    build_formal_run_identity,
    validate_bundle,
)
from rl.training.run_artifacts import write_run_config_json  # noqa: E402

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)


@pytest.fixture
def live_env():
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=2_500_001,
        obstacle_obs_channel=True, **V2)
    e = GPUCTFVecEnv(cfg)
    e.reset()
    yield e
    e.close()


@dataclass
class _StubPPOConfig:
    """Minimal config surface the run-config writer touches.

    A dataclass because the production writer calls ``asdict(cfg)``.
    """

    run_tag: str = "smoke_wiring"
    checkpoint_dir: str = ""
    total_timesteps: int = 50_000
    metrics_csv_path: str = ""
    episode_csv_path: str = ""
    strategy_experience_csv_path: str = ""
    load_path: str = ""
    cli_preset: str = "no_latent_baseline"

    @classmethod
    def at(cls, tmp_path):
        return cls(checkpoint_dir=str(tmp_path),
                   metrics_csv_path=str(tmp_path / "metrics.csv"),
                   episode_csv_path=str(tmp_path / "episodes.csv"))


# --- the mandatory-identity contract ---------------------------------------

def test_run_config_writer_refuses_without_identity(tmp_path):
    """Must fail BEFORE the run produces unstampable artifacts."""
    cfg = _StubPPOConfig.at(tmp_path)
    with pytest.raises(RunIdentityError, match="requires the run's resolved RunIdentity"):
        write_run_config_json(cfg, argv=["train_ppo.py"])
    assert not list(tmp_path.glob("*_run_config.json")), (
        "no artifact may be written when identity is missing")


def test_run_config_writer_stamps_the_supplied_identity(tmp_path, live_env):
    cfg = _StubPPOConfig.at(tmp_path)
    ident = build_formal_run_identity(live_env, run_id=cfg.run_tag)
    path = write_run_config_json(cfg, argv=["train_ppo.py"], run_identity=ident)
    payload = json.loads(open(path, encoding="utf-8").read())

    ai = payload[ARTIFACT_IDENTITY_KEY]
    assert ai["run_id"] == cfg.run_tag
    assert ai["canonical_map"] == "map_a"
    assert ai["resolved_map"] == "map_a_open"
    assert ai["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    assert ai["ruleset_fingerprint"] == ident.ruleset_fingerprint
    assert ai["formal_result_eligible"] is True
    assert ai["identity_override_used"] is False


def test_writer_stamp_matches_the_live_environment_not_defaults(tmp_path):
    """A non-default ruleset must appear in the artifact, proving no fallback."""
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=2_500_002,
        obstacle_obs_channel=True,
        taggers_required=2, tag_nearest_only=False,
        tag_min_interval_seconds=0.0, tag_channel_seconds=1.0,
        suppression_attackers_required=2)
    env = GPUCTFVecEnv(cfg)
    env.reset()
    try:
        pcfg = _StubPPOConfig.at(tmp_path)
        ident = build_formal_run_identity(env, run_id=pcfg.run_tag)
        assert ident.ruleset_id == "RULESET_V1_TWO_TAGGER", (
            "identity must reflect the live env, not the V2 default")
        path = write_run_config_json(pcfg, argv=["x"], run_identity=ident)
        ai = json.loads(open(path, encoding="utf-8").read())[ARTIFACT_IDENTITY_KEY]
        assert ai["ruleset_id"] == "RULESET_V1_TWO_TAGGER"
    finally:
        env.close()


# --- bundle-level integration ----------------------------------------------

def test_real_run_config_validates_inside_a_bundle(tmp_path, live_env):
    """The real production artifact participates in bundle validation."""
    cfg = _StubPPOConfig.at(tmp_path)
    ident = build_formal_run_identity(live_env, run_id=cfg.run_tag)
    path = write_run_config_json(cfg, argv=["train_ppo.py"], run_identity=ident)
    run_config = json.loads(open(path, encoding="utf-8").read())

    from rl.ruleset_identity import stamp_csv_row, stamp_json_artifact

    bundle = {
        "run_config.json": run_config,
        "training_manifest.json": stamp_json_artifact({"seeds": [1]}, ident),
        "result_summary.json": stamp_json_artifact({"verdict": "SMOKE"}, ident),
    }
    rows = [stamp_csv_row({"episode_index": i}, ident) for i in range(3)]
    ref = validate_bundle(bundle, {"episode_rows.csv": rows})
    assert ref["run_id"] == cfg.run_tag
    assert ref["ruleset_fingerprint"] == ident.ruleset_fingerprint


def test_foreign_run_config_is_rejected_by_the_bundle(tmp_path, live_env):
    """Two production runs' artifacts must not validate together."""
    cfg = _StubPPOConfig.at(tmp_path)
    ident_a = build_formal_run_identity(live_env, run_id="run_a")
    ident_b = build_formal_run_identity(live_env, run_id="run_b")
    pa = write_run_config_json(cfg, argv=["a"], run_identity=ident_a)
    a = json.loads(open(pa, encoding="utf-8").read())

    from rl.ruleset_identity import stamp_json_artifact

    bundle = {"run_config.json": a,
              "training_manifest.json": stamp_json_artifact({}, ident_b)}
    with pytest.raises(RunIdentityError, match="run_id differs"):
        validate_bundle(bundle)


def test_orchestrator_resolves_identity_after_env_build():
    """Ordering guard: startup artifacts must be written after the env exists.

    If ``write_startup_formal_artifacts`` ever moves back above
    ``build_training_env``, identity would have to come from config defaults again.
    """
    import inspect

    from rl.training import orchestrator

    src = inspect.getsource(orchestrator)
    i_env = src.index("env = build_training_env(")
    i_ident = src.index("build_formal_run_identity(")
    i_rc = src.index("write_startup_formal_artifacts(")
    assert i_env < i_ident < i_rc, (
        "identity must be resolved from the live env, and startup artifacts "
        "written only after that")
