from __future__ import annotations

import dataclasses

from experiments.run_g0_v5_long import build_config as build_g0_v5_config
from experiments.run_r1_repertoire_training import (
    ALLOWED_CONFIG_DIFFS,
    G0_V5_CANONICAL_PARENT_SEED,
    POLICIES,
    build_r1_config,
    configure_r1_live_environment,
)
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.env_factory import build_training_env
from rl.training.resolved_config import resolve_training_config


def test_r1_resolved_configs_differ_from_g0_v5_only_on_frozen_fields():
    for policy, spec in POLICIES.items():
        parent = dataclasses.asdict(build_g0_v5_config(G0_V5_CANONICAL_PARENT_SEED))
        cfg, contract = build_r1_config(policy)
        child = dataclasses.asdict(cfg)
        actual = {k for k in parent if parent[k] != child[k]}
        assert actual == set(contract["resolved_config_diff"])
        assert actual <= ALLOWED_CONFIG_DIFFS
        assert cfg.use_latent_strategy is False
        assert cfg.load_path is None
        assert cfg.additional_timesteps == 0
        assert cfg.load_weights_only is False


def test_production_env_factory_reaches_pole_A_without_search_helper():
    """Drive the real PPO env factory, then the pre-rollout R0 seam."""
    cfg, contract = build_r1_config("A")
    cfg.device = "cpu"
    cfg.n_envs = 2
    cfg = normalize_and_validate_training_config(cfg)
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        extra = configure_r1_live_environment(
            env, cfg, policy="A", config_contract=contract,
        )
        rows = extra["r1_protocol"]["resolved_opponent_rows"]
        assert len(rows) == 2
        assert all(r["live_opponent_key"] == "OP6" for r in rows)
        assert all(int(r["resolved_profile"]["min_alive_for_defender"]) == 2
                   for r in rows)
        assert extra["r1_protocol"]["ruleset_id"] == "RULESET_V3_M1_OWN_FLAG_HOME"
    finally:
        env.close()


def test_generalist_live_batch_is_balanced_and_not_cross_contaminated():
    cfg, contract = build_r1_config("G")
    cfg.device = "cpu"
    cfg.n_envs = 4
    cfg = normalize_and_validate_training_config(cfg)
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        extra = configure_r1_live_environment(
            env, cfg, policy="G", config_contract=contract,
        )
        rows = extra["r1_protocol"]["resolved_opponent_rows"]
        assert extra["r1_protocol"]["initial_live_batch_counts"] == {"OP6": 2, "OP7": 2}
        for row in rows:
            profile = row["resolved_profile"]
            if row["live_opponent_key"] == "OP6":
                assert int(profile["min_alive_for_defender"]) == 2
                assert abs(profile["defender_zone_frac"] - 0.35) < 1e-6
            else:
                assert abs(profile["defender_zone_frac"] - 0.05) < 1e-6
                assert abs(profile["threat_radius"] - 12.0) < 1e-6
    finally:
        env.close()
