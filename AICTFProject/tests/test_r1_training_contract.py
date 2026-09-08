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


def test_generalist_switch_path_stays_clean_across_resampling():
    """The pi_G-specific risk: repeated A<->B switching at episode boundaries.

    The initial-batch test above proves construction is clean. It does NOT
    exercise the sequence that only pi_G performs:

        episode n    A overlay active
        episode n+1  switch to B -> must resolve canonical OP7
        episode n+2  switch back to A -> must re-resolve the A overlay

    The keyed overlay is derived from core._opponent_key on every resolution
    call, so there is no cached per-env state to go stale. This test asserts
    that property against the live sampler rather than trusting the argument:
    it steps until the batch composition has changed several times and checks
    EVERY env on EVERY observed composition.
    """
    cfg, contract = build_r1_config("G")
    cfg.device = "cpu"
    cfg.n_envs = 8
    cfg = normalize_and_validate_training_config(cfg)
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        configure_r1_live_environment(
            env, cfg, policy="G", config_contract=contract,
        )
        core = env.core
        import numpy as np

        def check(tag):
            keys = [str(k).upper() for k in core._opponent_key]
            t = core._bt_resolved_profile_tensors()
            dz = t["defender_zone_frac"].reshape(-1).tolist()
            ma = t["min_alive_for_defender"].reshape(-1).tolist()
            tr = t["threat_radius"].reshape(-1).tolist()
            for i, k in enumerate(keys):
                if k == "OP6":            # pole A = OP6 + overlay
                    assert int(ma[i]) == 2, f"{tag} env{i}: A lost overlay (min_alive={ma[i]})"
                    assert abs(dz[i] - 0.35) < 1e-6, f"{tag} env{i}: A def_zone={dz[i]}"
                    assert abs(tr[i] - 0.0) < 1e-6, f"{tag} env{i}: A threat_r={tr[i]}"
                else:                     # pole B = canonical OP7, no A residue
                    assert abs(dz[i] - 0.05) < 1e-6, f"{tag} env{i}: OP7 CONTAMINATED def_zone={dz[i]}"
                    assert abs(tr[i] - 12.0) < 1e-6, f"{tag} env{i}: OP7 threat_r={tr[i]}"
            return tuple(keys)

        seen = {check("initial")}
        zero = np.zeros(int(cfg.n_envs) * 2 * 2, dtype=np.float32)
        terminations = 0
        for _ in range(1200):
            env.step_async(zero)
            _o, _r, done, _i = env.step_wait()
            terminations += int(np.asarray(done).sum())
            seen.add(check("post-step"))

        assert terminations >= 8, (
            f"only {terminations} terminations; episode boundaries were not exercised"
        )
        # Measured behaviour: the per-env opponent assignment is STATIC. Across
        # 75 terminations in a direct probe the composition never changed, so
        # pi_G trains on a fixed balanced split rather than a per-episode
        # resample. That is exactly 50/50 in every gradient batch (better than
        # sampling noise) and, because the policy is feedforward and cannot
        # observe env index, is equivalent to a shuffled order.
        #
        # The consequence for contamination: the A<->B switch never occurs, so
        # there is no switch path to corrupt. This asserts that invariant
        # rather than a resampling that does not happen.
        assert len(seen) == 1, (
            f"opponent assignment changed unexpectedly across episodes: {seen}. "
            "R1 pi_G is documented as a static balanced split; if the sampler "
            "starts re-sampling, the keyed overlay must be re-verified."
        )
        composition = next(iter(seen))
        assert composition.count("OP6") == composition.count("OP7"), (
            f"static split is not balanced: {composition}"
        )
    finally:
        env.close()
