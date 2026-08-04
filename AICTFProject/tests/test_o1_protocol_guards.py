"""The O1 launcher must refuse to train anything the frozen protocol forbids.

A preregistration only binds if something enforces it. These tests check that
the guards in ``run_o1_response_oracle`` actually fire, rather than trusting
that a future edit will keep the config honest -- the failure mode being a run
that trains under a quietly different setup and reports the frozen constants
anyway.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")

from experiments.run_o1_response_oracle import (  # noqa: E402
    build_config,
    load_prereg,
    verify_against_prereg,
    verify_reward_matches_g0,
)


@pytest.fixture(scope="module")
def prereg() -> dict:
    return load_prereg()


@pytest.fixture
def cfg(prereg):
    return build_config(int(prereg["training"]["seeds"][0]), prereg)


def test_the_declared_configuration_passes(cfg, prereg):
    seed = int(prereg["training"]["seeds"][0])
    assert verify_against_prereg(cfg, seed, prereg) == []
    problems, verified = verify_reward_matches_g0(cfg)
    assert problems == []
    # 7 env reward overrides plus gamma.
    assert len(verified) == 8
    assert verified["env_sparse_tag_no_flag_points"] == 0.0
    assert verified["env_sparse_own_oob_points"] == -70.0
    assert verified["gamma"] == 0.995


def test_fresh_init_is_what_build_config_produces(cfg):
    assert not cfg.load_path
    assert int(cfg.additional_timesteps) == 0
    assert bool(cfg.use_latent_strategy) is False
    assert str(getattr(cfg, "phase_pod_id", "")) == ""


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda c: setattr(c, "load_path", "some/ckpt.zip"), "warm start is FORBIDDEN"),
        (lambda c: setattr(c, "total_timesteps", 250_000), "total_timesteps"),
        (lambda c: setattr(c, "phase_pod_id", "defend_lead"), "phase pods are prohibited"),
        (lambda c: setattr(c, "use_latent_strategy", True), "independent policy"),
        (lambda c: setattr(c, "map_layout", "map_b"), "map"),
        (lambda c: setattr(c, "max_decision_steps", 120), "horizon"),
        (lambda c: setattr(c, "train_domain_randomization", True), "domain randomization"),
        (lambda c: setattr(c, "opponent_pool", ("OP6",)), "opponent pool"),
    ],
)
def test_protocol_violations_are_caught(cfg, prereg, mutate, expect):
    mutate(cfg)
    problems = verify_against_prereg(cfg, int(prereg["training"]["seeds"][0]), prereg)
    assert any(expect in p for p in problems), f"expected {expect!r} in {problems}"


def test_undeclared_seed_is_caught(cfg, prereg):
    problems = verify_against_prereg(cfg, 4_242_424, prereg)
    assert any("not one of the declared O1 seeds" in p for p in problems)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("env_sparse_tag_no_flag_points", 100.0),
        ("env_sparse_own_oob_points", -100.0),
        ("env_action_failed_punishment", -0.2),
        ("env_sparse_mine_tag_points", 100.0),
    ],
)
def test_reward_drift_from_g0_is_caught(cfg, field, bad):
    """The one thing that must NOT differ from G0-V5 is the reward."""
    setattr(cfg, field, bad)
    problems, _ = verify_reward_matches_g0(cfg)
    assert any(field in p for p in problems), f"undetected reward drift in {field}"


def test_gamma_drift_is_caught(cfg):
    cfg.gamma = 0.99
    problems, _ = verify_reward_matches_g0(cfg)
    assert any("gamma" in p for p in problems)
