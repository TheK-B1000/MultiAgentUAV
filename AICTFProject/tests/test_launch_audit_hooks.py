"""Tests for the runtime audit hooks.

The static gate proves a run may start. These prove the treatment still exists
while it runs. Torch-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl import launch_audit_hooks as hooks
from rl.launch_gate import LaunchGateError

OPP = {7: "A", 9: "B"}          # scripted opponent id -> pole
ZTP = {0: "A", 1: "B"}          # latent -> expected pole


class FakeTrainer:
    """Stands in for CustomPPOTrainer; the hooks only ever setattr/getattr."""


def _attach(tmp_path=None, hard_fail=True):
    return hooks.attach(FakeTrainer(), ZTP, OPP, hard_fail=hard_fail, artifact_dir=tmp_path)


# ------------------------------------------------------------------- no-op path

def test_hooks_are_noops_when_nothing_is_attached():
    """An unaudited run must pay only a getattr and never crash."""
    trainer = FakeTrainer()
    assert hooks.get(trainer) is None
    hooks.observe_episode_close(trainer, 0, 0, 7)
    hooks.bump(trainer, "ppo")


def test_attach_then_module_level_helpers_reach_the_bundle():
    trainer = FakeTrainer()
    hooks.attach(trainer, ZTP, OPP)
    hooks.observe_episode_close(trainer, 0, 0, 7)
    hooks.bump(trainer, "ppo", 3)
    aud = hooks.get(trainer)
    assert aud.episodes_observed == 1
    assert aud.counters.counts["ppo"] == 3


# ------------------------------------------------------------ z -> pole drift

def test_correct_assignment_survives_many_episodes():
    aud = _attach()
    for _ in range(200):
        aud.observe_episode_close(0, 0, 7)
        aud.observe_episode_close(1, 1, 9)
    aud.require_runtime_clean(["ppo"] if False else [], min_resets=400)
    assert aud.pole.telemetry()["violations"] == 0


def test_drift_raises_immediately_not_at_the_end():
    """EXP2B's shape. The run must die on the offending episode, not 1M steps later."""
    aud = _attach()
    aud.observe_episode_close(0, 0, 7)                     # correct
    with pytest.raises(LaunchGateError, match="Z_POLE_ASSIGNMENT_DRIFT"):
        aud.observe_episode_close(0, 0, 9)                 # z0 now facing pole B
    assert aud.episodes_observed == 2                      # died on episode 2


def test_unknown_opponent_reports_its_specific_cause_not_generic_drift():
    """An unknown opponent also registers as drift; the precise cause must win."""
    aud = _attach()
    with pytest.raises(LaunchGateError, match="UNKNOWN_OPPONENT_ID"):
        aud.observe_episode_close(0, 0, 999)


def test_unknown_opponent_is_reported_at_run_end_when_not_hard_failing():
    aud = _attach(hard_fail=False)
    aud.observe_episode_close(0, 0, 999)
    assert aud.unknown_opponents == {999: 1}
    aud.hard_fail = True
    with pytest.raises(LaunchGateError, match="UNKNOWN_OPPONENT_IDS"):
        aud.require_runtime_clean([])


def test_soft_mode_records_without_raising():
    """hard_fail=False is for fixtures and diagnostics only, never production."""
    aud = _attach(hard_fail=False)
    aud.observe_episode_close(0, 0, 9)
    assert len(aud.pole.violations) == 1
    assert aud.classified is True


# ---------------------------------------------------------- treatment liveness

def test_component_that_never_updated_fails_the_run():
    aud = _attach()
    aud.observe_episode_close(0, 0, 7)
    aud.observe_episode_close(1, 1, 9)
    aud.bump("ppo", 10)
    with pytest.raises(LaunchGateError, match="TREATMENT_NEVER_UPDATED"):
        aud.require_runtime_clean(["ppo", "selective_ranking"], min_resets=2)


def test_zero_episodes_means_treatment_never_instantiated():
    aud = _attach()
    with pytest.raises(LaunchGateError, match="POLE_AUDIT_FAILED"):
        aud.require_runtime_clean([], min_resets=1)


def test_only_one_latent_exercised_fails():
    aud = _attach()
    for _ in range(10):
        aud.observe_episode_close(0, 0, 7)          # z1 never runs
    with pytest.raises(LaunchGateError, match="POLE_AUDIT_FAILED"):
        aud.require_runtime_clean([], min_resets=10)


def test_clean_run_passes_everything():
    aud = _attach()
    for _ in range(5):
        aud.observe_episode_close(0, 0, 7)
        aud.observe_episode_close(1, 1, 9)
    for name in ("ppo", "selective_ranking", "scorer"):
        aud.bump(name, 4)
    aud.require_runtime_clean(["ppo", "selective_ranking", "scorer"], min_resets=10)


# -------------------------------------------------- no pressure on unresolved

def test_ranking_pressure_on_unresolved_is_fatal():
    """The operational content of 'preference not established'."""
    aud = _attach()
    aud.bump("ranking_on_unresolved", 1)
    with pytest.raises(LaunchGateError, match="RANKING_PRESSURE_ON_UNRESOLVED"):
        aud.require_no_pressure("ranking_on_unresolved")


def test_no_pressure_passes_when_ties_are_left_alone():
    aud = _attach()
    aud.bump("ranking_on_resolved", 25)
    aud.require_no_pressure("ranking_on_unresolved")


# ----------------------------------------------------------- self-classification

def test_failure_writes_a_classification_artifact(tmp_path):
    """The run must explain itself on the way down."""
    aud = _attach(tmp_path)
    aud.observe_episode_close(0, 0, 7)
    with pytest.raises(LaunchGateError):
        aud.observe_episode_close(0, 0, 9)
    rec = json.loads((tmp_path / "RUNTIME_AUDIT_FAILURE.json").read_text())
    assert rec["classification"] == "Z_POLE_ASSIGNMENT_DRIFT"
    assert rec["verdict"].startswith("INVALID_TREATMENT")
    assert rec["episodes_observed"] == 2
    assert "occupancy" in rec["pole_telemetry"]


def test_classification_is_skipped_when_no_artifact_dir():
    aud = _attach(None)
    aud.observe_episode_close(0, 0, 7)
    with pytest.raises(LaunchGateError):
        aud.observe_episode_close(0, 0, 9)
    assert aud.classified is True          # flagged even with nowhere to write


def test_unwritable_artifact_dir_does_not_mask_the_real_error(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("this is a file, not a directory")
    aud = _attach(blocker / "sub")
    aud.observe_episode_close(0, 0, 7)
    with pytest.raises(LaunchGateError, match="Z_POLE_ASSIGNMENT_DRIFT"):
        aud.observe_episode_close(0, 0, 9)


# ------------------------------------------------------------------- telemetry

def test_telemetry_exposes_occupancy_for_reporting():
    aud = _attach()
    for _ in range(3):
        aud.observe_episode_close(0, 0, 7)
    aud.observe_episode_close(1, 1, 9)
    tel = aud.telemetry()
    assert tel["episodes_observed"] == 4
    assert tel["pole"]["occupancy"]["z0->A"] == 3
    assert tel["pole"]["occupancy"]["z1->B"] == 1
