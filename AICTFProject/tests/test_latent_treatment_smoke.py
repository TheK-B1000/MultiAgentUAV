"""Tests for the end-to-end treatment smoke harness.

Fixtures exercise the plumbing. The live path is separately proven to refuse on
unearned artifacts -- nothing synthetic may ever satisfy it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.latent_treatment_smoke import (
    RESOLVED,
    UNRESOLVED,
    ParameterWitness,
    require_live_artifacts,
    run_smoke,
    write_receipt,
)
from rl.launch_gate import LaunchGateError

ZTP = {0: "A", 1: "B"}
OPP = {7: "A", 9: "B"}


class FixtureProbe:
    """A minimal, honest stand-in for the treatment. Never used by --live."""

    def __init__(self, *, episodes=8, drift=False, resolved=6, unresolved=0,
                 move_frozen=False, move_trainable=True, steps=1):
        self._episodes, self._drift = episodes, drift
        self._resolved, self._unresolved = resolved, unresolved
        self._move_frozen, self._move_trainable, self._steps = move_frozen, move_trainable, steps
        self.params = {
            "trunk.frozen_w": np.ones(4, dtype=np.float32),
            "head_z0.w": np.zeros(4, dtype=np.float32),
            "head_z1.w": np.zeros(4, dtype=np.float32),
        }

    def episode_boundaries(self):
        for i in range(self._episodes):
            z = i % 2
            opponent = 7 if z == 0 else 9
            if self._drift and i == self._episodes - 1:
                opponent = 9 if z == 0 else 7        # silently swapped
            yield i, z, opponent

    def named_parameters(self):
        return self.params

    def frozen_parameter_names(self):
        return ["trunk.frozen_w"]

    def trainable_parameter_names(self):
        return ["head_z0.w", "head_z1.w"]

    def apply_supervision(self, bump):
        bump("ppo", 1)
        if self._resolved:
            bump(RESOLVED, self._resolved)
        if self._unresolved:
            bump(UNRESOLVED, self._unresolved)

    def optimizer_step(self):
        if self._move_trainable:
            self.params["head_z0.w"] = self.params["head_z0.w"] + 0.1
            self.params["head_z1.w"] = self.params["head_z1.w"] + 0.1
        if self._move_frozen:
            self.params["trunk.frozen_w"] = self.params["trunk.frozen_w"] + 0.1
        return self._steps


def _run(**kw):
    return run_smoke(FixtureProbe(**kw), z_to_pole=ZTP, opponent_to_pole=OPP)


# ------------------------------------------------------------------ happy path

def test_clean_treatment_passes():
    r = _run()
    assert r.verdict == "PASS", r.failures
    assert r.z0_episodes == 4 and r.z1_episodes == 4
    assert r.pole_assignment_violations == 0
    assert r.resolved_pressure_count == 6
    assert r.unresolved_pressure_count == 0
    assert r.optimizer_steps == 1
    assert r.frozen_parameter_hash_match is True
    assert set(r.changed_parameter_groups) == {"head_z0.w", "head_z1.w"}


# --------------------------------------------- 1 & 2: treatment exists, persists

def test_pole_drift_fails_the_smoke():
    r = _run(drift=True)
    assert r.verdict == "INVALID_BEFORE_TRAINING"
    assert r.pole_assignment_violations == 1
    assert any("pole assignment violated" in f for f in r.failures)


def test_too_few_resets_fails():
    r = _run(episodes=2)
    assert any("resets observed" in f for f in r.failures)


def test_a_latent_that_never_executes_fails():
    class OneLatent(FixtureProbe):
        def episode_boundaries(self):
            for i in range(8):
                yield i, 0, 7                      # z1 never runs
    r = run_smoke(OneLatent(), z_to_pole=ZTP, opponent_to_pole=OPP)
    assert any("z1 never executed" in f for f in r.failures)
    assert r.z1_episodes == 0


# ------------------------------------------ 3: selective supervision is real

def test_any_pressure_on_unresolved_fails():
    """The whole point of the third class."""
    r = _run(unresolved=1)
    assert r.verdict == "INVALID_BEFORE_TRAINING"
    assert any("exactly zero" in f for f in r.failures)


def test_no_pressure_on_resolved_fails():
    """Abstaining on everything is the dual degenerate solution."""
    r = _run(resolved=0)
    assert any("no strategic pressure" in f for f in r.failures)


# --------------------------------------------------- 4: optimizer path is real

def test_frozen_parameters_moving_fails():
    r = _run(move_frozen=True)
    assert r.frozen_parameter_hash_match is False
    assert any("frozen parameters changed" in f for f in r.failures)


def test_trainable_parameters_not_moving_fails():
    """A loss that is computed but never reaches the optimizer looks exactly like this."""
    r = _run(move_trainable=False)
    assert any("did not move" in f for f in r.failures)
    assert r.changed_parameter_groups == []


def test_zero_optimizer_steps_fails():
    r = _run(steps=0)
    assert any("no optimizer step" in f for f in r.failures)


# ------------------------------------------------------------ ParameterWitness

def test_witness_requires_snapshot_first():
    w = ParameterWitness(lambda: {"a": np.zeros(2, dtype=np.float32)})
    with pytest.raises(LaunchGateError, match="before snapshot"):
        w.compare()


def test_witness_detects_bit_level_change():
    store = {"a": np.zeros(2, dtype=np.float32)}
    w = ParameterWitness(lambda: store)
    w.snapshot()
    store["a"] = np.array([0.0, 1e-8], dtype=np.float32)
    changed, unchanged = w.compare()
    assert changed == ["a"] and unchanged == []


def test_witness_reports_unchanged_parameters():
    store = {"a": np.zeros(2, dtype=np.float32), "b": np.ones(2, dtype=np.float32)}
    w = ParameterWitness(lambda: store)
    w.snapshot()
    store["a"] = np.ones(2, dtype=np.float32)
    changed, unchanged = w.compare()
    assert changed == ["a"] and unchanged == ["b"]


def test_witness_fails_when_parameters_disappear():
    store = {"a": np.zeros(2, dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    w = ParameterWitness(lambda: store)
    w.snapshot()
    del store["b"]
    with pytest.raises(LaunchGateError, match="vanished"):
        w.compare()


def test_digest_is_order_independent_but_value_sensitive():
    a = {"x": np.zeros(2, dtype=np.float32), "y": np.ones(2, dtype=np.float32)}
    w1 = ParameterWitness(lambda: a)
    w2 = ParameterWitness(lambda: {"y": a["y"], "x": a["x"]})
    assert w1.digest(["x", "y"]) == w2.digest(["y", "x"])
    b = {"x": np.zeros(2, dtype=np.float32), "y": np.full(2, 2.0, dtype=np.float32)}
    assert ParameterWitness(lambda: b).digest(["x", "y"]) != w1.digest(["x", "y"])


# ------------------------------------------------- 5: live path stays barred

def test_live_refuses_without_earned_artifacts(tmp_path):
    """No collection, no audit, no thresholds -- and no way around it."""
    with pytest.raises(LaunchGateError, match="LIVE SMOKE REFUSED"):
        require_live_artifacts(tmp_path, tmp_path / "a.json", tmp_path / "t.json")


def test_live_refuses_on_real_repo_state_today():
    """The genuine current state must refuse. If this ever passes, re-read why."""
    with pytest.raises(LaunchGateError, match="LIVE SMOKE REFUSED"):
        require_live_artifacts()


def test_fixture_mode_never_invents_a_threshold_sha():
    r = _run()
    assert r.threshold_artifact_sha == ""
    assert any("none may be invented" in n for n in r.notes)


def test_live_mode_without_threshold_artifact_is_invalid():
    r = run_smoke(FixtureProbe(), z_to_pole=ZTP, opponent_to_pole=OPP, mode="live")
    assert r.verdict == "INVALID_BEFORE_TRAINING"
    assert any("no threshold artifact" in f for f in r.failures)


# ---------------------------------------------------------------- the receipt

def test_receipt_contains_every_required_field(tmp_path):
    path = write_receipt(_run(), tmp_path / "LATENT_TREATMENT_SMOKE.json")
    rec = json.loads(Path(path).read_text())
    for key in ("verdict", "resets_observed", "z0_episodes", "z1_episodes",
                "pole_assignment_violations", "resolved_pressure_count",
                "unresolved_pressure_count", "optimizer_steps",
                "changed_parameter_groups", "frozen_parameter_hash_match",
                "threshold_artifact_sha"):
        assert key in rec, key
    assert rec["verdict"] == "PASS"
    assert "NOTHING about whether learning works" in rec["meaning"]


def test_receipt_records_failure_verdict(tmp_path):
    path = write_receipt(_run(drift=True), tmp_path / "r.json")
    assert json.loads(Path(path).read_text())["verdict"] == "INVALID_BEFORE_TRAINING"
