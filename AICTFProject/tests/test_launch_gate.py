"""Tests for the K=2 latent launch gate.

Each test is named for the failure it would have caught. Torch-free by design so
the gate is verifiable without a GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rl.launch_gate import (
    CALIB_BLOCK,
    COLLECTION_BLOCK,
    EXPERIMENT_CLASSES,
    NOT_APPLICABLE,
    OPPONENT_RANDOMIZE_FIELD,
    Check,
    LaunchGateError,
    PoleAssignmentAuditor,
    UpdateCounters,
    check_calib_split,
    check_collection_complete,
    check_content_hashes,
    check_fresh_training,
    check_final_untouched,
    check_opponent_mode,
    check_rollout_overshoot,
    check_seed_block,
    check_support_floor,
    check_thresholds_frozen,
    require_launch_authorized,
    use_time_lookup,
)


@dataclass
class FakeCfg:
    mode: str = "FIXED_OPPONENT"
    opponent_randomize: bool = False
    load_path: str | None = None
    n_envs: int = 32
    n_steps: int = 2048
    total_timesteps: int = 1_000_000


def _make_collection(tmp_path: Path, *, seeds=None, complete=True) -> Path:
    data = tmp_path / "stratified_regime_data"
    (data / "seed_shards").mkdir(parents=True)
    lo, hi = COLLECTION_BLOCK
    for seed in (seeds if seeds is not None else range(lo, hi + 1)):
        (data / "seed_shards" / f"seed_{seed}.npz").write_bytes(b"")
    if complete:
        (data / "COLLECTION_COMPLETE.json").write_text(
            json.dumps({"verdict": "COLLECTION_COMPLETE", "utc": "2026-08-29T00:00:00Z"}))
    return data


def _make_audit(tmp_path: Path, *, verdict="VALID", worst=41) -> Path:
    cells = {f"{p}_r{r}_{b}": {"n_states": 100, "n_distinct_seeds": worst + i,
                               "valid": (worst + i) >= 32}
             for i, (p, r, b) in enumerate(
                 (p, r, b) for p in "AB" for r in range(4) for b in ("not_late", "late"))}
    path = tmp_path / "SUPPORT_VALIDITY.json"
    path.write_text(json.dumps({
        "cells": cells,
        "invalid_cells": [k for k, v in cells.items() if not v["valid"]],
        "VERDICT": verdict}))
    return path


def _make_thresholds(tmp_path: Path, **over) -> Path:
    rec = {"status": "FROZEN", "calibrated_on": "CALIB",
           "experiment_class": "ONLINE_ABSTENTION",
           "thresholds": {"tau": 0.8, "rho": 0.5, "o_max": 0.1, "kappa": 0.6}}
    rec.update(over)
    path = tmp_path / "ABSTENTION_THRESHOLDS.json"
    path.write_text(json.dumps(rec))
    return path


# ------------------------------------------------------------------ happy path

def test_full_gate_passes_when_everything_is_in_order(tmp_path):
    checks = require_launch_authorized(
        FakeCfg(), _make_collection(tmp_path), _make_audit(tmp_path),
        _make_thresholds(tmp_path), strict_process_check=False)
    assert all(c.passed or not c.blocking for c in checks)


# ---------------------------------------------- collection / support integrity

def test_refuses_before_collection_completes(tmp_path):
    """The barrier is COLLECTION_COMPLETE, not 'enough seeds look done'."""
    data = _make_collection(tmp_path, complete=False)
    assert not check_collection_complete(data).passed
    with pytest.raises(LaunchGateError, match="collection_complete"):
        require_launch_authorized(FakeCfg(), data, _make_audit(tmp_path),
                                  _make_thresholds(tmp_path), strict_process_check=False)


def test_refuses_when_seed_block_has_extra_seeds(tmp_path):
    """'Just add 20 more seeds' is the repair the protocol prohibits by name."""
    lo, hi = COLLECTION_BLOCK
    data = _make_collection(tmp_path, seeds=list(range(lo, hi + 21)))
    check = check_seed_block(data)
    assert not check.passed and "unexpected" in check.detail


def test_refuses_when_seed_block_is_short(tmp_path):
    lo, hi = COLLECTION_BLOCK
    data = _make_collection(tmp_path, seeds=list(range(lo, hi - 4)))
    assert not check_seed_block(data).passed


def test_refuses_on_invalid_support_audit(tmp_path):
    """RASR died here: 5 of 8 cells short. The gate must not shrug it off."""
    assert not check_support_floor(_make_audit(tmp_path, verdict="INVALID")).passed


def test_refuses_when_a_cell_is_below_floor_despite_valid_verdict(tmp_path):
    """Defence in depth: trust the cells, not only the summary label."""
    path = _make_audit(tmp_path, worst=41)
    rec = json.loads(path.read_text())
    rec["cells"]["B_r2_late"]["n_distinct_seeds"] = 12
    path.write_text(json.dumps(rec))
    check = check_support_floor(path)
    assert not check.passed and "below floor" in check.detail


def test_refuses_when_audit_missing(tmp_path):
    assert not check_support_floor(tmp_path / "nope.json").passed


def test_calib_split_must_be_complete(tmp_path):
    lo, _ = CALIB_BLOCK
    data = _make_collection(tmp_path)
    (data / "seed_shards" / f"seed_{lo}.npz").unlink()
    assert not check_calib_split(data).passed


def test_final_seeds_anywhere_are_fatal(tmp_path):
    """FINAL is sealed. Its appearance in a training-side artifact is disqualifying."""
    root = tmp_path / "artifacts"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "seed_10600001.npz").write_bytes(b"")
    check = check_final_untouched(root)
    assert not check.passed and "FINAL" in check.detail


def test_final_check_passes_on_clean_tree(tmp_path):
    assert check_final_untouched(tmp_path).passed


# ------------------------------------------------------------------ thresholds

def test_refuses_when_thresholds_absent(tmp_path):
    """Calibration must have happened. Unset is not 'default to something'."""
    assert not check_thresholds_frozen(tmp_path / "missing.json").passed


def test_refuses_when_thresholds_not_frozen(tmp_path):
    assert not check_thresholds_frozen(_make_thresholds(tmp_path, status="DRAFT")).passed


@pytest.mark.parametrize("missing", ["tau", "rho", "o_max", "kappa"])
def test_refuses_when_any_single_threshold_is_missing(tmp_path, missing):
    """All four are jointly frozen; three out of four is not a partial pass.

    kappa matters most here: a run launched without a commit/abstain cutoff has no
    defined abstention behaviour, which collapses three classes back to two.
    """
    rec = {"status": "FROZEN", "calibrated_on": "CALIB",
           "experiment_class": "ONLINE_ABSTENTION",
           "thresholds": {"tau": 0.8, "rho": 0.5, "o_max": 0.1, "kappa": 0.6}}
    del rec["thresholds"][missing]
    path = tmp_path / "t.json"
    path.write_text(json.dumps(rec))
    check = check_thresholds_frozen(path)
    assert not check.passed and missing in check.detail


def test_refuses_when_thresholds_calibrated_on_wrong_split(tmp_path):
    """Calibrating on EVAL is evaluation-block leakage."""
    check = check_thresholds_frozen(_make_thresholds(tmp_path, calibrated_on="EVAL"))
    assert not check.passed and "CALIB" in check.detail


# -------------------------------------------------------------- config hygiene

def test_refuses_opponent_pool_mode():
    """EXP2B/EXP2C never instantiated their treatment because of this."""
    check = check_opponent_mode(FakeCfg(mode="OPPONENT_POOL"))
    assert not check.passed and "FIXED_OPPONENT" in check.detail


def test_refuses_fixed_opponent_with_randomization_flag():
    """FIXED_OPPONENT + randomize reproduces OPPONENT_POOL behaviour exactly."""
    assert not check_opponent_mode(FakeCfg(opponent_randomize=True)).passed


def test_the_randomize_field_actually_exists_on_the_real_config():
    """Regression: the guard once read a field name that did not exist.

    getattr(cfg, "randomize_scripted_opponent", False) returned False for every
    config, so the check passed the exact EXP2C setting it was written to catch.
    The unit test passed too, because its fixture used the same wrong name -- test
    and code shared one misconception. Binding to the REAL config is the only thing
    that detects that class of error.
    """
    from rl.config.ppo_config import PPOConfig
    assert hasattr(PPOConfig(), OPPONENT_RANDOMIZE_FIELD), (
        f"{OPPONENT_RANDOMIZE_FIELD} is gone from PPOConfig; the opponent guard is "
        "now inert and must be repointed, not deleted")


def test_missing_randomize_field_fails_closed():
    """A guard that cannot verify its precondition must refuse, not assume."""
    class NoField:
        mode = "FIXED_OPPONENT"
    check = check_opponent_mode(NoField())
    assert not check.passed and "cannot verify" in check.detail


def test_real_exp2c_config_would_be_refused():
    """The archived EXP2C run had opponent_randomize=True. It must not pass."""
    from rl.config.ppo_config import PPOConfig
    cfg = PPOConfig()
    cfg.mode = "FIXED_OPPONENT"
    cfg.opponent_randomize = True
    check = check_opponent_mode(cfg)
    assert not check.passed and "EXP2B/EXP2C defect" in check.detail


def test_clean_real_config_passes():
    from rl.config.ppo_config import PPOConfig
    cfg = PPOConfig()
    cfg.mode = "FIXED_OPPONENT"
    cfg.opponent_randomize = False
    assert check_opponent_mode(cfg).passed


def test_refuses_accidental_resume():
    """Fresh means step 0."""
    assert not check_fresh_training(FakeCfg(load_path="ckpt.zip")).passed


def test_rollout_overshoot_is_reported_not_blocking():
    check = check_rollout_overshoot(FakeCfg())
    assert check.passed and not check.blocking
    assert "overshoot" in check.detail


def test_rollout_overshoot_computes_the_real_number():
    # 1_000_000 / 65_536 -> 16 rollouts = 1_048_576, overshoot 48_576
    assert "48576" in check_rollout_overshoot(FakeCfg()).detail


def test_content_hash_drift_is_fatal(tmp_path):
    """Frozen documents that silently changed must stop the run."""
    doc = tmp_path / "protocol.json"
    doc.write_text("frozen")
    assert check_content_hashes({str(doc): "0" * 64}).passed is False


def test_content_hash_match_passes(tmp_path):
    import hashlib
    doc = tmp_path / "protocol.json"
    doc.write_text("frozen")
    digest = hashlib.sha256(doc.read_bytes()).hexdigest()
    assert check_content_hashes({str(doc): digest}).passed


# --------------------------------------------------------- runtime: z -> pole

def test_pole_auditor_accepts_persistent_assignment():
    aud = PoleAssignmentAuditor()
    for _ in range(50):
        aud.observe_reset(0, "A")
        aud.observe_reset(1, "B")
    aud.require_clean(min_resets=100)
    assert aud.telemetry()["violations"] == 0


def test_pole_auditor_catches_drift_after_the_first_episode():
    """The exact EXP2B failure: episode 1 correct, later episodes ~50/50."""
    aud = PoleAssignmentAuditor()
    aud.observe_reset(0, "A")
    for _ in range(20):
        aud.observe_reset(0, "B")        # silently became random
    with pytest.raises(LaunchGateError, match="persistence violated"):
        aud.require_clean()


def test_pole_auditor_catches_never_instantiated_treatment():
    """Zero resets means the treatment may never have run at all."""
    with pytest.raises(LaunchGateError, match="at least"):
        PoleAssignmentAuditor().require_clean(min_resets=1)


def test_pole_auditor_requires_both_latents_exercised():
    aud = PoleAssignmentAuditor()
    for _ in range(10):
        aud.observe_reset(0, "A")        # z1 never used
    with pytest.raises(LaunchGateError, match="never exercised"):
        aud.require_clean()


def test_pole_auditor_reports_occupancy_fractions():
    aud = PoleAssignmentAuditor()
    for _ in range(3):
        aud.observe_reset(0, "A")
    aud.observe_reset(1, "B")
    frac = aud.telemetry()["occupancy_frac"]
    assert frac["z0->A"] == pytest.approx(0.75)


# ------------------------------------------------------ runtime: update counts

def test_update_counters_catch_a_loss_that_never_reached_the_optimizer():
    counters = UpdateCounters()
    counters.bump("ppo", 100)
    counters.bump("ranking", 0)
    with pytest.raises(LaunchGateError, match="never updated"):
        counters.require_nonzero(["ppo", "ranking", "scorer"])


def test_update_counters_pass_when_everything_ran():
    counters = UpdateCounters()
    for name in ("ppo", "ranking", "identity", "scorer"):
        counters.bump(name, 5)
    counters.require_nonzero(["ppo", "ranking", "identity", "scorer"])


def test_unresolved_examples_must_receive_no_ranking_pressure():
    """The whole point of the third class: no false A/B pressure on ties."""
    counters = UpdateCounters()
    counters.bump("ranking_on_unresolved", 3)
    with pytest.raises(LaunchGateError, match="no ranking pressure"):
        counters.require_no_pressure("ranking_on_unresolved")


def test_no_pressure_check_passes_when_ties_are_left_alone():
    UpdateCounters().require_no_pressure("ranking_on_unresolved")


# --------------------------------------------------- runtime: use-time lookup

def test_use_time_lookup_rejects_missing_hook():
    """The cached-runner SAPPO bug: hook bound at construction, stale at use."""
    with pytest.raises(LaunchGateError, match="not registered at use time"):
        use_time_lookup({}, "selective_supervision")


def test_use_time_lookup_rejects_non_callable():
    with pytest.raises(LaunchGateError, match="not callable"):
        use_time_lookup({"h": "a string, not a hook"}, "h")


def test_use_time_lookup_returns_the_live_hook():
    registry = {"h": lambda: "first"}
    registry["h"] = lambda: "second"          # rebound after registration
    assert use_time_lookup(registry, "h")() == "second"


# ------------------------------------------------- experiment-class awareness

def _oracle_thresholds(tmp_path: Path, **over) -> Path:
    """ORACLE_GATED_REHEARSAL: only tau applies; the rest are N/A by design."""
    rec = {"status": "FROZEN", "calibrated_on": "CALIB",
           "experiment_class": "ORACLE_GATED_REHEARSAL",
           "thresholds": {"tau": 0.70, "rho": NOT_APPLICABLE,
                          "o_max": NOT_APPLICABLE, "kappa": NOT_APPLICABLE}}
    rec.update(over)
    path = tmp_path / "ABSTENTION_THRESHOLDS.json"
    path.write_text(json.dumps(rec))
    return path


def test_oracle_gated_rehearsal_needs_only_tau(tmp_path):
    """kappa/rho/o_max have nothing to measure without a test-time abstention rule."""
    check = check_thresholds_frozen(_oracle_thresholds(tmp_path), "ORACLE_GATED_REHEARSAL")
    assert check.passed, check.detail
    assert "N/A by design" in check.detail


def test_online_abstention_still_requires_all_four(tmp_path):
    """The permissive class must not weaken the strict one."""
    assert not check_thresholds_frozen(_oracle_thresholds(tmp_path), "ONLINE_ABSTENTION").passed


def test_permissive_artifact_cannot_launch_a_stricter_run(tmp_path):
    """Artifact and caller must agree on the class."""
    check = check_thresholds_frozen(_oracle_thresholds(tmp_path), "ONLINE_ABSTENTION")
    assert not check.passed and "may not launch a stricter run" in check.detail


def test_strict_artifact_cannot_be_used_for_a_different_class(tmp_path):
    check = check_thresholds_frozen(_make_thresholds(tmp_path), "ORACLE_GATED_REHEARSAL")
    assert not check.passed and "experiment_class" in check.detail


def test_unused_thresholds_must_be_disclaimed_not_omitted(tmp_path):
    """Absence-by-design must be distinguishable from absence-by-omission."""
    path = _oracle_thresholds(tmp_path)
    rec = json.loads(path.read_text())
    del rec["thresholds"]["kappa"]                 # omitted rather than disclaimed
    path.write_text(json.dumps(rec))
    check = check_thresholds_frozen(path, "ORACLE_GATED_REHEARSAL")
    assert not check.passed and "NOT_APPLICABLE_BY_DESIGN" in check.detail


def test_unknown_experiment_class_is_refused(tmp_path):
    check = check_thresholds_frozen(_oracle_thresholds(tmp_path), "SOMETHING_INVENTED")
    assert not check.passed and "unknown experiment class" in check.detail


def test_default_class_is_the_strictest():
    """Forgetting to pass a class can only ever over-require, never under-require."""
    assert EXPERIMENT_CLASSES["ONLINE_ABSTENTION"] == ("tau", "rho", "o_max", "kappa")
    strictest = max(EXPERIMENT_CLASSES.values(), key=len)
    assert EXPERIMENT_CLASSES["ONLINE_ABSTENTION"] == strictest
