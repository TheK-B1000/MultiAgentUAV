"""Regression cases from the rollout-assignment smoke.

Three real defects, all the same shape: an evidence check that could not see what it
claimed to check, and reported a clean number anyway. Torch-free.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from experiments.smoke_oracle_rollout_assignment import (
    REQUIRED_ROW_FIELDS,
    overlay_evidence_check,
)

# OP7's BASE profile genuinely contains this, independent of any overlay. It is the
# exact value the Pole-A overlay sets, which is what makes effect-based checking unsafe.
OP7_BASE_PROFILE = {"bt_level": 7, "min_alive_for_defender": 2, "enable_mines": True}
OP6_WITH_OVERLAY = {"bt_level": 6, "min_alive_for_defender": 2, "enable_defender": True}


def _row(env, key, *, genome=None, overlay=None, profile=None):
    return {"env_index": env, "live_opponent_key": key,
            "genome_id": genome, "requested_overlay": overlay or {},
            "resolved_profile": profile or {}}


def _correct_rows():
    return ([_row(i, "OP6", genome="SDS2_A_payoff_INIT_3",
                  overlay={"min_alive_for_defender": 2}, profile=OP6_WITH_OVERLAY)
             for i in range(16)]
            + [_row(i, "OP7", profile=OP7_BASE_PROFILE) for i in range(16, 32)])


# ------------------------------------------------- 1. auditor liveness

def test_audit_observation_is_not_gated_behind_a_training_feature_flag():
    """The auditor attached cleanly and observed 0/32 envs.

    observe_episode_close reached it only via record_episode_strategy_outcome, which
    sits behind `episode_strategy_ppo_on or forced_z_logging_on`. EXP2C has both OFF,
    so the auditor was inert and reported zero rather than erroring -- indistinguishable
    from nothing having gone wrong.
    """
    from rl.custom_ppo.rollout import collector
    src = inspect.getsource(collector)
    assert "launch_audit_hooks.observe_episode_close" in src, (
        "the collector no longer observes episode closes; the auditor is inert")

    call = src.index("launch_audit_hooks.observe_episode_close")
    guard = src.index("if episode_strategy_ppo_on or v3i3_finalize_on or forced_z_logging_on:")
    assert call < guard, (
        "the audit observation moved back inside (or after) the feature-gated block; "
        "it must run from the UNGATED done-path so auditor liveness never depends on "
        "latent_episode_strategy_ppo or forced-z logging")


def test_audit_observation_is_guarded_only_by_auditor_presence():
    """Its sole condition must be whether an auditor is attached."""
    from rl.custom_ppo.rollout import collector
    src = inspect.getsource(collector)
    call = src.index("launch_audit_hooks.observe_episode_close")
    preceding = src[:call]
    assert "launch_audit_hooks.get(self.runtime) is not None" in preceding, (
        "the observation must be gated on auditor presence alone")


# --------------------------------------- 2. real-field binding / fail closed

def test_missing_required_fields_fail_closed_rather_than_counting_zero():
    """bool(None) once flagged all 16 OP6 envs; the same mechanism can pass silently."""
    rows = [{k: v for k, v in r.items() if k != "genome_id"} for r in _correct_rows()]
    mismatches, failures = overlay_evidence_check(rows)
    assert mismatches is None, "a check that cannot see its evidence must not report a count"
    assert failures and "genome_id" in failures[0]


@pytest.mark.parametrize("dropped", REQUIRED_ROW_FIELDS)
def test_each_required_field_individually_fails_closed(dropped):
    rows = [{k: v for k, v in r.items() if k != dropped} for r in _correct_rows()]
    mismatches, failures = overlay_evidence_check(rows)
    assert mismatches is None and dropped in failures[0]


def test_no_rows_at_all_fails_closed():
    mismatches, failures = overlay_evidence_check([])
    assert mismatches is None and "captured at all" in failures[0]


def test_none_result_is_distinguishable_from_zero():
    """None means 'could not check'; 0 means 'checked, all correct'."""
    unknown, _ = overlay_evidence_check([])
    clean, _ = overlay_evidence_check(_correct_rows())
    assert unknown is None and clean == 0
    assert unknown != clean


# ------------------------------------------- 3. overlay provenance, not effect

def test_correct_installation_passes():
    mismatches, failures = overlay_evidence_check(_correct_rows())
    assert mismatches == 0 and failures == []


def test_effect_based_check_cannot_distinguish_the_poles_at_all():
    """OP6-with-overlay and OP7-base BOTH read min_alive_for_defender=2.

    Verified against the real captured evidence, not assumed. So a check reading the
    resolved profile cannot separate the poles in either direction: on the CORRECT
    configuration it flags all 16 OP7 envs as falsely carrying the overlay (a false
    FAIL that would send someone debugging a non-existent problem), and a one-sided
    variant that only asserted "OP6 has the value" would pass even with no overlay
    installed anywhere (a false PASS). Provenance is the only discriminator.
    """
    rows = [_row(i, "OP6", profile={"min_alive_for_defender": 2}) for i in range(16)] \
         + [_row(i, "OP7", profile=OP7_BASE_PROFILE) for i in range(16, 32)]
    # the effect-based reading every row satisfies:
    assert all(r["resolved_profile"].get("min_alive_for_defender") == 2 for r in rows)
    # provenance disagrees, correctly
    mismatches, failures = overlay_evidence_check(rows)
    assert mismatches == 16, "provenance must reject OP6 rows with no installed genome"
    assert failures


def test_overlay_on_the_wrong_pole_is_caught():
    rows = [_row(i, "OP6", profile=OP6_WITH_OVERLAY) for i in range(16)] \
         + [_row(i, "OP7", genome="SDS2_A_payoff_INIT_3",
                 overlay={"min_alive_for_defender": 2}) for i in range(16, 32)]
    mismatches, _ = overlay_evidence_check(rows)
    assert mismatches == 32, "OP6 missing it and OP7 carrying it are both mismatches"


def test_empty_requested_overlay_does_not_count_as_installed():
    """A genome id with no actual overlay payload is not an installation."""
    rows = [_row(i, "OP6", genome="SDS2_A_payoff_INIT_3", overlay={}) for i in range(16)] \
         + [_row(i, "OP7", profile=OP7_BASE_PROFILE) for i in range(16, 32)]
    mismatches, _ = overlay_evidence_check(rows)
    assert mismatches == 16


# ---------------------------------------------------- seam remains inert by default

def test_post_trainer_setup_defaults_to_none():
    """Experiments not using this treatment must acquire no new runtime behaviour."""
    from rl.training.orchestrator import orchestrate_training_run
    sig = inspect.signature(orchestrate_training_run)
    assert sig.parameters["post_trainer_setup"].default is None


def test_orchestrator_returns_the_env_setup_manifest():
    """A throwing check protects the run; the record has to carry the evidence."""
    from rl.training.orchestrator import orchestrate_training_run
    src = inspect.getsource(orchestrate_training_run)
    assert "return training_manifest_extra" in src
