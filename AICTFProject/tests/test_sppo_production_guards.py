"""Refusal surface for the SPPPO 1M production run.

The runner is part of the treatment: its job is to guarantee the run is
genuinely fresh. These tests inject each forbidden condition and require a hard
failure, because a guard that has never been seen to fire is not evidence.

The positive control is the important one. The selected lambda = 1.0 development
checkpoint shares the training block, the lambda and the architecture, so a
resume from it would look entirely correct while silently invalidating the 1M
claim. That is the most tempting accidental continuation path in the project.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from experiments import run_sppo_production as P  # noqa: E402


def _cfg():
    cfg, _ = P.build_production_config()
    return cfg


# ----------------------------------------------------------- baseline passes
def test_clean_production_config_validates():
    checks = P.validate(_cfg())
    assert all(c["ok"] for c in checks.values())
    assert len(checks) >= 18


def test_contract_declares_fresh_and_terminal_only():
    cfg = _cfg()
    c = P.launch_contract(cfg, P.validate(cfg))
    assert c["fresh_initialisation"] is True
    assert c["resumed_from"] is None
    assert c["candidate_checkpoints_reused"] is False
    assert c["development_steps_counted_toward_budget"] == 0
    assert c["terminal_checkpoint_only"] is True
    assert c["total_timesteps"] == 1_000_000
    assert c["lambda_R"] == 1.0


# ------------------------------------------- POSITIVE CONTROL: the temptation
def test_POSITIVE_CONTROL_resume_from_selected_dev_checkpoint_is_refused():
    """Point load_path at the winning lambda=1.0 dev checkpoint. Must hard-fail.

    Same training block, same lambda, same architecture -- the one continuation
    that would look right and be wrong.
    """
    from experiments.run_sppo_lambda_sweep import OUT as SWEEP_OUT, _tag
    ck_dir = SWEEP_OUT / _tag(1.0) / "ckpts"
    ckpts = sorted(ck_dir.glob("*"))
    assert ckpts, "the selected candidate checkpoint should exist for provenance"

    cfg = _cfg()
    cfg.load_path = str(ckpts[-1])
    with pytest.raises(RuntimeError, match="PRODUCTION LAUNCH REFUSED"):
        P.validate(cfg)
    # and specifically for the right reasons
    try:
        P.validate(cfg)
    except RuntimeError as e:
        msg = str(e)
        assert "no_checkpoint_input__load_path" in msg
        assert "no_input_resolves_into_the_sweep_tree" in msg


def test_candidate_checkpoints_existing_does_NOT_block_a_fresh_run():
    """Existence is provenance. Only resolution as an INPUT is forbidden."""
    from experiments.run_sppo_lambda_sweep import OUT as SWEEP_OUT, _tag
    assert (SWEEP_OUT / _tag(1.0) / "ckpts").exists(), "candidates must still exist"
    checks = P.validate(_cfg())          # unchanged config still validates
    assert all(c["ok"] for c in checks.values())


# ------------------------------------------------------------ each refusal
@pytest.mark.parametrize("field,value,key", [
    ("total_timesteps", 500_000, "budget_exactly_1M"),
    ("total_timesteps", 1_098_304, "budget_exactly_1M"),        # 1M + dev steps
    ("sppo_lambda_rank", 0.3, "lambda_R_is_frozen_selection"),
    ("sppo_lambda_rank", 0.0, "lambda_R_is_frozen_selection"),
    ("sppo_ranking_margin", 0.05, "margin_frozen"),
    ("sppo_ranking_cadence", 4, "ranking_cadence_frozen"),
    ("mode", "OPPONENT_POOL", "mode_is_FIXED_OPPONENT"),
    ("opponent_randomize", True, "opponent_randomize_off"),
    ("seed", 10_200_001, "training_seed_is_the_training_block"),
    ("checkpoint_run_start_step", 98_304, "starts_at_timestep_zero"),
    ("load_weights_only", True, "load_weights_only_off"),
    ("periodic_checkpoint_steps", 100_000, "periodic_checkpointing_disabled"),
    ("exp2_teacher_lambda", 0.5, "teacher_lambda_unchanged"),
    ("exp2_teacher_cadence", 1, "teacher_cadence_unchanged"),
    ("n_envs", 16, "n_envs_is_32"),
])
def test_each_forbidden_condition_is_refused(field, value, key):
    cfg = _cfg()
    setattr(cfg, field, value)
    with pytest.raises(RuntimeError, match="PRODUCTION LAUNCH REFUSED") as ei:
        P.validate(cfg)
    assert key in str(ei.value), f"expected {key} to fire, got:\n{ei.value}"


def test_production_output_inside_the_sweep_tree_is_refused():
    from experiments.run_sppo_lambda_sweep import OUT as SWEEP_OUT
    cfg = _cfg()
    cfg.checkpoint_dir = str(SWEEP_OUT / "sneaky" / "ckpts")
    with pytest.raises(RuntimeError, match="output_outside_sweep_tree__checkpoint_dir"):
        P.validate(cfg)


def test_wrong_qpsi_is_refused():
    cfg = _cfg()
    cfg.sppo_qpsi_path = "artifacts/strategic_demand/does_not_exist.pt"
    with pytest.raises(RuntimeError, match="qpsi_sha_matches_frozen"):
        P.validate(cfg)


def test_teacher_checkpoints_are_not_treated_as_forbidden_inputs():
    """The frozen SAPPO teachers are legitimate inputs and must not be rejected."""
    cfg = _cfg()
    assert cfg.exp2_teacher_checkpoints, "teachers must be configured"
    assert "exp2_teacher_checkpoints" not in P.CHECKPOINT_INPUT_FIELDS
    checks = P.validate(cfg)
    assert all(c["ok"] for c in checks.values())
