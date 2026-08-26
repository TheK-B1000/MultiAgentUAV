"""Regression guards on the SPPPO development evaluator's env construction.

This bug family has now appeared twice in this branch: code that looks correct at
setup and is wrong at runtime.

  1. strategy_anchor._masked_heads silently took its unmasked fallback because it
     was handed the inference wrapper rather than the inner model.
  2. the first dev evaluator used R2.build_env -- the SINGLE-env Phase 0 scoring
     builder -- which would have produced a length-1 pole vector scored against a
     32-wide batch.

Neither would have raised. Both would have produced plausible numbers. These
tests pin the properties that distinguish the correct construction, so the next
instance fails in CI instead of in a frozen result.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

SRC = Path("experiments/eval_sppo_dev_candidates.py")


def _source() -> str:
    return SRC.read_text(encoding="utf-8")


def test_evaluator_uses_the_training_env_builder_not_the_phase0_one():
    """R2.build_env is single-env; the dev evaluation needs the 32-env layout."""
    src = _source()
    assert "build_training_env" in src, "evaluator must use training's env construction"
    assert "r2_learned_crossover" not in src, (
        "R2.build_env is the SINGLE-env Phase 0 scoring builder; using it here "
        "yields a length-1 pole vector against a 32-wide batch")


def test_evaluator_asserts_pole_vector_width_against_env_count():
    """A width mismatch must abort, not broadcast silently."""
    src = _source()
    assert "pole.shape[0] != int(env.core.B)" in src, (
        "evaluator must verify the pole vector spans every env")


def test_evaluator_asserts_live_z_pole_layout():
    """The per-env cell assertion is what EXP2B/EXP2C lacked after construction."""
    src = _source()
    assert "configure_exp2b_live_environment" in src
    assert "live z/pole assignment is broken" in src


def test_evaluator_verifies_qpsi_is_unmutated():
    src = _source()
    assert "Q_psi mutated during evaluation" in src, (
        "the frozen scorer must be proven bit-identical after measurement")


def test_evaluator_acts_under_the_assigned_z_with_masking():
    """Actions must use PPO's own masking, not raw logits."""
    src = _source()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_masked_argmax"), None)
    assert fn is not None, "evaluator must route actions through a masked path"
    body = ast.get_source_segment(src, fn)
    assert "_mask_logits" in body and "policy_logits" in body
    assert "z_idx=z_idx" in body, "actions must be taken under the ASSIGNED z"


def test_evaluator_reads_the_development_block_only():
    src = _source()
    assert "10_200_001" in src or "10200001" in src
    # the final evaluation block must never appear in a development scorer
    assert "10_300_001" not in src and "10300001" not in src, (
        "the untouched final block must not be reachable from development scoring")


def test_selection_source_is_the_dev_evaluation_not_training_rows():
    sweep = Path("experiments/run_sppo_lambda_sweep.py").read_text(encoding="utf-8")
    assert "DEV_EVAL" in sweep
    assert "training_telemetry_used_for_selection" in sweep, (
        "the selector must record that it did not consume training telemetry")
