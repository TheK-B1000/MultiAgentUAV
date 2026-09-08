"""Contract guards on the SPPPO V1 final evaluator, BEFORE 103... is consumed.

Every check the PI required before seed consumption, plus the construction
regression this project has now learned to value twice:

    number of pole/latent assignments == evaluation batch width

These run without touching a single 10300001..10300192 seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import eval_sppo_v1_terminal as W  # noqa: E402
from experiments import eval_exp2_k2_terminal as E  # noqa: E402

PROD_SHA = "1260e420b85ad01b1aecf433d65688adf153538195d30511e26a3585b7285f1f"


def test_student_checkpoint_sha_is_the_frozen_production_terminal():
    assert E.EXPECTED_HASHES["student"] == PROD_SHA
    assert E.STUDENT.name.startswith("final_")
    assert E.STUDENT.is_file(), "the frozen production terminal must exist"
    assert E._sha256(E.STUDENT) == PROD_SHA, "on-disk checkpoint does not match the frozen SHA"


def test_block_is_exactly_192_seeds_of_103():
    assert E.SEED_BASE == 10_300_001
    assert E.SEED_BLOCK == "10300001..10300192"
    assert E.N_PAIRED == 192
    lo, hi = (int(x) for x in E.SEED_BLOCK.split(".."))
    assert hi - lo + 1 == 192


def test_qpsi_is_structurally_absent_from_the_evaluation_path():
    """Not merely unused -- not importable, loadable or callable from here.

    Scans CODE, not prose: the wrapper's docstring legitimately explains that
    Q_psi is absent, and a raw text scan would flag that explanation as a
    violation. Imports, attribute access and call targets are what matter.
    """
    import ast
    banned = ("qpsi", "q_psi", "scorer", "strategic_contrast", "ranking_loss",
              "load_frozen_qpsi", "expected_value")
    for mod in (E, W):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        low = {s.lower() for s in names}
        hits = [b for b in banned if any(b in s for s in low)]
        assert not hits, f"{hits} reachable as code in {Path(mod.__file__).name}"
        # and rl.scorer must not be importable from this path at all
        assert not any("rl.scorer" in s for s in names), "rl.scorer imported by the evaluator"


def test_gate_definitions_are_inherited_not_redefined():
    """The wrapper may rebind identity only. Gate logic must come from EXP2."""
    src = Path(W.__file__).read_text(encoding="utf-8")
    for forbidden in ("N_BOOT", "ALPHA =", "BOOTSTRAP_SEED", "N_PAIRED",
                      "POLICY_CELLS", "def _paired", "def _ratio"):
        assert forbidden not in src, f"wrapper redefines {forbidden!r}; gates must be inherited"


def test_bootstrap_convention_identical_to_the_frozen_exp2c_run():
    assert E.N_BOOT == 20_000
    assert E.ALPHA == 0.05
    assert E.BOOTSTRAP_SEED == 7
    assert E.MAX_STEPS == 240


def test_both_latent_modes_scored_on_both_poles():
    assert set(E.POLES) == {"A", "B"}
    assert set(E.POLICY_CELLS) == {"z0", "z1", "pi_A", "pi_B"}
    cells = [f"{p}|{k}" for p in E.POLICY_CELLS for k in E.POLES]
    for required in ("z0|A", "z0|B", "z1|A", "z1|B"):
        assert required in cells, f"{required} missing: a mode is not scored off its training pole"
    assert len(cells) == 8


def test_retention_references_are_the_same_frozen_sappo_checkpoints():
    """Read EXP2C's hashes from SOURCE, never by importing its wrapper.

    Importing a second wrapper rebinds the shared evaluator module and would
    clobber the SPPPO identity under test.
    """
    src = Path("experiments/eval_exp2c_k2_terminal.py").read_text(encoding="utf-8")
    for name in ("pi_A", "pi_B"):
        assert E.EXPECTED_HASHES[name] in src, (
            f"{name} reference differs from the frozen EXP2C retention reference")


def test_protocol_declares_the_same_terminal_contract_as_exp2c():
    sppo = json.loads(Path("artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json")
                      .read_text(encoding="utf-8"))["terminal_evaluation"]
    exp2c = json.loads(Path("artifacts/strategic_demand/"
                            "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_PROTOCOL.json")
                       .read_text(encoding="utf-8"))["terminal_evaluation"]
    for k in ("episodes_per_cell", "total_episodes", "bootstrap",
              "student_cells", "teacher_reference_cells", "deterministic"):
        assert sppo[k] == exp2c[k], f"terminal contract drift on {k!r}"


def test_evaluator_cannot_read_development_or_sweep_results():
    src = (Path(E.__file__).read_text(encoding="utf-8")
           + Path(W.__file__).read_text(encoding="utf-8"))
    for leak in ("SPPPO_DEV_EVALUATION", "SPPPO_LAMBDA_SELECTION", "lambda_sweep",
                 "SPPPO_PRODUCTION_FIRST_INTERVAL"):
        assert leak not in src, f"evaluator can read {leak!r} and alter execution"


def test_development_seed_is_not_the_evaluation_block():
    assert E.DEVELOPMENT_SEED == 10_200_001
    assert not (10_300_001 <= E.DEVELOPMENT_SEED <= 10_300_192)


def test_one_attempt_guard_blocks_a_second_launch():
    """Spend-once: existing outputs must refuse a second --launch."""
    src = Path(E.__file__).read_text(encoding="utf-8")
    assert "spend-once" in src
    assert "summary.json" in src and "episode_rows.csv" in src


def test_evaluation_output_is_outside_the_training_and_sweep_trees():
    out = E.OUT.resolve()
    for tree in (E.TRAIN_DIR.resolve(),
                 (E.SD / "sppo" / "lambda_sweep").resolve()):
        with pytest.raises(ValueError):
            out.relative_to(tree)


def test_assignment_width_matches_batch_width_regression():
    """The construction bug this project has now hit twice.

    A pole/latent assignment vector shorter than the batch would broadcast
    silently and score the wrong cells. Assert the counts are pinned.
    """
    assert len(E.POLICY_CELLS) * len(E.POLES) == 8
    proto = json.loads(Path("artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json")
                       .read_text(encoding="utf-8"))["terminal_evaluation"]
    n_cells = len(proto["student_cells"]) + len(proto["teacher_reference_cells"])
    assert n_cells == 8
    assert proto["episodes_per_cell"] * n_cells == proto["total_episodes"] == 1536


def test_seed_and_episode_accounting_is_declared():
    proto = json.loads(Path("artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json")
                       .read_text(encoding="utf-8"))["terminal_evaluation"]
    assert proto["raw_rows_required"] is True, "raw rows needed to prove each cell scored once"
    assert proto["episodes_per_cell"] == 192


def test_binding_clobber_is_fatal_not_silent():
    """Importing another wrapper must abort scoring, not redirect it."""
    import importlib
    W._assert_bindings_intact()                      # clean: passes
    other = importlib.import_module("experiments.eval_exp2c_k2_terminal")
    assert E.SEED_BASE == 8_900_001, "precondition: the clobber actually happened"
    with pytest.raises(RuntimeError, match="clobbered"):
        W._assert_bindings_intact()
    importlib.reload(W)                              # restore for other tests
    W._assert_bindings_intact()
