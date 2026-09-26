"""Rule 9 at training time: experiments/train_specialist_scale.py must REFUSE a seed the
registry has not reserved for the named experiment, or that another run already trained.

Driven through the real ``main()`` with a temporary registry and a temporary artifacts
root, so every refusal is the trainer's own, not a re-implementation of it. The
positive control (a legal seed passes the Rule-9 checks and completes a dry run) is what
shows the refusals are specific rather than a trainer that refuses everything.
"""
from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("torch")

from experiments import seed_registry as SR  # noqa: E402
import experiments.train_specialist_scale as T  # noqa: E402

MANIFEST = "train_specialist_scale run manifest"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    monkeypatch.setattr(SR, "ROOT", tmp_path)          # prior-use scan reads tmp artifacts
    SR.allocate("T_PAIR", 20500001, 20500002, "exploratory", "pi_A, pi_B")
    SR.allocate("T_DONE", 20600001, 20600002, "exploratory", "finished")
    SR.set_status("T_DONE", "SPENT")
    SR.allocate("T_GONE", 20700001, 20700002, "exploratory", "superseded")
    SR.set_status("T_GONE", "RETIRED")
    return tmp_path


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["train_specialist_scale.py", "--team-size", "2",
                                      "--policy", "A", "--device", "cpu", "--dry-run",
                                      *argv])
    return T.main()


@pytest.mark.parametrize("argv,needle", [
    (("--seed", "20500001"), "no experiment id"),
    (("--seed", "20500001", "--experiment-id", "NOT_REGISTERED"), "not registered"),
    (("--seed", "20500009", "--experiment-id", "T_PAIR"), "outside"),
    (("--seed", "20600001", "--experiment-id", "T_DONE"), "SPENT"),
    (("--seed", "20700001", "--experiment-id", "T_GONE"), "RETIRED"),
])
def test_trainer_refuses_before_any_work(registry, monkeypatch, argv, needle):
    with pytest.raises(SystemExit, match=r"Rule 9 seed registry\): .*" + needle):
        _main(monkeypatch, *argv)


def test_trainer_refuses_a_seed_another_run_already_trained(registry, monkeypatch):
    moved = registry / "artifacts" / "scale_2v2_specialists" / "INVALID_moved_aside_run"
    moved.mkdir(parents=True)
    (moved / "run_manifest.json").write_text(
        json.dumps({"record": MANIFEST, "seed": 20500001}), encoding="utf-8")
    with pytest.raises(SystemExit, match="already trained by 1 other run"):
        _main(monkeypatch, "--seed", "20500001", "--experiment-id", "T_PAIR")


def test_legal_seed_passes_rule9_and_completes_a_dry_run(registry, monkeypatch, capsys):
    """Positive control: the other seed of the same pair is untouched and must launch."""
    moved = registry / "artifacts" / "some_run"
    moved.mkdir(parents=True)
    (moved / "run_manifest.json").write_text(
        json.dumps({"record": MANIFEST, "seed": 20500001}), encoding="utf-8")
    assert _main(monkeypatch, "--seed", "20500002", "--experiment-id", "T_PAIR") == 0
    out = capsys.readouterr().out
    assert "RULE 9: seed 20500002 in T_PAIR 20500001..20500002" in out
    assert "--dry-run: config resolved and checks passed" in out


def test_smoke_runs_do_not_consult_the_registry(registry, monkeypatch, capsys):
    """Smokes use the reserved 999xxxxx family, never a registered block."""
    monkeypatch.setattr(sys, "argv", ["train_specialist_scale.py", "--team-size", "2",
                                      "--policy", "A", "--device", "cpu", "--dry-run",
                                      "--smoke", "--seed", str(SR.SMOKE_LO + 21),
                                      # own label: the real smoke_pi_A_specialist_2v2 dir holds
                                      # checkpoints, and the overwrite guard would fire first.
                                      # A dry run never creates this directory.
                                      "--run-label-suffix", "_rule9_pytest_dry_run_only"])
    assert T.main() == 0
    assert "RULE 9" not in capsys.readouterr().out
