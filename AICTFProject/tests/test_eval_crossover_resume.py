"""Crash-safe crossover evaluation: an interrupted run resumes from its PARTIAL rows file and
yields exactly the rows an uninterrupted run yields.

Motivation (2026-09-27): a Windows Update restart killed a 4.5 h diagnostic arm with nothing on
disk, because rows were written only at the end.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from experiments import eval_specialist_crossover_scaled as E  # noqa: E402
from experiments import seed_registry as SR  # noqa: E402

FP = {"label": "L", "seeds": [1, 2, 2]}


def _write(p: Path, lines: list[str]) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_partial_with_matching_fingerprint_loads_rows(tmp_path):
    p = _write(tmp_path / "x.PARTIAL.jsonl", [json.dumps({"fingerprint": FP}),
                                               json.dumps({"policy": "pi_A", "pole": "A", "seed": 1, "win": 1})])
    assert E.load_partial(p, FP) == {("pi_A", "A", 1): {"policy": "pi_A", "pole": "A", "seed": 1, "win": 1}}


def test_partial_from_a_different_configuration_is_refused(tmp_path):
    p = _write(tmp_path / "x.PARTIAL.jsonl", [json.dumps({"fingerprint": {**FP, "label": "OTHER"}})])
    with pytest.raises(SystemExit, match="different run configuration"):
        E.load_partial(p, FP)


def test_torn_final_line_is_dropped_but_a_malformed_middle_line_refuses(tmp_path):
    good = json.dumps({"policy": "pi_A", "pole": "A", "seed": 1, "win": 1})
    p = _write(tmp_path / "a.PARTIAL.jsonl", [json.dumps({"fingerprint": FP}), good, '{"policy": "pi_A", "po'])
    assert len(E.load_partial(p, FP)) == 1
    q = _write(tmp_path / "b.PARTIAL.jsonl", [json.dumps({"fingerprint": FP}), '{"broken', good])
    with pytest.raises(SystemExit, match="line 2 is malformed"):
        E.load_partial(q, FP)


# ---------------------------------------------------------------- end to end (real episodes)
ROOT = Path(__file__).resolve().parents[1]
PI_A = ROOT / "artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip"
PI_B = ROOT / "artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip"
SPEC = ROOT / "artifacts/strategic_demand/sppo/STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json"


class _Crash(RuntimeError):
    pass


def _argv(label, resume=False):
    a = ["eval", "--team-size", "2", "--spec", str(SPEC), "--pi-a-path", str(PI_A), "--pi-b-path", str(PI_B),
         "--seed-base", "20800001", "--n-seeds", "2", "--label", label, "--device", "cpu",
         "--registry-experiment-id", "T_RESUME_SHARED"]
    return a + (["--resume"] if resume else [])


def _rows(sd: Path, label: str):
    with (sd / f"{label.lower()}_specialist_crossover_eval_rows.csv").open(encoding="utf-8") as fh:
        return sorted((r["policy"], r["pole"], r["seed"], r["win"], r["blue"], r["red"], r["margin"])
                      for r in csv.DictReader(fh))


@pytest.mark.skipif(not (PI_A.is_file() and PI_B.is_file()), reason="2v2 repaired finals not on disk")
def test_interrupted_run_resumes_to_the_identical_rows(tmp_path, monkeypatch):
    import experiments.tqdm_loop as TL
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    monkeypatch.setattr(E, "SD", tmp_path)
    monkeypatch.setattr(E, "ROOT", tmp_path)   # record paths are written relative to ROOT
    SR.allocate("T_RESUME_SHARED", 20800001, 20800002, "exploratory", "resume test",
                shared_by_labels=["T_RESUME_REF", "T_RESUME_CRASH"])

    # reference: uninterrupted
    monkeypatch.setattr(sys, "argv", _argv("T_RESUME_REF"))
    E.main()

    # crash after 3 of 8 episodes
    real = TL.tqdm_iter

    def crashing(it, **kw):
        for i, x in enumerate(real(it, **kw)):
            if i == 3:
                raise _Crash("simulated host restart")
            yield x
    monkeypatch.setattr(TL, "tqdm_iter", crashing)
    monkeypatch.setattr(sys, "argv", _argv("T_RESUME_CRASH"))
    with pytest.raises(_Crash):
        E.main()
    partial = tmp_path / "t_resume_crash_specialist_crossover_eval_rows.PARTIAL.jsonl"
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 1 + 3

    # without --resume: refused, nothing overwritten
    monkeypatch.setattr(TL, "tqdm_iter", real)
    with pytest.raises(SystemExit, match="pass --resume"):
        E.main()

    # resume: completes; rows identical to the uninterrupted run; PARTIAL removed after sealing
    monkeypatch.setattr(sys, "argv", _argv("T_RESUME_CRASH", resume=True))
    E.main()
    assert _rows(tmp_path, "T_RESUME_CRASH") == _rows(tmp_path, "T_RESUME_REF")
    assert not partial.exists()
    rec = json.loads((tmp_path / "T_RESUME_CRASH_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    assert rec["status"] == "SEALED"
    assert SR.load()["blocks"][0]["status"] == "SPENT"          # both shared labels sealed
