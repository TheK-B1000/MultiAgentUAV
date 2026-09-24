"""Rule 8 self-test: the exporter must FAIL on an incomplete bundle.

The motivating incident was silent: the exporter ran, printed success, and left
the training curves and every intermediate checkpoint behind. By the time anyone
needed them the source machine was gone. So the tests that matter are the ones
where something is missing or corrupted and the tool refuses to say "complete".
"""

from __future__ import annotations

import json

import pytest

from experiments import export_run_bundle as EX


def _run_dir(root, scale, P, suffix="", *, curves=True, intermediates=2):
    label = f"pi_{P}_specialist_{scale}{suffix}"
    d = root / "artifacts" / f"scale_{scale}_specialists" / label
    (d / "ckpts").mkdir(parents=True)
    (d / "ckpts" / f"final_{label}.zip").write_bytes(b"terminal" + P.encode())
    for i in range(1, intermediates + 1):
        (d / "ckpts" / f"ckpt_{label}_{i}00000.zip").write_bytes(b"ck%d" % i)
    for f in ("training_manifest.json", "result_summary.json"):
        (d / f).write_text(json.dumps({"who": label}), encoding="utf-8")
    if curves:
        (d / "metrics.csv").write_text("step,win_rate\n1000,0.5\n", encoding="utf-8")
        (d / "episode_rows.csv").write_text("ep,win\n0,1\n", encoding="utf-8")
    return d


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(EX, "ROOT", tmp_path)
    monkeypatch.setattr(EX, "SD", tmp_path / "artifacts" / "strategic_demand" / "sppo")
    EX.SD.mkdir(parents=True)
    (EX.SD / "SCALE_4V4_THING_SPEC.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_complete_run_exports_and_verifies(fake):
    _run_dir(fake, "4v4", "A"); _run_dir(fake, "4v4", "B")
    out = fake / "bundle"
    assert EX.export_scale("4v4", "", out, strict=True, extra_specs=[]) == 0
    mf = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert mf["complete"] is True and mf["n_missing"] == 0
    # the files the original exporter dropped are present, with hashes
    for P in ("A", "B"):
        lbl = f"pi_{P}_specialist_4v4"
        assert mf["inventory"][f"specialists/{lbl}__metrics.csv"]["present"]
        assert mf["inventory"][f"specialists/{lbl}__episode_rows.csv"]["sha256"]
        assert mf["inventory"][f"specialists/{lbl}__ckpts/"]["n_files"] == 2
    assert EX.verify(out) == 0


def test_missing_training_curves_fail_strict(fake):
    """The exact 2026-09-12 loss: terminal checkpoint present, curves gone."""
    _run_dir(fake, "6v6", "A", curves=False, intermediates=0)
    _run_dir(fake, "6v6", "B", curves=False, intermediates=0)
    out = fake / "b6"
    assert EX.export_scale("6v6", "", out, strict=True, extra_specs=[]) == 1
    mf = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert mf["complete"] is False
    names = [k for k, v in mf["inventory"].items() if not v.get("present")]
    assert any("metrics.csv" in n for n in names)
    assert any("episode_rows.csv" in n for n in names)
    assert any(k.endswith("__ckpts/") and not v["present"]
               for k, v in mf["inventory"].items())


def test_permissive_mode_still_records_the_gap(fake):
    """Without --strict the exporter returns 0, but must never claim complete."""
    _run_dir(fake, "6v6", "A", curves=False); _run_dir(fake, "6v6", "B", curves=False)
    out = fake / "b7"
    assert EX.export_scale("6v6", "", out, strict=False, extra_specs=[]) == 0
    assert json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["complete"] is False


def test_verify_detects_a_file_lost_in_transfer(fake):
    _run_dir(fake, "4v4", "A"); _run_dir(fake, "4v4", "B")
    out = fake / "b2"
    EX.export_scale("4v4", "", out, strict=True, extra_specs=[])
    (out / "specialists" / "pi_A_specialist_4v4__metrics.csv").unlink()
    assert EX.verify(out) == 1


def test_verify_detects_corruption(fake):
    _run_dir(fake, "4v4", "A"); _run_dir(fake, "4v4", "B")
    out = fake / "b3"
    EX.export_scale("4v4", "", out, strict=True, extra_specs=[])
    (out / "specialists" / "final_pi_A_specialist_4v4.zip").write_bytes(b"truncated")
    assert EX.verify(out) == 1


def test_verify_refuses_a_bundle_with_no_manifest(fake):
    (fake / "nomanifest").mkdir()
    with pytest.raises(SystemExit, match="cannot be verified"):
        EX.verify(fake / "nomanifest")


def test_verify_fails_a_bundle_that_was_incomplete_at_creation(fake):
    """Every file matches its hash, but the bundle was already missing things when
    it was made. Hash-matching is not completeness -- it must still fail."""
    _run_dir(fake, "6v6", "A", curves=False); _run_dir(fake, "6v6", "B", curves=False)
    out = fake / "b4"
    EX.export_scale("6v6", "", out, strict=False, extra_specs=[])
    assert EX.verify(out) == 1


def test_plan_mode_copies_nothing(fake):
    _run_dir(fake, "4v4", "A"); _run_dir(fake, "4v4", "B")
    out = fake / "b5"
    assert EX.export_scale("4v4", "", out, strict=True, extra_specs=[],
                           plan_only=True) == 0
    assert not out.exists()


def test_plan_mode_reports_missing_under_strict(fake):
    _run_dir(fake, "6v6", "A", curves=False); _run_dir(fake, "6v6", "B", curves=False)
    assert EX.export_scale("6v6", "", fake / "b8", strict=True, extra_specs=[],
                           plan_only=True) == 1


def test_suffixed_run_labels_are_supported(fake):
    """4v4 runs carry suffixes like _b3; the exporter must follow them."""
    _run_dir(fake, "4v4", "A", "_b3"); _run_dir(fake, "4v4", "B", "_b3")
    out = fake / "b9"
    assert EX.export_scale("4v4", "_b3", out, strict=True, extra_specs=[]) == 0
    assert (out / "specialists" / "final_pi_A_specialist_4v4_b3.zip").is_file()


def test_missing_run_directory_is_a_named_failure(fake):
    _run_dir(fake, "4v4", "A")            # pi_B never trained
    out = fake / "b10"
    assert EX.export_scale("4v4", "", out, strict=True, extra_specs=[]) == 1
    mf = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert mf["inventory"]["specialists/pi_B_specialist_4v4/"]["present"] is False
