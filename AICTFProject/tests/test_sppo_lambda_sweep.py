"""Guards on the SPPPO lambda_R development sweep driver.

A single-axis claim is worthless unless the check that enforces it fails when
the axis is violated. These tests inject the violations deliberately.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from experiments import run_sppo_lambda_sweep as SW  # noqa: E402


def test_contract_resolves_single_axis():
    c = SW.build_sweep_contract()
    assert c["single_axis_verified"] is True
    assert c["grid"] == [0.0, 0.03, 0.1, 0.3, 1.0]
    assert c["grid_extension"] == "PROHIBITED"
    assert c["steps_per_candidate"] == 98_304
    for lam, sci in c["scientific_diff_fields"].items():
        assert sci == ["sppo_lambda_rank"], f"lambda={lam} is not single-axis: {sci}"


def test_all_candidates_share_one_base_config():
    """Different seeds/PPO settings between candidates must be impossible."""
    import dataclasses
    hashes = set()
    for lam in SW.LAMBDA_GRID:
        _, base, _ = SW.build_candidate(lam)
        hashes.add(SW._stable_hash(base))
    assert len(hashes) == 1, "candidates were built from different base configs"


def test_every_candidate_carries_the_frozen_budget_and_seed():
    for lam in SW.LAMBDA_GRID:
        cfg, _, _ = SW.build_candidate(lam)
        assert int(cfg.total_timesteps) == 98_304
        assert int(cfg.seed) == 10_100_001, 'candidates must TRAIN on the training block'
        assert float(cfg.sppo_ranking_margin) == 0.04
        assert int(cfg.sppo_ranking_cadence) == 1
        assert float(cfg.sppo_lambda_rank) == lam


def test_control_carries_exactly_zero_lambda():
    cfg, _, _ = SW.build_candidate(0.0)
    assert cfg.sppo_lambda_rank == 0.0


def test_drift_outside_the_frozen_axes_aborts(monkeypatch):
    """Inject a second differing field; the contract must refuse to build."""
    real = SW.build_candidate

    def poisoned(lam):
        cfg, base, contract = real(lam)
        if lam != SW.CONTROL:
            cfg.clip_range = 0.9              # a real scientific difference
        return cfg, base, contract

    monkeypatch.setattr(SW, "build_candidate", poisoned)
    with pytest.raises(RuntimeError, match="outside the frozen axes"):
        SW.build_sweep_contract()


def test_base_config_divergence_aborts(monkeypatch):
    """If candidates stop sharing one base, the sweep must not run."""
    real = SW.build_candidate

    def poisoned(lam):
        cfg, base, contract = real(lam)
        if lam == 1.0:
            base = dict(base); base["seed"] = 999
        return cfg, base, contract

    monkeypatch.setattr(SW, "build_candidate", poisoned)
    with pytest.raises(RuntimeError, match="do not share one base config"):
        SW.build_sweep_contract()


def test_protocol_grid_drift_is_detected(monkeypatch, tmp_path):
    """The driver's grid must match the frozen protocol, not just look right."""
    payload = json.loads(SW.PROTOCOL.read_text(encoding="utf-8"))
    payload["lambda_R_SELECTION_EXPERIMENT_FROZEN_BEFORE_ANY_CANDIDATE_RUNS"][
        "candidates"] = [0.0, 0.03, 0.1, 0.3, 3.0]        # extended grid
    p = tmp_path / "drifted.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(SW, "PROTOCOL", p)
    with pytest.raises(RuntimeError, match="grid drift"):
        SW._load_protocol()


def test_protocol_budget_drift_is_detected(monkeypatch, tmp_path):
    payload = json.loads(SW.PROTOCOL.read_text(encoding="utf-8"))
    payload["lambda_R_SELECTION_EXPERIMENT_FROZEN_BEFORE_ANY_CANDIDATE_RUNS"][
        "development_budget_per_candidate"]["value_env_steps"] = 150_000
    p = tmp_path / "drifted.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(SW, "PROTOCOL", p)
    with pytest.raises(RuntimeError, match="budget drift"):
        SW._load_protocol()


def test_selection_refuses_a_short_candidate(monkeypatch, tmp_path):
    """A candidate that stopped early must abort selection, not be scored."""
    d = tmp_path / SW._tag(0.03)
    d.mkdir(parents=True)
    (d / "metrics.csv").write_text(
        "timesteps,sppo_delta_A,sppo_delta_B,ep_rew_mean,sppo_n_rank_updates\n"
        "49152,0.1,0.1,1.0,12\n", encoding="utf-8")
    monkeypatch.setattr(SW, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="run to completion"):
        SW._terminal_training_steps_guard = None
        if SW._terminal_training_steps(0.03) < SW.DEV_STEPS:
            raise RuntimeError("lambda=0.03 must run to completion")


def test_selection_refuses_a_missing_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(SW, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="did not run"):
        SW._terminal_training_steps(0.3)


def _write(tmp_path, lam, dA, dB, rew, n_rank):
    """Write the training run AND its separate development evaluation.

    The training metrics deliberately carry absurd -99 contrast values. If the
    selector ever regresses to reading TRAINING telemetry instead of the
    out-of-sample development evaluation, these make it fail loudly rather than
    quietly select an in-sample winner.
    """
    d = tmp_path / SW._tag(lam)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.csv").write_text(
        "timesteps,sppo_delta_A,sppo_delta_B,ep_rew_mean,sppo_n_rank_updates\n"
        f"98304,-99,-99,-99,{n_rank}\n", encoding="utf-8")
    ev = SW.DEV_EVAL
    payload = json.loads(ev.read_text(encoding="utf-8")) if ev.is_file() else {"results": {}}
    payload["results"][str(lam)] = {"delta_A": dA, "delta_B": dB, "ep_rew_mean": rew}
    ev.write_text(json.dumps(payload), encoding="utf-8")


def test_selection_picks_the_smallest_qualifying_lambda(monkeypatch, tmp_path):
    monkeypatch.setattr(SW, "OUT", tmp_path)
    monkeypatch.setattr(SW, "SELECTION", tmp_path / "sel.json")
    monkeypatch.setattr(SW, "DEV_EVAL", tmp_path.parent / "SPPPO_DEV_EVALUATION.json")
    _write(tmp_path, 0.0, 0.01, 0.01, 1.00, 0)      # control
    _write(tmp_path, 0.03, 0.00, 0.05, 1.00, 24)    # fails: delta_A flat
    _write(tmp_path, 0.1, 0.05, 0.05, 0.98, 24)     # qualifies
    _write(tmp_path, 0.3, 0.09, 0.09, 0.99, 24)     # qualifies, but larger
    _write(tmp_path, 1.0, 0.20, 0.20, 0.50, 24)     # fails: return collapse
    rec = SW.select()
    assert rec["SELECTED_LAMBDA_R"] == 0.1, rec["candidates"]
    assert rec["qualifying"] == [0.1, 0.3]
    assert rec["VERDICT"] == "LAMBDA_SELECTED"


def test_selection_reports_not_well_posed_when_none_qualify(monkeypatch, tmp_path):
    monkeypatch.setattr(SW, "OUT", tmp_path)
    monkeypatch.setattr(SW, "SELECTION", tmp_path / "sel.json")
    monkeypatch.setattr(SW, "DEV_EVAL", tmp_path.parent / "SPPPO_DEV_EVALUATION.json")
    _write(tmp_path, 0.0, 0.05, 0.05, 1.00, 0)
    for lam in (0.03, 0.1, 0.3, 1.0):
        _write(tmp_path, lam, 0.01, 0.01, 1.00, 24)   # both contrasts WORSE
    rec = SW.select()
    assert rec["SELECTED_LAMBDA_R"] is None
    assert rec["VERDICT"] == "SPPPO_V1_NOT_WELL_POSED"
    assert "No second grid" in rec["consequence"]


def test_selection_rejects_a_control_that_ran_ranking(monkeypatch, tmp_path):
    """If the control recorded ranking updates it was not structurally absent."""
    monkeypatch.setattr(SW, "OUT", tmp_path)
    monkeypatch.setattr(SW, "SELECTION", tmp_path / "sel.json")
    monkeypatch.setattr(SW, "DEV_EVAL", tmp_path.parent / "SPPPO_DEV_EVALUATION.json")
    _write(tmp_path, 0.0, 0.01, 0.01, 1.0, 24)        # control ran ranking!
    for lam in (0.03, 0.1, 0.3, 1.0):
        _write(tmp_path, lam, 0.05, 0.05, 1.0, 24)
    with pytest.raises(RuntimeError, match="structurally absent"):
        SW.select()


def test_return_tolerance_boundary_is_five_percent(monkeypatch, tmp_path):
    monkeypatch.setattr(SW, "OUT", tmp_path)
    monkeypatch.setattr(SW, "SELECTION", tmp_path / "sel.json")
    monkeypatch.setattr(SW, "DEV_EVAL", tmp_path.parent / "SPPPO_DEV_EVALUATION.json")
    _write(tmp_path, 0.0, 0.01, 0.01, 1.00, 0)
    _write(tmp_path, 0.03, 0.05, 0.05, 0.949, 24)     # 5.1% drop -> fails
    _write(tmp_path, 0.1, 0.05, 0.05, 0.951, 24)      # 4.9% drop -> passes
    _write(tmp_path, 0.3, 0.05, 0.05, 1.00, 24)
    _write(tmp_path, 1.0, 0.05, 0.05, 1.00, 24)
    rec = SW.select()
    assert rec["SELECTED_LAMBDA_R"] == 0.1
    assert rec["candidates"][0]["return_within_tolerance"] is False


def test_selection_refuses_without_the_development_evaluation(monkeypatch, tmp_path):
    """Selection must not fall back to training telemetry."""
    monkeypatch.setattr(SW, "OUT", tmp_path)
    monkeypatch.setattr(SW, "DEV_EVAL", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="development evaluation"):
        SW._terminal_row(0.1)
