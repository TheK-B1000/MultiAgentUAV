"""Deployment-noise mode of the Separated crossover evaluator (2v2 noise suite, 2026-09-28).

Tier values come only from DEPLOYMENT_ROBUSTNESS_SPEC.json; localization noise also reaches the
entity pathway and the CLOSEST_DEFENDS allocator; every noise stream is seeded by the episode seed,
so a crash + --resume reproduces the uninterrupted rows exactly.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments import eval_specialist_crossover_scaled as E  # noqa: E402
from experiments import seed_registry as SR  # noqa: E402


def test_tiers_come_from_the_frozen_spec_only():
    assert E.resolve_perturbation("nominal", "") is None
    loc = E.resolve_perturbation("localization_noise", "medium")
    assert (loc["sensor_noise"], loc["drift"], loc["delay_ticks"]) == (0.06, 0.0, 0)
    mot = E.resolve_perturbation("motion_error", "medium")
    assert (mot["sensor_noise"], mot["drift"], mot["delay_ticks"]) == (0.0, 0.06, 0)
    dly = E.resolve_perturbation("control_delay", "medium")
    assert (dly["sensor_noise"], dly["drift"], dly["delay_ticks"]) == (0.0, 0.0, 2)


@pytest.mark.parametrize("fam,sev,needle", [("nominal", "medium", "meaningless"),
                                            ("motion_error", "", "needs --severity")])
def test_bad_tier_requests_refuse(fam, sev, needle):
    with pytest.raises(SystemExit, match=needle):
        E.resolve_perturbation(fam, sev)


def test_entity_noise_moves_positions_recomputes_distance_and_is_reproducible():
    rng = np.random.default_rng(0)
    obs = {"teammates": rng.normal(size=(1, 2, 1, 6)).astype(np.float32),
           "enemies": rng.normal(size=(1, 2, 2, 6)).astype(np.float32), "grid": np.zeros(3)}
    a = E.perturb_entities(obs, 0.06, np.random.default_rng([7, 1]))
    b = E.perturb_entities(obs, 0.06, np.random.default_rng([7, 1]))
    for k in ("teammates", "enemies"):
        assert np.array_equal(a[k], b[k])                                   # seeded -> identical
        assert not np.array_equal(a[k][..., :2], obs[k][..., :2])           # dx, dy moved
        assert np.array_equal(a[k][..., 3:], obs[k][..., 3:])               # alive/carry/tag untouched
        assert np.allclose(a[k][..., 2], np.sqrt(a[k][..., 0] ** 2 + a[k][..., 1] ** 2 + 1e-8))
    assert a["grid"] is obs["grid"]


# ---------------------------------------------------------------- end to end (real episodes)
ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "artifacts/scale_2v2_specialists"
PI_D = CK / "pi_A_specialist_2v2_std_split_defend_k1/ckpts/final_pi_A_specialist_2v2_std_split_defend_k1.zip"
PI_A = CK / "pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip"
PI_B = CK / "pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip"
SPEC = ROOT / "artifacts/strategic_demand/sppo/STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json"
HAVE = all(p.is_file() for p in (PI_D, PI_A, PI_B, SPEC))


class _Crash(RuntimeError):
    pass


def _argv(label, fam, sev, resume=False):
    a = ["eval", "--team-size", "2", "--spec", str(SPEC), "--pi-a-path", str(PI_D), "--pi-b-path", str(PI_B),
         "--seed-base", "20900001", "--n-seeds", "2", "--label", label, "--device", "cpu",
         "--role-fixed-for-episode", "--frozen-attack-path", str(PI_A), "--role-k-defend", "1",
         "--registry-experiment-id", "T_NOISE_SHARED", "--perturbation", fam, "--severity", sev]
    return a + (["--resume"] if resume else [])


def _rows(sd: Path, label: str):
    with (sd / f"{label.lower()}_specialist_crossover_eval_rows.csv").open(encoding="utf-8") as fh:
        return sorted((r["policy"], r["pole"], r["seed"], r["win"], r["blue"], r["red"], r["margin"])
                      for r in csv.DictReader(fh))


@pytest.mark.skipif(not HAVE, reason="2v2 split composite checkpoints not on disk")
def test_noisy_split_composite_resumes_to_identical_rows_and_control_delay_runs(tmp_path, monkeypatch):
    import experiments.tqdm_loop as TL
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    monkeypatch.setattr(E, "SD", tmp_path)
    monkeypatch.setattr(E, "ROOT", tmp_path)
    SR.allocate("T_NOISE_SHARED", 20900001, 20900002, "exploratory", "noise test",
                shared_by_labels=["T_LOC_REF", "T_LOC_CRASH", "T_DELAY"])

    monkeypatch.setattr(sys, "argv", _argv("T_LOC_REF", "localization_noise", "medium"))
    E.main()

    real = TL.tqdm_iter

    def crashing(it, **kw):
        for i, x in enumerate(real(it, **kw)):
            if i == 3:
                raise _Crash("simulated host restart")
            yield x
    monkeypatch.setattr(TL, "tqdm_iter", crashing)
    monkeypatch.setattr(sys, "argv", _argv("T_LOC_CRASH", "localization_noise", "medium"))
    with pytest.raises(_Crash):
        E.main()
    monkeypatch.setattr(TL, "tqdm_iter", real)
    monkeypatch.setattr(sys, "argv", _argv("T_LOC_CRASH", "localization_noise", "medium", resume=True))
    E.main()
    assert _rows(tmp_path, "T_LOC_CRASH") == _rows(tmp_path, "T_LOC_REF")
    rec = json.loads((tmp_path / "T_LOC_CRASH_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    assert rec["status"] == "SEALED"
    assert rec["perturbation"]["family"] == "localization_noise" and rec["perturbation"]["sensor_noise"] == 0.06
    assert rec["perturbation"]["entity_noise"] and rec["perturbation"]["allocator_noise"]

    monkeypatch.setattr(sys, "argv", _argv("T_DELAY", "control_delay", "medium"))
    E.main()
    d = json.loads((tmp_path / "T_DELAY_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    assert d["status"] == "SEALED" and d["perturbation"]["delay_ticks"] == 2
    assert SR.load()["blocks"][0]["status"] == "SPENT"
