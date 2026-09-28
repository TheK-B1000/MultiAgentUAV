"""Deployment-noise table: per-condition crossover and paired change vs nominal on matched seeds."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiments import collect_baseline_results as C

LABELS = [c[0] for c in C.NOISE["2v2"]["conditions"]]
FAMILIES = [c[1] for c in C.NOISE["2v2"]["conditions"]]


def _write(sd, label, family, wins, seeds):
    """wins: {(policy, pole): list of 0/1 per seed}."""
    p = sd / f"{label.lower()}_specialist_crossover_eval_rows.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["policy", "pole", "seed", "win"])
        w.writeheader()
        for (pol, pole), ws in wins.items():
            for s, x in zip(seeds, ws):
                w.writerow({"policy": pol, "pole": pole, "seed": s, "win": x})
    v = {k: np.array(x, dtype=float) for k, x in wins.items()}
    da = float((v[("pi_A", "A")] - v[("pi_B", "A")]).mean())
    db = float((v[("pi_B", "B")] - v[("pi_A", "B")]).mean())
    (sd / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").write_text(json.dumps({
        "status": "SEALED", "perturbation": {"family": family},
        "pole_attestations": {"A": {"hashes_match": True}, "B": {"hashes_match": True}},
        "PRIMARY_GATE": {"delta_A": {"mean": da}, "delta_B": {"mean": db}},
    }), encoding="utf-8")


def _suite(sd, seeds, drop_last_seed_in=None, wrong_family_at=None):
    rng = np.random.default_rng(3)
    for i, (label, fam) in enumerate(zip(LABELS, FAMILIES)):
        ss = seeds[:-1] if label == drop_last_seed_in else seeds
        wins = {(p, q): rng.integers(0, 2, len(ss)).tolist() for p in ("pi_A", "pi_B") for q in ("A", "B")}
        _write(sd, label, "motion_error" if i == wrong_family_at else fam, wins, ss)


def test_paired_change_equals_per_seed_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SD", tmp_path)
    seeds = list(range(100, 132))
    _suite(tmp_path, seeds)
    out = C.collect_noise("2v2")
    assert out["n_seeds"] == 32 and [c["family"] for c in out["conditions"]] == FAMILIES
    assert "paired_change_vs_nominal" not in out["conditions"][0]
    nom, loc = out["conditions"][0], out["conditions"][1]
    assert loc["paired_change_vs_nominal"]["delta_A"]["mean"] == pytest.approx(
        loc["delta_A"]["mean"] - nom["delta_A"]["mean"])            # paired mean == difference of means
    tex = C._tex({"utc": "t", "rows": [], "noise": {"2v2": out}}, "2v2")
    assert "\\newcommand{\\TwoNoiseLocDDA}" in tex and "\\TwoNoiseNomDDA" not in tex


def test_pending_until_every_condition_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SD", tmp_path)
    _suite(tmp_path, list(range(100, 110)))
    (tmp_path / f"{LABELS[3]}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").unlink()
    assert C.collect_noise("2v2") is None


def test_unmatched_seeds_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SD", tmp_path)
    _suite(tmp_path, list(range(100, 110)), drop_last_seed_in=LABELS[2])
    with pytest.raises(SystemExit, match="not matched"):
        C.collect_noise("2v2")


def test_wrong_perturbation_family_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SD", tmp_path)
    _suite(tmp_path, list(range(100, 110)), wrong_family_at=1)
    with pytest.raises(SystemExit, match="is not localization_noise"):
        C.collect_noise("2v2")
