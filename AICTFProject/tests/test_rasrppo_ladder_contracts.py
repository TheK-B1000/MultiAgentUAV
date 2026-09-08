from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import experiments.run_rasrppo_ladder as ladder
from experiments.qualify_rasr_regime_qpsi import (
    paired_seed_bootstrap_lcb,
    support_counts,
)
from experiments.run_rasrppo_ladder import OLD_QPSI_SHA, ROOT, build_config
from experiments.run_sppo_production import build_production_config
from rl.config.ppo_config import PPOConfig, TrainMode
from rl.custom_ppo.exp2_teacher_compression import (
    directed_identity_kl,
    teacher_student_kl,
)
from rl.presets import PRESET_REGISTRY, apply_preset
from rl.scorer.qpsi import QPsi, QPsiConfig


def _valid_vec(batch: int) -> torch.Tensor:
    vec = torch.zeros(batch, 2, 20)
    vec[..., 6] = 2.0 / 20.0
    vec[..., 7] = 10.0 / 20.0
    return vec


class _ToyPolicy(nn.Module):
    action_dims = (2, 2)
    n_agents = 2

    def __init__(self, *, teacher_logits: tuple[float, float] | None = None):
        super().__init__()
        if teacher_logits is None:
            self.mode_logits = nn.Parameter(torch.zeros(2, 2, 2))
            self.teacher_logits = None
        else:
            self.mode_logits = None
            self.teacher_logits = nn.Parameter(
                torch.tensor(teacher_logits, dtype=torch.float32).repeat(2, 1)
            )

    def policy_logits(self, obs, *, z_idx=None):
        batch = obs["mask"].shape[0]
        if self.mode_logits is None:
            return self.teacher_logits.reshape(1, -1).repeat(batch, 1)
        return self.mode_logits.index_select(0, z_idx).reshape(batch, -1)

    @staticmethod
    def _mask_logits(logits, mask):
        return logits.masked_fill(mask <= 0, -1e9)


IDENTITY_FIELDS = {
    "checkpoint_dir",
    "episode_csv_path",
    "metrics_csv_path",
    "run_tag",
}


def _dict(arm: str) -> dict:
    return dataclasses.asdict(build_config(arm)[0])


def _changed(left: dict, right: dict) -> set[str]:
    return {key for key in left if left[key] != right[key]}


def test_s0_is_scientifically_identical_to_spppo_production():
    production = dataclasses.asdict(build_production_config()[0])
    s0 = _dict("S0")
    assert _changed(production, s0) == IDENTITY_FIELDS | {"seed"}
    for field in (
        "sppo_lambda_rank",
        "sppo_ranking_margin",
        "sppo_ranking_cadence",
        "opponent_randomize",
        "mode",
        "exp2_teacher_checkpoints",
        "exp2_teacher_sha256",
        "sppo_qpsi_path",
        "sppo_qpsi_sha256",
    ):
        assert s0[field] == production[field]
    assert s0["mode"] == TrainMode.FIXED_OPPONENT.value
    assert s0["sppo_qpsi_sha256"] == OLD_QPSI_SHA
    assert not s0["rasr_regime_qpsi"]
    assert not s0["rasr_private_critic_heads"]
    assert not s0["rasr_directed_identity"]


def test_each_successor_has_only_its_frozen_scientific_delta():
    s0, r1, r2, r3 = (_dict(arm) for arm in ("S0", "R1", "R2", "R3"))
    expected_r1 = IDENTITY_FIELDS | {
        "rasr_regime_qpsi",
        "rasr_regime_qpsi_path",
    }
    if r1["rasr_regime_qpsi_sha256"] != s0["rasr_regime_qpsi_sha256"]:
        expected_r1.add("rasr_regime_qpsi_sha256")
    assert _changed(s0, r1) == expected_r1
    assert _changed(r1, r2) == IDENTITY_FIELDS | {"rasr_private_critic_heads"}
    assert _changed(r2, r3) == IDENTITY_FIELDS | {"rasr_directed_identity"}
    assert r1["rasr_regime_qpsi_path"]
    assert r1["rasr_regime_qpsi_sha256"] == (
        "44c0680e037939de287ad4201fead6312bc92b6bcd1fd902f568868cb24b760a"
    )


def test_dev_bootstrap_resamples_whole_seed_clusters_deterministically():
    values = np.asarray([1.0, 3.0, -2.0, 2.0])
    seeds = np.asarray([10, 10, 11, 11])
    lcb1, draws1 = paired_seed_bootstrap_lcb(
        values, seeds, samples=200, alpha=0.05, rng_seed=7
    )
    lcb2, draws2 = paired_seed_bootstrap_lcb(
        values, seeds, samples=200, alpha=0.05, rng_seed=7
    )
    assert lcb1 == lcb2
    assert np.array_equal(draws1, draws2)
    assert set(np.unique(draws1)).issubset({0.0, 1.0, 2.0})


def test_dev_support_counts_distinct_seeds_per_pole_regime():
    poles = np.repeat([0, 1], 8)
    regimes = np.tile(np.repeat(np.arange(4), 2), 2)
    seeds = np.tile([10500001, 10500002], 8)
    counts = support_counts(poles, regimes, seeds)
    assert set(counts) == {
        f"pole_{pole}_regime_{regime}"
        for pole in ("A", "B")
        for regime in range(4)
    }
    assert all(cell["n_states"] == 2 for cell in counts.values())
    assert all(cell["n_distinct_dev_seeds"] == 2 for cell in counts.values())


def test_aliases_resolve_identically_without_faithful_labels():
    pairs = {
        "rasrppo_s0_same_block_control": "rasrppo_s0",
        "rasrppo_r1_regime_scorer": "rasrppo_r1",
        "rasrppo_r2_private_critic": "rasrppo_r2",
        "rasrppo_r3_directed_identity": "rasrppo_r3",
    }
    for long_name, short_name in pairs.items():
        assert dataclasses.asdict(apply_preset(PPOConfig(), long_name)) == dataclasses.asdict(
            apply_preset(PPOConfig(), short_name)
        )
    rasr_aliases = {name for name in PRESET_REGISTRY if name.startswith("rasrppo_")}
    assert rasr_aliases == set(pairs) | set(pairs.values())
    assert not any(
        token in alias
        for alias in rasr_aliases
        for token in ("paper_faithful", "summer_faithful", "plan_faithful")
    )


def test_directed_identity_adam_step_increases_both_teacher_gaps():
    student = _ToyPolicy()
    teachers = {
        0: _ToyPolicy(teacher_logits=(3.0, -3.0)),
        1: _ToyPolicy(teacher_logits=(-3.0, 3.0)),
    }
    obs = {"mask": torch.ones(8, 4)}
    z = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    decision = torch.ones(8, 2, dtype=torch.bool)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.05)
    _, before = directed_identity_kl(student, teachers, obs, z, decision)
    loss, _ = directed_identity_kl(student, teachers, obs, z, decision)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    _, after = directed_identity_kl(student, teachers, obs, z, decision)
    assert after["identity_gap_A"] > before["identity_gap_A"]
    assert after["identity_gap_B"] > before["identity_gap_B"]
    assert all(
        parameter.grad is None
        for teacher in teachers.values()
        for parameter in teacher.parameters()
    )


def test_default_off_uses_ordinary_positive_teacher_kl():
    cfg = PPOConfig()
    assert cfg.rasr_directed_identity is False
    student = _ToyPolicy()
    teachers = {
        0: _ToyPolicy(teacher_logits=(3.0, -3.0)),
        1: _ToyPolicy(teacher_logits=(-3.0, 3.0)),
    }
    obs = {"mask": torch.ones(4, 4)}
    z = torch.tensor([0, 0, 1, 1])
    decision = torch.ones(4, 2, dtype=torch.bool)
    loss, metrics = teacher_student_kl(student, teachers, obs, z, decision)
    assert torch.isfinite(loss)
    assert "identity_gap_A" not in metrics
    assert {"kl_z0", "kl_z1"} <= set(metrics)


def test_four_regime_heads_have_selected_head_gradient_isolation():
    model = QPsi(QPsiConfig(n_regimes=4, hidden=16, conv_width=8, action_dim=4, rank=2))
    vec = _valid_vec(1)
    # Regime 2: own flag stolen, not carrying.
    vec[..., 6] = 5.0 / 20.0
    vec[..., 7] = 5.0 / 20.0
    value = model(
        torch.zeros(1, 2, 7, 20, 20),
        vec,
        torch.ones(1, 2),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
    )
    value.sum().backward()
    for index, heads in enumerate(model.regime_heads):
        grads = [parameter.grad for parameter in heads.parameters()]
        if index == 2:
            assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)
        else:
            assert all(grad is None or torch.count_nonzero(grad) == 0 for grad in grads)


def test_final_evaluation_outputs_are_absent():
    rasr_dir = ROOT / "artifacts" / "strategic_demand" / "rasrppo"
    forbidden_names = [
        path
        for path in rasr_dir.rglob("*")
        if path.is_file()
        and (
            "final_eval" in path.name.lower()
            or "terminal_evaluation" in path.name.lower()
            or path.suffix.lower() == ".csv"
            and "106" in path.name
        )
    ]
    assert forbidden_names == []
    assert not (rasr_dir / "FINAL").exists()


def test_dev_and_policy_launch_gate_semantics(tmp_path, monkeypatch):
    implementation = tmp_path / "implementation.json"
    qualification = tmp_path / "qualification.json"
    monkeypatch.setattr(ladder, "IMPLEMENTATION_GATE", implementation)
    monkeypatch.setattr(ladder, "SCORER_QUALIFICATION", qualification)

    implementation.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "dev_collection_authorized": True,
                "policy_launch_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    assert ladder.require_dev_collection_gate()["verdict"] == "PASS"
    with pytest.raises(RuntimeError, match="policy_launch_authorized"):
        ladder._launch_gate()

    implementation.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "dev_collection_authorized": True,
                "policy_launch_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="qualification artifact is absent"):
        ladder._launch_gate()
    qualification.write_text(json.dumps({"verdict": "FAIL"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="verdict must be PASS"):
        ladder._launch_gate()
    qualification.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    assert ladder._launch_gate()["policy_launch_authorized"] is True
