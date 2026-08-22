from __future__ import annotations

import copy
import random

import numpy as np
import torch

from experiments.run_exp2b_specialization_preserving_compression import (
    CELL_KEYS,
    CELL_Z,
    EXPECTED_CELLS,
    build_exp2b_config,
)
from rl.custom_ppo.exp2_teacher_compression import (
    Exp2TeacherCompressionRunner,
    exp2b_actor_gradient_cosine,
)
from tests.test_exp2_k2_compression import _batch, _student, _teacher


def _ppo_batch(batch_size: int = 8):
    batch = _batch(batch_size)
    model = _student()
    batch.update({
        "global_state": torch.zeros(batch_size, model.global_state_dim),
        "actions": torch.zeros(batch_size, 4, dtype=torch.long),
        "log_probs": torch.zeros(batch_size),
        "advantages": torch.linspace(-1.0, 1.0, batch_size),
    })
    return model, batch


def test_exp2b_config_diff_is_identity_paths_only():
    cfg, contract = build_exp2b_config()
    assert cfg.seed == 8_400_001
    assert cfg.total_timesteps == 2_000_000
    assert tuple(cfg.forced_latent_env_ids) == CELL_Z
    assert set(contract["resolved_config_diff_vs_EXP2"]) == {
        "checkpoint_dir", "episode_csv_path", "exp2_protocol_path",
        "metrics_csv_path", "run_tag", "seed",
    }
    assert contract["single_scientific_delta_external_assignment"]["EXP2B"] == EXPECTED_CELLS


def test_exp2b_assignment_is_exactly_16_0_0_16():
    realized = {
        "z0_A": sum(z == 0 and key == "OP6" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z0_B": sum(z == 0 and key == "OP7" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z1_A": sum(z == 1 and key == "OP6" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z1_B": sum(z == 1 and key == "OP7" for z, key in zip(CELL_Z, CELL_KEYS)),
    }
    assert realized == {"z0_A": 16, "z0_B": 0, "z1_A": 0, "z1_B": 16}


def test_gradient_cosine_is_finite_and_mutation_free():
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    model, batch = _ppo_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    params_before = [p.detach().clone() for p in model.parameters()]
    grads_before = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    python_before, numpy_before = random.getstate(), np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    cosine = exp2b_actor_gradient_cosine(model, batch, clip_range=0.2)

    assert np.isfinite(cosine) and -1.000001 <= cosine <= 1.000001
    assert all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    for before, parameter in zip(grads_before, model.parameters()):
        assert (before is None and parameter.grad is None) or torch.equal(before, parameter.grad)
    assert optimizer.state_dict() == optimizer_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_exp2b_runner_reports_assigned_cells_and_one_cosine_per_step():
    model, batch = _ppo_batch()
    runner = Exp2TeacherCompressionRunner(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        {0: _teacher(10), 1: _teacher(11)},
        lambda_teacher=0.1,
        cadence=4,
        batch_size=64,
        max_grad_norm=None,
        seed=99,
        device="cpu",
        cell_counts=(16, 0, 0, 16),
        gradient_cosine_enabled=True,
        clip_range=0.2,
    )
    runner.realized_environment_steps = 8192
    runner.note_ppo_minibatch(batch)
    runner.note_ppo_minibatch(batch)
    telemetry = runner.telemetry()
    assert telemetry["exp2_cell_count_z0_A"] == 16
    assert telemetry["exp2_cell_count_z0_B"] == 0
    assert telemetry["exp2_cell_count_z1_A"] == 0
    assert telemetry["exp2_cell_count_z1_B"] == 16
    assert telemetry["exp2_cell_steps_z0_A"] == 4096
    assert telemetry["exp2_cell_steps_z1_B"] == 4096
    assert telemetry["exp2b_gradient_cosine_count"] == 1
