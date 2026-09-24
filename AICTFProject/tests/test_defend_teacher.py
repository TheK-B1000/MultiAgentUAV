"""Executable contracts for DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC.

C2 (fixed_for_episode role hold), C4 (teacher loss gating), C5 (lambda
schedule), C8 (mutual exclusion / isolation), and C9 (structurally absent at
lambda<=0, and teacher-free at a mid-schedule resolved lambda of exactly
0.0) are pure/unit contracts here. C3 (physics-port parity against the
sealed N' controller) is a heavier gpu_env-backed contract in its own
section at the bottom.

C6/C7 (real pi_A checkpoint warm-start t0-equivalence, fresh optimizer
identity) are exercised by
experiments/probe_defend_teacher_role_warmstart_contract.py against the real
checkpoint; the synthetic-model version of the actor-expansion mechanism
itself (zero-init role columns preserve pretrained logits for any r, fresh
Adam survives the width change) is already pinned generically by
tests/test_rule_role_conditioning.py and is not duplicated here, since
DEFEND-teacher reuses that exact mechanism unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl.custom_ppo.defend_teacher import (
    DefendTeacherRunner,
    defend_teacher_loss,
    masked_macro_and_waypoint_logits,
)
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.rule_role_assignment import RoleHoldState
from rl.custom_ppo.schedules import resolve_defend_teacher_lambda

N_CH, ROWS, COLS, VEC_DIM = 7, 20, 20, 20


def _spaces(n=4):
    obs = spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(n, N_CH, ROWS, COLS), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(n, VEC_DIM), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, shape=(n * (5 + 50),), dtype=np.float32),
    })
    act = spaces.MultiDiscrete([5, 50] * n)
    return obs, act


def _rand_obs(n=4, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "grid": torch.tensor(rng.random((batch, n, N_CH, ROWS, COLS)), dtype=torch.float32),
        "vec": torch.tensor(rng.uniform(-1, 1, (batch, n, VEC_DIM)), dtype=torch.float32),
        "agent_mask": torch.ones(batch, n),
        "mask": torch.ones(batch, n * 55),
    }


def _lock_agent(mask: torch.Tensor, n: int, agent_idx: int, *, macro: int = 1, waypoint: int = 3) -> torch.Tensor:
    """Set one agent's mask row to a committed one-hot (macro, waypoint) pair."""
    batch = mask.shape[0]
    m3 = mask.clone().view(batch, n, 55)
    m3[:, agent_idx, :] = 0.0
    m3[:, agent_idx, macro] = 1.0
    m3[:, agent_idx, 5 + waypoint] = 1.0
    return m3.view(batch, n * 55)


def _home(B=1, x=0.0, y=0.0):
    return torch.tensor([[x, y]] * B, dtype=torch.float32)


# ---------------------------------------------------------------------------
# C2: role_fixed_for_episode
# ---------------------------------------------------------------------------


def test_fixed_for_episode_ignores_death():
    hold = RoleHoldState(1, 4, hold_ticks=8, fixed_for_episode=True, device="cpu")
    home = _home(1)
    pos_x = torch.tensor([[0.0, 1.0, 5.0, 6.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    r0 = hold.update(pos_x, pos_y, home, alive)
    alive2 = torch.tensor([[False, True, True, True]])
    r1 = hold.update(pos_x, pos_y, home, alive2)
    assert torch.equal(r1, r0), "fixed_for_episode must not reassign on a synthetic mid-episode death"


def test_fixed_for_episode_ignores_force_kwarg():
    """The collector calls update(force=True) at the top of every collect() cycle,
    not only at genuine episode boundaries -- force must not trigger reassignment."""
    hold = RoleHoldState(1, 4, hold_ticks=8, fixed_for_episode=True, device="cpu")
    home = _home(1)
    pos_x = torch.tensor([[0.0, 1.0, 5.0, 6.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    r0 = hold.update(pos_x, pos_y, home, alive, force=True)
    pos_x_flip = torch.tensor([[6.0, 5.0, 1.0, 0.0]])
    r1 = hold.update(pos_x_flip, pos_y, home, alive, force=True)
    assert torch.equal(r1, r0), "force=True must not reassign under fixed_for_episode"


def test_fixed_for_episode_reset_envs_triggers_exactly_one_reassignment():
    hold = RoleHoldState(1, 4, hold_ticks=8, fixed_for_episode=True, device="cpu")
    home = _home(1)
    pos_x = torch.tensor([[0.0, 1.0, 5.0, 6.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    r0 = hold.update(pos_x, pos_y, home, alive)

    hold.reset_envs(torch.tensor([True]))
    pos_x_flip = torch.tensor([[6.0, 5.0, 1.0, 0.0]])
    r1 = hold.update(pos_x_flip, pos_y, home, alive)
    assert not torch.equal(r1, r0), "reset_envs must trigger exactly one reassignment on the next update()"

    pos_x_flip2 = torch.tensor([[0.0, 6.0, 1.0, 5.0]])
    r2 = hold.update(pos_x_flip2, pos_y, home, alive)
    assert torch.equal(r2, r1), "reassignment must not repeat until the next reset_envs"


def test_fixed_for_episode_role_hold_ticks_is_inert():
    home = _home(1)
    pos_x = torch.tensor([[0.0, 1.0, 5.0, 6.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    for hold_ticks in (1, 8, 1000):
        hold = RoleHoldState(1, 4, hold_ticks=hold_ticks, fixed_for_episode=True, device="cpu")
        r0 = hold.update(pos_x, pos_y, home, alive)
        for _ in range(20):
            r = hold.update(pos_x, pos_y, home, alive)
            assert torch.equal(r, r0)


def test_fixed_for_episode_per_env_reset_only_affects_that_env():
    hold = RoleHoldState(2, 4, hold_ticks=8, fixed_for_episode=True, device="cpu")
    home = _home(2)
    pos_x = torch.tensor([[0.0, 1.0, 5.0, 6.0], [0.0, 1.0, 5.0, 6.0]])
    pos_y = torch.zeros(2, 4)
    alive = torch.ones(2, 4, dtype=torch.bool)
    r0 = hold.update(pos_x, pos_y, home, alive)

    hold.reset_envs(torch.tensor([True, False]))
    pos_x_flip = torch.tensor([[6.0, 5.0, 1.0, 0.0], [6.0, 5.0, 1.0, 0.0]])
    r1 = hold.update(pos_x_flip, pos_y, home, alive)
    assert not torch.equal(r1[0], r0[0]), "env 0 was reset and must reassign"
    assert torch.equal(r1[1], r0[1]), "env 1 was not reset and must not reassign"


# ---------------------------------------------------------------------------
# C4: teacher loss gating (role==DEFEND, decision-eligible, alive)
# ---------------------------------------------------------------------------


def test_masked_macro_and_waypoint_logits_shapes():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=True,
    )
    obs = _rand_obs()
    obs["roles"] = torch.zeros(2, 4)
    macro, waypoint = masked_macro_and_waypoint_logits(m, obs)
    assert tuple(macro.shape) == (2, 4, 5)
    assert tuple(waypoint.shape) == (2, 4, 50)


def test_defend_teacher_loss_gates_by_role_decision_and_alive():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=True,
    )
    n, batch = 4, 2
    obs = _rand_obs(n=n, batch=batch)
    # agents 0,1 DEFEND; 2,3 ATTACK.
    obs["roles"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * batch)
    # Agent 1 (DEFEND) is locked mid-commit -> not decision-eligible -> excluded.
    obs["mask"] = _lock_agent(obs["mask"], n, agent_idx=1)
    # Agent 2 (ATTACK) is dead -> exercises alive-masking, but role already excludes it.
    agent_mask = torch.ones(batch, n)
    agent_mask[:, 2] = 0.0
    obs["agent_mask"] = agent_mask

    waypoint_target = torch.randint(0, 50, (batch, n))
    loss, tel = defend_teacher_loss(m, obs, waypoint_target)
    assert int(tel["n_gated"]) == 1 * batch, "only agent 0 (DEFEND, eligible, alive) should be gated"
    assert loss.requires_grad
    assert float(loss.detach()) > 0.0


def test_defend_teacher_loss_empty_gate_returns_zero_no_grad_dependency():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=True,
    )
    n, batch = 4, 2
    obs = _rand_obs(n=n, batch=batch)
    obs["roles"] = torch.ones(batch, n)  # everyone ATTACK -> gate always empty
    waypoint_target = torch.randint(0, 50, (batch, n))
    loss, tel = defend_teacher_loss(m, obs, waypoint_target)
    assert int(tel["n_gated"]) == 0
    assert float(loss.detach()) == 0.0


def test_defend_teacher_loss_requires_roles_key():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=True,
    )
    obs = _rand_obs()
    with pytest.raises(KeyError):
        defend_teacher_loss(m, obs, torch.zeros(2, 4, dtype=torch.long))


# ---------------------------------------------------------------------------
# C5: lambda schedule (linear_anneal boundary values from LAMBDA_SCHEDULE_locked)
# ---------------------------------------------------------------------------


def _schedule_cfg(peak=0.1, end=0.0, start=50_000, stop=150_000):
    return SimpleNamespace(
        defend_teacher_lambda=peak,
        defend_teacher_lambda_end=end,
        defend_teacher_decay_start_step=start,
        defend_teacher_decay_end_step=stop,
    )


def test_resolve_defend_teacher_lambda_boundary_values():
    cfg = _schedule_cfg()
    cases = {0: 0.1, 49_999: 0.1, 50_000: 0.1, 100_000: 0.05, 150_000: 0.0, 200_000: 0.0}
    for step, expected in cases.items():
        got = resolve_defend_teacher_lambda(cfg, global_step=step)
        assert abs(got - expected) < 1e-9, (step, got, expected)
    v = resolve_defend_teacher_lambda(cfg, global_step=149_999)
    assert 0.0 < v < 1e-3, "t=149999 must be strictly positive and near zero"


def test_resolve_defend_teacher_lambda_disabled_returns_zero():
    cfg = _schedule_cfg(peak=0.0)
    assert resolve_defend_teacher_lambda(cfg, global_step=0) == 0.0
    assert resolve_defend_teacher_lambda(cfg, global_step=100_000) == 0.0


def test_resolve_defend_teacher_lambda_monotonic_nonincreasing_on_decay_window():
    cfg = _schedule_cfg()
    steps = list(range(50_000, 150_001, 5_000))
    vals = [resolve_defend_teacher_lambda(cfg, global_step=s) for s in steps]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_resolve_defend_teacher_lambda_constant_elsewhere():
    cfg = _schedule_cfg()
    for s in range(0, 50_000, 10_000):
        assert resolve_defend_teacher_lambda(cfg, global_step=s) == 0.1
    for s in range(150_000, 200_001, 10_000):
        assert resolve_defend_teacher_lambda(cfg, global_step=s) == 0.0


# ---------------------------------------------------------------------------
# C9: structurally absent (lambda<=0 -> no runner; resolved lambda==0 mid-run -> no step)
# ---------------------------------------------------------------------------


def _teacher_batch(n=4, batch=2, seed=0, roles=None):
    obs = _rand_obs(n=n, batch=batch, seed=seed)
    if roles is None:
        roles = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * batch) if n == 4 else torch.zeros(batch, n)
    return {
        "obs_grid": obs["grid"],
        "obs_vec": obs["vec"],
        "obs_agent_mask": obs["agent_mask"],
        "obs_mask": obs["mask"],
        "obs_roles": roles,
        "obs_defend_teacher_waypoint": torch.randint(0, 50, (batch, n)),
    }


def _teacher_model_and_optimizer():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=True,
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    return m, opt


def test_runner_rejects_nonpositive_lambda():
    m, opt = _teacher_model_and_optimizer()
    with pytest.raises(ValueError, match="lambda_teacher <= 0"):
        DefendTeacherRunner(m, opt, lambda_teacher=0.0)
    with pytest.raises(ValueError, match="lambda_teacher <= 0"):
        DefendTeacherRunner(m, opt, lambda_teacher=-0.1)


def test_runner_requires_role_conditioning():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=False,
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="role_conditioning_enabled"):
        DefendTeacherRunner(m, opt, lambda_teacher=0.1)


def test_runner_zero_lambda_mid_schedule_skips_backward_and_step():
    m, opt = _teacher_model_and_optimizer()
    runner = DefendTeacherRunner(m, opt, lambda_teacher=0.1, cadence=1)
    runner.lambda_teacher = 0.0  # simulate the [150000,200000] consolidation phase
    params_before = [p.detach().clone() for p in m.parameters()]
    batch = _teacher_batch()
    fired = runner.note_ppo_minibatch(batch)
    assert fired is False
    assert runner.n_teacher_updates == 1
    assert runner.n_skipped_zero_lambda == 1
    assert runner.n_updates == 0
    for p_before, p_after in zip(params_before, m.parameters()):
        assert torch.equal(p_before, p_after), "zero-lambda consolidation phase must not touch params"


def test_runner_cadence_gating_and_nonzero_lambda_updates_params():
    m, opt = _teacher_model_and_optimizer()
    runner = DefendTeacherRunner(m, opt, lambda_teacher=0.1, cadence=4)
    batch = _teacher_batch()
    fired = [runner.note_ppo_minibatch(batch) for _ in range(3)]
    assert fired == [False, False, False]
    assert runner.n_ppo_actor_minibatches == 3
    assert runner.n_teacher_updates == 0

    params_before = [p.detach().clone() for p in m.parameters()]
    fired_4th = runner.note_ppo_minibatch(batch)
    assert runner.n_ppo_actor_minibatches == 4
    assert runner.n_teacher_updates == 1
    if fired_4th:
        assert runner.n_updates == 1
        changed = any(
            not torch.equal(a, b) for a, b in zip(params_before, m.parameters())
        )
        assert changed, "a fired teacher update with lambda>0 and a nonempty gate must change params"


def test_runner_note_ppo_minibatch_requires_batch_on_cadence_tick():
    m, opt = _teacher_model_and_optimizer()
    runner = DefendTeacherRunner(m, opt, lambda_teacher=0.1, cadence=1)
    with pytest.raises(RuntimeError, match="requires the PPO minibatch"):
        runner.note_ppo_minibatch(None)


# ---------------------------------------------------------------------------
# C8: mutual exclusion / isolation
# ---------------------------------------------------------------------------


def test_isolation_defend_teacher_absent_by_default():
    from rl.config.ppo_config import PPOConfig

    cfg = PPOConfig()
    assert float(cfg.defend_teacher_lambda) == 0.0
    assert bool(cfg.role_fixed_for_episode) is False


def test_isolation_role_fixed_for_episode_does_not_enable_teacher():
    from rl.config.ppo_config import PPOConfig

    cfg = PPOConfig()
    cfg.role_conditioning_enabled = True
    cfg.role_fixed_for_episode = True
    assert float(cfg.defend_teacher_lambda) == 0.0


def test_maybe_attach_defend_teacher_noop_when_disabled():
    from rl.training.orchestrator import _maybe_attach_defend_teacher

    cfg = SimpleNamespace(defend_teacher_lambda=0.0)
    trainer = SimpleNamespace()
    _maybe_attach_defend_teacher(cfg, trainer)
    assert not hasattr(trainer, "defend_teacher_runner")


def test_maybe_attach_defend_teacher_requires_role_conditioning():
    from rl.training.orchestrator import _maybe_attach_defend_teacher

    cfg = SimpleNamespace(defend_teacher_lambda=0.1, role_conditioning_enabled=False)
    trainer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="role_conditioning_enabled"):
        _maybe_attach_defend_teacher(cfg, trainer)


@pytest.mark.parametrize(
    "attr", ["sappo_anchor_runner", "exp2_teacher_compression_runner",
             "sibling_sep_runner", "role_pres_runner", "getflag_preserve_runner"],
)
def test_maybe_attach_defend_teacher_mutual_exclusion(attr):
    from rl.training.orchestrator import _maybe_attach_defend_teacher

    cfg = SimpleNamespace(defend_teacher_lambda=0.1, role_conditioning_enabled=True)
    trainer = SimpleNamespace(**{attr: object()})
    with pytest.raises(RuntimeError):
        _maybe_attach_defend_teacher(cfg, trainer)


def test_maybe_attach_getflag_preservation_rejects_defend_teacher_lambda():
    from rl.training.orchestrator import _maybe_attach_getflag_preservation

    cfg = SimpleNamespace(
        getflag_preserve_lambda=0.1, getflag_preserve_ckpt="x",
        sibling_sep_lambda=0.0, role_pres_lambda=0.0, defend_teacher_lambda=0.1,
    )
    trainer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="DEFEND-teacher"):
        _maybe_attach_getflag_preservation(cfg, trainer)


def test_maybe_attach_sibling_separation_rejects_defend_teacher_lambda():
    from rl.training.orchestrator import _maybe_attach_sibling_separation

    cfg = SimpleNamespace(
        sibling_sep_lambda=0.1, sibling_sep_ckpt="x", sibling_sep_dataset="y",
        defend_teacher_lambda=0.1,
    )
    trainer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="DEFEND-teacher"):
        _maybe_attach_sibling_separation(cfg, trainer)


def test_maybe_attach_role_preservation_rejects_defend_teacher_lambda():
    from rl.training.orchestrator import _maybe_attach_role_preservation

    cfg = SimpleNamespace(
        role_pres_lambda=0.1, role_pres_targets="x", role_pres_style="GUARD",
        sibling_sep_lambda=0.0, getflag_preserve_lambda=0.0, defend_teacher_lambda=0.1,
    )
    trainer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="DEFEND-teacher"):
        _maybe_attach_role_preservation(cfg, trainer)


# ---------------------------------------------------------------------------
# C3: teacher parity against the sealed N' controller (physics port)
# ---------------------------------------------------------------------------


def _packed_bits(core, agent_idx: int) -> int:
    nM, nT = int(core.cfg.n_macros), int(core.cfg.n_targets)
    mask = core._build_action_mask(side="blue").view(1, -1, nM + nT)
    row = (mask[0, agent_idx].detach().cpu().numpy() > 0)
    return int(sum(int(row[i]) << i for i in range(len(row))))


def _row_for_agent(A, core, agent_idx: int) -> "np.ndarray":
    row = np.zeros(len(A.STATE_FIELDS))
    row[A.IX["x"]] = float(core.blue_x[0, agent_idx])
    row[A.IX["y"]] = float(core.blue_y[0, agent_idx])
    row[A.IX["h"]] = float(core.blue_heading[0, agent_idx])
    row[A.IX["v"]] = float(core.blue_speed[0, agent_idx])
    row[A.IX["Fx"]] = float(core.blue_flag_pos[0, 0])
    row[A.IX["Fy"]] = float(core.blue_flag_pos[0, 1])
    return row


def test_c3_teacher_parity_against_sealed_n_prime_controller():
    """Bit-identical: the batched training-time port must reproduce N''s own
    sealed controller_select on the same state, not merely approximate it."""
    A = pytest.importorskip("experiments.audit_scaffold_to_native_representability_4v4")
    Nmod = pytest.importorskip("experiments.run_goto_only_defend_substitution_4v4")
    from rl.custom_ppo.defend_teacher import compute_defend_teacher_waypoints

    eng = A.Engine()
    core = eng.core
    n_agents = int(core.blue_x.shape[1])

    def check_all_agents(label: str) -> None:
        batched = compute_defend_teacher_waypoints(core)
        for agent_idx in range(n_agents):
            bits = _packed_bits(core, agent_idx)
            row = _row_for_agent(A, core, agent_idx)
            ref_w = Nmod.controller_select(eng, row, bits)
            got_w = int(batched[0, agent_idx].item())
            assert got_w == ref_w, f"{label} agent={agent_idx}: got {got_w}, sealed controller says {ref_w}"

    check_all_agents("reset")

    # A few stepped ticks with GO_TO actions for state diversity.
    for step_i in range(3):
        nT = int(core.cfg.n_targets)
        actions = np.zeros((1, n_agents * 2), dtype=np.int64)
        for a in range(n_agents):
            actions[0, 2 * a] = 0  # GO_TO
            actions[0, 2 * a + 1] = (step_i * 7 + a * 3) % nT
        eng.env.step_async(actions)
        eng.env.step_wait()
        check_all_agents(f"step{step_i}")

    # Synthetic boundary states for agent 0: at the flag (degenerate), just
    # inside / just outside the tag radius, and far away.
    from gpu_env._core._rules import _pyquaticus_defender_radius_cells

    radius = _pyquaticus_defender_radius_cells(float(core.cfg.tag_range_cells))
    fx0, fy0 = float(core.blue_flag_pos[0, 0]), float(core.blue_flag_pos[0, 1])
    cols, rows = float(core.cols - 1), float(core.rows - 1)
    for i, (x, y, h) in enumerate([
        (fx0 + 0.01, fy0 + 0.01, 0.0),
        (fx0 + radius * 0.5, fy0, 0.0),
        (fx0 + radius * 3.0, fy0, 2.5),
        (min(cols, fx0 + radius * 5.0), min(rows, fy0 + radius * 5.0), 0.7),
    ]):
        with torch.no_grad():
            core.blue_x[0, 0] = float(np.clip(x, 0.0, cols))
            core.blue_y[0, 0] = float(np.clip(y, 0.0, rows))
            core.blue_heading[0, 0] = float(h)
            core.blue_speed[0, 0] = 0.3
        check_all_agents(f"synth{i}")


def test_c3_teacher_port_is_deterministic():
    A = pytest.importorskip("experiments.audit_scaffold_to_native_representability_4v4")
    from rl.custom_ppo.defend_teacher import compute_defend_teacher_waypoints

    eng = A.Engine()
    core = eng.core
    a = compute_defend_teacher_waypoints(core)
    b = compute_defend_teacher_waypoints(core)
    assert torch.equal(a, b)
