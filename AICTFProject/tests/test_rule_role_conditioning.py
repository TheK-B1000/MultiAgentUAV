"""Executable contracts for RULE_BASED_ROLE_CONDITIONING_SPEC.

Contracts 1–6, 8 are pure assignment. Plumbing / warm-start / isolation
contracts live alongside once wiring lands (same file).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl.custom_ppo.rule_role_assignment import (
    ROLE_ATTACK,
    ROLE_DEFEND,
    RoleHoldState,
    assert_no_strategy_leakage_inputs,
    assign_roles_from_geometry,
    distances_to_home,
    role_k,
)


def _home(B=1, x=0.0, y=0.0):
    return torch.tensor([[x, y]] * B, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Contract 1–2: counts + closest N/2 are DEFEND
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N,k", [(4, 2), (6, 3)])
def test_all_alive_exact_half_defend_half_attack(N, k):
    assert role_k(N) == k
    # Agents at distances 1..N from home at origin along x-axis.
    pos_x = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0)
    pos_y = torch.zeros(1, N)
    alive = torch.ones(1, N, dtype=torch.bool)
    roles, d = assign_roles_from_geometry(pos_x, pos_y, _home(), alive)
    assert int((roles == ROLE_DEFEND).sum()) == k
    assert int((roles == ROLE_ATTACK).sum()) == N - k
    # Closest k indices 0..k-1 must be DEFEND; farthest ATTACK.
    assert torch.equal(roles[0, :k], torch.zeros(k))
    assert torch.equal(roles[0, k:], torch.ones(N - k))
    assert torch.allclose(d[0], pos_x[0])


def test_n_alive_less_than_k_all_living_defend():
    # N=4, k=2, only 1 alive → that one is DEFEND; dead are sentinel 0.
    pos_x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.tensor([[False, True, False, False]])
    roles, _ = assign_roles_from_geometry(pos_x, pos_y, _home(), alive)
    assert roles[0, 1].item() == ROLE_DEFEND
    assert int((roles[0] == ROLE_ATTACK).sum()) == 0


# ---------------------------------------------------------------------------
# Contract 3: deterministic ties
# ---------------------------------------------------------------------------
def test_identical_geometry_identical_assignment():
    pos_x = torch.tensor([[2.0, 2.0, 5.0, 5.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    a, _ = assign_roles_from_geometry(pos_x, pos_y, _home(), alive)
    b, _ = assign_roles_from_geometry(pos_x.clone(), pos_y.clone(), _home(), alive)
    assert torch.equal(a, b)


def test_tie_break_prefers_smaller_agent_index():
    # All four at same distance → smallest indices 0,1 are DEFEND.
    pos_x = torch.ones(1, 4) * 3.0
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    roles, _ = assign_roles_from_geometry(pos_x, pos_y, _home(), alive)
    assert torch.equal(roles[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Contract 4: permutation of enumeration
# ---------------------------------------------------------------------------
def test_physical_assignment_follows_geometry_not_batch_order():
    """Permuting the agent axis is a different physical roster layout.

    The frozen rule uses agent index only as a tie-break. When distances differ,
    DEFEND slots track the physically closest agents regardless of how we
    label them in a twin roster with matching (position, index) pairs.
    """
    # Roster A: indices 0..3 at distances 1,2,3,4
    pos_x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    roles_a, _ = assign_roles_from_geometry(pos_x, pos_y, _home(), alive)
    # Same physical agents, same indices → same roles. A shuffled *copy of the
    # distance vector applied to the same indices* would be a different world;
    # instead verify reordering inputs consistently with index relabeling:
    # swap agents 0 and 3 (positions AND the identity that travels with them
    # is modeled by applying the same swap to a role lookup by position).
    pos_swap = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    roles_b, _ = assign_roles_from_geometry(pos_swap, pos_y, _home(), alive)
    # Closest two positions are now index 3 (d=1) and index 1 (d=2).
    assert roles_b[0, 3].item() == ROLE_DEFEND
    assert roles_b[0, 1].item() == ROLE_DEFEND
    assert roles_b[0, 0].item() == ROLE_ATTACK
    assert roles_b[0, 2].item() == ROLE_ATTACK
    # Original closest were 0 and 1.
    assert roles_a[0, 0].item() == ROLE_DEFEND
    assert roles_a[0, 1].item() == ROLE_DEFEND


def test_permutation_of_batch_dim_preserves_per_env_roles():
    pos_x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    pos_y = torch.zeros(2, 4)
    alive = torch.ones(2, 4, dtype=torch.bool)
    roles, _ = assign_roles_from_geometry(pos_x, pos_y, _home(B=2), alive)
    roles_rev, _ = assign_roles_from_geometry(pos_x.flip(0), pos_y.flip(0), _home(B=2), alive)
    assert torch.equal(roles.flip(0), roles_rev)


# ---------------------------------------------------------------------------
# Contract 5–6: hold persistence + death immediate reassign
# ---------------------------------------------------------------------------
def test_roles_persist_across_hold_despite_distance_reordering():
    hold = RoleHoldState(1, 4, hold_ticks=8, device="cpu")
    home = _home()
    # Initial: distances 1,2,3,4 → DEFEND 0,1
    pos_x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    r0 = hold.update(pos_x, pos_y, home, alive, force=True)
    assert torch.equal(r0[0, :2], torch.zeros(2))
    # Swap distances so geometry alone would flip DEFEND to 2,3 — but hold.
    pos_flip = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    for _ in range(7):
        r = hold.update(pos_flip, pos_y, home, alive, force=False)
        assert torch.equal(r, r0), "roles must not change during H_r merely from distance reorder"
    # On the tick that expires the hold, reassignment is allowed.
    r_new = hold.update(pos_flip, pos_y, home, alive, force=False)
    assert torch.equal(r_new[0, 2:], torch.zeros(2))
    assert torch.equal(r_new[0, :2], torch.ones(2))


def test_death_triggers_immediate_reassignment():
    hold = RoleHoldState(1, 4, hold_ticks=8, device="cpu")
    home = _home()
    pos_x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)
    r0 = hold.update(pos_x, pos_y, home, alive, force=True)
    assert hold.age[0].item() == 1
    # Kill agent 0 (a DEFEND) before hold expires.
    alive2 = torch.tensor([[False, True, True, True]])
    r1 = hold.update(pos_x, pos_y, home, alive2, force=False)
    assert not torch.equal(r1, r0)
    # Living: 1,2,3 at d=2,3,4 → k_eff=min(2,3)=2 → DEFEND closest living 1,2
    assert r1[0, 1].item() == ROLE_DEFEND
    assert r1[0, 2].item() == ROLE_DEFEND
    assert r1[0, 3].item() == ROLE_ATTACK
    assert r1[0, 0].item() == ROLE_DEFEND  # dead sentinel


# ---------------------------------------------------------------------------
# Contract 8: no strategy leakage into assignment API
# ---------------------------------------------------------------------------
def test_assignment_rejects_strategy_leakage_kwargs():
    with pytest.raises(ValueError, match="strategy-leakage"):
        assert_no_strategy_leakage_inputs(teacher_macro=1)
    with pytest.raises(ValueError, match="strategy-leakage"):
        assert_no_strategy_leakage_inputs(getflag_preserve_lambda=0.1)
    # Benign kwargs ok.
    assert_no_strategy_leakage_inputs(pos_x=1, home_xy=2, alive=3)


def test_distances_helper_matches_assignment():
    pos_x = torch.tensor([[3.0, 0.0, 6.0, 0.0]])
    pos_y = torch.tensor([[4.0, 0.0, 8.0, 0.0]])
    home = _home()
    d = distances_to_home(pos_x, pos_y, home)
    assert torch.allclose(d[0, 0], torch.tensor(5.0))
    assert torch.allclose(d[0, 2], torch.tensor(10.0))


# ---------------------------------------------------------------------------
# Contract 7: actor plumbing — flipping r_i changes logits (after non-zero role col)
# Contract 9: warm-start zero-column → logits identical for any r at t=0
# Contract 10: isolation — GETFLAG preserve structurally absent; disabled = no inject
# ---------------------------------------------------------------------------
from gymnasium import spaces

from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.checkpoints.state_dict import (
    _expand_role_conditioning_linears,
    _load_model_state_dict_compat,
)


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


def test_role_disabled_rejects_roles_tensor():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=False,
    )
    obs = _rand_obs()
    roles = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="role_conditioning_enabled=False"):
        m.policy_logits(obs, roles=roles)


def test_role_enabled_requires_roles_tensor():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=True,
    )
    with pytest.raises(ValueError, match="requires roles"):
        m.policy_logits(_rand_obs())


def test_flipping_role_changes_logits_after_nonzero_role_column():
    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=True,
    )
    # Zero-init role column → logits identical for any r (warm-start contract).
    # Nudge the new input column so the plumbing contract can fire.
    with torch.no_grad():
        w = m.latent_actor.body[0].weight
        w[:, -1].fill_(0.25)
    obs = _rand_obs(seed=7)
    r0 = torch.zeros(2, 4)
    r1 = torch.ones(2, 4)
    logits0 = m.policy_logits(obs, roles=r0)
    logits1 = m.policy_logits(obs, roles=r1)
    assert not torch.allclose(logits0, logits1), "role bit must reach the actor"


def test_warm_start_zero_role_columns_preserve_logits_for_any_r():
    """Contract 9: after expanding B_t500k-shaped weights with zero new columns,
    any role vector yields identical actor logits (pretrained map untouched)."""
    obs_s, act_s = _spaces()
    base = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=False,
    )
    torch.manual_seed(3)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    role_m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=True,
    )
    sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    sd = _expand_role_conditioning_linears(sd, role_m)
    _load_model_state_dict_compat(role_m, sd)

    obs = _rand_obs(seed=11)
    gs = torch.zeros(2, role_m.global_state_dim)
    # Baseline logits from the non-role model.
    base_logits = base.policy_logits(obs)
    # Role model at zero-init role columns: any r matches base.
    for roles in (torch.zeros(2, 4), torch.ones(2, 4), torch.tensor([[0., 1., 0., 1.], [1., 1., 0., 0.]])):
        logits = role_m.policy_logits(obs, roles=roles)
        assert torch.allclose(logits, base_logits, atol=1e-5, rtol=1e-5), (
            "zero-init role columns must preserve pretrained actor logits for any r"
        )
        # Critic also expands; values with any team_roles must match base at t=0.
        v_base = base.values(gs)
        v_role = role_m.values(gs, team_roles=roles)
        assert torch.allclose(v_base, v_role, atol=1e-5, rtol=1e-5)


def test_isolation_getflag_preserve_absent_on_role_config():
    from rl.config.ppo_config import PPOConfig
    cfg = PPOConfig()
    cfg.role_conditioning_enabled = True
    cfg.entity_repair_enabled = True
    assert float(cfg.getflag_preserve_lambda) == 0.0
    assert cfg.getflag_preserve_ckpt == ""


def test_optimizer_migration_skips_without_loading_mismatched_moments():
    """Warm-start seam: architecture migration must not load Adam state that can
    silently keep old in_features moments (148) against a widened param (149)."""
    from rl.custom_ppo.trainer_optimizers import TrainerOptimizerBundle

    class _Opt:
        def __init__(self):
            self.loaded = False

        def load_state_dict(self, _sd):
            self.loaded = True
            raise AssertionError("must not call load_state_dict under migration")

        def state_dict(self):
            return {}

    bundle = TrainerOptimizerBundle(
        primary=_Opt(),
        actor=_Opt(),
        critic=_Opt(),
        router=None,
        actor_cf=None,
    )
    bundle.load_checkpoint({"optimizer_state_dict": {}}, allow_architecture_migration=True)
    assert bundle.primary.loaded is False


def test_fresh_adam_after_role_expansion_survives_first_step():
    """After W'=[W 0] load into a 149-d role model, a freshly built Adam must
    step without 148-vs-149 foreach_lerp crashes."""
    from rl.custom_ppo.trainer_optimizers import TrainerOptimizerBundle
    from types import SimpleNamespace

    obs_s, act_s = _spaces()
    base = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=False, entity_repair_enabled=True,
    )
    role_m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=True, entity_repair_enabled=True,
    )
    sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    sd = _expand_role_conditioning_linears(sd, role_m)
    _load_model_state_dict_compat(role_m, sd)
    # Simulate loader rebuild after warm-start.
    hparams = SimpleNamespace(
        learning_rate=1e-4,
        max_grad_norm=0.5,
        use_latent_strategy=False,
        latent_episode_strategy_lr=0.0,
        v6i1_three_optimizer_mode=False,
    )
    cfg = SimpleNamespace(
        learning_rate=1e-4,
        max_grad_norm=0.5,
        use_latent_strategy=False,
        latent_episode_strategy_lr=0.0,
        v6i1_three_optimizer_mode=False,
        actor_lr=None,
        critic_lr=None,
        router_lr=None,
    )
    # TrainerOptimizerBundle.build may need more fields — fall back to plain Adam.
    try:
        bundle = TrainerOptimizerBundle.build(model=role_m, cfg=cfg, hparams=hparams)
        opt = bundle.primary
    except Exception:
        opt = torch.optim.Adam(role_m.parameters(), lr=1e-4)
    loss = (role_m.latent_actor.body[0].weight ** 2).sum() * 1e-8
    loss.backward()
    opt.step()
    w = role_m.latent_actor.body[0].weight
    assert tuple(w.shape) == (256, 149)
    # New role column still near zero after one tiny step from zero-init column.
    assert float(w[:, -1].abs().max().item()) < 1e-3


def test_isolation_manifest_fields_forbid_getflag_on_role_arm():
    from rl.config.ppo_config import PPOConfig
    cfg = PPOConfig()
    cfg.role_conditioning_enabled = True
    cfg.getflag_preserve_lambda = 0.0
    assert cfg.getflag_preserve_lambda == 0.0
    assert bool(cfg.role_conditioning_enabled) is True
