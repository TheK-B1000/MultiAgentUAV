"""Rule 12 contract for the entity-residual repair: three anchors, run against
the REAL SharedActorCentralizedCritic, not a stand-in network.

A guarantee proven on a toy net does not transfer to production code unless
the production fusion point is exercised directly -- this is the check the PI
asked for after the earlier `blue_scripted` and `CarrierRestore` incidents.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl.custom_ppo.entity_residual import (ENTITY_FEATURES, EntityResidualEncoder,
                                           augmented_local_in)
from rl.custom_ppo.policy import SharedActorCentralizedCritic

N_AGENTS, N_CH, ROWS, COLS, VEC_DIM = 4, 7, 20, 20, 20
K_TEAM, K_ENEMY = N_AGENTS - 1, N_AGENTS


def _obs_space():
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(N_AGENTS, N_CH, ROWS, COLS), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(N_AGENTS, VEC_DIM), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(N_AGENTS,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, shape=(N_AGENTS * (5 + 50),), dtype=np.float32),
    })


def _act_space():
    return spaces.MultiDiscrete([5, 50] * N_AGENTS)


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return SharedActorCentralizedCritic(_obs_space(), _act_space(),
                                        strategy_encoder_enabled=False, latent_k=0)


@pytest.fixture
def entity_module(policy):
    torch.manual_seed(1)
    return EntityResidualEncoder(out_dim=policy._local_actor_in_dim)


def _rand_obs(batch=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "grid": torch.tensor(rng.random((batch, N_AGENTS, N_CH, ROWS, COLS)), dtype=torch.float32),
        "vec": torch.tensor(rng.uniform(-1, 1, (batch, N_AGENTS, VEC_DIM)), dtype=torch.float32),
        "agent_mask": torch.ones(batch, N_AGENTS),
    }


def _rand_entities(batch, K, seed, all_invalid=False):
    rng = np.random.default_rng(seed)
    ent = torch.tensor(rng.uniform(-5, 5, (batch * N_AGENTS, K, ENTITY_FEATURES)),
                       dtype=torch.float32)
    valid = (torch.zeros(batch * N_AGENTS, K, dtype=torch.bool) if all_invalid
            else torch.ones(batch * N_AGENTS, K, dtype=torch.bool))
    return ent, valid


# ------------------------------------------------------- anchor 1: EMPTY_IDENTITY --
def test_empty_identity_holds_at_random_untrained_weights(policy, entity_module):
    """g(empty, empty) == 0 EXACTLY, for random (untrained) weights -- not just
    at zero-init. Proves the bias-free construction, not the zero-init trick."""
    torch.manual_seed(2)                     # perturb weights away from init
    with torch.no_grad():
        for p in entity_module.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    obs = _rand_obs()
    tm, tm_v = _rand_entities(3, K_TEAM, 10, all_invalid=True)
    en, en_v = _rand_entities(3, K_ENEMY, 11, all_invalid=True)
    aug, base, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, entity_module)
    assert torch.equal(aug, base), "g(empty, empty) must be EXACTLY zero, any weights"


def test_empty_identity_survives_many_random_weight_draws(policy):
    """Same property, several independent weight draws -- not a lucky seed."""
    obs = _rand_obs()
    tm, tm_v = _rand_entities(3, K_TEAM, 20, all_invalid=True)
    en, en_v = _rand_entities(3, K_ENEMY, 21, all_invalid=True)
    for seed in range(5):
        torch.manual_seed(seed)
        mod = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)
        with torch.no_grad():
            for p in mod.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        aug, base, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, mod)
        assert torch.equal(aug, base), f"failed at weight-draw seed {seed}"


def test_zero_init_gives_warm_start_equivalence_with_real_entities(policy, entity_module):
    """The SEPARATE, stronger guarantee: at t=0 (fresh EntityResidualEncoder),
    g(T, E) = 0 even with REAL non-empty entities -- enables bit-identical
    warm start from a pre-repair checkpoint."""
    obs = _rand_obs()
    tm, tm_v = _rand_entities(3, K_TEAM, 30)     # real entities, not empty
    en, en_v = _rand_entities(3, K_ENEMY, 31)
    aug, base, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, entity_module)
    assert torch.equal(aug, base), "fresh module must contribute exactly 0 at t=0"


# --------------------------------------------------------- anchor 2: PERMUTATION --
def test_permutation_invariance_teammates_and_enemies(policy, entity_module):
    torch.manual_seed(3)
    with torch.no_grad():
        for p in entity_module.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    obs = _rand_obs()
    tm, tm_v = _rand_entities(3, K_TEAM, 40)
    en, en_v = _rand_entities(3, K_ENEMY, 41)

    g1 = entity_module(tm, tm_v, en, en_v)

    perm_t = torch.randperm(K_TEAM)
    perm_e = torch.randperm(K_ENEMY)
    g2 = entity_module(tm[:, perm_t], tm_v[:, perm_t], en[:, perm_e], en_v[:, perm_e])
    assert torch.allclose(g1, g2, atol=1e-6), "shuffling entity order changed g"


def test_permutation_invariance_with_partial_validity(policy, entity_module):
    """Permutation invariance must hold with a MIXED valid/invalid mask too,
    not only when every entity is valid."""
    torch.manual_seed(4)
    tm, tm_v = _rand_entities(2, K_TEAM, 50)
    tm_v[:, 0] = False                        # one invalid teammate slot
    en, en_v = _rand_entities(2, K_ENEMY, 51)
    en_v[:, -1] = False
    g1 = entity_module(tm, tm_v, en, en_v)
    perm_t = torch.randperm(K_TEAM)
    perm_e = torch.randperm(K_ENEMY)
    g2 = entity_module(tm[:, perm_t], tm_v[:, perm_t], en[:, perm_e], en_v[:, perm_e])
    assert torch.allclose(g1, g2, atol=1e-6)


# --------------------------------------------------------- anchor 3: NO_MUTATION --
def test_base_path_bit_identical_regardless_of_entity_content(policy, entity_module):
    """The CNN/vec path (local_in_base) must be UNCHANGED by entity content --
    entities are strictly additive, never a replacement or a side channel into
    the base encoder."""
    obs = _rand_obs()
    tm1, tm_v1 = _rand_entities(3, K_TEAM, 60)
    en1, en_v1 = _rand_entities(3, K_ENEMY, 61)
    tm2, tm_v2 = _rand_entities(3, K_TEAM, 999)     # completely different entities
    en2, en_v2 = _rand_entities(3, K_ENEMY, 998)

    _, base1, cnn1, mask1 = augmented_local_in(policy, obs, tm1, tm_v1, en1, en_v1, entity_module)
    _, base2, cnn2, mask2 = augmented_local_in(policy, obs, tm2, tm_v2, en2, en_v2, entity_module)
    assert torch.equal(base1, base2), "local_in_base changed with entity content"
    assert torch.equal(cnn1, cnn2)
    assert torch.equal(mask1, mask2)


def test_base_path_matches_unmodified_encode_local_obs(policy):
    """local_in_base from the shim must equal calling the REAL, untouched
    policy._encode_local_obs directly -- the shim reads it, never alters it."""
    obs = _rand_obs()
    direct, direct_cnn, direct_mask = policy._encode_local_obs(obs)
    mod = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)
    tm, tm_v = _rand_entities(3, K_TEAM, 70)
    en, en_v = _rand_entities(3, K_ENEMY, 71)
    _, base, cnn, mask = augmented_local_in(policy, obs, tm, tm_v, en, en_v, mod)
    assert torch.equal(base, direct)
    assert torch.equal(cnn, direct_cnn)
    assert torch.equal(mask, direct_mask)


# ------------------------------------------------- end-to-end: logits identity ----
def test_policy_logits_identical_at_warm_start_with_entities_present(policy):
    """The property that actually matters for deployment: with a fresh entity
    module, POLICY_LOGITS on the augmented pathway equal the base policy's
    logits, even with real entity content -- a literal 2v2/4v4 checkpoint can
    be warm-started with provably zero behaviour change at step 0."""
    obs = _rand_obs()
    mod = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)
    tm, tm_v = _rand_entities(3, K_TEAM, 80)
    en, en_v = _rand_entities(3, K_ENEMY, 81)

    base_logits = policy.policy_logits(obs)

    aug_local_in, _, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, mod)
    batch = int(obs["grid"].shape[0])
    aug_flat = policy.latent_actor(aug_local_in)
    aug_logits = aug_flat.reshape(batch, policy.n_agents * policy.per_agent_logits) \
                         .reshape(batch, policy.n_agents * policy.per_agent_logits)
    assert torch.allclose(aug_logits, base_logits, atol=1e-5)


def test_output_width_matches_local_actor_input(policy, entity_module):
    obs = _rand_obs()
    tm, tm_v = _rand_entities(3, K_TEAM, 90)
    en, en_v = _rand_entities(3, K_ENEMY, 91)
    aug, base, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, entity_module)
    assert aug.shape == base.shape == (3 * N_AGENTS, policy._local_actor_in_dim)


def test_no_bias_parameters_anywhere_in_entity_path(entity_module):
    """Structural guard against a future edit silently adding a bias term,
    which would break the empty-identity guarantee for ANY trained weights."""
    for name, p in entity_module.named_parameters():
        assert "bias" not in name, f"found a bias parameter: {name} -- breaks EMPTY_IDENTITY"
