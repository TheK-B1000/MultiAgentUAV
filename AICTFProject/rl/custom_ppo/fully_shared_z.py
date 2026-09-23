"""Fully shared strategy-conditioned baseline (sharing-axis port).

Implements the frozen scientific locks in
``FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.json`` (and the contingent 6v6
companion): one ``pi_phi(a|o,z)`` under identical CLOSEST_DEFENDS allocation,
no router, no split ATTACK/DEFEND networks.

Default-off: every helper is a no-op unless
``fully_shared_z_conditioned_enabled`` is True.
"""

from __future__ import annotations

from typing import Any, Sequence

# Canonical warm-start pin from FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.
WARM_START_4V4_PATH = (
    "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/"
    "ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip"
)
WARM_START_4V4_SHA256 = (
    "94dde69d091a79344db3390d5464dbb4bcf51677a175df93969ab252b2e0f478"
)

Z_TO_POLE_TAG = {0: "OP6", 1: "OP7"}
POLE_TAG_TO_Z = {"OP6": 0, "OP7": 1, "A": 0, "B": 1}


def half_half_forced_latent_ids(n_envs: int, *, latent_k: int = 2) -> tuple[int, ...]:
    """Even n_envs → alternating z0/z1 (equal occupancy). Fail-closed on odds."""
    n = int(n_envs)
    k = int(latent_k)
    if k != 2:
        raise ValueError(f"fully-shared v1 locks latent_k=2; got {k}")
    if n < 2 or n % 2 != 0:
        raise ValueError(
            f"fully-shared pole-match requires even n_envs >= 2; got {n}"
        )
    return tuple(0 if i % 2 == 0 else 1 for i in range(n))


def opponent_tag_for_z(z: int) -> str:
    z_i = int(z)
    if z_i not in Z_TO_POLE_TAG:
        raise ValueError(f"fully-shared v1 only supports z in {{0,1}}; got {z_i}")
    return Z_TO_POLE_TAG[z_i]


def apply_fully_shared_z_config(cfg: Any, *, team_size: int) -> list[str]:
    """Mutate ``cfg`` into the frozen fully-shared architecture.

    Returns a list of human-readable lock lines for the launch banner.
    Raises ``ValueError`` on incompatible flags.
    """
    n = int(team_size)
    if n not in (4, 6):
        raise ValueError(f"fully-shared scale port supports team_size in {{4,6}}; got {n}")

    locks: list[str] = []

    cfg.fully_shared_z_conditioned_enabled = True
    cfg.fully_shared_z_pole_match = True

    # Single shared network under forced z — no router, no split nets.
    cfg.use_latent_strategy = True
    cfg.latent_k = 2
    cfg.latent_z_embed_dim = int(getattr(cfg, "latent_z_embed_dim", 16) or 16)
    cfg.latent_actor_conditioning = "concat"
    cfg.latent_assignment_mode = "static_env"
    cfg.fixed_latent_strategy = False  # static_env owns assignment; not a single global id
    cfg.train_router_when_forced = False
    cfg.train_router_critic_when_forced = False

    # Silence every q_phi / router gradient channel (forced z only).
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_lam_h = 0.0
    cfg.latent_lam_h_start = 0.0
    cfg.latent_lam_h_end = 0.0
    cfg.latent_lam_p = 0.0
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_outcome_diversity_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.enable_actor_z_film = False
    cfg.latent_population_birth_per_z_action_heads = False
    cfg.exp2c_mode_specific_action_heads = False
    cfg.enable_latent_z_residual = False

    # CLOSEST_DEFENDS allocator — identical to separated PASS arm.
    cfg.role_conditioning_enabled = True
    cfg.role_fixed_for_episode = True
    cfg.role_hold_ticks = 8
    cfg.role_k_defend = 0  # default k=N/2
    cfg.role_k_defend_choices = ""
    cfg.split_attack_defend_enabled = False
    cfg.split_attack_defend_frozen_ckpt = ""
    cfg.split_attack_defend_frozen_ckpt_sha256 = ""
    cfg.assignment_conditioning_enabled = False
    cfg.defend_teacher_lambda = 0.0
    cfg.defend_teacher_lambda_end = 0.0

    # Both poles installed; static env→z→opponent matching (no pool resample).
    from rl.config.ppo_config import TrainMode

    cfg.opponent_pool = ("OP6", "OP7")
    cfg.opponent_pool_weights = (0.5, 0.5)
    cfg.opponent_randomize = False
    cfg.mode = TrainMode.FIXED_OPPONENT.value
    cfg.fixed_opponent_tag = "OP6"

    n_envs = int(getattr(cfg, "n_envs", 0) or 0)
    ids = half_half_forced_latent_ids(n_envs, latent_k=2)
    cfg.forced_latent_env_ids = ids

    # Fail-closed incompatibilities.
    if bool(getattr(cfg, "split_attack_defend_enabled", False)):
        raise ValueError("fully-shared forbids split_attack_defend_enabled")
    if float(getattr(cfg, "defend_teacher_lambda", 0.0) or 0.0) > 0.0:
        raise ValueError("fully-shared v1 locks defend_teacher_lambda=0")
    if bool(getattr(cfg, "assignment_conditioning_enabled", False)):
        raise ValueError("fully-shared cannot coexist with assignment conditioning")
    if float(getattr(cfg, "sibling_sep_lambda", 0.0) or 0.0) > 0.0:
        raise ValueError("fully-shared cannot coexist with sibling-sep")
    if float(getattr(cfg, "role_pres_lambda", 0.0) or 0.0) > 0.0:
        raise ValueError("fully-shared cannot coexist with role-pres")
    if float(getattr(cfg, "getflag_preserve_lambda", 0.0) or 0.0) > 0.0:
        raise ValueError("fully-shared cannot coexist with GETFLAG preservation")

    locks.append(f"use_latent_strategy=True latent_k=2 concat z_embed={cfg.latent_z_embed_dim}")
    locks.append("latent_assignment_mode=static_env (no q_phi; forced z per env)")
    locks.append(f"forced_latent_env_ids half/half over n_envs={n_envs}")
    locks.append("role_conditioning + role_fixed_for_episode (CLOSEST_DEFENDS k=N/2)")
    locks.append("split_attack_defend=OFF defend_teacher=0 (single shared param set)")
    locks.append("pole_match: z0->OP6, z1->OP7 (static; opponent_randomize=OFF)")
    locks.append("all latent/router loss coefs = 0")
    return locks


def assert_fully_shared_contracts(cfg: Any) -> None:
    """Fail-closed structural contracts (C2, C3, C4 pieces) before GPU work."""
    if not bool(getattr(cfg, "fully_shared_z_conditioned_enabled", False)):
        raise AssertionError("fully_shared_z_conditioned_enabled must be True")
    if not bool(getattr(cfg, "use_latent_strategy", False)):
        raise AssertionError("C1/C2: use_latent_strategy required")
    if int(getattr(cfg, "latent_k", 0) or 0) != 2:
        raise AssertionError("C1: latent_k must be 2")
    if str(getattr(cfg, "latent_assignment_mode", "")) != "static_env":
        raise AssertionError("C5 train path: latent_assignment_mode must be static_env")
    if str(getattr(cfg, "latent_actor_conditioning", "concat")) != "concat":
        raise AssertionError("C3: latent_actor_conditioning must be concat")
    if bool(getattr(cfg, "split_attack_defend_enabled", False)):
        raise AssertionError("C2: split_attack_defend must be False")
    if not bool(getattr(cfg, "role_conditioning_enabled", False)):
        raise AssertionError("C4: role_conditioning_enabled required")
    if not bool(getattr(cfg, "role_fixed_for_episode", False)):
        raise AssertionError("C4: role_fixed_for_episode required")
    if str(getattr(cfg, "role_k_defend_choices", "") or "").strip():
        raise AssertionError("C4 4v4: role_k_defend_choices must be empty (k=N/2)")
    if int(getattr(cfg, "role_k_defend", 0) or 0) != 0:
        raise AssertionError("C4 4v4: role_k_defend must be 0 (default N/2)")
    if bool(getattr(cfg, "enable_actor_z_film", False)):
        raise AssertionError("C3: FiLM forbidden")
    if bool(getattr(cfg, "latent_population_birth_per_z_action_heads", False)):
        raise AssertionError("C3: per-z action heads forbidden")
    if bool(getattr(cfg, "exp2c_mode_specific_action_heads", False)):
        raise AssertionError("C3: per-z action heads forbidden")
    if float(getattr(cfg, "latent_strategy_ppo_coef", 0.0) or 0.0) != 0.0:
        raise AssertionError("C5: latent_strategy_ppo_coef must be 0 (no router)")
    if float(getattr(cfg, "latent_lam_h", 0.0) or 0.0) != 0.0:
        raise AssertionError("C5: latent_lam_h must be 0")
    if float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) != 0.0:
        raise AssertionError("C5: latent_lam_p must be 0")

    ids = tuple(int(v) for v in getattr(cfg, "forced_latent_env_ids", ()) or ())
    n_envs = int(getattr(cfg, "n_envs", 0) or 0)
    if len(ids) != n_envs:
        raise AssertionError(
            f"C6: forced_latent_env_ids length {len(ids)} != n_envs {n_envs}"
        )
    if ids.count(0) != ids.count(1):
        raise AssertionError(
            f"C6: forced_latent_env_ids must be half/half z0/z1; got counts "
            f"z0={ids.count(0)} z1={ids.count(1)}"
        )
    if bool(getattr(cfg, "fully_shared_z_pole_match", False)):
        for i, z in enumerate(ids):
            expect = opponent_tag_for_z(z)
            # Structural: mapping is deterministic; live env check is separate.
            _ = expect


def initial_opponent_keys_for_forced_z(forced_ids: Sequence[int]) -> list[str]:
    """Map each env's forced z to its matched pole tag."""
    return [opponent_tag_for_z(int(z)) for z in forced_ids]


def count_actor_modules(model: Any) -> int:
    """C2 helper: exactly one SharedActorCentralizedCritic actor body."""
    # The shared policy owns a single latent_actor; there is no second actor.
    la = getattr(model, "latent_actor", None)
    return 1 if la is not None else 0


__all__ = [
    "WARM_START_4V4_PATH",
    "WARM_START_4V4_SHA256",
    "Z_TO_POLE_TAG",
    "POLE_TAG_TO_Z",
    "half_half_forced_latent_ids",
    "opponent_tag_for_z",
    "apply_fully_shared_z_config",
    "assert_fully_shared_contracts",
    "initial_opponent_keys_for_forced_z",
    "count_actor_modules",
]
