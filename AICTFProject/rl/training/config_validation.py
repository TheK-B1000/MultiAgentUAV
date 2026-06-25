"""Training config normalization and validation.

Extracted from :mod:`rl.train_ppo` so the trainer body, presets, and CLI
all share one source of truth for "what does it mean for this PPOConfig to
be valid?". No PPO math, no env construction; pure config rules.

Public API:

* :data:`EVAL_ONLY_TRAINING_OPPONENT_TAGS` -- scripted opponents stripped
  from training opponent pools unless explicitly allowed.
* :func:`_normalize_train_mode` -- map CLI / config aliases onto
  :class:`TrainMode` string values.
* :func:`_strip_eval_only_opponents_from_training_pool` -- in-place removal
  of eval-only opponents from ``cfg.opponent_pool``.
* :func:`normalize_and_validate_training_config` -- the canonical entry
  point used by ``train_ppo``: applies mode normalization, mutual-
  exclusion checks, latent-strategy invariants, and opponent-pool
  scrubbing, then returns the same ``cfg`` for chaining.

Reproducibility contract: this module is allowed to *coerce* fields that
the user could already have written incorrectly (e.g. legacy mode aliases,
mixed-case opponent tags) but it must never silently change a valid value
of a field that is part of the on-disk run-tag/checkpoint contract.
"""

from __future__ import annotations

import json
import os

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.custom_ppo.update.update_order import validate_actor_cf_update_mode


# Scripted tags dropped from training pools unless ``allow_op4_in_training_pool``
# (eval / zero-shot default).
EVAL_ONLY_TRAINING_OPPONENT_TAGS: frozenset[str] = frozenset({"OP4"})


def _normalize_train_mode(mode: str) -> str:
    """Map CLI / legacy mode aliases onto ``TrainMode`` string values."""
    raw = str(mode).upper().strip()
    aliases = {
        "FIXED": TrainMode.FIXED_OPPONENT.value,
        "FIXED_OPPONENT": TrainMode.FIXED_OPPONENT.value,
        "OPPONENT_POOL": TrainMode.OPPONENT_POOL.value,
        "POOL": TrainMode.OPPONENT_POOL.value,
        "OPPONENT_RANDOM": TrainMode.OPPONENT_POOL.value,
        "RANDOM_OPPONENT": TrainMode.OPPONENT_POOL.value,
        "PAPER": TrainMode.CURRICULUM.value,
        "NO_LEAGUE": TrainMode.CURRICULUM.value,
        "CURRICULUM": TrainMode.CURRICULUM.value,
        "CURRICULUM_NO_LEAGUE": TrainMode.CURRICULUM.value,
        "JACOB": TrainMode.CURRICULUM.value,
    }
    removed = {"LEAGUE", "CURRICULUM_LEAGUE", "SELF_PLAY"}
    if raw in removed:
        print(f"[PPO] Train mode {raw!r} is not in the local PPO audit path; using FIXED_OPPONENT.")
        return TrainMode.FIXED_OPPONENT.value
    return aliases.get(raw, raw)


def _normalize_opponent_pool_weights(cfg: PPOConfig) -> None:
    """Validate and normalize ``cfg.opponent_pool_weights`` to sum to 1.0.

    Empty weights ⇒ uniform sampling (no-op). Non-empty must have the same length
    as ``cfg.opponent_pool``; entries must be finite and non-negative with positive
    sum. The tuple is rewritten in-place after normalization so downstream code can
    pass it straight to ``np.random.Generator.choice(p=...)``.
    """
    raw = tuple(getattr(cfg, "opponent_pool_weights", ()) or ())
    if not raw:
        return
    pool = tuple(getattr(cfg, "opponent_pool", ()) or ())
    if len(raw) != len(pool):
        raise ValueError(
            f"opponent_pool_weights has length {len(raw)} but opponent_pool has length "
            f"{len(pool)} (pool={pool!r}, weights={raw!r}). Weights must be positionally "
            "aligned with the pool."
        )
    try:
        floats = tuple(float(w) for w in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"opponent_pool_weights entries must be numeric; got {raw!r}."
        ) from exc
    if any(not (w >= 0.0) for w in floats):  # rejects NaN and negatives
        raise ValueError(
            f"opponent_pool_weights entries must be finite and non-negative; got {floats!r}."
        )
    total = sum(floats)
    if total <= 0.0:
        raise ValueError(
            f"opponent_pool_weights sum must be > 0; got {floats!r} (sum={total})."
        )
    normalized = tuple(w / total for w in floats)
    if any(abs(a - b) > 1e-9 for a, b in zip(normalized, floats)):
        named = ", ".join(f"{tag}={w:.4f}" for tag, w in zip(pool, normalized))
        print(f"[PPO] opponent_pool_weights normalized to sum=1.0: {named}")
    cfg.opponent_pool_weights = normalized


def _strip_eval_only_opponents_from_training_pool(cfg: PPOConfig) -> None:
    """Remove eval-only scripted opponents from ``cfg.opponent_pool`` when training samples that pool."""
    if bool(getattr(cfg, "allow_op4_in_training_pool", False)):
        return
    pool = tuple(str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ()) if str(x).strip())
    banned = EVAL_ONLY_TRAINING_OPPONENT_TAGS
    filt = tuple(x for x in pool if x not in banned)
    if not filt:
        raise ValueError(
            "opponent_pool is empty after removing eval-only scripted tags "
            f"{sorted(banned)}. Use OP1–OP3, OP5–OP7 (and aliases OP5, OP6_TURTLE, OP7_SWITCHER) for training, or pass "
            "--allow-op4-in-training-pool together with OP4 in --opponent-pool."
        )
    removed = sorted(set(pool) - set(filt))
    if removed:
        print(
            "[PPO] opponent_pool excludes "
            f"{removed} (eval-only by default). "
            "Pass --allow-op4-in-training-pool to include those tags in training."
        )
        cfg.opponent_pool = filt


def _validate_v6i6_expansion_config(cfg: PPOConfig) -> None:
    if not bool(getattr(cfg, "use_v6i6_expansion", False)):
        return
    if str(getattr(cfg, "v6i6_expansion_stage", "")) != "E1":
        raise ValueError("v6i6_expansion_stage must be 'E1'; do not reuse Phase B for actor-side expansion.")
    if not bool(getattr(cfg, "v6i6_fixed_z_episode_attribution", False)):
        raise ValueError("v6i6 requires fixed-z episodes so outcome attribution is not mislabeled by mid-episode switching.")
    if int(getattr(cfg, "latent_resample_every_n", 0) or 0) != 0:
        raise ValueError("v6i6 requires latent_resample_every_n=0 for fixed-z episode attribution.")
    if not bool(getattr(cfg, "v6i6_use_reference_critic_for_opportunity", False)):
        raise ValueError("v6i6 opportunity weights must use a frozen reference critic.")
    if str(getattr(cfg, "v6i6_trainable_scope", "")) != "target_embedding_gate_adapter_only":
        raise ValueError("v6i6 trainable scope must be target_embedding_gate_adapter_only.")
    if bool(getattr(cfg, "v6i6_require_validated_anchors", True)):
        manifest_path = getattr(cfg, "v6i6_anchor_validation_manifest", None)
        if not manifest_path:
            raise ValueError(
                "v6i6 requires a validated anchor manifest before training. "
                "Run the forced-z and branch evaluations first, then pass the manifest path."
            )
        _load_v6i6_anchor_manifest(cfg, str(manifest_path))


def _load_v6i6_anchor_manifest(cfg: PPOConfig, manifest_path: str) -> None:
    if not os.path.isfile(manifest_path):
        raise ValueError(f"v6i6_anchor_validation_manifest does not exist: {manifest_path!r}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    verdict = str(manifest.get("verdict", "")).strip().upper()
    if verdict != "VALIDATED":
        raise ValueError(f"v6i6 anchor manifest verdict must be VALIDATED; got {verdict!r}.")

    latent_k = int(getattr(cfg, "latent_k", 4))
    anchors = tuple(int(z) for z in manifest.get("anchors", ()))
    target = int(manifest.get("expansion_target", -1))
    dormant = tuple(int(z) for z in manifest.get("dormant", ()))
    selected = anchors + (target,) + dormant
    if not anchors:
        raise ValueError("v6i6 anchor manifest must select at least one anchor latent.")
    if target < 0:
        raise ValueError("v6i6 anchor manifest must select expansion_target.")
    if any(z < 0 or z >= latent_k for z in selected):
        raise ValueError(f"v6i6 manifest latents must be in [0, {latent_k - 1}]; got {selected!r}.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"v6i6 manifest anchors/target/dormant must be disjoint; got {selected!r}.")

    cfg.v6i6_anchor_latents = anchors
    cfg.v6i6_target_latent = target
    cfg.v6i6_dormant_latents = dormant


def normalize_and_validate_training_config(cfg: PPOConfig) -> PPOConfig:
    """Normalize ``cfg.mode`` and reject inconsistent latent / opponent settings.

    Behaviour mirrors the original inline block at the top of
    :func:`rl.train_ppo.train_ppo` -- ``cfg`` is mutated in place and also
    returned so callers can chain ``cfg = normalize_and_validate_training_config(cfg)``.

    Order matters and matches the legacy block exactly:

    1. Mode alias canonicalization (``cfg.mode``).
    2. ``OPPONENT_POOL`` implies ``opponent_randomize=True``.
    3. Reject unsupported modes.
    4. ``CURRICULUM`` forces latent strategy off (Jacob paper baseline).
    5. ``opponent_randomize`` requires non-curriculum mode + a valid scripted pool;
       eval-only tags are stripped.
    6. Latent-strategy invariants (``latent_k`` >= 1, fixed-z bounds,
       ``latent_resample_every_n`` != 1).
    """
    cfg.mode = _normalize_train_mode(cfg.mode)
    if cfg.mode == TrainMode.OPPONENT_POOL.value:
        cfg.opponent_randomize = True
    supported_modes = {
        TrainMode.FIXED_OPPONENT.value,
        TrainMode.OPPONENT_POOL.value,
        TrainMode.CURRICULUM.value,
    }
    if cfg.mode not in supported_modes:
        raise ValueError(
            "The local PPO trainer currently supports FIXED_OPPONENT, OPPONENT_POOL, and "
            "CURRICULUM training."
        )
    if cfg.mode == TrainMode.CURRICULUM.value:
        if bool(getattr(cfg, "use_latent_strategy", False)) or bool(getattr(cfg, "fixed_latent_strategy", False)):
            print("[PPO] Curriculum mode is the Jacob paper baseline; forcing latent strategy OFF.")
        cfg.use_latent_strategy = False
        cfg.fixed_latent_strategy = False

    if bool(getattr(cfg, "opponent_randomize", False)):
        if cfg.mode == TrainMode.CURRICULUM.value:
            raise ValueError(
                "opponent_randomize=True is incompatible with CURRICULUM mode "
                "(curriculum already sequences scripted opponents). "
                "Use mode=FIXED_OPPONENT or OPPONENT_POOL with opponent_randomize, or turn opponent_randomize off."
            )
        pool = tuple(str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ()) if str(x).strip())
        allowed = {
            "OP1",
            "OP2",
            "OP3",
            "OP4",
            "OP5_RUSHER",
            "OP5",
            "OP6",
            "OP6_TURTLE",
            "OP7",
            "OP7_SWITCHER",
            "OP8",
            "OP8_INTERCEPTOR",
            "OP9",
            "OP9_FORTRESS",
            "OP10",
            "OP10_ESCORT",
        }
        pool = tuple(x for x in pool if x in allowed)
        if not pool:
            raise ValueError(f"opponent_pool must contain at least one of {sorted(allowed)}; got {getattr(cfg, 'opponent_pool', ())!r}.")
        cfg.opponent_pool = pool
        _strip_eval_only_opponents_from_training_pool(cfg)
        _normalize_opponent_pool_weights(cfg)

    if bool(getattr(cfg, "use_latent_strategy", False)):
        update_mode = str(getattr(cfg, "actor_cf_update_mode", "combined") or "combined")
        cfg.actor_cf_update_mode = validate_actor_cf_update_mode(update_mode)
        k = int(getattr(cfg, "latent_k", 4))
        if k < 1:
            raise ValueError("latent_k must be >= 1.")
        if bool(getattr(cfg, "fixed_latent_strategy", False)):
            fixed_id = int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0)
            if fixed_id < 0 or fixed_id >= k:
                raise ValueError(f"fixed_latent_strategy_id must be in [0, {k - 1}] when latent_k={k}.")
        res_n = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
        if res_n == 1:
            raise ValueError(
                "latent_resample_every_n=1 is disallowed (do not resample z every decision step). "
                "Use 0 (sample at episode start) or N>=2 (sparse refresh)."
            )
        _validate_v6i6_expansion_config(cfg)
    elif bool(getattr(cfg, "fixed_latent_strategy", False)):
        raise ValueError("fixed_latent_strategy requires use_latent_strategy=True.")

    return cfg


__all__ = [
    "EVAL_ONLY_TRAINING_OPPONENT_TAGS",
    "_normalize_opponent_pool_weights",
    "_normalize_train_mode",
    "_strip_eval_only_opponents_from_training_pool",
    "normalize_and_validate_training_config",
]
